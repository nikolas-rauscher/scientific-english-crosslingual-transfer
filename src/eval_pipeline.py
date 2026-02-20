#!/usr/bin/env python3
"""
Universal Evaluation Pipeline with Hydra

A flexible, configurable pipeline for evaluating language models on various benchmarks.
Uses Hydra for configuration management and automatically extracts metadata from checkpoints.

Usage:
    python src/eval_pipeline.py experiment=three_models
    python src/eval_pipeline.py experiment=checkpoint_progression
    python src/eval_pipeline.py models=[t5-base,path/to/checkpoint.ckpt] benchmarks=[mmlu]
"""

import os
import json
import re
import sys
import logging
import subprocess
import time
import csv
import threading
import queue
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import hydra
import rootutils
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def convert_dictconfig_to_dict(obj):
    """Recursively convert DictConfig objects to regular dicts for JSON serialization"""
    if isinstance(obj, DictConfig):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dictconfig_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_dictconfig_to_dict(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle dataclasses and other objects with __dict__
        try:
            return convert_dictconfig_to_dict(obj.__dict__)
        except:
            return str(obj)  # Fallback to string representation
    else:
        return obj


class CheckpointAnalyzer:
    """Analyzes checkpoints to extract metadata automatically"""
    
    @staticmethod
    def extract_step_and_ppl(checkpoint_path: Path) -> Tuple[Optional[int], Optional[float]]:
        """Extract step number and validation perplexity from checkpoint filename/path"""
        
        filename = checkpoint_path.name
        
        # Pattern for best checkpoints: step-262500-val_ppl-1.402.ckpt
        best_pattern = r'step-(\d+)-val_ppl-([0-9.]+)\.ckpt'
        match = re.search(best_pattern, filename)
        
        if match:
            step = int(match.group(1))
            ppl = float(match.group(2))
            return step, ppl
        
        # Pattern for step checkpoints: step-step=375000.ckpt
        step_pattern = r'step-step=(\d+)\.ckpt'
        match = re.search(step_pattern, filename)
        
        if match:
            step = int(match.group(1))
            return step, None
            
        # Pattern for epoch checkpoints: epoch-1-step-10000.ckpt
        epoch_step_pattern = r'epoch-\d+-step-(\d+)\.ckpt'
        match = re.search(epoch_step_pattern, filename)
        
        if match:
            step = int(match.group(1))
            return step, None
        
        return None, None
    
    @staticmethod
    def extract_run_info(checkpoint_path: Path) -> Dict[str, Any]:
        """Extract training run information from checkpoint path"""
        
        metadata = {}
        
        # Extract step and validation perplexity
        step, val_ppl = CheckpointAnalyzer.extract_step_and_ppl(checkpoint_path)
        if step:
            metadata['training_steps'] = step
        if val_ppl:
            metadata['val_perplexity'] = val_ppl
            
        # Extract run date from path (format: 2025-08-01_20-57-52)
        path_str = str(checkpoint_path)
        date_pattern = r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})'
        match = re.search(date_pattern, path_str)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            metadata['run_date'] = date_str
            metadata['run_time'] = time_str.replace('-', ':')
            
        # Detect training style from path
        if 'scifive' in path_str.lower():
            metadata['training_style'] = 'scifive'
            metadata['learning_rate'] = 0.001  # SciFive uses constant 0.001
        elif 'clean_restart' in path_str.lower():
            metadata['training_style'] = 'restart'
            
        # Extract run directory name
        run_dir_pattern = r'runs/([^/]+)'
        match = re.search(run_dir_pattern, path_str)
        if match:
            metadata['run_directory'] = match.group(1)
            
        return metadata
    
    @staticmethod
    def analyze_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """Comprehensive checkpoint analysis"""
        
        metadata = CheckpointAnalyzer.extract_run_info(checkpoint_path)
        
        # Try to load checkpoint for additional metadata
        try:
            if checkpoint_path.exists():
                # Load checkpoint header without full weights
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Extract hyperparameters if available
                if 'hyper_parameters' in checkpoint:
                    hp = checkpoint['hyper_parameters']
                    metadata['learning_rate'] = hp.get('lr', hp.get('learning_rate'))
                    metadata['batch_size'] = hp.get('batch_size')
                    metadata['model_name'] = hp.get('model_name')
                    
                # Extract training info
                if 'global_step' in checkpoint:
                    metadata['actual_global_step'] = checkpoint['global_step']
                if 'epoch' in checkpoint:
                    metadata['epoch'] = checkpoint['epoch']
                    
                # Extract Lightning version info
                if 'pytorch-lightning_version' in checkpoint:
                    metadata['lightning_version'] = checkpoint['pytorch-lightning_version']
                    
        except Exception as e:
            log.warning(f"Could not load checkpoint metadata from {checkpoint_path}: {e}")
            
        return metadata


class ModelManager:
    """Handles model loading, conversion, and management with automatic metadata extraction"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.root_dir)
        temp_root = self.project_root / "evaluation" / "models" / "temp"
        run_tokens = []
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            run_tokens.append(f"job{slurm_job_id}")
        try:
            hydra_run_dirname = Path(HydraConfig.get().runtime.output_dir).name
            hydra_run_dirname = re.sub(r"[^A-Za-z0-9._-]", "_", hydra_run_dirname)
            if hydra_run_dirname:
                run_tokens.append(hydra_run_dirname)
        except Exception:
            pass
        run_tokens.append(f"pid{os.getpid()}")
        run_tokens.append(datetime.now().strftime("%H%M%S_%f"))
        self.temp_dir = temp_root / "_".join(run_tokens)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Using isolated temp model directory: {self.temp_dir}")
        
        # Aggressive background conversion for performance optimization
        self._conversion_queue = queue.Queue()
        self._conversion_results = {}
        self._conversion_threads = []
        self._stop_conversion = threading.Event()
        self._conversion_cache = {}  # Cache for already converted models
        self._max_conversion_workers = min(12, max(4, int(os.cpu_count() * 0.75)))  # Use up to 75% of CPU cores for conversion
        
    def prepare_model(self, model_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Prepare model for evaluation and return path + enriched metadata"""
        # Use optimized conversion (cache + background conversion)
        return self.get_converted_model(model_config)
    
    def _detect_source_type(self, source_path: str) -> str:
        """Auto-detect source type from path"""
        
        if source_path.endswith('.ckpt'):
            return 'checkpoint'
        elif '/' in source_path and not source_path.startswith('/'):
            # Looks like HuggingFace model name
            return 'huggingface'
        elif Path(source_path).exists() or '/' in source_path:
            # Existing path or absolute path
            return 'converted_path'
        else:
            # Default to HuggingFace
            return 'huggingface'
    
    def _generate_model_name(self, model_config: Dict, metadata: Dict) -> str:
        """Generate descriptive model name from source path structure"""
        
        # Use explicit name if provided
        if 'name' in model_config and model_config['name']:
            return model_config['name']
        
        source_path = model_config['source_path']
        name_parts = []
        
        # Handle HuggingFace models
        if source_path == 't5-base':
            return "t5-base"
        elif source_path.startswith('t5-') or source_path.startswith('google/'):
            return source_path.replace('/', '_')
        
        # Extract meaningful parts from path structure
        path_parts = Path(source_path).parts
        
        # Look for key identifiers in path
        for part in path_parts:
            part_lower = part.lower()
            
            # Training type identifiers
            if 'scifive' in part_lower:
                name_parts.append('scifive_style')
            elif 'clean_restart' in part_lower:
                name_parts.append('clean_restart')
            elif 'pretrained' in part_lower:
                name_parts.append('pretrained')
            
            # Model base identifiers  
            if 't5-base-pretrained' in part_lower:
                if not any('t5' in existing for existing in name_parts):
                    name_parts.append('t5_base_pretrained')
            elif 't5' in part_lower and len(part_lower) > 2:  # Avoid matching just "t5"
                if not any('t5' in existing for existing in name_parts):
                    name_parts.append('t5_base')
        
        # If no specific identifiers found, use generic path-based name
        if not name_parts:
            # Use parent directory name
            if len(path_parts) > 1:
                parent_dir = path_parts[-2] if path_parts[-1].endswith('.ckpt') else path_parts[-1]
                if parent_dir not in ['checkpoints', 'best', 'steps']:
                    name_parts.append(parent_dir.replace('-', '_'))
        
        # Add training steps for disambiguation
        if 'training_steps' in metadata:
            steps = metadata['training_steps']
            if steps >= 1000:
                name_parts.append(f"{steps//1000}k")
            else:
                name_parts.append(f"{steps}")
        
        # Add validation perplexity if available (for best checkpoints)
        if 'val_perplexity' in metadata:
            ppl = metadata['val_perplexity']
            # Format: 1.397 → ppl1397, 1.40 → ppl140
            ppl_str = f"ppl{ppl:.3f}".replace('.', '')
            name_parts.append(ppl_str)
        
        # Fallback to timestamp if nothing found
        if not name_parts:
            name_parts.append(f"model_{int(time.time())}")
        
        return "_".join(name_parts)
    
    def _convert_checkpoint(
        self,
        ckpt_path: Path,
        output_path: Path,
        metadata: Dict,
        tokenizer_override: Optional[str] = None,
    ) -> str:
        """Convert Lightning checkpoint to HuggingFace format"""
        
        log.info(f"Converting checkpoint: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        state_dict = checkpoint['state_dict']
        
        # Remove nested Lightning prefixes ('model.model.' first, then fallback to 'model.')
        clean_state_dict = {}
        stripped_model_model = 0
        stripped_model = 0
        for key, value in state_dict.items():
            if key.startswith('model.model.'):
                clean_key = key[len('model.model.'):]
                clean_state_dict[clean_key] = value
                stripped_model_model += 1
            elif key.startswith('model.'):
                clean_key = key[len('model.'):]
                clean_state_dict[clean_key] = value
                stripped_model += 1
            else:
                clean_state_dict[key] = value
        if stripped_model_model or stripped_model:
            log.info(
                f"Stripped prefixes from state_dict -> model.model.: {stripped_model_model}, model.: {stripped_model}"
            )

        # Infer vocab size from weights if available
        ckpt_vocab_size = None
        try:
            # Standard HF key for shared embeddings
            if 'shared.weight' in clean_state_dict:
                ckpt_vocab_size = clean_state_dict['shared.weight'].shape[0]
            else:
                # Fallback: search any key ending with 'shared.weight'
                for k, v in clean_state_dict.items():
                    if k.endswith('shared.weight') and hasattr(v, 'shape') and len(v.shape) == 2:
                        ckpt_vocab_size = v.shape[0]
                        break
        except Exception:
            pass

        # Decide base model id (FLAN vs vanilla T5) based on path hints
        def _infer_base_model_id(p: Path) -> str:
            pl = str(p).lower()
            if 'flan' in pl:
                return 'google/flan-t5-base'
            return 't5-base'

        base_model_id = _infer_base_model_id(ckpt_path)
        log.info(f"Base model assumption for config/tokenizer: {base_model_id}")

        # Try to locate a tokenizer near the checkpoint path (parent dirs), else fallback to base model
        def _resolve_tokenizer_source(p: Path, fallback: str) -> Union[str, Path]:
            candidates = []
            # Check the run directory and a few parents for tokenizer files
            for cand in [p.parent, p.parent.parent if p.parent.parent else None, p.parent.parent.parent if p.parent.parent and p.parent.parent.parent else None]:
                if cand:
                    candidates.append(cand)
                    # also common subdir names
                    for sub in ['tokenizer', 'spm', 'sentencepiece', 'hf_tokenizer']:
                        candidates.append(cand / sub)
            token_files = {'tokenizer.json', 'spiece.model', 'sentencepiece.bpe.model', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json'}
            for c in candidates:
                try:
                    if c and c.exists() and c.is_dir():
                        files = {x.name for x in c.iterdir() if x.is_file()}
                        if files & token_files:
                            return c
                except Exception:
                    continue
            return fallback

        tokenizer_src: Union[str, Path] = _resolve_tokenizer_source(ckpt_path, base_model_id)
        if tokenizer_override:
            override_path = Path(tokenizer_override)
            if not override_path.is_absolute():
                override_path = self.project_root / override_path
            tokenizer_src = override_path if override_path.exists() else tokenizer_override
            log.info(f"Using tokenizer override for checkpoint conversion: {tokenizer_src}")
        def _load_auto_tokenizer(src: Union[str, Path], use_fast: bool) -> Any:
            kwargs: Dict[str, Any] = {
                "use_fast": use_fast,
                "trust_remote_code": True,
            }
            if isinstance(src, Path):
                kwargs["local_files_only"] = True
                return AutoTokenizer.from_pretrained(str(src), **kwargs)
            return AutoTokenizer.from_pretrained(str(src), **kwargs)

        try:
            tokenizer = _load_auto_tokenizer(tokenizer_src, use_fast=True)
            log.info(f"Tokenizer loaded ({tokenizer.__class__.__name__}) from: {tokenizer_src}")
        except Exception as fast_err:
            # If an explicit tokenizer path is provided, fail hard instead of silently masking the issue.
            if tokenizer_override:
                raise RuntimeError(
                    f"Explicit tokenizer_path failed to load with fast tokenizer: {tokenizer_src} | error: {fast_err}"
                ) from fast_err
            try:
                tokenizer = _load_auto_tokenizer(tokenizer_src, use_fast=False)
                log.info(f"Tokenizer loaded ({tokenizer.__class__.__name__}) from: {tokenizer_src}")
            except Exception as slow_err:
                log.warning(
                    "Falling back to t5-base tokenizer due to error loading %s "
                    "(fast=%s, slow=%s)",
                    tokenizer_src,
                    fast_err,
                    slow_err,
                )
                tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)

        # Build T5 config from the best available source (checkpoint metadata / tokenizer override / fallback)
        def _load_t5_config_from_candidate(candidate: Union[str, Path, None]) -> Optional[Tuple[T5Config, str]]:
            if candidate is None:
                return None

            # Local path candidates
            if isinstance(candidate, Path):
                path_candidate = candidate.expanduser()
                if not path_candidate.is_absolute():
                    path_candidate = self.project_root / path_candidate
                if path_candidate.is_dir() and (path_candidate / 'config.json').exists():
                    return T5Config.from_pretrained(path_candidate, local_files_only=True), str(path_candidate)
                return None

            cand = str(candidate).strip()
            if not cand:
                return None

            # Resolve possible local path (absolute or project-relative)
            local_path = Path(cand).expanduser()
            if local_path.is_absolute() or cand.startswith('.'):
                resolved_local = local_path if local_path.is_absolute() else (self.project_root / local_path)
                if resolved_local.exists():
                    if resolved_local.is_dir() and (resolved_local / 'config.json').exists():
                        return T5Config.from_pretrained(resolved_local, local_files_only=True), str(resolved_local)
                    # Local path exists but does not contain config.json -> not a valid config source.
                    return None
                return None

            resolved_local = (self.project_root / local_path).resolve()
            if resolved_local.exists():
                if resolved_local.is_dir() and (resolved_local / 'config.json').exists():
                    return T5Config.from_pretrained(resolved_local, local_files_only=True), str(resolved_local)
                # Path resolves locally but has no config; do not treat it as an HF id.
                return None

            # Hugging Face model id candidates
            looks_like_hf_id = ('/' in cand and not cand.startswith('/')) or cand.startswith('t5') or cand.startswith('google/')
            if looks_like_hf_id:
                try:
                    return T5Config.from_pretrained(cand), cand
                except Exception:
                    return None

            return None

        config_candidates: List[Union[str, Path]] = []
        for candidate in [
            metadata.get('base_model_id'),
            metadata.get('model_name'),
            tokenizer_override if tokenizer_override else None,
            tokenizer_src if isinstance(tokenizer_src, str) else None,
            base_model_id,
            't5-base',
        ]:
            if candidate and candidate not in config_candidates:
                config_candidates.append(candidate)

        config = None
        config_source = None
        for candidate in config_candidates:
            loaded = _load_t5_config_from_candidate(candidate)
            if loaded is not None:
                config, config_source = loaded
                break

        if config is None:
            config = T5Config.from_pretrained('t5-base')
            config_source = 't5-base'

        log.info(f"Using T5 config source for checkpoint conversion: {config_source}")

        # Infer T5 architecture hints from checkpoint tensor shapes to avoid shape mismatch on non-standard variants.
        def _infer_t5_arch_hints() -> Dict[str, Any]:
            hints: Dict[str, Any] = {}

            def _find_shape(keys: List[str]) -> Optional[Tuple[int, ...]]:
                for key in keys:
                    tensor = clean_state_dict.get(key)
                    if tensor is not None and hasattr(tensor, 'shape'):
                        return tuple(int(x) for x in tensor.shape)
                for existing_key, tensor in clean_state_dict.items():
                    if any(existing_key.endswith(key) for key in keys) and hasattr(tensor, 'shape'):
                        return tuple(int(x) for x in tensor.shape)
                return None

            shared_shape = _find_shape(['shared.weight'])
            if shared_shape and len(shared_shape) == 2:
                hints['vocab_size'] = shared_shape[0]
                hints['d_model'] = shared_shape[1]

            wo_shape = _find_shape(['encoder.block.0.layer.1.DenseReluDense.wo.weight'])
            if wo_shape and len(wo_shape) == 2:
                hints['d_ff'] = wo_shape[1]

            wi_shape = _find_shape(['encoder.block.0.layer.1.DenseReluDense.wi.weight'])
            if wi_shape and len(wi_shape) == 2:
                hints.setdefault('d_ff', wi_shape[0])

            rel_bias_shape = _find_shape(['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'])
            if rel_bias_shape and len(rel_bias_shape) == 2:
                hints['relative_attention_num_buckets'] = rel_bias_shape[0]
                hints['num_heads'] = rel_bias_shape[1]

            q_shape = _find_shape(['encoder.block.0.layer.0.SelfAttention.q.weight'])
            if q_shape and len(q_shape) == 2:
                q_out = q_shape[0]
                if 'num_heads' in hints and hints['num_heads'] > 0 and q_out % hints['num_heads'] == 0:
                    hints['d_kv'] = q_out // hints['num_heads']
                else:
                    # Fallback: guess a plausible (num_heads, d_kv) pair from q projection size.
                    for head_count in [64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]:
                        if q_out % head_count != 0:
                            continue
                        d_kv = q_out // head_count
                        if 8 <= d_kv <= 256:
                            hints['num_heads'] = head_count
                            hints['d_kv'] = d_kv
                            break

            encoder_blocks = set()
            decoder_blocks = set()
            for key in clean_state_dict.keys():
                enc_match = re.match(r'encoder\.block\.(\d+)\.', key)
                if enc_match:
                    encoder_blocks.add(int(enc_match.group(1)))
                dec_match = re.match(r'decoder\.block\.(\d+)\.', key)
                if dec_match:
                    decoder_blocks.add(int(dec_match.group(1)))

            if encoder_blocks:
                hints['num_layers'] = max(encoder_blocks) + 1
            if decoder_blocks:
                hints['num_decoder_layers'] = max(decoder_blocks) + 1

            if any('.DenseReluDense.wi_0.weight' in key for key in clean_state_dict):
                hints['feed_forward_proj'] = 'gated-gelu'
            elif any('.DenseReluDense.wi.weight' in key for key in clean_state_dict):
                hints.setdefault('feed_forward_proj', 'relu')

            return hints

        arch_hints = _infer_t5_arch_hints()
        hint_changes = []
        for attr, inferred_value in arch_hints.items():
            if not hasattr(config, attr):
                continue
            current_value = getattr(config, attr)
            if current_value != inferred_value:
                setattr(config, attr, inferred_value)
                hint_changes.append(f"{attr}: {current_value} -> {inferred_value}")
        if hint_changes:
            log.info(f"Adjusted T5 config from checkpoint shapes: {', '.join(hint_changes)}")

        # Align config vocab size with checkpoint weights if known
        if ckpt_vocab_size is not None and getattr(config, 'vocab_size', None) != ckpt_vocab_size:
            log.info(f"Adjusting config.vocab_size from {config.vocab_size} -> {ckpt_vocab_size} (from weights)")
            config.vocab_size = ckpt_vocab_size
        # Ensure special token ids are set from tokenizer
        try:
            if getattr(config, 'eos_token_id', None) is None and tokenizer.eos_token_id is not None:
                config.eos_token_id = tokenizer.eos_token_id
            if getattr(config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
                config.pad_token_id = tokenizer.pad_token_id
            if getattr(config, 'decoder_start_token_id', None) is None and tokenizer.pad_token_id is not None:
                # T5 commonly uses pad_token_id as decoder_start_token_id
                config.decoder_start_token_id = tokenizer.pad_token_id
        except Exception:
            pass
        
        # Create model and load state dict
        try:
            model = T5ForConditionalGeneration(config)
            load_res = model.load_state_dict(clean_state_dict, strict=False)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed loading checkpoint with config source '{config_source}' (hints={arch_hints}): {e}"
            ) from e
        # Log potential mapping issues to surface bad conversions early
        missing_cnt = len(getattr(load_res, 'missing_keys', []) or [])
        unexpected_cnt = len(getattr(load_res, 'unexpected_keys', []) or [])
        if missing_cnt or unexpected_cnt:
            log.warning(
                f"While loading checkpoint into HF model: missing_keys={missing_cnt}, unexpected_keys={unexpected_cnt}"
            )
        # Optional: warn if tokenizer vocab doesn't match weights
        try:
            emb_shape = tuple(model.get_input_embeddings().weight.shape)
            if tokenizer.vocab_size is not None and emb_shape[0] != tokenizer.vocab_size:
                log.warning(
                    f"Tokenizer vocab_size ({tokenizer.vocab_size}) != model embeddings ({emb_shape[0]})."
                    " Ensure you use the original training tokenizer to avoid generation issues."
                )
        except Exception:
            pass
        
        # Save in HuggingFace format
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save comprehensive metadata
        conversion_metadata = {
            'conversion_time': datetime.now().isoformat(),
            'original_checkpoint': str(ckpt_path),
            'extracted_metadata': metadata,
            'pipeline_temp_model': True
        }
        
        with open(output_path / 'conversion_metadata.json', 'w') as f:
            json.dump(conversion_metadata, f, indent=2)
            
        log.info(f"Converted model saved to: {output_path}")
        return str(output_path)
    
    def cleanup_temp_models(self):
        """Remove temporary converted models"""
        # Stop all background conversion threads
        if self._conversion_threads:
            self._stop_conversion.set()
            for thread in self._conversion_threads:
                if thread.is_alive():
                    thread.join(timeout=10)
            log.info(f"Stopped {len(self._conversion_threads)} conversion threads")
            
        if self.cfg.eval.get('cleanup_temp_models', True):
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                log.info(f"Cleaned up temporary models: {self.temp_dir}")

    def _format_model_args_extra(self, model_config: Optional[Dict[str, Any]]) -> str:
        """Convert optional per-model extra args to a deterministic model_args string for cache keys."""
        if not model_config:
            return ""
        extra = model_config.get("model_args")
        if extra is None:
            return ""

        if isinstance(extra, (DictConfig, ListConfig)):
            extra = OmegaConf.to_container(extra, resolve=True)

        if isinstance(extra, str):
            return extra.strip().strip(",")

        if isinstance(extra, dict):
            parts = []
            for key, value in extra.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                else:
                    value_str = str(value)
                parts.append(f"{key}={value_str}")
            return ",".join(parts)

        if isinstance(extra, (list, tuple)):
            return ",".join(str(item) for item in extra if str(item).strip())

        return str(extra)
    
    def _get_cache_key(self, model_config: Dict[str, Any]) -> str:
        """Generate unique cache key for model"""
        source_path = model_config.get('source_path', model_config.get('path', ''))
        tokenizer_path = model_config.get('tokenizer_path')
        extra_model_args = self._format_model_args_extra(model_config)

        key_payload: Dict[str, Any] = {
            "source_path": source_path,
            "tokenizer_path": tokenizer_path,
            "model_args": extra_model_args,
        }

        # For checkpoints, include file metadata so key changes when checkpoint updates.
        if isinstance(source_path, str) and source_path.endswith('.ckpt'):
            full_path = Path(source_path)
            if not full_path.is_absolute():
                full_path = self.project_root / full_path
            if full_path.exists():
                stat = full_path.stat()
                key_payload["source_abs"] = str(full_path.resolve())
                key_payload["source_mtime"] = stat.st_mtime
                key_payload["source_size"] = stat.st_size

        # If tokenizer path is local, include metadata too (important for same checkpoint + different tokenizer).
        if tokenizer_path:
            tok_path = Path(str(tokenizer_path))
            if not tok_path.is_absolute():
                tok_path = self.project_root / tok_path
            if tok_path.exists():
                try:
                    stat = tok_path.stat()
                    key_payload["tokenizer_abs"] = str(tok_path.resolve())
                    key_payload["tokenizer_mtime"] = stat.st_mtime
                    key_payload["tokenizer_size"] = stat.st_size
                except Exception:
                    pass

        key_string = json.dumps(key_payload, sort_keys=True, ensure_ascii=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_conversion_cache(self, model_config: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check if model is already converted and cached"""
        cache_key = self._get_cache_key(model_config)
        
        if cache_key in self._conversion_cache:
            cached_path, cached_metadata = self._conversion_cache[cache_key]
            # Verify cached model still exists
            if Path(cached_path).exists():
                log.info(f"Using cached conversion: {cached_path}")
                return cached_path, cached_metadata
            else:
                # Remove invalid cache entry
                del self._conversion_cache[cache_key]
        
        return None
    
    def _background_conversion_worker(self):
        """Background thread worker for model conversion"""
        while not self._stop_conversion.is_set():
            try:
                # Wait for model to convert (with timeout)
                model_config = self._conversion_queue.get(timeout=1.0)
                if model_config is None:  # Poison pill to stop
                    break
                
                cache_key = self._get_cache_key(model_config)
                log.info(f"Background converting: {model_config.get('name', 'unnamed')}")
                
                try:
                    # Perform actual conversion
                    model_path, metadata = self._convert_model_internal(model_config)
                    
                    # Cache result
                    self._conversion_cache[cache_key] = (model_path, metadata)
                    self._conversion_results[cache_key] = (model_path, metadata)
                    
                    log.info(f"Background conversion completed: {model_config.get('name', 'unnamed')}")
                    
                except Exception as e:
                    log.error(f"Background conversion failed: {e}")
                    self._conversion_results[cache_key] = None
                
                self._conversion_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check stop flag and continue
            except Exception as e:
                log.error(f"Background conversion worker error: {e}")
    
    def start_background_conversion(self):
        """Start multiple background conversion threads"""
        if not self._conversion_threads or not any(t.is_alive() for t in self._conversion_threads):
            self._stop_conversion.clear()
            self._conversion_threads = []
            
            for i in range(self._max_conversion_workers):
                thread = threading.Thread(
                    target=self._background_conversion_worker, 
                    daemon=True,
                    name=f"ConversionWorker-{i+1}"
                )
                thread.start()
                self._conversion_threads.append(thread)
            
            log.info(f"Started {self._max_conversion_workers} background conversion threads (CPU cores: {os.cpu_count()})")
    
    def queue_model_for_conversion(self, model_config: Dict[str, Any]):
        """Queue model for background conversion"""
        # Only queue if not already cached
        if self._check_conversion_cache(model_config) is None:
            self._conversion_queue.put(model_config)
            log.info(f"Queued for background conversion: {model_config.get('name', 'unnamed')}")
    
    def get_converted_model(self, model_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Get converted model, either from cache, background conversion, or convert now"""
        cache_key = self._get_cache_key(model_config)
        
        # Check cache first
        cached_result = self._check_conversion_cache(model_config)
        if cached_result:
            return cached_result
        
        # Check if background conversion completed
        if cache_key in self._conversion_results:
            result = self._conversion_results[cache_key]
            if result is not None:
                log.info(f"Using background-converted model: {model_config.get('name', 'unnamed')}")
                return result
            else:
                log.warning(f"Background conversion failed, converting synchronously: {model_config.get('name', 'unnamed')}")
        
        # Fall back to synchronous conversion
        log.info(f"Converting synchronously: {model_config.get('name', 'unnamed')}")
        model_path, metadata = self._convert_model_internal(model_config)
        
        # Cache result
        self._conversion_cache[cache_key] = (model_path, metadata)
        return model_path, metadata
    
    def _convert_model_internal(self, model_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Internal method for actual model conversion (called by both sync and async paths)"""
        source_type = model_config.get('source_type', 'auto')
        source_path = model_config.get('source_path', model_config.get('path'))
        
        if not source_path:
            raise KeyError("Model config must include 'source_path' (or legacy alias 'path')")
        
        # Auto-detect source type if not specified
        if source_type == 'auto':
            source_type = self._detect_source_type(source_path)
            
        # Prepare metadata
        metadata = model_config.get('metadata', {})
        
        if source_type == 'huggingface':
            # Use HuggingFace model directly
            model_path = source_path
            metadata.update({
                'source_type': 'huggingface',
                'original_source': source_path
            })
            
        elif source_type == 'converted_path':
            # Use existing converted model
            path = Path(source_path)
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise FileNotFoundError(f"Converted model not found: {path}")
            
            # Handle common local layouts
            # Case 1: HF layout at root (config.json present) -> use as-is
            # Case 2: Split layout with model/ and tokenizer/ subfolders -> merge into a temp HF folder
            model_path = None
            if (path / 'config.json').exists():
                # Already HF-compliant at root
                model_path = str(path)
                metadata.update({'layout': 'hf_root'})
            elif (path / 'model' / 'config.json').exists():
                # Merge into temporary directory
                import shutil, tempfile
                safe_name = hashlib.md5(str(path).encode()).hexdigest()[:10]
                merged_dir = self.temp_dir / f"merged_{safe_name}"
                merged_dir.mkdir(parents=True, exist_ok=True)

                model_dir = path / 'model'
                tokenizer_dir = path / 'tokenizer'

                # Copy all files from model_dir into merged root
                for item in model_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, merged_dir / item.name)

                # Copy tokenizer files if present
                if tokenizer_dir.exists() and tokenizer_dir.is_dir():
                    for item in tokenizer_dir.iterdir():
                        if item.is_file():
                            # avoid overwriting files with same name from model_dir
                            dest = merged_dir / item.name
                            if not dest.exists():
                                shutil.copy2(item, dest)

                # Write small marker for debugging
                with open(merged_dir / 'LOCAL_LAYOUT_MERGED', 'w') as f:
                    f.write(str(path))

                model_path = str(merged_dir)
                metadata.update({'layout': 'split_model_tokenizer_merged', 'merged_from': str(path)})
            else:
                # Unknown layout; keep path (lm-eval may still handle it)
                model_path = str(path)
                metadata.update({'layout': 'unknown'})

            metadata.update({
                'source_type': 'converted_path',
                'original_source': source_path
            })
            
        elif source_type == 'checkpoint':
            # Convert checkpoint and extract metadata
            ckpt_path = Path(source_path)
            if not ckpt_path.is_absolute():
                ckpt_path = self.project_root / ckpt_path
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                
            # Extract metadata from checkpoint
            checkpoint_metadata = CheckpointAnalyzer.analyze_checkpoint(ckpt_path)
            metadata.update(checkpoint_metadata)
            metadata.update({
                'source_type': 'checkpoint',
                'original_checkpoint': str(ckpt_path)
            })
            
            # Generate model name based on metadata
            model_name = self._generate_model_name(model_config, checkpoint_metadata)
            
            # Convert checkpoint
            output_path = self.temp_dir / model_name
            tokenizer_override = model_config.get("tokenizer_path")
            if not tokenizer_override:
                raise ValueError(
                    "Checkpoint evaluation requires an explicit tokenizer_path. "
                    "Auto tokenizer resolution is disabled to avoid distorted comparisons. "
                    f"Model='{model_name}', checkpoint='{ckpt_path}'"
                )
            model_path = self._convert_checkpoint(
                ckpt_path,
                output_path,
                metadata,
                tokenizer_override=tokenizer_override,
            )
            
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
            
        return model_path, metadata


class BenchmarkRunner:
    """Handles benchmark execution"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.root_dir)
        self.setup_environment()
        
    def setup_environment(self):
        """Setup environment variables for cached datasets"""
        data_dir = self.project_root / self.cfg.paths.data_dir
        mmlu_cache = data_dir / "mmlu_datasets"
        hf_cache = data_dir / "huggingface_cache"
        
        os.environ['HF_DATASETS_CACHE'] = str(mmlu_cache)
        os.environ['HF_HOME'] = str(hf_cache)
        os.environ['HF_HUB_CACHE'] = str(hf_cache / "hub")
        os.environ['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        # Determinism-related environment variables for sub-process
        try:
            seed_value = self.cfg.get('seed', 42)
        except Exception:
            seed_value = 42
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
        # Avoid tokenizer thread races changing order
        os.environ['TOKENIZERS_PARALLELISM'] = os.environ.get('TOKENIZERS_PARALLELISM', 'false')
        
        log.info(f"Using MMLU datasets cache: {mmlu_cache}")
        log.info(f"Using HuggingFace cache: {hf_cache}")
    
    def run_benchmark(
        self,
        model_path: str,
        model_name: str,
        benchmark_config: Dict,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Run a benchmark evaluation on a model"""
        
        results = {}
        # Use Hydra's automatic output directory
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_base = hydra_output_dir / "evaluation/results/universal" / self.cfg.experiment_name
        output_base.mkdir(parents=True, exist_ok=True)

        # Create friendlier convenience link grouped by date only
        try:
            # Derive date/time from hydra run dir name if possible
            stamp = hydra_output_dir.name
            m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", stamp)
            if m:
                date_str, time_str = m.group(1), m.group(2)
            else:
                now = datetime.now()
                date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")

            project_root = Path(self.cfg.paths.root_dir)

            # by_date: evaluation/results/by_date/YYYY-MM-DD/HH-MM-SS/<experiment_name> -> output_base
            by_date_link = project_root / "evaluation" / "results" / "by_date" / date_str / time_str / self.cfg.experiment_name
            by_date_link.parent.mkdir(parents=True, exist_ok=True)
            if not by_date_link.exists():
                os.symlink(output_base, by_date_link, target_is_directory=True)
                log.info(f"📁 Linked results (by_date): {by_date_link}")

            # Note: We intentionally do NOT create by_experiment links to keep layout minimal
        except Exception as e:
            log.warning(f"Could not create convenience result links: {e}")
        
        shots = benchmark_config.get('shots', [0, 5])
        
        for shot in shots:
            log.info(f"Running {benchmark_config['name']} {shot}-shot evaluation for {model_name}")
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_base / f"{model_name}_{benchmark_config['name']}_{shot}shot_{timestamp}"
            
            # Build command
            cmd = self._build_evaluation_command(
                model_path, model_name, benchmark_config, shot, output_dir, model_config
            )
            
            # Run evaluation
            start_time = time.time()
            try:
                # Check if wandb_args is present - if so, use shell=True for proper quoting
                if "--wandb_args" in cmd:
                    # Find the wandb_args value and quote it properly
                    wandb_idx = cmd.index("--wandb_args")
                    wandb_value = cmd[wandb_idx + 1]
                    # Reconstruct command as string with quoted wandb_args
                    cmd_parts = cmd[:wandb_idx] + ["--wandb_args", f'"{wandb_value}"'] + cmd[wandb_idx + 2:]
                    cmd_string = " ".join(cmd_parts)
                    log.info(f"Executing (shell=True): {cmd_string}")
                    
                    # Prepare isolated env to avoid distributed port collisions across parallel evals
                    local_env = os.environ.copy()
                    local_env.setdefault('MASTER_ADDR', '127.0.0.1')
                    # Find an ephemeral free port for this eval process group
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('127.0.0.1', 0))
                            free_port = s.getsockname()[1]
                    except Exception:
                        free_port = 29500
                    local_env['MASTER_PORT'] = str(free_port)
                    local_env['RANK'] = '0'
                    local_env['WORLD_SIZE'] = '1'
                    local_env['LOCAL_RANK'] = '0'

                    result = subprocess.run(
                        cmd_string,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        check=True,
                        shell=True,
                        env=local_env
                    )
                    
                    # lm-eval handles W&B integration natively
                else:
                    # Use normal list-based execution for non-wandb commands
                    log.info(f"Executing: {' '.join(cmd)}")
                    
                    # Prepare isolated env to avoid distributed port collisions across parallel evals
                    local_env = os.environ.copy()
                    local_env.setdefault('MASTER_ADDR', '127.0.0.1')
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('127.0.0.1', 0))
                            free_port = s.getsockname()[1]
                    except Exception:
                        free_port = 29500
                    local_env['MASTER_PORT'] = str(free_port)
                    local_env['RANK'] = '0'
                    local_env['WORLD_SIZE'] = '1'
                    local_env['LOCAL_RANK'] = '0'

                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        check=True,
                        env=local_env
                    )
                
                duration = time.time() - start_time
                
                # Parse results
                eval_results = self._parse_results(output_dir, benchmark_config)
                eval_results.update({
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60,
                    'command': ' '.join(cmd),
                    'output_dir': str(output_dir),
                    'status': 'success'
                })
                
                results[f"{shot}_shot"] = eval_results
                
                log.info(f"Completed {shot}-shot in {duration/60:.1f} minutes")
                
            except subprocess.CalledProcessError as e:
                log.error(f"Evaluation failed: {e}")
                log.error(f"Command: {' '.join(cmd)}")
                log.error(f"STDOUT: {e.stdout}")
                log.error(f"STDERR: {e.stderr}")
                results[f"{shot}_shot"] = {
                    'status': 'failed',
                    'error': str(e),
                    'stdout': e.stdout,
                    'stderr': e.stderr,
                    'command': ' '.join(cmd)
                }
                
        return results
    
    def _format_model_args_extra(self, model_config: Optional[Dict[str, Any]]) -> str:
        """Convert optional per-model extra args to an lm-eval model_args string."""
        if not model_config:
            return ""
        extra = model_config.get("model_args")
        if extra is None:
            return ""

        if isinstance(extra, (DictConfig, ListConfig)):
            extra = OmegaConf.to_container(extra, resolve=True)

        if isinstance(extra, str):
            return extra.strip().strip(",")

        if isinstance(extra, dict):
            parts = []
            for key, value in extra.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                else:
                    value_str = str(value)
                parts.append(f"{key}={value_str}")
            return ",".join(parts)

        if isinstance(extra, (list, tuple)):
            return ",".join(str(item) for item in extra if str(item).strip())

        log.warning(f"Unsupported model_args type: {type(extra)} (ignored)")
        return ""

    def _build_evaluation_command(
        self,
        model_path: str,
        model_name: str,
        benchmark_config: Dict,
        shots: int,
        output_dir: Path,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Build the evaluation command for lm-eval with native W&B integration"""
        
        # Check if model_path is a local filesystem path (directory or file)
        # Important: Hugging Face model IDs also contain '/', so do NOT use that heuristic.
        local_path = Path(model_path).expanduser()
        is_local_path = local_path.exists()

        if is_local_path:
            # For local paths, build robust hf args:
            # - handle layouts where model/ and tokenizer/ live in separate subfolders
            # - prefer using local files only to avoid accidental network calls
            extra_kv = []

            # Detect common folder structure variants
            subfolder = None
            tokenizer_dir = None
            # We may rewrite the model_path to directly point to the folder that contains config.json
            # to avoid passing a `subfolder` argument altogether (lm-eval shares subfolder for tokenizer too).
            rewritten_pretrained = None

            if local_path.is_dir():
                # Case A: top-level contains config.json -> use as-is
                if (local_path / "config.json").exists():
                    # Try to detect tokenizer colocated
                    if (local_path / "tokenizer.json").exists() or (local_path / "tokenizer_config.json").exists():
                        tokenizer_dir = str(local_path)
                    # Else: check sibling tokenizer folder
                    elif (local_path.parent / "tokenizer" / "tokenizer.json").exists() or (
                        local_path.parent / "tokenizer" / "tokenizer_config.json"
                    ).exists():
                        tokenizer_dir = str(local_path.parent / "tokenizer")

                # Case B: config lives in subfolder `model/`
                elif (local_path / "model" / "config.json").exists():
                    # Prefer rewriting `pretrained` to the actual model subdir so we don't need `subfolder`
                    rewritten_pretrained = str(local_path / "model")
                    # Tokenizer folder at sibling `tokenizer/` if present
                    if (local_path / "tokenizer" / "tokenizer.json").exists() or (
                        local_path / "tokenizer" / "tokenizer_config.json"
                    ).exists():
                        tokenizer_dir = str(local_path / "tokenizer")
                # Case C: we were pointed directly at the `model/` subdir
                elif local_path.name == "model" and (local_path / "config.json").exists():
                    # Look for sibling tokenizer directory one level up
                    if (local_path.parent / "tokenizer" / "tokenizer.json").exists() or (
                        local_path.parent / "tokenizer" / "tokenizer_config.json"
                    ).exists():
                        tokenizer_dir = str(local_path.parent / "tokenizer")

            # Compose model_args
            resolved_pretrained = rewritten_pretrained if rewritten_pretrained else model_path
            parts = [f"pretrained={resolved_pretrained}", "trust_remote_code=True", "local_files_only=True"]
            # IMPORTANT: Do NOT pass `subfolder` when we also pass a custom `tokenizer` path.
            # lm-eval forwards the same subfolder to tokenizer loading which breaks non-mirrored layouts.
            if subfolder and not tokenizer_dir:
                parts.append(f"subfolder={subfolder}")
            if tokenizer_dir and tokenizer_dir != resolved_pretrained:
                parts.append(f"tokenizer={tokenizer_dir}")
            model_args = ",".join(parts)
        else:
            # For HuggingFace model names, use standard format
            model_args = f"pretrained={model_path}"

        # Append optional per-model args from config
        extra_model_args = self._format_model_args_extra(model_config)
        if extra_model_args:
            model_args = f"{model_args},{extra_model_args}"
        
        # Use FLAN T5 template with loglikelihood for MMLU instead of default generative template
        task_name = benchmark_config['name']
        if task_name == 'mmlu':
            task_name = 'mmlu_flan_n_shot_loglikelihood'
        
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task_name,
            "--device", str(benchmark_config.get('device', 'cuda')),
            "--batch_size", str(benchmark_config.get('batch_size', 'auto')),
            "--output_path", str(output_dir),
            "--num_fewshot", str(shots),
            "--trust_remote_code"
        ]
        
        # Add --write_out flag for diagnostic purposes if enabled
        # Priority: benchmark-specific > global diagnostics config > default (False)
        write_out_enabled = benchmark_config.get('write_out', 
                                                self.cfg.get('diagnostics', {}).get('write_out', False))
        if write_out_enabled:
            cmd.append("--write_out")
            log.info(f"📝 Enabled write_out diagnostics for {benchmark_config['name']} - will show prompts and targets")
        
        # Ensure a deterministic seed is always set (priority: benchmark_config.seed > cfg.seed > 42)
        try:
            cfg_seed = self.cfg.get('seed', None)
        except Exception:
            cfg_seed = None
        seed_value = benchmark_config.get('seed', cfg_seed if cfg_seed is not None else 42)
        cmd.extend(["--seed", str(seed_value)])
        
        # Add specific tasks if configured
        if 'tasks' in benchmark_config:
            cmd[cmd.index("--tasks") + 1] = ",".join(benchmark_config['tasks'])
        
        # ✅ NATIVE W&B INTEGRATION - Syntax validated and working!
        if self.cfg.logger.get('wandb', {}).get('project'):
            # Use the actual model name from config for W&B run naming
            model_name_for_wandb = model_name
            
            # Build W&B args using the validated comma-separated syntax
            wandb_args_parts = [
                f"project={self.cfg.logger.wandb.project}",
            ]
            
            # Add entity if configured
            if self.cfg.logger.wandb.get('entity'):
                wandb_args_parts.append(f"entity={self.cfg.logger.wandb.entity}")
                
            # Add group if configured
            if self.cfg.logger.wandb.get('group'):
                wandb_args_parts.append(f"group={self.cfg.logger.wandb.group}")
                
            # Add descriptive run name with model and benchmark info
            run_name = f"{model_name_for_wandb}_{benchmark_config['name']}_{shots}shot"
            wandb_args_parts.append(f"name={run_name}")
            
            # Skip tags in wandb_args - they cause parsing issues
            # We'll add better W&B organization after the evaluation completes
            
            # Combine into single comma-separated string
            wandb_args_string = ",".join(wandb_args_parts)
            
            # Add wandb_args without quotes (subprocess handles arguments correctly)
            cmd.extend(["--wandb_args", wandb_args_string])
            
            log.info(f"🚀 Using native W&B integration: {wandb_args_string}")
        else:
            log.info("📊 W&B logging disabled (no project configured)")
            
        # Add extra arguments
        extra_args = benchmark_config.get('extra_args', {})
        for key, value in extra_args.items():
            cmd.extend([f"--{key}", str(value)])
            
        return cmd
    
    def _parse_results(self, output_dir: Path, benchmark_config: Dict) -> Dict:
        """Parse evaluation results from output directory"""
        
        # lm-eval creates subdirectory with model name, so search for results file
        results_file = None
        
        # First try direct path
        direct_results = output_dir / "results.json"
        if direct_results.exists():
            results_file = direct_results
        else:
            # Search in subdirectories for results_*.json files
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    for json_file in subdir.glob("results_*.json"):
                        results_file = json_file
                        break
                    if results_file:
                        break
        
        if not results_file:
            return {'parse_status': 'failed', 'error': 'Results file not found'}
            
        try:
            with open(results_file) as f:
                data = json.load(f)
                
            # Extract results based on benchmark type
            if benchmark_config['name'] == 'mmlu':
                return self._parse_mmlu_results(data)
            else:
                # Generic parsing for other benchmarks
                return self._parse_generic_results(data, benchmark_config['name'])
                
        except Exception as e:
            return {'parse_status': 'failed', 'error': f'Failed to parse results: {e}'}
    
    def _parse_mmlu_results(self, data: Dict) -> Dict:
        """Parse MMLU-specific results (handles both full MMLU and subsets)"""
        try:
            results = data['results']
            
            # Check if we have the full MMLU aggregated results
            if 'mmlu' in results:
                # Full MMLU evaluation
                return {
                    'parse_status': 'success',
                    'overall_accuracy': results['mmlu']['acc,none'],
                    'overall_stderr': results['mmlu']['acc_stderr,none'],
                    'humanities': results.get('mmlu_humanities', {}).get('acc,none', 'N/A'),
                    'social_sciences': results.get('mmlu_social_sciences', {}).get('acc,none', 'N/A'),
                    'stem': results.get('mmlu_stem', {}).get('acc,none', 'N/A'),
                    'other': results.get('mmlu_other', {}).get('acc,none', 'N/A'),
                    'raw_results': data
                }
            else:
                # MMLU subset evaluation - calculate average across available tasks
                mmlu_tasks = {k: v for k, v in results.items() if k.startswith('mmlu_')}
                
                if not mmlu_tasks:
                    return {
                        'parse_status': 'failed',
                        'error': 'No MMLU tasks found in results',
                        'raw_results': data
                    }
                
                # Calculate average accuracy across subset tasks
                accuracies = []
                task_results = {}
                
                for task_name, task_result in mmlu_tasks.items():
                    if 'acc,none' in task_result:
                        accuracy = task_result['acc,none']
                        accuracies.append(accuracy)
                        task_results[task_name] = accuracy
                
                if not accuracies:
                    return {
                        'parse_status': 'failed',
                        'error': 'No accuracy metrics found in MMLU tasks',
                        'raw_results': data
                    }
                
                avg_accuracy = sum(accuracies) / len(accuracies)
                
                return {
                    'parse_status': 'success',
                    'overall_accuracy': avg_accuracy,
                    'overall_stderr': 'N/A',  # Can't calculate stderr for subset
                    'humanities': 'N/A',
                    'social_sciences': 'N/A', 
                    'stem': 'N/A',
                    'other': 'N/A',
                    'subset_tasks': task_results,
                    'num_tasks': len(accuracies),
                    'raw_results': data
                }
                
        except KeyError as e:
            return {
                'parse_status': 'failed', 
                'error': f'Missing key in MMLU results: {e}',
                'raw_results': data
            }
        except Exception as e:
            return {
                'parse_status': 'failed',
                'error': f'Error parsing MMLU results: {e}',
                'raw_results': data
            }
    
    def _parse_generic_results(self, data: Dict, benchmark_name: str) -> Dict:
        """Parse generic benchmark results"""
        try:
            results = data['results']
            
            # Find the main metric for this benchmark
            if benchmark_name in results:
                main_result = results[benchmark_name]
                # Common metric names
                for metric in ['acc,none', 'acc_norm,none', 'exact_match,none']:
                    if metric in main_result:
                        return {
                            'parse_status': 'success',
                            'primary_metric': metric,
                            'primary_score': main_result[metric],
                            'raw_results': data
                        }
                # Perplexity-style metrics (prefer word_perplexity -> byte_perplexity -> bits_per_byte)
                for metric in ['word_perplexity,none', 'byte_perplexity,none', 'bits_per_byte,none']:
                    if metric in main_result:
                        return {
                            'parse_status': 'success',
                            'primary_metric': metric,
                            'primary_score': main_result[metric],
                            'raw_results': data
                        }
                # Fallback: pick first metric containing a known key
                for key in main_result.keys():
                    if any(k in key for k in ['word_perplexity', 'byte_perplexity', 'bits_per_byte']):
                        return {
                            'parse_status': 'success',
                            'primary_metric': key,
                            'primary_score': main_result[key],
                            'raw_results': data
                        }
            
            return {
                'parse_status': 'partial',
                'error': 'Could not find primary metric',
                'raw_results': data
            }
            
        except Exception as e:
            return {
                'parse_status': 'failed',
                'error': f'Failed to parse {benchmark_name} results: {e}',
                'raw_results': data
            }


class WandBLogger:
    """W&B logging coordinator - works with native lm-eval integration"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run = None
        self.native_integration_active = False
        
    def start_experiment(self):
        """Initialize W&B experiment with native lm-eval integration"""
        
        if not self.cfg.logger.get('wandb', {}).get('project'):
            log.info("W&B logging disabled (no project configured)")
            return
        
        log.info("🚀 Using native lm-eval W&B integration")
        self.native_integration_active = True
        
    def log_model_results(self, model_name: str, model_metadata: Dict, 
                         benchmark_name: str, results: Dict):
        """Log results - enhanced metadata and plots with native integration"""
        
        # If using native integration, only log additional metadata
        if self.native_integration_active:
            # Try to get the active run from lm-eval
            if not self.run:
                self.run = wandb.run
                
            if self.run and model_metadata:
                # Log additional metadata that lm-eval doesn't capture
                metadata_metrics = {}
                
                if 'training_steps' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/training_steps"] = model_metadata['training_steps']
                if 'val_perplexity' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/val_perplexity"] = model_metadata['val_perplexity']
                if 'source_type' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/source_type"] = model_metadata['source_type']
                if 'training_style' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/training_style"] = model_metadata['training_style']
                if 'run_date' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/run_date"] = model_metadata['run_date']
                if 'learning_rate' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/learning_rate"] = model_metadata['learning_rate']
                if 'run_directory' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/run_directory"] = model_metadata['run_directory']
                
                # Add performance enhancements and create plots
                for shot_key, shot_results in results.items():
                    if shot_results.get('status') == 'success' and 'duration_minutes' in shot_results:
                        shot_num = shot_key.replace('_shot', '')
                        metadata_metrics[f"{model_name}/{benchmark_name}/{shot_num}shot/duration_min"] = shot_results['duration_minutes']
                        
                        # Add improvement over random baseline for MMLU
                        if benchmark_name == 'mmlu' and 'overall_accuracy' in shot_results:
                            random_baseline = 0.25  # 25% for 4-choice questions
                            improvement = shot_results['overall_accuracy'] - random_baseline
                            metadata_metrics[f"{model_name}/{benchmark_name}/{shot_num}shot/improvement_over_random"] = improvement
                        
                        # 🎨 CREATE W&B PLOTS for MMLU subtasks/categories
                        if benchmark_name == 'mmlu':
                            self._create_mmlu_plots(model_name, shot_num, shot_results)
                
                if metadata_metrics:
                    log.info(f"📊 Logging {len(metadata_metrics)} enhanced metadata fields to W&B for {model_name}")
                    self.run.log(metadata_metrics)
                else:
                    log.info(f"ℹ️ No additional metadata to log for {model_name} (using native lm-eval integration)")
    
    def _create_mmlu_plots(self, model_name: str, shot_num: str, shot_results: Dict):
        """Create W&B plots for MMLU results"""
        
        if not self.run:
            return
            
        try:
            # 📊 MMLU Categories Bar Chart
            categories = ['humanities', 'social_sciences', 'stem', 'other']
            category_data = []
            category_labels = []
            
            for category in categories:
                value = shot_results.get(category)
                if value is not None and value != "N/A" and isinstance(value, (int, float)):
                    category_data.append(value)
                    category_labels.append(category.replace('_', ' ').title())
            
            if category_data:
                # Create bar chart data for W&B
                data = [[label, value] for label, value in zip(category_labels, category_data)]
                table = wandb.Table(data=data, columns=["Category", "Accuracy"])
                
                bar_chart = wandb.plot.bar(
                    table, "Category", "Accuracy",
                    title=f"{model_name} - MMLU Categories ({shot_num}-shot)"
                )
                
                self.run.log({f"{model_name}/plots/mmlu_categories_{shot_num}shot": bar_chart})
                log.info(f"📊 Created MMLU categories plot for {model_name} ({shot_num}-shot)")
            
            # 📊 MMLU Subtasks Bar Chart (if available)
            if 'subset_tasks' in shot_results:
                subtask_data = []
                for task_name, accuracy in shot_results['subset_tasks'].items():
                    clean_name = task_name.replace('mmlu_', '').replace('_', ' ').title()
                    subtask_data.append([clean_name, accuracy])
                
                if subtask_data:
                    # Sort by accuracy for better visualization
                    subtask_data.sort(key=lambda x: x[1], reverse=True)
                    
                    table = wandb.Table(data=subtask_data, columns=["Task", "Accuracy"])
                    
                    bar_chart = wandb.plot.bar(
                        table, "Task", "Accuracy", 
                        title=f"{model_name} - MMLU Subtasks ({shot_num}-shot)"
                    )
                    
                    self.run.log({f"{model_name}/plots/mmlu_subtasks_{shot_num}shot": bar_chart})
                    log.info(f"📊 Created MMLU subtasks plot for {model_name} ({shot_num}-shot) with {len(subtask_data)} tasks")
            
            # 📊 Performance vs Random Baseline
            if 'overall_accuracy' in shot_results:
                overall_acc = shot_results['overall_accuracy']
                random_baseline = 0.25
                
                comparison_data = [
                    ["Random Baseline", random_baseline],
                    [f"{model_name}", overall_acc],
                    ["Improvement", overall_acc - random_baseline]
                ]
                
                table = wandb.Table(data=comparison_data, columns=["Metric", "Value"])
                
                bar_chart = wandb.plot.bar(
                    table, "Metric", "Value",
                    title=f"{model_name} - Performance vs Baseline ({shot_num}-shot)"
                )
                
                self.run.log({f"{model_name}/plots/performance_vs_baseline_{shot_num}shot": bar_chart})
                log.info(f"📊 Created performance comparison plot for {model_name} ({shot_num}-shot)")
                
        except Exception as e:
            log.warning(f"Failed to create plots for {model_name}: {e}")
    
    def finish_experiment(self):
        """Finish W&B experiment - handled natively by lm-eval"""
        log.info("🚀 Native W&B integration - run lifecycle managed by lm-eval")


@task_wrapper
def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """Main evaluation function"""
    
    log.info(f"Starting Universal Evaluation Pipeline: {cfg.experiment_name}")
    log.info(f"Models: {len(cfg.models)}")
    log.info(f"Benchmarks: {len(cfg.benchmarks)}")
    
    # Initialize components
    model_manager = ModelManager(cfg)
    benchmark_runner = BenchmarkRunner(cfg)
    wandb_logger = WandBLogger(cfg)
    
    # Start W&B experiment
    wandb_logger.start_experiment()
    
    all_results = {}
    
    try:
        # Start aggressive background conversion and queue ALL models immediately
        model_manager.start_background_conversion()
        
        # Queue ALL models for background conversion (aggressive preloading)
        for i, model_config in enumerate(cfg.models):
            model_manager.queue_model_for_conversion(model_config)
        
        log.info(f"Started aggressive conversion: {model_manager._max_conversion_workers} threads converting {len(cfg.models)} models in parallel")
        
        # GPU-Parallel Evaluation Setup (adaptive)
        # Priority: env override -> config override -> auto by GPU memory -> conservative default
        max_gpu_parallel_env = os.environ.get('EVAL_MAX_PARALLEL')
        max_gpu_parallel_cfg = None
        try:
            max_gpu_parallel_cfg = int(cfg.get('max_gpu_parallel', 0) or 0)
        except Exception:
            max_gpu_parallel_cfg = None

        if max_gpu_parallel_env and str(max_gpu_parallel_env).strip():
            try:
                max_gpu_parallel = int(str(max_gpu_parallel_env))
            except Exception:
                max_gpu_parallel = 1
        elif max_gpu_parallel_cfg and max_gpu_parallel_cfg > 0:
            max_gpu_parallel = max_gpu_parallel_cfg
        else:
            # Auto-detect GPU memory and choose a safe parallelism
            mem_total = None
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                line = res.stdout.strip().splitlines()[0]
                mem_total = int(line)
            except Exception:
                pass

            if mem_total is not None:
                if mem_total >= 80000:
                    max_gpu_parallel = 4
                elif mem_total >= 60000:
                    max_gpu_parallel = 2
                else:
                    max_gpu_parallel = 1
            else:
                max_gpu_parallel = 2  # fallback conservative default

        max_gpu_parallel = max(1, min(max_gpu_parallel, len(cfg.models)))
        log.info(f"🚀 Parallel evaluation: {max_gpu_parallel} models in parallel (override via EVAL_MAX_PARALLEL)")
        
        # Prepare evaluation tasks
        eval_tasks = []
        for i, model_config in enumerate(cfg.models):
            # Generate model name if not provided
            if 'name' in model_config and model_config['name']:
                model_name = model_config['name']
            else:
                model_name = f"model_{i+1}"  # Temporary, will be updated after metadata extraction
            
            eval_tasks.append((i, model_config, model_name))
        
        # Execute evaluations in parallel batches
        def evaluate_model_task(task_info):
            i, model_config, model_name = task_info
            
            log.info(f"\n{'='*60}")
            log.info(f"🔄 GPU-Parallel Processing model {i+1}/{len(cfg.models)}: {model_name}")
            log.info(f"Source: {model_config.get('source_path', model_config.get('path', 'unknown'))}")
            log.info(f"{'='*60}")
            
            try:
                # Prepare model (convert if necessary) and extract metadata
                # This will use cache/background conversion when available
                model_path, metadata = model_manager.prepare_model(model_config)
                
                # Generate proper model name from metadata if not provided
                if 'name' not in model_config or not model_config['name']:
                    model_name = model_manager._generate_model_name(model_config, metadata)
                
                log.info(f"✅ Model ready: {model_name}")
                log.info(f"📂 Model path: {model_path}")
                log.info(f"📋 Extracted metadata: {metadata}")
                
                model_results = {
                    'model_config': model_config,
                    'model_path': model_path,
                    'metadata': metadata,
                    'benchmarks': {}
                }
                
                # Run each benchmark
                for benchmark_config in cfg.benchmarks:
                    log.info(f"\n🧪 Running benchmark: {benchmark_config['name']} on {model_name}")
                    
                    benchmark_results = benchmark_runner.run_benchmark(
                        model_path, model_name, benchmark_config, model_config=model_config
                    )
                    
                    model_results['benchmarks'][benchmark_config['name']] = benchmark_results
                    
                    # Log enhanced metadata to complement native lm-eval W&B integration
                    wandb_logger.log_model_results(
                        model_name, metadata, benchmark_config['name'], benchmark_results
                    )
                
                log.info(f"✅ Completed evaluation for {model_name}")
                return model_name, model_results
                
            except Exception as e:
                log.error(f"❌ Failed to process model {model_name}: {e}")
                return model_name, {
                    'model_config': model_config,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Execute evaluations in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_gpu_parallel, thread_name_prefix="GPUEval") as executor:
            # Submit all tasks
            future_to_task = {executor.submit(evaluate_model_task, task): task for task in eval_tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    model_name, model_results = future.result()
                    all_results[model_name] = model_results
                    log.info(f"🎯 Completed {len(all_results)}/{len(cfg.models)} models")
                except Exception as e:
                    task_info = task
                    log.error(f"❌ Task failed for {task_info}: {e}")
        
        log.info(f"🎉 All {len(cfg.models)} models evaluated in parallel!")
        
        # Save results
        save_results(cfg, all_results)
        
        # Save compact summary
        save_compact_summary(cfg, all_results)
        
        # Print summary
        print_summary(cfg, all_results)
        
    finally:
        # Cleanup
        model_manager.cleanup_temp_models()
        wandb_logger.finish_experiment()
        
    log.info("Pipeline completed!")
    
    # Return results in format expected by task_wrapper
    return {}, all_results


def save_results(cfg: DictConfig, results: Dict):
    """Save results to JSON file"""
    
    # Use Hydra's automatic output directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir = hydra_output_dir / "evaluation/results/universal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{cfg.experiment_name}_{timestamp}.json"
    
    # Add metadata
    results_with_metadata = {
        'experiment_metadata': {
            'name': cfg.experiment_name,
            'description': cfg.get('description', ''),
            'timestamp': timestamp,
            'config': OmegaConf.to_container(cfg, resolve=True)
        },
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(convert_dictconfig_to_dict(results_with_metadata), f, indent=2)
        
    log.info(f"Results saved to: {results_file}")


def find_baseline_model(cfg: DictConfig, results: Dict) -> str:
    """Automatically detect baseline model from config"""
    
    for model_config in cfg.models:
        model_name = model_config.get('name', '')
        model_path = model_config.get('source_path', model_config.get('path', ''))
        
        # Check if path contains t5-base (most common baseline)
        if 't5-base' in model_path:
            return model_name or 'baseline'
        
        # Check if name contains baseline  
        if 'baseline' in model_name.lower():
            return model_name
            
        # Check if name contains 'original' (like t5_base_original)
        if 'original' in model_name.lower():
            return model_name
    
    # Fallback: first model in config
    if cfg.models:
        first_model = cfg.models[0]
        return first_model.get('name', 'baseline')
    
    return None


def extract_all_subtasks(shot_results: Dict, benchmark_name: str) -> Dict:
    """Extract all available subtasks and categories for any benchmark type"""
    
    result = {
        'subtasks': {},
        'categories': {},
        'metadata': {'benchmark_type': benchmark_name}
    }
    
    if benchmark_name == 'mmlu' or benchmark_name.startswith('mmlu'):
        # MMLU: Extract subset tasks and categories
        if 'subset_tasks' in shot_results:
            result['subtasks'] = shot_results['subset_tasks']
            result['metadata']['num_tasks'] = shot_results.get('num_tasks', len(shot_results['subset_tasks']))
        
        # MMLU categories
        mmlu_categories = ['humanities', 'social_sciences', 'stem', 'other']
        for category in mmlu_categories:
            value = shot_results.get(category)
            if value is not None and value != "N/A" and isinstance(value, (int, float)):
                result['categories'][f'mmlu_{category}'] = value
    
    elif 'raw_results' in shot_results and 'results' in shot_results['raw_results']:
        # Generic extraction from raw lm-eval results
        raw_results = shot_results['raw_results']['results']
        
        # Extract all individual task results
        for task_name, task_data in raw_results.items():
            if isinstance(task_data, dict):
                # Find primary metric for this task
                for metric in ['acc,none', 'acc_norm,none', 'exact_match,none', 'rouge1,none', 'rouge2,none', 'rougeL,none']:
                    if metric in task_data:
                        result['subtasks'][task_name] = task_data[metric]
                        break
        
        result['metadata']['num_tasks'] = len(result['subtasks'])
    
    return result


def calculate_baseline_comparison(all_model_scores: Dict, baseline_model_name: str) -> Dict:
    """Calculate differences and improvements relative to detected baseline model"""
    
    comparisons = {}
    
    # Find baseline scores using detected baseline model name
    baseline_scores = all_model_scores.get(baseline_model_name, {})
    
    if not baseline_scores:
        log.warning(f"Baseline model '{baseline_model_name}' not found in current results. Skipping baseline comparisons.")
        return {}
    
    log.info(f"📊 Using '{baseline_model_name}' as baseline for comparisons")
    
    # Calculate comparisons for all models
    for model_name, scores in all_model_scores.items():
        comparisons[model_name] = {}
        
        for benchmark_shot, score in scores.items():
            baseline_score = baseline_scores.get(benchmark_shot)
            
            if baseline_score and isinstance(baseline_score, (int, float)) and baseline_score > 0:
                difference = score - baseline_score
                improvement_percent = (difference / baseline_score) * 100
                
                comparisons[model_name][f'{benchmark_shot}_vs_baseline_diff'] = difference
                comparisons[model_name][f'{benchmark_shot}_vs_baseline_improvement_pct'] = improvement_percent
            else:
                comparisons[model_name][f'{benchmark_shot}_vs_baseline_diff'] = None
                comparisons[model_name][f'{benchmark_shot}_vs_baseline_improvement_pct'] = None
    
    return comparisons


def export_results_to_csv(summary_data: Dict, csv_file: Path, cfg: DictConfig = None):
    """Enhanced CSV export with comprehensive subtask details and baseline comparisons"""
    
    try:
        rows = []
        all_model_scores = {}  # For baseline calculations
        all_subtasks = set()   # Track all unique subtasks across models
        all_categories = set() # Track all unique categories across models
        
        # Extract experiment info
        exp_info = summary_data.get('experiment_info', {})
        experiment_name = exp_info.get('name', 'unknown')
        timestamp = exp_info.get('timestamp', 'unknown')
        
        # Detect baseline model from config
        baseline_model_name = None
        if cfg:
            baseline_model_name = find_baseline_model(cfg, summary_data.get('models', {}))
            log.info(f"🎯 Detected baseline model: {baseline_model_name}")
        
        # First pass: collect all model scores and identify all subtasks/categories
        for model_name, model_data in summary_data.get('models', {}).items():
            if model_data.get('status') != 'success':
                continue
                
            all_model_scores[model_name] = {}
            benchmarks = model_data.get('benchmarks', {})
            
            for benchmark_name, benchmark_data in benchmarks.items():
                for shot_type, shot_data in benchmark_data.items():
                    if isinstance(shot_data, dict) and shot_data.get('status') == 'success':
                        # Collect scores for baseline comparison
                        if 'overall_accuracy' in shot_data:
                            key = f"{benchmark_name}_{shot_type.replace('_shot', '')}_shot"
                            all_model_scores[model_name][key] = shot_data['overall_accuracy']
                        
                        # Extract and catalog all subtasks/categories
                        subtask_data = extract_all_subtasks(shot_data, benchmark_name)
                        all_subtasks.update(subtask_data['subtasks'].keys())
                        all_categories.update(subtask_data['categories'].keys())
        
        # Calculate baseline comparisons if baseline detected
        baseline_comparisons = {}
        if baseline_model_name:
            baseline_comparisons = calculate_baseline_comparison(all_model_scores, baseline_model_name)
        
        # Second pass: create comprehensive CSV rows
        for model_name, model_data in summary_data.get('models', {}).items():
            if model_data.get('status') != 'success':
                continue
                
            metadata = model_data.get('metadata', {})
            benchmarks = model_data.get('benchmarks', {})
            
            # Create base row with model info
            base_row = {
                'experiment_name': experiment_name,
                'timestamp': timestamp,
                'model_name': model_name,
                'source_path': model_data.get('source_path', ''),
                'source_type': metadata.get('source_type', ''),
                'training_steps': metadata.get('training_steps', ''),
                'val_perplexity': metadata.get('val_perplexity', ''),
                'learning_rate': metadata.get('learning_rate', ''),
                'run_date': metadata.get('run_date', ''),
                'training_style': metadata.get('training_style', ''),
                'baseline_model': baseline_model_name or ''
            }
            
            # Add benchmark results
            for benchmark_name, benchmark_data in benchmarks.items():
                for shot_type, shot_data in benchmark_data.items():
                    if isinstance(shot_data, dict) and shot_data.get('status') == 'success':
                        row = base_row.copy()
                        
                        # Basic benchmark info
                        row.update({
                            'benchmark': benchmark_name,
                            'shot_type': shot_type,
                            'overall_accuracy': shot_data.get('overall_accuracy'),
                            'primary_metric_name': shot_data.get('parse_status', 'overall_accuracy'),
                            'primary_metric_value': shot_data.get('overall_accuracy') or shot_data.get('primary_score'),
                            'duration_minutes': shot_data.get('duration_minutes', ''),
                            'status': shot_data.get('status', 'success')
                        })
                        
                        # Add baseline comparisons if available
                        if baseline_comparisons and model_name in baseline_comparisons:
                            shot_key = f"{benchmark_name}_{shot_type.replace('_shot', '')}_shot"
                            comparison_data = baseline_comparisons[model_name]
                            row.update({
                                'baseline_difference': comparison_data.get(f'{shot_key}_vs_baseline_diff', ''),
                                'baseline_improvement_percent': comparison_data.get(f'{shot_key}_vs_baseline_improvement_pct', '')
                            })
                        else:
                            row.update({
                                'baseline_difference': '',
                                'baseline_improvement_percent': ''
                            })
                        
                        # Extract comprehensive subtask and category data
                        subtask_data = extract_all_subtasks(shot_data, benchmark_name)
                        
                        # Add all subtasks (with empty values for missing ones)
                        for subtask_name in sorted(all_subtasks):
                            clean_name = subtask_name.replace('mmlu_', '').replace('_', ' ')
                            row[f'subtask_{clean_name}'] = subtask_data['subtasks'].get(subtask_name, '')
                        
                        # Add all categories (with empty values for missing ones)  
                        for category_name in sorted(all_categories):
                            row[f'category_{category_name}'] = subtask_data['categories'].get(category_name, '')
                        
                        # Add metadata
                        row['num_subtasks'] = subtask_data['metadata'].get('num_tasks', len(subtask_data['subtasks']))
                        
                        rows.append(row)
        
        if not rows:
            log.warning("No successful results to export to CSV")
            return
            
        # Write comprehensive CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        log.info(f"📄 Enhanced CSV with {len(all_subtasks)} subtasks and {len(all_categories)} categories exported to: {csv_file}")
        if baseline_model_name and baseline_comparisons:
            log.info(f"📊 Baseline comparisons calculated relative to: {baseline_model_name}")
        
    except Exception as e:
        log.error(f"Failed to export enhanced CSV: {e}")
        import traceback
        log.error(traceback.format_exc())


def save_compact_summary(cfg: DictConfig, results: Dict):
    """Save compact evaluation summary with key metrics and metadata"""
    
    # Use Hydra's automatic output directory
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir = hydra_output_dir / "evaluation/results/universal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"{cfg.experiment_name}_summary_{timestamp}.json"
    
    # Build compact summary
    compact_summary = {
        'experiment_info': {
            'name': cfg.experiment_name,
            'description': cfg.get('description', ''),
            'timestamp': timestamp,
            'evaluation_date': datetime.now().isoformat(),
            'total_models': len(results),
            'benchmarks': [b['name'] for b in cfg.benchmarks],
            'shots_evaluated': [b.get('shots', [0, 5]) for b in cfg.benchmarks]
        },
        'models': {},
        'benchmark_rankings': {},
        'performance_summary': {}
    }
    
    # Extract compact model info and results
    all_benchmark_scores = {}  # For rankings
    
    for model_name, model_results in results.items():
        if model_results.get('status') == 'failed':
            compact_summary['models'][model_name] = {
                'status': 'failed',
                'error': model_results.get('error', 'Unknown error'),
                'metadata': model_results.get('metadata', {})
            }
            continue
            
        # Model metadata
        metadata = model_results.get('metadata', {})
        model_info = {
            'status': 'success',
            'source_path': model_results.get('model_config', {}).get('source_path', 'unknown'),
            'metadata': {
                'training_steps': metadata.get('training_steps'),
                'val_perplexity': metadata.get('val_perplexity'),
                'training_style': metadata.get('training_style'),
                'learning_rate': metadata.get('learning_rate'),
                'run_date': metadata.get('run_date'),
                'source_type': metadata.get('source_type')
            },
            'benchmarks': {}
        }
        
        # Extract benchmark results
        if 'benchmarks' in model_results:
            for benchmark_name, benchmark_results in model_results['benchmarks'].items():
                model_info['benchmarks'][benchmark_name] = {}
                
                for shot_key, shot_results in benchmark_results.items():
                    if shot_results.get('status') == 'success':
                        shot_num = shot_key.replace('_shot', '')
                        
                        # Core metrics
                        shot_summary = {
                            'overall_accuracy': shot_results.get('overall_accuracy'),
                            'status': 'success'
                        }
                        
                        # MMLU specific metrics (clean, no redundant raw data)
                        if benchmark_name == 'mmlu':
                            # Only include non-null category scores
                            categories = {}
                            for cat in ['humanities', 'social_sciences', 'stem', 'other']:
                                value = shot_results.get(cat)
                                if value is not None and value != "N/A":
                                    categories[cat] = value
                            
                            if categories:
                                shot_summary['categories'] = categories
                            
                            # Include subset tasks summary if available
                            if shot_results.get('subset_tasks'):
                                shot_summary['subset_tasks'] = shot_results['subset_tasks']
                                shot_summary['num_tasks'] = shot_results.get('num_tasks')
                        
                        # Add to rankings data
                        ranking_key = f"{benchmark_name}_{shot_num}shot"
                        if ranking_key not in all_benchmark_scores:
                            all_benchmark_scores[ranking_key] = {}
                        
                        score = shot_results.get('overall_accuracy') or shot_results.get('primary_score')
                        if score is not None:
                            all_benchmark_scores[ranking_key][model_name] = score
                        
                        model_info['benchmarks'][benchmark_name][shot_key] = shot_summary
                    else:
                        model_info['benchmarks'][benchmark_name][shot_key] = {
                            'status': 'failed',
                            'error': shot_results.get('error', 'Unknown error')
                        }
        
        compact_summary['models'][model_name] = model_info
    
    # Generate rankings
    for benchmark_key, scores in all_benchmark_scores.items():
        if len(scores) > 1:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            compact_summary['benchmark_rankings'][benchmark_key] = [
                {
                    'rank': rank,
                    'model': model_name,
                    'score': score,
                    'improvement_over_last': score - sorted_models[rank][1] if rank > 0 else 0.0
                }
                for rank, (model_name, score) in enumerate(sorted_models)
            ]
    
    # Performance summary statistics
    if all_benchmark_scores:
        for benchmark_key, scores in all_benchmark_scores.items():
            if scores:
                compact_summary['performance_summary'][benchmark_key] = {
                    'best_score': max(scores.values()),
                    'worst_score': min(scores.values()),
                    'average_score': sum(scores.values()) / len(scores),
                    'score_range': max(scores.values()) - min(scores.values()),
                    'models_count': len(scores)
                }
    
    # Save compact summary
    with open(summary_file, 'w') as f:
        json.dump(compact_summary, f, indent=2, default=str)
        
    log.info(f"📊 Compact summary saved to: {summary_file}")
    
    # Create CSV export for easy analysis
    csv_file = output_dir / f"{cfg.experiment_name}_results_{timestamp}.csv"
    export_results_to_csv(compact_summary, csv_file, cfg)
    
    return summary_file


def print_summary(cfg: DictConfig, results: Dict):
    """Enhanced evaluation summary with comparisons and task details"""
    
    log.info(f"\n{'='*80}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*80}")
    
    # Print table header
    header = f"{'Model':<30}"
    for benchmark_config in cfg.benchmarks:
        shots = benchmark_config.get('shots', [0, 5])
        for shot in shots:
            header += f" {benchmark_config['name']}_{shot}shot"[:12].ljust(12)
    header += " Status"
    
    log.info(header)
    log.info("-" * len(header))
    
    # Print results for each model
    model_scores = {}  # For ranking
    
    for model_name, model_results in results.items():
        if 'benchmarks' in model_results:
            row = f"{model_name:<30}"
            
            for benchmark_config in cfg.benchmarks:
                benchmark_name = benchmark_config['name']
                shots = benchmark_config.get('shots', [0, 5])
                
                if benchmark_name in model_results['benchmarks']:
                    benchmark_results = model_results['benchmarks'][benchmark_name]
                    
                    for shot in shots:
                        shot_key = f"{shot}_shot"
                        if shot_key in benchmark_results:
                            result = benchmark_results[shot_key]
                            if result.get('status') == 'success':
                                if 'overall_accuracy' in result:
                                    score = result['overall_accuracy']
                                    row += f" {score:.3f}"
                                    
                                    # Store for ranking
                                    key = f"{benchmark_name}_{shot}shot"
                                    if key not in model_scores:
                                        model_scores[key] = {}
                                    model_scores[key][model_name] = score
                                    
                                elif 'primary_score' in result:
                                    score = result['primary_score']
                                    row += f" {score:.3f}"
                                    
                                    # Store for ranking
                                    key = f"{benchmark_name}_{shot}shot"
                                    if key not in model_scores:
                                        model_scores[key] = {}
                                    model_scores[key][model_name] = score
                                else:
                                    row += f" {'N/A':>11}"
                            else:
                                row += f" {'Failed':>11}"
                        else:
                            row += f" {'N/A':>11}"
                else:
                    for shot in shots:
                        row += f" {'N/A':>11}"
            
            status = "Success" if model_results.get('status') != 'failed' else "Failed"
            row += f" {status}"
            
            log.info(row)
        else:
            row = f"{model_name:<30}"
            # Fill with N/A for all benchmarks
            total_cols = sum(len(b.get('shots', [0, 5])) for b in cfg.benchmarks)
            row += " N/A" * (total_cols * 12) + " Failed"
            log.info(row)
    
    # ✨ NEW: Print rankings and comparisons
    log.info(f"\n{'='*80}")
    log.info("MODEL RANKINGS & COMPARISONS")
    log.info(f"{'='*80}")
    
    for benchmark_key, scores in model_scores.items():
        if len(scores) > 1:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            log.info(f"\n📊 {benchmark_key.upper()}:")
            for rank, (model_name, score) in enumerate(sorted_models, 1):
                if rank == 1:
                    log.info(f"  🥇 #{rank}: {model_name:<25} {score:.3f}")
                elif rank == 2:
                    log.info(f"  🥈 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
                elif rank == 3:
                    log.info(f"  🥉 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
                else:
                    log.info(f"  📍 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
    
    # ✨ DYNAMIC: Print task details and all available metrics
    log.info(f"\n{'='*80}")
    log.info("DETAILED METRICS")
    log.info(f"{'='*80}")
    
    for model_name, model_results in results.items():
        if 'benchmarks' in model_results and model_results.get('status') != 'failed':
            log.info(f"\n📋 {model_name.upper()}:")
            
            for benchmark_name, benchmark_results in model_results['benchmarks'].items():
                for shot_key, shot_results in benchmark_results.items():
                    if shot_results.get('status') == 'success':
                        shot_num = shot_key.replace('_shot', '')
                        
                        # Core metrics
                        if 'overall_accuracy' in shot_results:
                            log.info(f"  {benchmark_name} {shot_num}-shot:")
                            log.info(f"    🎯 Overall Accuracy:    {shot_results['overall_accuracy']:.3f}")
                            
                            # ✨ DYNAMIC: Show all available category scores
                            mmlu_categories = ['humanities', 'social_sciences', 'stem', 'other']
                            found_categories = []
                            
                            for category in mmlu_categories:
                                value = shot_results.get(category)
                                if value is not None and value != "N/A" and isinstance(value, (int, float)):
                                    found_categories.append((category, value))
                            
                            if found_categories:
                                log.info(f"    📊 Categories:")
                                for category, value in found_categories:
                                    log.info(f"      • {category.replace('_', ' ').title():<15} {value:.3f}")
                            
                            # Show subset task details if available
                            if 'subset_tasks' in shot_results:
                                log.info(f"    📝 Tasks (subset of {shot_results.get('num_tasks', 0)}):")
                                
                                for task_name, accuracy in shot_results['subset_tasks'].items():
                                    task_short = task_name.replace('mmlu_', '')
                                    log.info(f"      • {task_short:<20} {accuracy:.3f}")
                            
                            # ✨ DYNAMIC: Show any additional metrics that were found
                            extra_metrics = []
                            for key, value in shot_results.items():
                                if (isinstance(value, (int, float)) and 
                                    key not in ['overall_accuracy', 'overall_stderr', 'duration_seconds', 'duration_minutes', 'num_tasks'] and
                                    key not in mmlu_categories and
                                    key not in ['subset_tasks'] and
                                    not key.startswith('_') and
                                    value != "N/A"):
                                    extra_metrics.append((key, value))
                            
                            if extra_metrics:
                                log.info(f"    ➕ Additional Metrics:")
                                for key, value in extra_metrics:
                                    clean_key = key.replace('_', ' ').title()
                                    log.info(f"      • {clean_key:<20} {value:.3f}")
                            
                            # Performance info
                            duration = shot_results.get('duration_minutes', 0)
                            log.info(f"    ⏱️  Duration:            {duration:.1f} min")
                            
                        elif 'primary_score' in shot_results:
                            log.info(f"  {benchmark_name} {shot_num}-shot:")
                            log.info(f"    🎯 Primary Score:       {shot_results['primary_score']:.3f}")
                            
                            duration = shot_results.get('duration_minutes', 0)
                            log.info(f"    ⏱️  Duration:            {duration:.1f} min")
    
    log.info(f"\n{'='*80}")
    log.info("EVALUATION COMPLETED")
    log.info(f"{'='*80}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_pipeline")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point"""
    
    # Execute evaluation
    results = evaluate(cfg)
    
    # Results are now automatically added to W&B run summaries during evaluation
    # for easy comparison in W&B's run comparison views
    
    # Return metric for hyperparameter optimization
    return None


if __name__ == "__main__":
    main()
