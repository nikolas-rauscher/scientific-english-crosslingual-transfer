import os
import sys
import logging
import json
from datetime import datetime


script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, proj_root)

import hydra
from omegaconf import DictConfig, OmegaConf
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from src.dataprep.sliding_window_processor import TextOnlySlidingWindowProcessor

log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - sliding window preprocessing analytics will not be logged")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="preprocessing/sliding_windows")
def main(cfg: DictConfig) -> None:
    log.info("Starting T5 Sliding Window Pipeline")
    log.info("Configuration:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    wandb_session = None
    if WANDB_AVAILABLE and cfg.sliding_windows.get("log_to_wandb", False):
        try:
            tags = ["sliding-windows", "preprocessing"]
            
            # Add output format tag
            output_format = cfg.sliding_windows.window_config.get("output_format", "text")
            tags.append(f"format-{output_format}")
            
            if cfg.sliding_windows.limit_documents != -1:
                tags.extend(["test_run", f"limit-{cfg.sliding_windows.limit_documents}"])
            else:
                tags.append("full-dataset")
            
            wandb_session = wandb.init(
                project=cfg.sliding_windows.wandb.project,
                group=cfg.sliding_windows.wandb.group,
                tags=tags,
                job_type="sliding-window-creation",
                notes=f"Create {output_format} sliding windows for T5 training",
                config=OmegaConf.to_container(cfg, resolve=True)
            )
            log.info(f"W&B session initialized: {wandb_session.url}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B session: {e}")
            wandb_session = None

    # Build pipeline dynamically: reader -> windows -> optional stats -> writer
    pipeline = []

    def _canonicalize_for_parquet(value):
        """Recursively sort dict keys for deterministic struct field ordering."""
        if isinstance(value, dict):
            return {k: _canonicalize_for_parquet(value[k]) for k in sorted(value.keys())}
        if isinstance(value, list):
            return [_canonicalize_for_parquet(v) for v in value]
        return value

    def _stable_window_adapter(_writer, document):
        """
        Emit a deterministic row layout to avoid pyarrow schema drift across batches.
        Keep metadata as struct, but with canonicalized key order.
        """
        metadata = getattr(document, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {"value": json.dumps(metadata, ensure_ascii=False)}
        metadata = _canonicalize_for_parquet(metadata)
        return {
            "text": document.text,
            "id": document.id,
            "metadata": metadata,
        }

    # Reader
    pipeline.append(
        ParquetReader(
            data_folder=cfg.sliding_windows.paths.input_folder,
            glob_pattern=cfg.sliding_windows.paths.input_pattern,
            limit=cfg.sliding_windows.limit_documents,
            text_key=cfg.sliding_windows.reader.text_key,
            id_key=cfg.sliding_windows.reader.id_key,
            default_metadata=OmegaConf.to_container(cfg.sliding_windows.reader.default_metadata, resolve=True)
        )
    )

    # Normalization + filters configuration
    normalization_cfg = cfg.sliding_windows.get("normalization", {}) if hasattr(cfg.sliding_windows, "normalization") else {}
    normalize_whitespace = bool(normalization_cfg.get("enable", True))
    filters_cfg = cfg.sliding_windows.get("filters", {}) if hasattr(cfg.sliding_windows, "filters") else {}

    # Window processor
    pipeline.append(
        TextOnlySlidingWindowProcessor(
            tokenizer_name_or_path=cfg.sliding_windows.tokenizer.name_or_path,
            target_tokens=cfg.sliding_windows.window_config.target_tokens,
            overlap_ratio=cfg.sliding_windows.window_config.overlap_ratio,
            output_format=cfg.sliding_windows.window_config.get("output_format", "text"),
            balance_last_window=cfg.sliding_windows.window_config.get("dynamic_last_window", {}).get("enable", False),
            min_last_tokens=cfg.sliding_windows.window_config.get("dynamic_last_window", {}).get("min_last_tokens", 384),
            log_to_wandb=cfg.sliding_windows.get("log_to_wandb", False) and bool(wandb_session),
            wandb_project=cfg.sliding_windows.wandb.project,
            wandb_group=cfg.sliding_windows.wandb.group,
            normalize_whitespace=normalize_whitespace,
            filters=filters_cfg,
        )
    )

    # Optional: ONLY DocStats on windows (no other stats)
    stats_cfg = cfg.sliding_windows.get("stats", {}) if hasattr(cfg.sliding_windows, "stats") else {}
    stats_enabled = bool(stats_cfg.get("enable", False))
    if stats_enabled:
        stats_output_folder = stats_cfg.get("output_folder", os.path.join(cfg.sliding_windows.paths.output_folder, "stats"))
        os.makedirs(stats_output_folder, exist_ok=True)

        modules = stats_cfg.get("modules", {})
        # Support both dict access and attribute access
        has_doc_stats = hasattr(modules, 'doc_stats') or (isinstance(modules, dict) and 'doc_stats' in modules)
        if has_doc_stats:
            from hydra.utils import instantiate
            doc_stats_cfg = getattr(modules, 'doc_stats') if hasattr(modules, 'doc_stats') else modules['doc_stats']
            out_dir = os.path.join(stats_output_folder, getattr(doc_stats_cfg, 'output_folder', 'doc_stats') if hasattr(doc_stats_cfg, 'output_folder') else doc_stats_cfg.get('output_folder', 'doc_stats'))
            os.makedirs(out_dir, exist_ok=True)
            pipeline.append(instantiate(doc_stats_cfg, output_folder=out_dir))
            log.info("Attached stats module: DocStats")
        else:
            log.info("Stats enabled but no doc_stats module configured; skipping stats")

        # Writer for enriched windows (or raw if disabled)
        if bool(stats_cfg.get("save_enriched_docs", True)):
            enriched_dir = os.path.join(cfg.sliding_windows.paths.output_folder, "enriched_windows")
            os.makedirs(enriched_dir, exist_ok=True)
            pipeline.append(ParquetWriter(enriched_dir, adapter=_stable_window_adapter))
            log.info(f"Enriched windows will be saved to: {enriched_dir}")
        else:
            pipeline.append(ParquetWriter(cfg.sliding_windows.paths.output_folder, adapter=_stable_window_adapter))
            log.info(f"Raw windows will be saved to: {cfg.sliding_windows.paths.output_folder}")
    else:
        # No stats: write raw windows directly
        pipeline.append(ParquetWriter(cfg.sliding_windows.paths.output_folder, adapter=_stable_window_adapter))
        log.info(f"Raw windows will be saved to: {cfg.sliding_windows.paths.output_folder}")

    log.info(f"Pipeline: {len(pipeline)} steps")
    log.info(f"Tasks: {cfg.sliding_windows.execution.tasks}, Workers: {cfg.sliding_windows.execution.workers}")
    
    # New API logging
    target_tokens = cfg.sliding_windows.window_config.target_tokens
    overlap_ratio = cfg.sliding_windows.window_config.overlap_ratio
    overlap_tokens = int(target_tokens * overlap_ratio)
    output_format = cfg.sliding_windows.window_config.get("output_format", "text")
    
    log.info(f"Target Tokens: {target_tokens}, Overlap: {overlap_ratio:.1%} ({overlap_tokens} tokens)")
    log.info(f"Output Format: {output_format}")
    log.info(f"Tokenizer: {cfg.sliding_windows.tokenizer.name_or_path}")
    
    if cfg.sliding_windows.limit_documents != -1:
        log.info(f"Test mode: {cfg.sliding_windows.limit_documents} docs")
    else:
        log.info("Full processing mode")
    
    log.info(f"Input: {cfg.sliding_windows.paths.input_folder}")
    log.info(f"Output: {cfg.sliding_windows.paths.output_folder}")
    
    if wandb_session:
        log.info(f"W&B: {wandb_session.name}")
    else:
        log.info("W&B: Disabled")

    # Execute pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=cfg.sliding_windows.execution.tasks,
        workers=cfg.sliding_windows.execution.workers,
        logging_dir=cfg.sliding_windows.paths.get("logging_dir", "logs/sliding_windows")
    )
    
    try:
        log.info("Starting pipeline...")
        start_time = datetime.now()
        
        executor.run()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        log.info("Pipeline completed!")
        log.info(f"Time: {execution_time:.1f}s")
        
        if wandb_session:
            wandb.log({
                "pipeline_execution_time": execution_time,
                "limit_documents": cfg.sliding_windows.limit_documents,
                "target_tokens": target_tokens,
                "overlap_ratio": overlap_ratio,
                "overlap_tokens": overlap_tokens,
                "output_format": output_format,
                "tasks": cfg.sliding_windows.execution.tasks,
                "workers": cfg.sliding_windows.execution.workers,
                "tokenizer": cfg.sliding_windows.tokenizer.name_or_path
            })
        
        if wandb_session:
            try:
                wandb.finish()
                log.info("W&B session finished")
            except Exception as e:
                log.warning(f"Error finishing W&B session: {e}")
            
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        if wandb_session:
            try:
                wandb.finish()
            except:
                pass
        raise


if __name__ == "__main__":
    main() 
