#!/usr/bin/env python3
"""Run paper-track WECHSEL transfer for all languages (no CLI args).

Paper-track specifics:
- fixed EN checkpoint
- fixed target tokenizer root: models/tokenizers_paper_spm32k/<LANG>
- output root: models/wechsel_init_paper_spm32k/<LANG>
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from wechsel import WECHSEL, load_embeddings


LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path("/netscratch/nrauscher/projects/BA-hydra")
SUBPROJECT_ROOT = PROJECT_ROOT / "cross_lingual_transfer_multilingual"
TARGETS_CONFIG = SUBPROJECT_ROOT / "configs" / "languages" / "targets.yaml"
TOKENIZER_ROOT = SUBPROJECT_ROOT / "models" / "tokenizers_paper_spm32k"
OUTPUT_ROOT = SUBPROJECT_ROOT / "models" / "wechsel_init_paper_spm32k"
ENGLISH_CHECKPOINT = (
    PROJECT_ROOT
    / "pretraining_logs_lr_001_OPTIMIZED_clean_restart"
    / "train"
    / "runs"
    / "2025-09-08_02-33-22"
    / "checkpoints"
    / "best"
    / "step-487500-val_ppl-3.72168.ckpt"
)
BASE_MODEL = "t5-base"
LANGUAGES = [
    "deu_Latn",
    "jpn_Jpan",
    "spa_Latn",
    "rus_Cyrl",
    "pol_Latn",
    "por_Latn",
]

# Needed when loading Lightning checkpoints that pickle objects from project modules.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LanguageTarget:
    dataset_code: str
    label: str
    fasttext_code: str
    wechsel_dictionary: str
    tokenizer_path: str


def _configure_caches() -> None:
    cache_root = PROJECT_ROOT / "cross_lingual_transfer" / "cache"
    wechsel_cache = cache_root / "wechsel"
    hf_cache = PROJECT_ROOT / ".hf_cache"

    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["WECHSEL_CACHE_DIR"] = str(wechsel_cache)
    os.environ["WECHSEL_CACHE"] = str(wechsel_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache)
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["PYTORCH_PRETRAINED_BERT_CACHE"] = str(hf_cache)

    wechsel_cache.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)


def _normalize_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _normalize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if is_dataclass(obj):
        return _normalize_json(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _normalize_json(obj.to_dict())
        except Exception:  # pylint: disable=broad-except
            pass
    if hasattr(obj, "__dict__"):
        return {
            str(k): _normalize_json(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_") and not callable(v)
        }
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _normalize_json(obj.tolist())
        except Exception:  # pylint: disable=broad-except
            pass
    return str(obj)


def _load_targets() -> list[LanguageTarget]:
    cfg = OmegaConf.load(TARGETS_CONFIG)
    by_code = {str(row.dataset_code): row for row in cfg.languages}

    targets: list[LanguageTarget] = []
    for code in LANGUAGES:
        if code not in by_code:
            raise ValueError(f"Language not found in targets config: {code}")
        row = by_code[code]
        tokenizer_path = TOKENIZER_ROOT / code
        targets.append(
            LanguageTarget(
                dataset_code=code,
                label=str(row.label),
                fasttext_code=str(row.fasttext_code),
                wechsel_dictionary=str(row.wechsel_dictionary),
                tokenizer_path=str(tokenizer_path),
            )
        )
    return targets


def _load_checkpoint_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        # Fallback for environments where full module graph is unavailable.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in checkpoint:
        return checkpoint

    out = {}
    stripped_model_model = 0
    stripped_model = 0
    for key, value in checkpoint["state_dict"].items():
        new_key = key
        if new_key.startswith("model.model."):
            new_key = new_key[len("model.model.") :]
            stripped_model_model += 1
        elif new_key.startswith("model."):
            new_key = new_key[len("model.") :]
            stripped_model += 1
        out[new_key] = value
    LOG.info(
        "Stripped checkpoint prefixes -> model.model.: %d, model.: %d",
        stripped_model_model,
        stripped_model,
    )
    return out


class PaperWechselTransferRunner:
    def __init__(self) -> None:
        self.english_tokenizer: T5Tokenizer | None = None
        self.english_model: T5ForConditionalGeneration | None = None
        self.english_embeddings_np = None

    def load_english_once(self) -> None:
        LOG.info("Loading English base tokenizer/model: %s", BASE_MODEL)
        self.english_tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
        self.english_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

        LOG.info("Loading source checkpoint: %s", ENGLISH_CHECKPOINT)
        state_dict = _load_checkpoint_state_dict(ENGLISH_CHECKPOINT)
        missing, unexpected = self.english_model.load_state_dict(state_dict, strict=False)
        if missing:
            LOG.warning("Missing keys while loading EN checkpoint: %d", len(missing))
        if unexpected:
            LOG.warning("Unexpected keys while loading EN checkpoint: %d", len(unexpected))

        self.english_embeddings_np = (
            self.english_model.get_input_embeddings().weight.detach().cpu().numpy()
        )

    @staticmethod
    def _ensure_t5_extra_ids(tokenizer: AutoTokenizer) -> int:
        needed = {f"<extra_id_{i}>" for i in range(100)}
        present = set(tokenizer.get_vocab().keys())
        missing = sorted(needed - present)
        if not missing:
            return 0
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        return len(missing)

    def run_single(self, target: LanguageTarget) -> dict[str, Any]:
        if self.english_model is None or self.english_tokenizer is None:
            raise RuntimeError("English model/tokenizer not initialized")

        tok_dir = Path(target.tokenizer_path)
        if not tok_dir.exists():
            raise FileNotFoundError(f"Missing paper tokenizer dir: {tok_dir}")

        out_dir = OUTPUT_ROOT / target.dataset_code
        out_model_dir = out_dir / "model"
        out_tok_dir = out_dir / "tokenizer"
        out_meta_path = out_dir / "transfer_metadata.json"

        # Always rebuild to avoid stale artifacts from prior failed/partial runs.
        if out_dir.exists():
            shutil.rmtree(out_dir)

        LOG.info(
            "Paper WECHSEL %s | tokenizer=%s | fasttext=%s | dict=%s",
            target.dataset_code,
            tok_dir,
            target.fasttext_code,
            target.wechsel_dictionary,
        )

        # Keep tokenizer loading independent from fast-tokenizer JSON format.
        target_tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False)
        extra_added = self._ensure_t5_extra_ids(target_tokenizer)

        if not hasattr(self.english_tokenizer, "vocab"):
            self.english_tokenizer.vocab = self.english_tokenizer.get_vocab()
        if not hasattr(target_tokenizer, "vocab"):
            target_tokenizer.vocab = target_tokenizer.get_vocab()

        wechsel = WECHSEL(
            load_embeddings("en"),
            load_embeddings(target.fasttext_code),
            bilingual_dictionary=target.wechsel_dictionary,
        )
        transferred_embeddings, transfer_info = wechsel.apply(
            self.english_tokenizer,
            target_tokenizer,
            self.english_embeddings_np,
        )

        model = copy.deepcopy(self.english_model)
        new_vocab_size = len(target_tokenizer)
        if model.config.vocab_size != new_vocab_size:
            model.resize_token_embeddings(new_vocab_size)

        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            W = torch.as_tensor(transferred_embeddings, dtype=emb.dtype, device=emb.device)
            emb.copy_(W)
            out = model.get_output_embeddings().weight
            out.copy_(W)

        cfg = model.config
        cfg.vocab_size = len(target_tokenizer)
        cfg.pad_token_id = target_tokenizer.pad_token_id
        cfg.eos_token_id = target_tokenizer.eos_token_id
        cfg.decoder_start_token_id = target_tokenizer.pad_token_id
        model.tie_weights()

        out_model_dir.mkdir(parents=True, exist_ok=True)
        out_tok_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_model_dir)
        # Preserve tokenizer artifacts exactly as produced in stage-01.
        shutil.rmtree(out_tok_dir, ignore_errors=True)
        shutil.copytree(tok_dir, out_tok_dir)

        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "transfer_method": "wechsel_paper_spm32k",
            "source_checkpoint": str(ENGLISH_CHECKPOINT),
            "base_model": BASE_MODEL,
            "target": {
                "dataset_code": target.dataset_code,
                "label": target.label,
                "tokenizer_path": str(tok_dir),
                "fasttext_code": target.fasttext_code,
                "wechsel_dictionary": target.wechsel_dictionary,
            },
            "vocab_size_final": len(target_tokenizer),
            "extra_id_tokens_added": extra_added,
            "transfer_info": _normalize_json(transfer_info),
        }
        out_meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "dataset_code": target.dataset_code,
            "status": "ok",
            "output_dir": str(out_dir),
            "vocab_size_final": len(target_tokenizer),
            "extra_id_tokens_added": extra_added,
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(name)s:%(lineno)d - %(message)s",
    )

    if not ENGLISH_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing EN checkpoint: {ENGLISH_CHECKPOINT}")
    if not TARGETS_CONFIG.exists():
        raise FileNotFoundError(f"Missing targets config: {TARGETS_CONFIG}")

    _configure_caches()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    targets = _load_targets()
    LOG.info("Paper track languages: %s", ", ".join(t.dataset_code for t in targets))
    LOG.info("Tokenizer root: %s", TOKENIZER_ROOT)
    LOG.info("Output root: %s", OUTPUT_ROOT)

    runner = PaperWechselTransferRunner()
    runner.load_english_once()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for target in targets:
        try:
            res = runner.run_single(target)
            results.append(res)
            LOG.info("Finished %s with status=%s", target.dataset_code, res.get("status"))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append({"dataset_code": target.dataset_code, "error": str(exc)})
            LOG.exception("Failed for %s", target.dataset_code)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "english_checkpoint": str(ENGLISH_CHECKPOINT),
        "base_model": BASE_MODEL,
        "targets_config": str(TARGETS_CONFIG),
        "tokenizer_root": str(TOKENIZER_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "results": results,
        "failures": failures,
    }
    summary_path = OUTPUT_ROOT / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOG.info("Summary written to %s", summary_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
