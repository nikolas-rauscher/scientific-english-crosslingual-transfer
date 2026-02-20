#!/usr/bin/env python3
"""Materialize paper-track A/B models into persistent HF folders (no CLI args)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
import yaml


PROJECT_ROOT = Path("/netscratch/anonymous_user/projects/BA-hydra")
SUBPROJECT_ROOT = PROJECT_ROOT / "cross_lingual_transfer_multilingual"
TARGETS_CONFIG = SUBPROJECT_ROOT / "configs" / "languages" / "targets.yaml"

# Needed when loading Lightning checkpoints that pickle objects from project modules.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OLD_PERMANENT_ROOT = PROJECT_ROOT / "evaluation" / "converted_checkpoints" / "clt_multilingual_permanent"
OUTPUT_ROOT = (
    PROJECT_ROOT / "evaluation" / "converted_checkpoints" / "clt_multilingual_paper_spm32k_permanent"
)

OLD_WECHSEL_INIT_ROOT = SUBPROJECT_ROOT / "models" / "wechsel_init"
PAPER_WECHSEL_INIT_ROOT = SUBPROJECT_ROOT / "models" / "wechsel_init_paper_spm32k"

OLD_WECHSEL_RUNS_ROOT = SUBPROJECT_ROOT / "logs" / "train"
PAPER_WECHSEL_RUNS_ROOT = SUBPROJECT_ROOT / "logs" / "train_paper_spm32k"
HF_CONT_RUNS_ROOT = SUBPROJECT_ROOT / "logs" / "train_hf_monolingual"

LANGUAGES = [
    "deu_Latn",
    "jpn_Jpan",
    "spa_Latn",
    "rus_Cyrl",
    "pol_Latn",
    "por_Latn",
]


def _copy_tree_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _has_minimal_hf_t5_artifacts(model_dir: Path) -> bool:
    required = [
        model_dir / "config.json",
        model_dir / "tokenizer_config.json",
        model_dir / "special_tokens_map.json",
    ]
    if not all(p.exists() for p in required):
        return False
    if not ((model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()):
        return False
    if not ((model_dir / "spiece.model").exists() or (model_dir / "spm.model").exists()):
        return False
    return True


def _merge_split_model_tokenizer(src_root: Path, out_dir: Path) -> None:
    model_dir = src_root / "model"
    tok_dir = src_root / "tokenizer"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing model config.json in {model_dir}")
    if not tok_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer directory in {tok_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree_files(model_dir, out_dir)
    _copy_tree_files(tok_dir, out_dir)


def _strip_lightning_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        nk = key
        if nk.startswith("model.model."):
            nk = nk[len("model.model.") :]
        elif nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = value
    return out


def _infer_vocab_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    for key in ("shared.weight", "lm_head.weight"):
        if key in state_dict and getattr(state_dict[key], "ndim", 0) == 2:
            return int(state_dict[key].shape[0])
    return None


def _find_hydra_config_for_ckpt(ckpt_path: Path) -> Path | None:
    cur = ckpt_path.parent
    for _ in range(10):
        cand = cur / ".hydra" / "config.yaml"
        if cand.exists():
            return cand
        cur = cur.parent
    return None


def _convert_checkpoint(source_ckpt: Path, tokenizer_path: str, out_dir: Path) -> None:
    if _has_minimal_hf_t5_artifacts(out_dir):
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        # Fallback for environments where full module graph is unavailable.
        ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(source_ckpt, map_location="cpu")
    raw_state = ckpt.get("state_dict", ckpt)
    state = _strip_lightning_prefixes(raw_state)

    base_model = "t5-base"
    hydra_cfg = _find_hydra_config_for_ckpt(source_ckpt)
    if hydra_cfg is not None:
        try:
            cfg = yaml.safe_load(hydra_cfg.read_text(encoding="utf-8")) or {}
            base_model = (
                cfg.get("model", {})
                .get("t5_model", {})
                .get("pretrained_model_name_or_path", "t5-base")
            )
        except Exception:  # pylint: disable=broad-except
            base_model = "t5-base"

    try:
        config = T5Config.from_pretrained(base_model, local_files_only=True)
    except Exception:  # pylint: disable=broad-except
        config = T5Config.from_pretrained(base_model)

    vocab_from_state = _infer_vocab_size(state)
    if vocab_from_state is not None and config.vocab_size != vocab_from_state:
        config.vocab_size = vocab_from_state

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False)
    if config.eos_token_id is None and tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if config.pad_token_id is None and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if config.decoder_start_token_id is None and tokenizer.pad_token_id is not None:
        config.decoder_start_token_id = tokenizer.pad_token_id

    model = T5ForConditionalGeneration(config)
    load_result = model.load_state_dict(state, strict=False)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "source_type": "checkpoint",
        "original_checkpoint": str(source_ckpt),
        "tokenizer_path": str(tokenizer_path),
        "hydra_config": str(hydra_cfg) if hydra_cfg else None,
        "base_model": base_model,
        "vocab_from_state": vocab_from_state,
        "missing_keys": list(getattr(load_result, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", []) or []),
    }
    (out_dir / "conversion_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _download_hf_model(repo_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _has_minimal_hf_t5_artifacts(out_dir):
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _parse_val_ppl(filename: str) -> float | None:
    marker = "val_ppl-"
    if marker not in filename or not filename.endswith(".ckpt"):
        return None
    raw = filename.split(marker, 1)[1].rsplit(".ckpt", 1)[0]
    try:
        return float(raw)
    except ValueError:
        return None


def _pick_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    scored: list[tuple[float, Path]] = []
    for path in ckpts:
        ppl = _parse_val_ppl(path.name)
        if ppl is not None:
            scored.append((ppl, path))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _find_latest_run_checkpoint(run_root: Path) -> Path | None:
    runs_root = run_root / "runs"
    if not runs_root.exists():
        return None

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        ckpt = _pick_checkpoint(run_dir / "checkpoints" / "best")
        if ckpt is not None:
            return ckpt
    return None


def _load_targets_hf_models() -> dict[str, str]:
    cfg = OmegaConf.load(TARGETS_CONFIG)
    out: dict[str, str] = {}
    for row in cfg.languages:
        out[str(row.dataset_code)] = str(row.hf_monolingual_model)
    return out


def _materialize(spec: dict[str, Any], out_dir: Path) -> None:
    source_path = spec["source_path"]
    tokenizer_path = spec.get("tokenizer_path")

    if isinstance(source_path, str):
        src = Path(source_path)
        if src.is_absolute() and not src.exists():
            raise FileNotFoundError(f"Missing local source path: {src}")
        if src.exists():
            if src.is_dir():
                if (src / "model" / "config.json").exists():
                    _merge_split_model_tokenizer(src, out_dir)
                    return
                if (src / "config.json").exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _copy_tree_files(src, out_dir)
                    return
                raise FileNotFoundError(f"Unsupported local model layout: {src}")

            if src.is_file() and src.suffix == ".ckpt":
                if not tokenizer_path:
                    raise ValueError(f"Checkpoint requires tokenizer_path: {src}")
                _convert_checkpoint(src, str(tokenizer_path), out_dir)
                return

        # Fallback: treat as HF repo id
        _download_hf_model(source_path, out_dir)
        return

    raise TypeError(f"Unsupported source_path type: {type(source_path)}")


def _build_specs() -> list[dict[str, Any]]:
    hf_models = _load_targets_hf_models()
    specs: list[dict[str, Any]] = []

    for lang in LANGUAGES:
        hf_model = hf_models[lang]

        old_root = OLD_PERMANENT_ROOT / lang
        old_hf_dir = old_root / f"hf-monolingual-{lang}"
        old_hf_cont_dir = old_root / f"hf-monolingual-continued-{lang}"
        old_init_dir = old_root / f"wechsel-init-{lang}"
        old_cont_dir = old_root / f"wechsel-continued-{lang}"

        old_init_split = OLD_WECHSEL_INIT_ROOT / lang
        paper_init_split = PAPER_WECHSEL_INIT_ROOT / lang

        old_cont_ckpt = _find_latest_run_checkpoint(OLD_WECHSEL_RUNS_ROOT / lang)
        hf_cont_ckpt = _find_latest_run_checkpoint(HF_CONT_RUNS_ROOT / lang)
        paper_cont_ckpt = _find_latest_run_checkpoint(PAPER_WECHSEL_RUNS_ROOT / lang)

        # 1) HF monolingual
        specs.append(
            {
                "lang": lang,
                "name": f"hf-monolingual-{lang}",
                "source_path": str(old_hf_dir) if (old_hf_dir / "config.json").exists() else hf_model,
            }
        )

        # 2) HF monolingual continued
        specs.append(
            {
                "lang": lang,
                "name": f"hf-monolingual-continued-{lang}",
                "source_path": str(old_hf_cont_dir)
                if (old_hf_cont_dir / "config.json").exists()
                else (str(hf_cont_ckpt) if hf_cont_ckpt else ""),
                "tokenizer_path": hf_model,
            }
        )

        # 3) old WECHSEL init
        specs.append(
            {
                "lang": lang,
                "name": f"wechsel-init-{lang}",
                "source_path": str(old_init_dir)
                if (old_init_dir / "config.json").exists()
                else str(old_init_split),
            }
        )

        # 4) old WECHSEL continued
        specs.append(
            {
                "lang": lang,
                "name": f"wechsel-continued-{lang}",
                "source_path": str(old_cont_dir)
                if (old_cont_dir / "config.json").exists()
                else (str(old_cont_ckpt) if old_cont_ckpt else ""),
                "tokenizer_path": str(old_init_split / "tokenizer"),
            }
        )

        # 5) paper WECHSEL init
        specs.append(
            {
                "lang": lang,
                "name": f"wechsel-init-paper-spm32k-{lang}",
                "source_path": str(paper_init_split),
            }
        )

        # 6) paper WECHSEL continued
        specs.append(
            {
                "lang": lang,
                "name": f"wechsel-continued-paper-spm32k-{lang}",
                "source_path": str(paper_cont_ckpt) if paper_cont_ckpt else "",
                "tokenizer_path": str(paper_init_split / "tokenizer"),
            }
        )

    return specs


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    specs = _build_specs()
    manifest: list[dict[str, Any]] = []

    for spec in specs:
        lang = spec["lang"]
        name = spec["name"]
        source_path = spec.get("source_path", "")
        tokenizer_path = spec.get("tokenizer_path")
        target_dir = OUTPUT_ROOT / lang / name

        entry: dict[str, Any] = {
            "lang": lang,
            "name": name,
            "source_path": source_path,
            "tokenizer_path": tokenizer_path,
            "target_dir": str(target_dir),
            "status": "pending",
        }

        try:
            if not source_path:
                raise FileNotFoundError(f"No source resolved for {lang}/{name}")
            _materialize(spec, target_dir)
            if not _has_minimal_hf_t5_artifacts(target_dir):
                raise RuntimeError(f"Incomplete model artifacts in {target_dir}")
            entry["status"] = "ok"
        except Exception as exc:  # pylint: disable=broad-except
            entry["status"] = "error"
            entry["error"] = str(exc)

        manifest.append(entry)
        print(f"[{entry['status']}] {lang} :: {name} -> {target_dir}")

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")

    errors = [m for m in manifest if m.get("status") != "ok"]
    if errors:
        print(f"Completed with {len(errors)} error(s).")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
