#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    subproject_root = project_root / "cross_lingual_transfer_multilingual"
    default_targets = subproject_root / "configs" / "languages" / "targets.yaml"
    default_out = subproject_root / "models" / "hf_raw"

    p = argparse.ArgumentParser(description="Download monolingual HF source models into hf_raw layout.")
    p.add_argument("--targets-config", default=str(default_targets))
    p.add_argument("--output-root", default=str(default_out))
    p.add_argument("--languages", default="all", help="Comma-separated dataset codes or 'all'.")
    p.add_argument("--force", action="store_true", help="Overwrite existing destination directories.")
    p.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    args = p.parse_args()

    targets_cfg = OmegaConf.load(args.targets_config)
    rows = list(targets_cfg.languages)
    by_lang = {str(r.dataset_code): r for r in rows}

    if args.languages.strip().lower() == "all":
        selected = list(by_lang.values())
    else:
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]
        missing = [x for x in langs if x not in by_lang]
        if missing:
            raise ValueError(f"Unknown languages in --languages: {missing}")
        selected = [by_lang[x] for x in langs]

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for row in selected:
        lang = str(row.dataset_code)
        model_id = str(row.hf_monolingual_model)
        model_dir = out_root / lang / _safe_model_id(model_id)

        print(f"[plan] {lang}: {model_id} -> {model_dir}")
        if args.dry_run:
            continue

        if model_dir.exists() and not args.force:
            print(f"[skip] exists: {model_dir}")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tok.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        meta = {
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_code": lang,
            "source_hf_model_id": model_id,
            "destination": str(model_dir),
        }
        (model_dir / "download_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ok] {model_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

