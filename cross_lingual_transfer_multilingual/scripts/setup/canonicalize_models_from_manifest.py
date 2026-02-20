#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _model_stats(model, tokenizer) -> dict[str, Any]:
    emb = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight
    return {
        "config_vocab_size": int(getattr(model.config, "vocab_size")),
        "tokenizer_len": int(len(tokenizer)),
        "input_embedding_shape": list(emb.shape),
        "output_embedding_shape": list(out.shape),
        "pad_token_id_cfg": getattr(model.config, "pad_token_id", None),
        "eos_token_id_cfg": getattr(model.config, "eos_token_id", None),
        "decoder_start_token_id_cfg": getattr(model.config, "decoder_start_token_id", None),
        "pad_token_id_tok": tokenizer.pad_token_id,
        "eos_token_id_tok": tokenizer.eos_token_id,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    default_manifest = project_root / "cross_lingual_transfer_multilingual" / "manifests" / "model_lineage_v1.json"

    p = argparse.ArgumentParser(description="Canonicalize hf_raw models to tokenizer-aligned vocab size.")
    p.add_argument("--manifest", default=str(default_manifest))
    p.add_argument("--languages", default="all", help="Comma-separated dataset codes or 'all'.")
    p.add_argument("--setups", default="all", help="Comma-separated setup names or 'all'.")
    p.add_argument("--force", action="store_true", help="Overwrite existing canonical directories.")
    args = p.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required for canonicalization. "
            "Run with: /netscratch/anonymous_user/projects/BA-hydra/.venv_pretraining/bin/python "
            "cross_lingual_transfer_multilingual/scripts/setup/canonicalize_models_from_manifest.py"
        ) from exc

    manifest_path = Path(args.manifest).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])

    allowed_langs = None if args.languages.strip().lower() == "all" else {
        x.strip() for x in args.languages.split(",") if x.strip()
    }
    allowed_setups = None if args.setups.strip().lower() == "all" else {
        x.strip() for x in args.setups.split(",") if x.strip()
    }

    # Dedupe by canonical destination path.
    unique: dict[str, dict[str, Any]] = {}
    for e in entries:
        if allowed_langs is not None and e["language"] not in allowed_langs:
            continue
        if allowed_setups is not None and e["setup"] not in allowed_setups:
            continue
        unique[str(e["planned_paths"]["canonical_v1"])] = e

    converted = 0
    skipped = 0
    failed = 0
    failed_paths: dict[str, str] = {}

    for canonical_path_str, e in sorted(unique.items()):
        src = Path(str(e["planned_paths"]["hf_raw"]))
        dst = Path(canonical_path_str)
        model_name = str(e["model_name"])

        if not src.exists():
            print(f"[missing] hf_raw source missing for {model_name}: {src}")
            failed += 1
            continue

        if dst.exists() and (dst / "canonicalization_metadata.json").exists() and not args.force:
            print(f"[skip] canonical exists: {dst}")
            skipped += 1
            continue

        print(f"[plan] canonicalize {src} -> {dst}")
        try:
            if dst.exists():
                # force=True path; remove old destination
                import shutil
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained(src, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(src)

            before = _model_stats(model, tokenizer)
            target_vocab = int(len(tokenizer))
            model.resize_token_embeddings(target_vocab)
            model.config.vocab_size = target_vocab
            after = _model_stats(model, tokenizer)

            model.save_pretrained(dst)
            tokenizer.save_pretrained(dst)

            meta = {
                "canonicalized_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_hf_raw_path": str(src),
                "canonical_path": str(dst),
                "model_name": model_name,
                "before": before,
                "after": after,
            }
            (dst / "canonicalization_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            converted += 1
            print(f"[ok] {dst}")
        except Exception as exc:
            failed += 1
            failed_paths[str(dst)] = str(exc)
            if dst.exists() and not (dst / "canonicalization_metadata.json").exists():
                import shutil
                shutil.rmtree(dst)
            print(f"[fail] {model_name}: {exc}")

    # Update manifest entries status if canonical path now exists.
    now = datetime.now(timezone.utc).isoformat()
    for e in entries:
        cpath = Path(str(e["planned_paths"]["canonical_v1"]))
        if str(cpath) in failed_paths:
            e["status"] = "canonicalization_failed"
            e["canonicalization_error"] = failed_paths[str(cpath)]
            e["canonicalized_at_utc"] = now
        elif cpath.exists() and (cpath / "canonicalization_metadata.json").exists():
            e["status"] = "canonicalized"
            e.pop("canonicalization_error", None)
            e["canonicalized_at_utc"] = now

    data["entries"] = entries
    data["last_canonicalization_run_utc"] = now
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"done: converted={converted} skipped={skipped} failed={failed}")
    print(f"manifest updated: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
