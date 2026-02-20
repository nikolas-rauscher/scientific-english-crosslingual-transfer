#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class SetupSpec:
    name: str
    cfg_suffix: str


SETUPS = [
    SetupSpec(name="hf_explicit_tok", cfg_suffix="test_ppl_hf_explicit_tok"),
    SetupSpec(name="t5base_tok", cfg_suffix="test_ppl_t5base_tok"),
    SetupSpec(name="paper_spm32k_tok", cfg_suffix="test_ppl_paper_spm32k_tok"),
]


def _load_models(cfg_path: Path) -> list[dict[str, Any]]:
    cfg = OmegaConf.load(cfg_path)
    models = OmegaConf.to_container(cfg.get("models"), resolve=True)
    if not isinstance(models, list):
        raise ValueError(f"{cfg_path} does not contain a top-level models list")
    out: list[dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        if "source_path" not in m:
            continue
        out.append(m)
    return out


def _safe_name(s: str) -> str:
    return s.replace("/", "__")


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    subproject_root = project_root / "cross_lingual_transfer_multilingual"
    models_root = subproject_root / "models"
    configs_root = subproject_root / "configs" / "experiments"
    manifests_root = subproject_root / "manifests"

    # Canonical top-level layout (no data migration yet).
    hf_raw_root = models_root / "hf_raw"
    canonical_root = models_root / "canonical_v1"
    variants_root = models_root / "eval_variants_v1"
    for d in [hf_raw_root, canonical_root, variants_root, manifests_root]:
        d.mkdir(parents=True, exist_ok=True)

    targets_cfg = subproject_root / "configs" / "languages" / "targets.yaml"
    targets = OmegaConf.load(targets_cfg)
    languages = [str(x.dataset_code) for x in targets.languages]

    entries: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for setup in SETUPS:
        (variants_root / setup.name).mkdir(parents=True, exist_ok=True)
        for lang in languages:
            cfg_path = configs_root / f"{lang}_{setup.cfg_suffix}.yaml"
            if not cfg_path.exists():
                continue

            models = _load_models(cfg_path)
            for m in models:
                source_path = str(m["source_path"])
                model_name = str(m.get("name") or Path(source_path).name)
                tokenizer_path = str(m.get("tokenizer_path") or source_path)
                key = (setup.name, lang, model_name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Planned location scheme
                model_dirname = _safe_name(model_name)
                raw_path = hf_raw_root / lang / model_dirname
                canonical_path = canonical_root / lang / model_dirname
                variant_path = variants_root / setup.name / lang / model_dirname

                # Create directory skeleton now; migration/copy happens in separate step.
                for d in [raw_path.parent, canonical_path.parent, variant_path.parent]:
                    d.mkdir(parents=True, exist_ok=True)

                entries.append(
                    {
                        "setup": setup.name,
                        "language": lang,
                        "model_name": model_name,
                        "source_path_current": source_path,
                        "tokenizer_path_current": tokenizer_path,
                        "planned_paths": {
                            "hf_raw": str(raw_path),
                            "canonical_v1": str(canonical_path),
                            "eval_variant_v1": str(variant_path),
                        },
                        "status": "pending_migration",
                    }
                )

    manifest = {
        "schema_version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "subproject_root": str(subproject_root),
        "layout": {
            "hf_raw_root": str(hf_raw_root),
            "canonical_root": str(canonical_root),
            "variants_root": str(variants_root),
        },
        "languages": languages,
        "setups": [s.name for s in SETUPS],
        "notes": [
            "This manifest only scaffolds the target layout and records lineage intent.",
            "No model files are copied/mutated by this script.",
            "Use a dedicated migration/normalization step before strict evaluation.",
        ],
        "entries": entries,
    }

    out_path = manifests_root / "model_lineage_v1.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_path}")
    print(f"Entries: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
