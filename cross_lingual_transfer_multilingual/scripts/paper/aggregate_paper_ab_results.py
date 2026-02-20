#!/usr/bin/env python3
"""Aggregate paper-track A/B Global-MMLU results into CSV (no CLI args)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/netscratch/nrauscher/projects/BA-hydra")
SUMMARY_GLOB = (
    "logs/eval_pipeline/runs/**/evaluation/results/universal/"
    "clt_multi_mmlu_paper_*_summary_*.json"
)
OUTPUT_DIR = PROJECT_ROOT / "logs" / "eval_pipeline" / "aggregated" / "paper_spm32k_ab"

TOP_CATEGORIES = ["overall", "stem", "humanities", "social_sciences", "other"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_latest_summary_per_experiment(paths: list[Path]) -> list[Path]:
    by_exp: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        marker = "_summary_"
        if marker not in stem:
            continue
        exp_name = stem.split(marker, 1)[0]
        prev = by_exp.get(exp_name)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            by_exp[exp_name] = path
    return sorted(by_exp.values())


def _find_raw_result(exp_dir: Path, model_name: str, lang_code: str) -> Path | None:
    pattern = f"{model_name}_global_mmlu_{lang_code}_0shot_*/*/results_*.json"
    candidates = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _extract_metrics(raw: dict[str, Any], lang_code: str) -> dict[str, float | None]:
    groups = raw.get("groups", {})
    prefix = f"global_mmlu_full_{lang_code}"

    keys = {
        "overall": prefix,
        "humanities": f"{prefix}_humanities",
        "social_sciences": f"{prefix}_social_sciences",
        "stem": f"{prefix}_stem",
        "other": f"{prefix}_other",
    }

    out: dict[str, float | None] = {}
    for label, key in keys.items():
        block = groups.get(key, {})
        value = block.get("acc,none")
        out[label] = float(value) if value is not None else None
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_paths = [Path(p) for p in PROJECT_ROOT.glob(SUMMARY_GLOB)]
    selected = _pick_latest_summary_per_experiment(summary_paths)

    rows: list[dict[str, Any]] = []
    for summary_path in selected:
        summary = _read_json(summary_path)
        exp_info = summary.get("experiment_info", {})
        exp_name = str(exp_info.get("name", ""))
        benchmarks = exp_info.get("benchmarks", []) or []
        if not exp_name or not benchmarks:
            continue

        benchmark = str(benchmarks[0])
        if not benchmark.startswith("global_mmlu_"):
            continue
        lang_code = benchmark.replace("global_mmlu_", "", 1)

        if not exp_name.startswith("clt_multi_mmlu_paper_"):
            continue

        lang = exp_name.replace("clt_multi_mmlu_paper_", "", 1)
        exp_dir = summary_path.parent / exp_name

        for model_name, model_block in (summary.get("models", {}) or {}).items():
            status = str(model_block.get("status", ""))
            raw_path = _find_raw_result(exp_dir, model_name, lang_code)
            metrics = {k: None for k in TOP_CATEGORIES}
            if raw_path is not None and raw_path.exists():
                metrics = _extract_metrics(_read_json(raw_path), lang_code)

            row = {
                "language": lang,
                "lang_code": lang_code,
                "experiment": exp_name,
                "model": model_name,
                "status": status,
                "summary_file": str(summary_path.relative_to(PROJECT_ROOT)),
                "raw_result_file": str(raw_path.relative_to(PROJECT_ROOT)) if raw_path else "",
                "overall": metrics["overall"],
                "stem": metrics["stem"],
                "humanities": metrics["humanities"],
                "social_sciences": metrics["social_sciences"],
                "other": metrics["other"],
            }
            rows.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"paper_spm32k_ab_{ts}.csv"

    fieldnames = [
        "language",
        "lang_code",
        "experiment",
        "model",
        "status",
        "overall",
        "stem",
        "humanities",
        "social_sciences",
        "other",
        "summary_file",
        "raw_result_file",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["language"], r["model"])):
            writer.writerow(row)

    print(f"Selected summary files: {len(selected)}")
    print(f"Rows written: {len(rows)}")
    print(f"CSV: {out_csv}")

    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
