#!/usr/bin/env python3
"""Report per-language stats for multilingual subsplits and sliding windows.

This script summarizes, per language and split (train/val/test):
- docs rows and size from: splits/sub/<subsplit_name>/<split>/docs.parquet
- sliding-window file/row/size stats from:
  sliding_windows/sub/<subsplit_name>/<split>/*.parquet
- optional token stats from metadata.actual_tokens (avg/min/max)

It is robust to partially written/corrupted parquet files and reports them as bad_files.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")


@dataclass
class SplitStats:
    language: str
    split: str
    docs_exists: bool
    docs_rows: int
    docs_size_bytes: int
    windows_dir_exists: bool
    windows_valid_files: int
    windows_bad_files: int
    windows_rows: int
    windows_size_bytes: int
    token_rows_seen: int
    avg_actual_tokens: float | None
    min_actual_tokens: int | None
    max_actual_tokens: int | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report stats for multilingual subsplits/windows.")
    ap.add_argument(
        "--languages-root",
        default="cross_lingual_transfer_multilingual/data/languages",
        help="Root directory containing per-language folders.",
    )
    ap.add_argument(
        "--subsplit-name",
        required=True,
        help="Subsplit name under splits/sub and sliding_windows/sub (e.g. sub_charcap43gb_seed42).",
    )
    ap.add_argument(
        "--languages",
        default="all",
        help="Comma-separated language codes or 'all'.",
    )
    ap.add_argument(
        "--token-stats",
        choices=("none", "sample", "full"),
        default="sample",
        help="How to compute metadata.actual_tokens stats.",
    )
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=200000,
        help="Per language/split row cap when --token-stats=sample.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Batch size used for parquet streaming.",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="Optional path to save full JSON output.",
    )
    return ap.parse_args()


def human_gb(size_bytes: int) -> float:
    return round(size_bytes / (1024**3), 4)


def iter_languages(root: Path, lang_arg: str) -> list[str]:
    if lang_arg.strip().lower() == "all":
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    return [x.strip() for x in lang_arg.split(",") if x.strip()]


def safe_parquet_rows(path: Path) -> tuple[int, bool]:
    try:
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows), True
    except Exception:
        return 0, False


def scan_actual_tokens(
    parquet_files: Iterable[Path],
    mode: str,
    sample_rows: int,
    batch_size: int,
) -> tuple[int, float | None, int | None, int | None]:
    if mode == "none":
        return 0, None, None, None

    seen = 0
    total = 0.0
    tmin: int | None = None
    tmax: int | None = None
    row_cap = sample_rows if mode == "sample" else math.inf

    for f in parquet_files:
        if seen >= row_cap:
            break
        try:
            pf = pq.ParquetFile(f)
        except Exception:
            continue
        try:
            batches = pf.iter_batches(batch_size=batch_size, columns=["metadata"])
            for rb in batches:
                md_col = rb.column(0)
                for md in md_col.to_pylist():
                    if seen >= row_cap:
                        break
                    if not isinstance(md, dict):
                        continue
                    tok = md.get("actual_tokens")
                    if not isinstance(tok, (int, float)):
                        continue
                    tok_i = int(tok)
                    total += tok_i
                    seen += 1
                    tmin = tok_i if tmin is None else min(tmin, tok_i)
                    tmax = tok_i if tmax is None else max(tmax, tok_i)
        except Exception:
            continue

    avg = (total / seen) if seen > 0 else None
    return seen, avg, tmin, tmax


def collect_split_stats(
    lang_root: Path,
    language: str,
    split: str,
    subsplit_name: str,
    token_stats_mode: str,
    sample_rows: int,
    batch_size: int,
) -> SplitStats:
    docs_path = lang_root / language / "splits" / "sub" / subsplit_name / split / "docs.parquet"
    docs_exists = docs_path.exists()
    docs_rows = 0
    docs_size_bytes = int(docs_path.stat().st_size) if docs_exists else 0
    if docs_exists:
        docs_rows, _ = safe_parquet_rows(docs_path)

    windows_dir = lang_root / language / "sliding_windows" / "sub" / subsplit_name / split
    windows_dir_exists = windows_dir.exists()
    files = sorted(windows_dir.glob("*.parquet")) if windows_dir_exists else []

    valid_files = 0
    bad_files = 0
    rows = 0
    size_bytes = 0
    valid_file_list: list[Path] = []
    for f in files:
        r, ok = safe_parquet_rows(f)
        if ok:
            valid_files += 1
            rows += r
            size_bytes += int(f.stat().st_size)
            valid_file_list.append(f)
        else:
            bad_files += 1

    tok_seen, tok_avg, tok_min, tok_max = scan_actual_tokens(
        valid_file_list,
        mode=token_stats_mode,
        sample_rows=sample_rows,
        batch_size=batch_size,
    )

    return SplitStats(
        language=language,
        split=split,
        docs_exists=docs_exists,
        docs_rows=docs_rows,
        docs_size_bytes=docs_size_bytes,
        windows_dir_exists=windows_dir_exists,
        windows_valid_files=valid_files,
        windows_bad_files=bad_files,
        windows_rows=rows,
        windows_size_bytes=size_bytes,
        token_rows_seen=tok_seen,
        avg_actual_tokens=tok_avg,
        min_actual_tokens=tok_min,
        max_actual_tokens=tok_max,
    )


def print_table(rows: list[SplitStats]) -> None:
    print(
        "language\tsplit\tdocs_rows\tdocs_gb\twin_files\tbad_files\twin_rows\twin_gb\t"
        "tok_rows\tavg_tok\tmin_tok\tmax_tok"
    )
    for r in rows:
        avg_tok = f"{r.avg_actual_tokens:.2f}" if r.avg_actual_tokens is not None else "-"
        min_tok = str(r.min_actual_tokens) if r.min_actual_tokens is not None else "-"
        max_tok = str(r.max_actual_tokens) if r.max_actual_tokens is not None else "-"
        print(
            f"{r.language}\t{r.split}\t{r.docs_rows}\t{human_gb(r.docs_size_bytes)}\t"
            f"{r.windows_valid_files}\t{r.windows_bad_files}\t{r.windows_rows}\t{human_gb(r.windows_size_bytes)}\t"
            f"{r.token_rows_seen}\t{avg_tok}\t{min_tok}\t{max_tok}"
        )


def main() -> int:
    args = parse_args()
    lang_root = Path(args.languages_root)
    languages = iter_languages(lang_root, args.languages)

    out_rows: list[SplitStats] = []
    for lang in languages:
        for split in SPLITS:
            out_rows.append(
                collect_split_stats(
                    lang_root=lang_root,
                    language=lang,
                    split=split,
                    subsplit_name=args.subsplit_name,
                    token_stats_mode=args.token_stats,
                    sample_rows=args.sample_rows,
                    batch_size=args.batch_size,
                )
            )

    print_table(out_rows)

    if args.out_json:
        payload = {
            "languages_root": str(lang_root),
            "subsplit_name": args.subsplit_name,
            "token_stats_mode": args.token_stats,
            "sample_rows": args.sample_rows,
            "rows": [asdict(r) for r in out_rows],
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

