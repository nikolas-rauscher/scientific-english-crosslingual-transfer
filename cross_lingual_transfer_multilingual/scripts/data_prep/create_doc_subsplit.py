#!/usr/bin/env python3
"""
Create deterministic document-level subsplits from existing split parquet files.

This script samples a fixed number of rows from each split (train/val/test)
without replacement and writes them to a new target split directory.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import shutil
from pathlib import Path
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test doc subsplits."
    )
    parser.add_argument(
        "--source-splits-dir",
        required=True,
        help="Source split directory containing train/val/test docs.parquet.",
    )
    parser.add_argument(
        "--target-splits-dir",
        required=True,
        help="Target split directory to write train/val/test docs.parquet.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Requested number of rows for train split (-1 keeps all rows).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        required=True,
        help="Requested number of rows for val split (-1 keeps all rows).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        required=True,
        help="Requested number of rows for test split (-1 keeps all rows).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic sampling.",
    )
    parser.add_argument(
        "--allow-smaller",
        action="store_true",
        help="If requested size exceeds available rows, clamp to available rows.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete target split directory before writing output.",
    )
    return parser.parse_args()


def resolve_target_size(
    requested: int, available: int, split: str, allow_smaller: bool
) -> int:
    if requested < 0:
        return available
    if requested <= available:
        return requested
    if allow_smaller:
        return available
    raise ValueError(
        f"Requested {requested} rows for split '{split}', but only {available} available. "
        "Use --allow-smaller to clamp automatically."
    )


def sample_table_rows(table: pa.Table, target_size: int, seed: int) -> pa.Table:
    total_rows = table.num_rows
    if target_size == total_rows:
        return table
    if target_size == 0:
        return table.slice(0, 0)

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total_rows), k=target_size))
    return table.take(pa.array(indices, type=pa.int64()))


def write_sampled_parquet(
    source_file: Path,
    out_file: Path,
    target_size: int,
    seed: int,
) -> None:
    """Write deterministic sampled rows from source parquet to output parquet.

    Implementation is streaming over record batches to avoid pyarrow offset
    overflows when taking from very large string columns.
    """
    parquet = pq.ParquetFile(source_file)
    total_rows = parquet.metadata.num_rows

    if target_size == total_rows:
        shutil.copyfile(source_file, out_file)
        return

    writer = pq.ParquetWriter(out_file, parquet.schema_arrow, compression="snappy")
    try:
        if target_size == 0:
            return

        rng = random.Random(seed)
        sample_indices = sorted(rng.sample(range(total_rows), k=target_size))
        sample_ptr = 0
        row_offset = 0

        for batch in parquet.iter_batches(batch_size=65536):
            batch_rows = batch.num_rows
            batch_end = row_offset + batch_rows

            left = sample_ptr
            right = bisect.bisect_left(sample_indices, batch_end, lo=sample_ptr)
            if right > left:
                local_indices = [sample_indices[i] - row_offset for i in range(left, right)]
                table = pa.Table.from_batches([batch])
                sampled_batch = table.take(pa.array(local_indices, type=pa.int64()))
                writer.write_table(sampled_batch)
                sample_ptr = right

            row_offset = batch_end
            if sample_ptr >= target_size:
                break
    finally:
        writer.close()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_splits_dir).resolve()
    target_dir = Path(args.target_splits_dir).resolve()

    requested_sizes = {
        "train": int(args.train_size),
        "val": int(args.val_size),
        "test": int(args.test_size),
    }

    for split in SPLITS:
        source_file = source_dir / split / "docs.parquet"
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source split file: {source_file}")

    if args.clean_output and target_dir.exists():
        shutil.rmtree(target_dir)

    summary: Dict[str, object] = {
        "source_splits_dir": str(source_dir),
        "target_splits_dir": str(target_dir),
        "requested_sizes": requested_sizes,
        "seed": int(args.seed),
        "allow_smaller": bool(args.allow_smaller),
        "splits": {},
    }

    for split_offset, split in enumerate(SPLITS):
        source_file = source_dir / split / "docs.parquet"
        out_dir = target_dir / split
        out_file = out_dir / "docs.parquet"
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet = pq.ParquetFile(source_file)
        available = parquet.metadata.num_rows
        target_size = resolve_target_size(
            requested=requested_sizes[split],
            available=available,
            split=split,
            allow_smaller=args.allow_smaller,
        )

        write_sampled_parquet(
            source_file=source_file,
            out_file=out_file,
            target_size=target_size,
            seed=args.seed + split_offset,
        )

        print(
            f"[{split}] source_rows={available} requested={requested_sizes[split]} "
            f"written={target_size} -> {out_file}"
        )

        summary["splits"][split] = {
            "source_file": str(source_file),
            "target_file": str(out_file),
            "source_rows": int(available),
            "requested_rows": int(requested_sizes[split]),
            "written_rows": int(target_size),
        }

    summary_path = target_dir / "subsplit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
