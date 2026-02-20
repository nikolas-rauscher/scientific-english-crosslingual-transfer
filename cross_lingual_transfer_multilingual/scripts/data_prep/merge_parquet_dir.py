#!/usr/bin/env python3
"""Merge multiple parquet files from a directory into a single parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge parquet files in a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--glob", default="*.parquet", help="Glob pattern inside input-dir")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--clean-output", action="store_true", help="Delete output path if it exists")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir} matching {args.glob}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.clean_output:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output exists: {out_path}. Use --clean-output.")

    writer = None
    total_rows = 0

    for f in files:
        pf = pq.ParquetFile(f, memory_map=True)
        for rb in pf.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([rb])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += table.num_rows

    if writer is not None:
        writer.close()

    print(f"Merged {len(files)} files -> {out_path} ({total_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
