#!/usr/bin/env python3
"""Split one parquet file into N balanced shard parquet files.

Sharding is deterministic by global row index:
  shard_id = global_row_idx % num_shards
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a parquet file into N shards.")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output-dir", required=True, help="Output shard directory")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of output shards")
    parser.add_argument("--clean-output", action="store_true", help="Remove existing shard parquet files")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for old in output_dir.glob("shard-*.parquet"):
            old.unlink()

    pf = pq.ParquetFile(input_path, memory_map=True)
    writers: Dict[int, pq.ParquetWriter] = {}
    shard_rows = {i: 0 for i in range(args.num_shards)}

    global_row_idx = 0
    for rb in pf.iter_batches(batch_size=65536):
        table = pa.Table.from_batches([rb])
        n_rows = table.num_rows
        if n_rows == 0:
            continue

        idxs_by_shard = {i: [] for i in range(args.num_shards)}
        for local_idx in range(n_rows):
            shard_id = global_row_idx % args.num_shards
            idxs_by_shard[shard_id].append(local_idx)
            global_row_idx += 1

        for shard_id, idxs in idxs_by_shard.items():
            if not idxs:
                continue
            shard_table = table.take(pa.array(idxs, type=pa.int64()))
            out_path = output_dir / f"shard-{shard_id:05d}.parquet"
            if shard_id not in writers:
                writers[shard_id] = pq.ParquetWriter(out_path, shard_table.schema, compression="snappy")
            writers[shard_id].write_table(shard_table)
            shard_rows[shard_id] += shard_table.num_rows

    for writer in writers.values():
        writer.close()

    non_empty = sum(1 for v in shard_rows.values() if v > 0)
    print(
        f"Sharded {input_path} -> {output_dir} "
        f"(requested={args.num_shards}, non_empty={non_empty}, rows={sum(shard_rows.values())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
