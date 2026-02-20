#!/usr/bin/env python3
"""Build a deterministic doc-level manifest with split assignments.

This script scans raw language parquet files and writes a row-level manifest with:
- deterministic `doc_id`
- deterministic split label (`train`/`val`/`test`)
- source file + row index metadata

The split assignment is hash-based and reproducible.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def _ordered_parquet_files(data_dir: Path, file_order: str) -> List[Path]:
    if file_order == "sorted":
        files = sorted(data_dir.glob("*.parquet"))
    elif file_order == "glob":
        files = [Path(p) for p in glob.glob(str(data_dir / "*.parquet"))]
    else:
        raise ValueError(f"Unsupported file order: {file_order}")
    return files


def _normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan"}:
        return ""
    return text


def _derive_doc_id(
    *,
    pdf_md5_hash: str,
    source_zip_content_path: str,
    doi: str,
    source_file_path: str,
    parquet_filename: str,
    row_idx_in_file: int,
) -> str:
    """Create a stable doc id from available metadata columns."""
    md5 = _normalize_value(pdf_md5_hash)
    if md5:
        return f"md5:{md5.lower()}"

    zip_path = _normalize_value(source_zip_content_path)
    if zip_path:
        return f"zip:{zip_path}"

    doi_norm = _normalize_value(doi)
    if doi_norm:
        return f"doi:{doi_norm.lower()}"

    source_path = _normalize_value(source_file_path)
    if source_path:
        return f"src:{source_path}::row:{row_idx_in_file}"

    # Last-resort fallback to retain determinism.
    return f"file:{parquet_filename}::row:{row_idx_in_file}"


def _hash_unit_interval(seed: int, doc_id: str) -> float:
    key = f"{seed}:{doc_id}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    int64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int64 / 18446744073709551616.0  # 2**64


def _assign_split(u: float, train_ratio: float, val_ratio: float) -> str:
    if u < train_ratio:
        return "train"
    if u < (train_ratio + val_ratio):
        return "val"
    return "test"


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic doc manifest with split labels.")
    parser.add_argument("--language-dir", required=True, help="Directory with raw language parquet files.")
    parser.add_argument("--out-manifest", required=True, help="Output parquet path for manifest.")
    parser.add_argument("--out-summary", required=True, help="Output JSON summary path.")
    parser.add_argument("--train-ratio", type=float, default=0.949615)
    parser.add_argument("--val-ratio", type=float, default=0.000385)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-order", choices=["sorted", "glob"], default="sorted")
    parser.add_argument("--max-files", type=int, default=None, help="Optional debug limit for number of parquet files.")
    args = parser.parse_args()

    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    language_dir = Path(args.language_dir)
    out_manifest = Path(args.out_manifest)
    out_summary = Path(args.out_summary)

    files = _ordered_parquet_files(language_dir, args.file_order)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {language_dir}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None

    row_counts = {"train": 0, "val": 0, "test": 0}
    unique_doc_counts = {"train": 0, "val": 0, "test": 0}
    seen_doc_ids: set[str] = set()

    required_cols = ["pdf_md5_hash", "source_zip_content_path", "doi", "source_file_path"]
    total_rows = 0

    for file_idx, parquet_path in enumerate(files):
        pf = pq.ParquetFile(parquet_path, memory_map=True)
        available_cols = set(pf.schema_arrow.names)
        read_cols = [col for col in required_cols if col in available_cols]

        row_offset = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_table = pf.read_row_group(rg_idx, columns=read_cols if read_cols else None)
            col_data = {name: rg_table.column(name).to_pylist() for name in rg_table.column_names}

            n_rows = rg_table.num_rows
            doc_ids: List[str] = []
            splits: List[str] = []
            source_parquets: List[str] = []
            row_indices: List[int] = []
            pdf_md5_values: List[str] = []
            zip_values: List[str] = []
            doi_values: List[str] = []
            source_file_values: List[str] = []

            for i in range(n_rows):
                row_idx_in_file = row_offset + i
                pdf_md5 = _normalize_value(col_data.get("pdf_md5_hash", [""] * n_rows)[i] if "pdf_md5_hash" in col_data else "")
                zip_path = _normalize_value(col_data.get("source_zip_content_path", [""] * n_rows)[i] if "source_zip_content_path" in col_data else "")
                doi = _normalize_value(col_data.get("doi", [""] * n_rows)[i] if "doi" in col_data else "")
                source_file = _normalize_value(col_data.get("source_file_path", [""] * n_rows)[i] if "source_file_path" in col_data else "")

                doc_id = _derive_doc_id(
                    pdf_md5_hash=pdf_md5,
                    source_zip_content_path=zip_path,
                    doi=doi,
                    source_file_path=source_file,
                    parquet_filename=parquet_path.name,
                    row_idx_in_file=row_idx_in_file,
                )
                u = _hash_unit_interval(args.seed, doc_id)
                split = _assign_split(u, args.train_ratio, args.val_ratio)

                doc_ids.append(doc_id)
                splits.append(split)
                source_parquets.append(parquet_path.name)
                row_indices.append(row_idx_in_file)
                pdf_md5_values.append(pdf_md5)
                zip_values.append(zip_path)
                doi_values.append(doi)
                source_file_values.append(source_file)

                row_counts[split] += 1
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    unique_doc_counts[split] += 1

            manifest_table = pa.table(
                {
                    "doc_id": pa.array(doc_ids, type=pa.string()),
                    "split": pa.array(splits, type=pa.string()),
                    "source_parquet": pa.array(source_parquets, type=pa.string()),
                    "row_idx_in_file": pa.array(row_indices, type=pa.int64()),
                    "pdf_md5_hash": pa.array(pdf_md5_values, type=pa.string()),
                    "source_zip_content_path": pa.array(zip_values, type=pa.string()),
                    "doi": pa.array(doi_values, type=pa.string()),
                    "source_file_path": pa.array(source_file_values, type=pa.string()),
                }
            )

            if writer is None:
                writer = pq.ParquetWriter(out_manifest, manifest_table.schema, compression="snappy")
            writer.write_table(manifest_table)

            total_rows += n_rows
            row_offset += n_rows

        print(f"[{file_idx + 1}/{len(files)}] scanned {parquet_path.name}")

    if writer is not None:
        writer.close()

    summary = {
        "language_dir": str(language_dir.resolve()),
        "manifest_path": str(out_manifest.resolve()),
        "file_order": args.file_order,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "files_scanned": len(files),
        "rows_total": total_rows,
        "rows_per_split": row_counts,
        "unique_docs_total": len(seen_doc_ids),
        "unique_docs_per_split": unique_doc_counts,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Manifest built")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
