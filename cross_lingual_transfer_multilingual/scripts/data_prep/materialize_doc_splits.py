#!/usr/bin/env python3
"""Materialize doc-level train/val/test parquet files from raw data + manifest.

Input:
- raw language parquet folder
- manifest parquet from build_doc_manifest.py

Output:
- <output-dir>/train/part-*.parquet
- <output-dir>/val/part-*.parquet
- <output-dir>/test/part-*.parquet
- summary JSON
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

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

    return f"file:{parquet_filename}::row:{row_idx_in_file}"


def _load_split_map(manifest_path: Path) -> Dict[str, str]:
    table = pq.read_table(manifest_path, columns=["doc_id", "split"])
    doc_ids = table.column("doc_id").to_pylist()
    splits = table.column("split").to_pylist()

    split_map: Dict[str, str] = {}
    for doc_id, split in zip(doc_ids, splits):
        if doc_id in split_map:
            if split_map[doc_id] != split:
                raise ValueError(f"Conflicting split for doc_id={doc_id}")
            continue
        split_map[doc_id] = split

    return split_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize train/val/test doc parquets from manifest.")
    parser.add_argument("--language-dir", required=True, help="Directory with raw language parquet files.")
    parser.add_argument("--manifest", required=True, help="Manifest parquet from build_doc_manifest.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for split parquet files.")
    parser.add_argument("--out-summary", required=True, help="Output JSON summary path.")
    parser.add_argument("--file-order", choices=["sorted", "glob"], default="sorted")
    parser.add_argument("--max-files", type=int, default=None, help="Optional debug limit for number of parquet files.")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing split parquet files before writing new ones.",
    )
    args = parser.parse_args()

    language_dir = Path(args.language_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    out_summary = Path(args.out_summary)

    files = _ordered_parquet_files(language_dir, args.file_order)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {language_dir}")

    split_map = _load_split_map(manifest_path)
    if not split_map:
        raise ValueError("Manifest is empty")

    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        if args.clean_output:
            for old_file in (output_dir / split).glob("*.parquet"):
                old_file.unlink()
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    chunk_counter = {"train": 0, "val": 0, "test": 0}
    row_counter = {"train": 0, "val": 0, "test": 0}
    missing_doc_id_rows = 0
    writers: Dict[str, pq.ParquetWriter] = {}
    output_files = {
        "train": output_dir / "train" / "docs.parquet",
        "val": output_dir / "val" / "docs.parquet",
        "test": output_dir / "test" / "docs.parquet",
    }

    for file_idx, parquet_path in enumerate(files):
        pf = pq.ParquetFile(parquet_path, memory_map=True)
        row_offset = 0

        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            n_rows = table.num_rows

            col_names = set(table.column_names)
            pdf_col = table.column("pdf_md5_hash").to_pylist() if "pdf_md5_hash" in col_names else [""] * n_rows
            zip_col = table.column("source_zip_content_path").to_pylist() if "source_zip_content_path" in col_names else [""] * n_rows
            doi_col = table.column("doi").to_pylist() if "doi" in col_names else [""] * n_rows
            src_col = table.column("source_file_path").to_pylist() if "source_file_path" in col_names else [""] * n_rows

            doc_ids: List[str] = []
            split_labels: List[str] = []

            for i in range(n_rows):
                row_idx_in_file = row_offset + i
                doc_id = _derive_doc_id(
                    pdf_md5_hash=pdf_col[i],
                    source_zip_content_path=zip_col[i],
                    doi=doi_col[i],
                    source_file_path=src_col[i],
                    parquet_filename=parquet_path.name,
                    row_idx_in_file=row_idx_in_file,
                )
                split = split_map.get(doc_id)
                if split is None:
                    missing_doc_id_rows += 1
                    # Strict mode for data integrity.
                    raise KeyError(
                        f"doc_id missing in manifest: {doc_id} (file={parquet_path.name}, row={row_idx_in_file})"
                    )

                doc_ids.append(doc_id)
                split_labels.append(split)

            table = table.append_column("id", pa.array(doc_ids, type=pa.string()))
            table = table.append_column("doc_split", pa.array(split_labels, type=pa.string()))

            indices_by_split = {"train": [], "val": [], "test": []}
            for i, split in enumerate(split_labels):
                indices_by_split[split].append(i)

            for split, indices in indices_by_split.items():
                if not indices:
                    continue
                split_table = table.take(pa.array(indices, type=pa.int64()))
                if split not in writers:
                    writers[split] = pq.ParquetWriter(output_files[split], split_table.schema, compression="snappy")
                writers[split].write_table(split_table)
                chunk_counter[split] += 1
                row_counter[split] += split_table.num_rows

            row_offset += n_rows

        print(f"[{file_idx + 1}/{len(files)}] materialized {parquet_path.name}")

    for writer in writers.values():
        writer.close()

    summary = {
        "language_dir": str(language_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "file_order": args.file_order,
        "files_processed": len(files),
        "chunks_per_split": chunk_counter,
        "output_files": {k: str(v.resolve()) for k, v in output_files.items()},
        "rows_per_split": row_counter,
        "rows_total_written": sum(row_counter.values()),
        "missing_doc_id_rows": missing_doc_id_rows,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Doc split materialization complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
