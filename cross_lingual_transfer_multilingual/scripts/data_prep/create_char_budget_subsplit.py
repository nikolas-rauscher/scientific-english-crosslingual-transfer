#!/usr/bin/env python3
"""Create deterministic char-budget-capped subsplits with re-splitting.

Flow:
1) Determine total character budget (fixed total or reference language total).
2) Keep a deterministic subset of source docs so total chars roughly match the budget.
3) Re-split kept docs into train/val/test with configured ratios (default 93/2/5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create char-budget-capped train/val/test doc subsplits with re-splitting."
    )
    parser.add_argument(
        "--source-splits-dir",
        required=True,
        help="Source split directory containing train/val/test docs.parquet.",
    )
    parser.add_argument(
        "--reference-splits-dir",
        default=None,
        help=(
            "Optional reference split directory used to derive the global char cap. "
            "Required only when no fixed target total is provided."
        ),
    )
    parser.add_argument(
        "--target-total-gb",
        type=float,
        default=None,
        help="Optional fixed total character budget in decimal GB (1 GB = 1e9 chars).",
    )
    parser.add_argument(
        "--target-total-chars",
        type=int,
        default=None,
        help="Optional fixed total character budget (overrides reference-based cap).",
    )
    parser.add_argument(
        "--fixed-train-gb",
        type=float,
        default=None,
        help="Optional fixed train budget in decimal GB (1 GB = 1e9 chars).",
    )
    parser.add_argument(
        "--fixed-val-gb",
        type=float,
        default=None,
        help="Optional fixed val budget in decimal GB (1 GB = 1e9 chars).",
    )
    parser.add_argument(
        "--fixed-test-gb",
        type=float,
        default=None,
        help="Optional fixed test budget in decimal GB (1 GB = 1e9 chars).",
    )
    parser.add_argument(
        "--target-splits-dir",
        required=True,
        help="Target split directory to write train/val/test docs.parquet.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Text column name used for character counting.",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Document id column name used for deterministic keep/split decisions.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.93)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--keep-seed",
        type=int,
        default=42,
        help="Seed for deterministic keep/drop decision for char cap.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=43,
        help="Seed for deterministic train/val/test assignment after capping.",
    )
    parser.add_argument(
        "--allow-smaller",
        action="store_true",
        help="If source split cannot reach budget, clamp instead of failing.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete target split directory before writing output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Parquet streaming batch size.",
    )
    return parser.parse_args()


def _column_index_or_raise(schema: pa.Schema, col_name: str) -> int:
    idx = schema.get_field_index(col_name)
    if idx < 0:
        raise ValueError(f"Column '{col_name}' not found in schema: {schema.names}")
    return idx


def _hash_unit_interval(seed: int, value: str) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    int64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int64 / 18446744073709551616.0


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


def sum_chars(parquet_file: Path, text_key: str, batch_size: int) -> Tuple[int, int]:
    pf = pq.ParquetFile(parquet_file)
    text_idx = _column_index_or_raise(pf.schema_arrow, text_key)
    total_rows = 0
    total_chars = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        text_col = table.column(text_idx)
        char_len = pc.utf8_length(text_col)
        batch_chars = pc.sum(char_len).as_py()
        total_rows += table.num_rows
        total_chars += int(batch_chars or 0)

    return total_rows, total_chars


def sum_chars_across_splits(
    splits_dir: Path, text_key: str, batch_size: int
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    total_rows = 0
    total_chars = 0
    per_split: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        f = splits_dir / split / "docs.parquet"
        rows, chars = sum_chars(f, text_key, batch_size)
        per_split[split] = {"rows": int(rows), "chars": int(chars)}
        total_rows += rows
        total_chars += chars
    return int(total_rows), int(total_chars), per_split


def _gb_to_chars(gb: float) -> int:
    if gb <= 0:
        raise ValueError(f"GB budget must be > 0, got {gb}")
    return int(round(gb * 1_000_000_000))


def _pick_indices_by_budget(
    keys: list[str],
    chars: list[int],
    seed: int,
    budget_chars: int | None,
    excluded: set[int],
) -> Tuple[set[int], int]:
    if budget_chars is not None and budget_chars <= 0:
        return set(), 0

    if budget_chars is None:
        selected = {i for i in range(len(keys)) if i not in excluded}
        total = sum(chars[i] for i in selected)
        return selected, int(total)

    candidates: list[Tuple[float, int]] = []
    for i, key in enumerate(keys):
        if i in excluded:
            continue
        candidates.append((_hash_unit_interval(seed, key), i))
    candidates.sort(key=lambda x: x[0])

    selected: set[int] = set()
    total = 0
    best_overshoot_idx: int | None = None
    best_overshoot_delta: int | None = None

    for _, i in candidates:
        if total >= budget_chars:
            break
        c = chars[i]
        remaining = budget_chars - total
        if c <= remaining:
            selected.add(i)
            total += c
            continue

        # Track closest overshoot candidate in deterministic hash order.
        delta = c - remaining
        if best_overshoot_delta is None or delta < best_overshoot_delta:
            best_overshoot_delta = delta
            best_overshoot_idx = i

    # If we are under budget and a single overshoot document gets us closer, add it.
    if total < budget_chars and best_overshoot_idx is not None and best_overshoot_idx not in selected:
        under_gap = budget_chars - total
        over_gap = chars[best_overshoot_idx] - under_gap
        if over_gap < under_gap:
            selected.add(best_overshoot_idx)
            total += chars[best_overshoot_idx]

    # If still empty (e.g., all docs are larger than budget), take one deterministic doc.
    if not selected and candidates:
        i = candidates[0][1]
        selected.add(i)
        total += chars[i]

    return selected, int(total)


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_dir = Path(args.source_splits_dir).resolve()
    reference_dir = Path(args.reference_splits_dir).resolve() if args.reference_splits_dir else None
    target_dir = Path(args.target_splits_dir).resolve()

    fixed_split_mode = any(
        v is not None for v in (args.fixed_train_gb, args.fixed_val_gb, args.fixed_test_gb)
    )

    if args.target_total_gb is not None and args.target_total_chars is not None:
        raise ValueError("Use only one of --target-total-gb or --target-total-chars.")
    if fixed_split_mode and (args.target_total_gb is not None or args.target_total_chars is not None):
        raise ValueError(
            "Do not combine fixed split budgets (--fixed-*-gb) with --target-total-gb/--target-total-chars."
        )

    target_total_chars: int | None = None
    if args.target_total_gb is not None:
        if args.target_total_gb <= 0:
            raise ValueError("--target-total-gb must be > 0.")
        target_total_chars = int(round(args.target_total_gb * 1_000_000_000))
    elif args.target_total_chars is not None:
        if args.target_total_chars <= 0:
            raise ValueError("--target-total-chars must be > 0.")
        target_total_chars = int(args.target_total_chars)

    if fixed_split_mode and (args.fixed_val_gb is None or args.fixed_test_gb is None):
        raise ValueError("When using fixed split mode, provide both --fixed-val-gb and --fixed-test-gb.")

    if not fixed_split_mode and reference_dir is None and target_total_chars is None:
        raise ValueError(
            "Provide either --reference-splits-dir or a fixed budget via "
            "--target-total-gb/--target-total-chars."
        )

    for split in SPLITS:
        source_file = source_dir / split / "docs.parquet"
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source split file: {source_file}")
        if reference_dir is not None and not fixed_split_mode:
            ref_file = reference_dir / split / "docs.parquet"
            if not ref_file.exists():
                raise FileNotFoundError(f"Missing reference split file: {ref_file}")

    if args.clean_output and target_dir.exists():
        shutil.rmtree(target_dir)

    src_total_rows, src_total_chars, src_per_split = sum_chars_across_splits(
        source_dir, args.text_key, args.batch_size
    )
    if src_total_chars <= 0:
        raise ValueError("Source total chars is 0; cannot build capped subset.")

    ref_total_rows = 0
    ref_total_chars = 0
    ref_per_split: Dict[str, Dict[str, int]] = {}

    if fixed_split_mode:
        budget_mode = "fixed_split_chars"
        budget_train_chars = _gb_to_chars(args.fixed_train_gb) if args.fixed_train_gb is not None else None
        budget_val_chars = _gb_to_chars(args.fixed_val_gb) if args.fixed_val_gb is not None else None
        budget_test_chars = _gb_to_chars(args.fixed_test_gb) if args.fixed_test_gb is not None else None
        budget_total_chars = int(
            (budget_train_chars or 0) + (budget_val_chars or 0) + (budget_test_chars or 0)
        )
    else:
        budget_mode = "fixed_total_chars" if target_total_chars is not None else "reference_total_chars"
        if target_total_chars is None:
            assert reference_dir is not None
            ref_total_rows, ref_total_chars, ref_per_split = sum_chars_across_splits(
                reference_dir, args.text_key, args.batch_size
            )
            budget_total_chars = int(ref_total_chars)
        else:
            budget_total_chars = int(target_total_chars)
        budget_train_chars = None
        budget_val_chars = None
        budget_test_chars = None

    keep_ratio = budget_total_chars / src_total_chars if src_total_chars > 0 else 0.0
    if keep_ratio > 1.0:
        keep_ratio = 1.0
    if keep_ratio < 0.0:
        keep_ratio = 0.0

    if src_total_chars < budget_total_chars and not args.allow_smaller:
        raise ValueError(
            f"Source total chars ({src_total_chars}) < requested total chars ({budget_total_chars}). "
            "Use --allow-smaller to clamp automatically."
        )

    summary: Dict[str, object] = {
        "source_splits_dir": str(source_dir),
        "target_splits_dir": str(target_dir),
        "budget_mode": budget_mode,
        "budget_total_chars": int(budget_total_chars),
        "target_total_gb": float(args.target_total_gb) if args.target_total_gb is not None else None,
        "target_total_chars": int(target_total_chars) if target_total_chars is not None else None,
        "fixed_split_budgets_chars": {
            "train": int(budget_train_chars) if budget_train_chars is not None else None,
            "val": int(budget_val_chars) if budget_val_chars is not None else None,
            "test": int(budget_test_chars) if budget_test_chars is not None else None,
        },
        "text_key": args.text_key,
        "id_key": args.id_key,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "keep_seed": int(args.keep_seed),
        "split_seed": int(args.split_seed),
        "allow_smaller": bool(args.allow_smaller),
        "batch_size": int(args.batch_size),
        "source_total_rows": int(src_total_rows),
        "source_total_chars": int(src_total_chars),
        "keep_ratio": float(keep_ratio),
        "source_per_split": src_per_split,
        "splits": {},
    }
    if reference_dir is not None and not fixed_split_mode:
        summary["reference_splits_dir"] = str(reference_dir)
        summary["reference_total_rows"] = int(ref_total_rows)
        summary["reference_total_chars"] = int(ref_total_chars)
        summary["reference_per_split"] = ref_per_split

    # Fixed split budgets mode: deterministic split-specific selection on full source.
    if fixed_split_mode:
        keys: list[str] = []
        chars: list[int] = []
        global_idx = 0
        for source_split in SPLITS:
            source_file = source_dir / source_split / "docs.parquet"
            pf = pq.ParquetFile(source_file)
            text_idx = _column_index_or_raise(pf.schema_arrow, args.text_key)
            id_idx = _column_index_or_raise(pf.schema_arrow, args.id_key)
            for batch in pf.iter_batches(batch_size=args.batch_size):
                table = pa.Table.from_batches([batch])
                ids = table.column(id_idx).to_pylist()
                chars_list = pc.utf8_length(table.column(text_idx)).to_pylist()
                for i, raw_id in enumerate(ids):
                    doc_id = "" if raw_id is None else str(raw_id)
                    key = f"{doc_id}|{global_idx}"
                    keys.append(key)
                    chars.append(int(chars_list[i] or 0))
                    global_idx += 1

        excluded: set[int] = set()
        selected_val, val_chars = _pick_indices_by_budget(
            keys, chars, args.split_seed + 101, budget_val_chars, excluded
        )
        excluded |= selected_val
        selected_test, test_chars = _pick_indices_by_budget(
            keys, chars, args.split_seed + 202, budget_test_chars, excluded
        )
        excluded |= selected_test
        selected_train, train_chars = _pick_indices_by_budget(
            keys, chars, args.split_seed + 303, budget_train_chars, excluded
        )

        if not args.allow_smaller:
            if budget_val_chars is not None and val_chars < budget_val_chars:
                raise ValueError(
                    f"Could not fill val budget: got {val_chars}, target {budget_val_chars} chars."
                )
            if budget_test_chars is not None and test_chars < budget_test_chars:
                raise ValueError(
                    f"Could not fill test budget: got {test_chars}, target {budget_test_chars} chars."
                )
            if budget_train_chars is not None and train_chars < budget_train_chars:
                raise ValueError(
                    f"Could not fill train budget: got {train_chars}, target {budget_train_chars} chars."
                )

        selected_by_split = {
            "train": selected_train,
            "val": selected_val,
            "test": selected_test,
        }

        writers: Dict[str, pq.ParquetWriter] = {}
        written_rows = {"train": 0, "val": 0, "test": 0}
        written_chars = {"train": 0, "val": 0, "test": 0}
        schema_with_doc_split: pa.Schema | None = None

        global_idx = 0
        for source_split in SPLITS:
            source_file = source_dir / source_split / "docs.parquet"
            pf = pq.ParquetFile(source_file)
            text_idx = _column_index_or_raise(pf.schema_arrow, args.text_key)
            for batch in pf.iter_batches(batch_size=args.batch_size):
                table = pa.Table.from_batches([batch])
                chars_list = pc.utf8_length(table.column(text_idx)).to_pylist()
                indices_by_target: Dict[str, list[int]] = {"train": [], "val": [], "test": []}
                for i in range(table.num_rows):
                    idx = global_idx + i
                    if idx in selected_train:
                        indices_by_target["train"].append(i)
                        written_rows["train"] += 1
                        written_chars["train"] += int(chars_list[i] or 0)
                    elif idx in selected_val:
                        indices_by_target["val"].append(i)
                        written_rows["val"] += 1
                        written_chars["val"] += int(chars_list[i] or 0)
                    elif idx in selected_test:
                        indices_by_target["test"].append(i)
                        written_rows["test"] += 1
                        written_chars["test"] += int(chars_list[i] or 0)
                global_idx += table.num_rows

                for target_split, indices in indices_by_target.items():
                    if not indices:
                        continue
                    subset = table.take(pa.array(indices, type=pa.int64()))
                    if "doc_split" in subset.column_names:
                        col_idx = subset.schema.get_field_index("doc_split")
                        subset = subset.remove_column(col_idx)
                    split_col = pa.array([target_split] * subset.num_rows, type=pa.string())
                    subset = subset.append_column("doc_split", split_col)
                    if schema_with_doc_split is None:
                        schema_with_doc_split = subset.schema

                    out_dir = target_dir / target_split
                    out_file = out_dir / "docs.parquet"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if target_split not in writers:
                        writers[target_split] = pq.ParquetWriter(
                            out_file, subset.schema, compression="snappy"
                        )
                    writers[target_split].write_table(subset)

        for writer in writers.values():
            writer.close()

        if schema_with_doc_split is None:
            raise ValueError("No rows selected; cannot write empty splits without schema.")

        for target_split in SPLITS:
            out_dir = target_dir / target_split
            out_file = out_dir / "docs.parquet"
            if not out_file.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    pa.Table.from_batches([], schema=schema_with_doc_split),
                    out_file,
                    compression="snappy",
                )

        summary["selection_mode"] = "fixed_split_budgets"
        summary["splits"] = {
            "train": {
                "target_file": str(target_dir / "train" / "docs.parquet"),
                "requested_chars": int(budget_train_chars) if budget_train_chars is not None else None,
                "written_rows": int(written_rows["train"]),
                "written_chars": int(written_chars["train"]),
            },
            "val": {
                "target_file": str(target_dir / "val" / "docs.parquet"),
                "requested_chars": int(budget_val_chars) if budget_val_chars is not None else None,
                "written_rows": int(written_rows["val"]),
                "written_chars": int(written_chars["val"]),
            },
            "test": {
                "target_file": str(target_dir / "test" / "docs.parquet"),
                "requested_chars": int(budget_test_chars) if budget_test_chars is not None else None,
                "written_rows": int(written_rows["test"]),
                "written_chars": int(written_chars["test"]),
            },
        }
        summary["kept_total_rows"] = int(
            written_rows["train"] + written_rows["val"] + written_rows["test"]
        )
        summary["kept_total_chars"] = int(
            written_chars["train"] + written_chars["val"] + written_chars["test"]
        )
        summary["scanned_total_rows"] = int(src_total_rows)
        summary["scanned_total_chars"] = int(src_total_chars)

        for split in SPLITS:
            out_file = target_dir / split / "docs.parquet"
            print(
                f"[{split}] written_rows={written_rows[split]} "
                f"written_chars={written_chars[split]} -> {out_file}"
            )

        summary_path = target_dir / "subsplit_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {summary_path}")
        return

    writers: Dict[str, pq.ParquetWriter] = {}
    written_rows = {"train": 0, "val": 0, "test": 0}
    written_chars = {"train": 0, "val": 0, "test": 0}
    scanned_rows_total = 0
    scanned_chars_total = 0
    kept_rows_total = 0
    kept_chars_total = 0
    schema_with_doc_split: pa.Schema | None = None

    for source_split in SPLITS:
        source_file = source_dir / source_split / "docs.parquet"
        pf = pq.ParquetFile(source_file)
        text_idx = _column_index_or_raise(pf.schema_arrow, args.text_key)
        id_idx = _column_index_or_raise(pf.schema_arrow, args.id_key)

        for batch in pf.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            ids = table.column(id_idx).to_pylist()
            chars_list = pc.utf8_length(table.column(text_idx)).to_pylist()

            batch_row_base = scanned_rows_total
            scanned_rows_total += table.num_rows
            scanned_chars_total += sum(int(v or 0) for v in chars_list)

            indices_by_target: Dict[str, list[int]] = {"train": [], "val": [], "test": []}

            for i, raw_id in enumerate(ids):
                doc_id = "" if raw_id is None else str(raw_id)
                if doc_id == "":
                    # Fallback keeps deterministic behavior if IDs are missing.
                    doc_id = f"{source_split}:{batch_row_base + i}"
                keep_u = _hash_unit_interval(args.keep_seed, doc_id)
                if keep_u >= keep_ratio:
                    continue
                split_u = _hash_unit_interval(args.split_seed, doc_id)
                target_split = _assign_split(split_u, args.train_ratio, args.val_ratio)

                indices_by_target[target_split].append(i)
                doc_chars = int(chars_list[i] or 0)
                kept_rows_total += 1
                kept_chars_total += doc_chars
                written_rows[target_split] += 1
                written_chars[target_split] += doc_chars

            for target_split, indices in indices_by_target.items():
                if not indices:
                    continue
                subset = table.take(pa.array(indices, type=pa.int64()))

                if "doc_split" in subset.column_names:
                    col_idx = subset.schema.get_field_index("doc_split")
                    subset = subset.remove_column(col_idx)
                split_col = pa.array([target_split] * subset.num_rows, type=pa.string())
                subset = subset.append_column("doc_split", split_col)

                if schema_with_doc_split is None:
                    schema_with_doc_split = subset.schema

                out_dir = target_dir / target_split
                out_file = out_dir / "docs.parquet"
                out_dir.mkdir(parents=True, exist_ok=True)
                if target_split not in writers:
                    writers[target_split] = pq.ParquetWriter(
                        out_file, subset.schema, compression="snappy"
                    )
                writers[target_split].write_table(subset)

    for writer in writers.values():
        writer.close()

    if schema_with_doc_split is None:
        raise ValueError("No rows were kept after capping; cannot write empty splits without schema.")

    # Ensure all split files exist even if a split got zero rows.
    for target_split in SPLITS:
        out_dir = target_dir / target_split
        out_file = out_dir / "docs.parquet"
        if not out_file.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_batches([], schema=schema_with_doc_split),
                out_file,
                compression="snappy",
            )

    for split in SPLITS:
        out_file = target_dir / split / "docs.parquet"
        summary["splits"][split] = {
            "target_file": str(out_file),
            "written_rows": int(written_rows[split]),
            "written_chars": int(written_chars[split]),
        }
        print(f"[{split}] written_rows={written_rows[split]} written_chars={written_chars[split]} -> {out_file}")

    summary["kept_total_rows"] = int(kept_rows_total)
    summary["kept_total_chars"] = int(kept_chars_total)
    summary["scanned_total_rows"] = int(scanned_rows_total)
    summary["scanned_total_chars"] = int(scanned_chars_total)

    summary_path = target_dir / "subsplit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
