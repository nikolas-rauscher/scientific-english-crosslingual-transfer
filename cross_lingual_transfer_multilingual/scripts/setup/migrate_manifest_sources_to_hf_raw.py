#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    manifest_path = project_root / "cross_lingual_transfer_multilingual" / "manifests" / "model_lineage_v1.json"

    p = argparse.ArgumentParser(description="Migrate source_path_current entries to hf_raw paths from manifest.")
    p.add_argument("--manifest", default=str(manifest_path))
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    manifest_file = Path(args.manifest).resolve()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = data.get("entries", [])

    seen: set[tuple[str, str]] = set()
    migrated = 0
    skipped = 0
    missing = 0

    for e in entries:
        src = Path(str(e["source_path_current"]))
        dst = Path(str(e["planned_paths"]["hf_raw"]))
        key = (str(src), str(dst))
        if key in seen:
            continue
        seen.add(key)

        if not src.exists():
            print(f"[missing] source does not exist: {src}")
            missing += 1
            continue

        if dst.exists():
            if args.force:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            else:
                print(f"[skip] destination exists: {dst}")
                skipped += 1
                continue

        print(f"[plan] {src} -> {dst} ({args.mode})")
        dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(src, dst, args.mode)
        migrated += 1
        print(f"[ok] {dst}")

    print(f"done: migrated={migrated} skipped={skipped} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
