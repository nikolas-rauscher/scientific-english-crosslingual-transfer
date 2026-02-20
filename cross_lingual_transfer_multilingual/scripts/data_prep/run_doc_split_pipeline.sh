#!/usr/bin/env bash
set -euo pipefail

# End-to-end doc split pipeline for one language:
# 1) build manifest + deterministic split labels
# 2) materialize train/val/test parquet folders
#
# Usage:
#   bash cross_lingual_transfer_multilingual/scripts/data_prep/run_doc_split_pipeline.sh spa_Latn

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
SUBPROJECT_ROOT="$PROJECT_ROOT/cross_lingual_transfer_multilingual"
RAW_ROOT_DEFAULT="/ds-slt/sci-LLM/scilons/unpaywall_texts_pq/unpaywall_texts_pq_3"

LANG_CODE="${1:?Missing language code, e.g. spa_Latn}"
RAW_ROOT="${2:-$RAW_ROOT_DEFAULT}"
FILE_ORDER="${3:-sorted}"
MAX_FILES="${4:-}"

LANG_RAW_DIR="$RAW_ROOT/$LANG_CODE"
LANG_DIR="$SUBPROJECT_ROOT/data/languages/$LANG_CODE"
MANIFEST_DIR="$LANG_DIR/manifests"
SPLIT_DIR="$LANG_DIR/splits_doc"

MANIFEST_PATH="$MANIFEST_DIR/docs_manifest.parquet"
MANIFEST_SUMMARY="$MANIFEST_DIR/docs_manifest_summary.json"
SPLIT_SUMMARY="$SPLIT_DIR/materialize_summary.json"

if [[ ! -d "$LANG_RAW_DIR" ]]; then
  echo "ERROR: raw language dir not found: $LANG_RAW_DIR" >&2
  exit 1
fi

mkdir -p "$MANIFEST_DIR" "$SPLIT_DIR"

cd "$PROJECT_ROOT"

BUILD_CMD=(
  python cross_lingual_transfer_multilingual/scripts/data_prep/build_doc_manifest.py
  --language-dir "$LANG_RAW_DIR"
  --out-manifest "$MANIFEST_PATH"
  --out-summary "$MANIFEST_SUMMARY"
  --train-ratio 0.949615
  --val-ratio 0.000385
  --test-ratio 0.05
  --seed 42
  --file-order "$FILE_ORDER"
)

if [[ -n "$MAX_FILES" ]]; then
  BUILD_CMD+=(--max-files "$MAX_FILES")
fi

echo "Running manifest build..."
"${BUILD_CMD[@]}"

MATERIALIZE_CMD=(
  python cross_lingual_transfer_multilingual/scripts/data_prep/materialize_doc_splits.py
  --language-dir "$LANG_RAW_DIR"
  --manifest "$MANIFEST_PATH"
  --output-dir "$SPLIT_DIR"
  --out-summary "$SPLIT_SUMMARY"
  --file-order "$FILE_ORDER"
  --clean-output
)

if [[ -n "$MAX_FILES" ]]; then
  MATERIALIZE_CMD+=(--max-files "$MAX_FILES")
fi

echo "Running split materialization..."
"${MATERIALIZE_CMD[@]}"

echo "Done."

echo "Manifest: $MANIFEST_PATH"
echo "Split docs: $SPLIT_DIR/{train,val,test}"

echo "Next step (manual):"
echo "bash cross_lingual_transfer_multilingual/scripts/data_prep/run_sliding_windows_for_doc_splits.sh $LANG_CODE <tokenizer_path_or_hf_id>"
