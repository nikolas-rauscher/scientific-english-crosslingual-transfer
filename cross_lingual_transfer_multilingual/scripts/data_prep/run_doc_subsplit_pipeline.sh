#!/usr/bin/env bash
set -euo pipefail

# Create fixed-size doc subsplits from an existing full split set.
#
# Usage:
#   bash cross_lingual_transfer_multilingual/scripts/data_prep/run_doc_subsplit_pipeline.sh \
#     spa_Latn \
#     sub_800k_17k_43k_seed42 \
#     800000 \
#     17000 \
#     43000
#
# Optional args:
#   [SEED] [SOURCE_SPLITS_BASE_DIR] [TARGET_SPLITS_BASE_DIR] [ALLOW_SMALLER] [CLEAN_OUTPUT]
#
# Defaults:
#   SEED=42
#   SOURCE_SPLITS_BASE_DIR=splits/full (falls back to splits_doc if missing)
#   TARGET_SPLITS_BASE_DIR=splits/sub/<SUBSPLIT_NAME>
#   ALLOW_SMALLER=false
#   CLEAN_OUTPUT=true

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
SUBPROJECT_ROOT="$PROJECT_ROOT/cross_lingual_transfer_multilingual"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv_pretraining/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_pretraining/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

LANG_CODE="${1:?Missing language code, e.g. spa_Latn}"
SUBSPLIT_NAME="${2:?Missing subsplit name, e.g. sub_800k_17k_43k_seed42}"
TRAIN_SIZE="${3:?Missing train size, e.g. 800000}"
VAL_SIZE="${4:?Missing val size, e.g. 17000}"
TEST_SIZE="${5:?Missing test size, e.g. 43000}"
SEED="${6:-42}"
SOURCE_SPLITS_BASE_DIR="${7:-splits/full}"
TARGET_SPLITS_BASE_DIR="${8:-splits/sub/$SUBSPLIT_NAME}"
ALLOW_SMALLER="${9:-false}"
CLEAN_OUTPUT="${10:-true}"

LANG_DIR="$SUBPROJECT_ROOT/data/languages/$LANG_CODE"
SOURCE_SPLITS_DIR="$LANG_DIR/$SOURCE_SPLITS_BASE_DIR"
TARGET_SPLITS_DIR="$LANG_DIR/$TARGET_SPLITS_BASE_DIR"

# Backwards-compatibility: existing runs may still store the full split under "splits_doc".
if [[ ! -f "$SOURCE_SPLITS_DIR/train/docs.parquet" ]]; then
  if [[ "$SOURCE_SPLITS_BASE_DIR" == "splits/full" && -f "$LANG_DIR/splits_doc/train/docs.parquet" ]]; then
    SOURCE_SPLITS_BASE_DIR="splits_doc"
    SOURCE_SPLITS_DIR="$LANG_DIR/$SOURCE_SPLITS_BASE_DIR"
    echo "INFO: Falling back to legacy source split dir: $SOURCE_SPLITS_BASE_DIR"
  fi
fi

if [[ ! -f "$SOURCE_SPLITS_DIR/train/docs.parquet" ]]; then
  echo "ERROR: Missing source split parquet: $SOURCE_SPLITS_DIR/train/docs.parquet" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" cross_lingual_transfer_multilingual/scripts/data_prep/create_doc_subsplit.py
  --source-splits-dir "$SOURCE_SPLITS_DIR"
  --target-splits-dir "$TARGET_SPLITS_DIR"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --test-size "$TEST_SIZE"
  --seed "$SEED"
)

if [[ "$ALLOW_SMALLER" == "true" ]]; then
  CMD+=(--allow-smaller)
fi
if [[ "$CLEAN_OUTPUT" == "true" ]]; then
  CMD+=(--clean-output)
fi

echo "=============================================="
echo "Create Document Subsplit"
echo "Language:         $LANG_CODE"
echo "Subsplit name:    $SUBSPLIT_NAME"
echo "Source splits:    $SOURCE_SPLITS_BASE_DIR"
echo "Target splits:    $TARGET_SPLITS_BASE_DIR"
echo "Requested sizes:  train=$TRAIN_SIZE val=$VAL_SIZE test=$TEST_SIZE"
echo "Seed:             $SEED"
echo "Allow smaller:    $ALLOW_SMALLER"
echo "Clean output:     $CLEAN_OUTPUT"
echo "=============================================="

"${CMD[@]}"

echo
echo "Done. Next step (build sliding windows from this subsplit):"
echo "bash cross_lingual_transfer_multilingual/scripts/data_prep/run_sliding_windows_for_doc_splits.sh \\"
echo "  $LANG_CODE \\"
echo "  <TOKENIZER_PATH_OR_HF_ID> \\"
echo "  75 75 75 true train,val,test \\"
echo "  \"$TARGET_SPLITS_BASE_DIR\" \\"
echo "  \"sliding_windows/sub/$SUBSPLIT_NAME\""
