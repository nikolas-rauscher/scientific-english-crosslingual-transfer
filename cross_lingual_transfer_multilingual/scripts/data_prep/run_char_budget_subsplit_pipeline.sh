#!/usr/bin/env bash
set -euo pipefail

# Create char-budget-capped doc subsplits for one language.
# Default mode uses fixed split budgets:
#   train=4.3 GB, val=0.2 GB, test=0.5 GB
# This keeps val/test comparable across languages and caps train size for runtime.
# Fallback mode: fixed total budget or reference-based total budget.
#
# Usage:
#   bash cross_lingual_transfer_multilingual/scripts/data_prep/run_char_budget_subsplit_pipeline.sh \
#     spa_Latn \
#     sub_charcap_pol_seed42 \
#     pol_Latn
#
# Optional args:
#   [SOURCE_SPLITS_BASE_DIR] [TARGET_SPLITS_BASE_DIR] [ALLOW_SMALLER] [CLEAN_OUTPUT]
#
# Defaults:
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
SUBSPLIT_NAME="${2:?Missing subsplit name, e.g. sub_charcap_pol_seed42}"
REFERENCE_LANG="${3:?Missing reference language, e.g. pol_Latn}"
SOURCE_SPLITS_BASE_DIR="${4:-splits/full}"
TARGET_SPLITS_BASE_DIR="${5:-splits/sub/$SUBSPLIT_NAME}"
ALLOW_SMALLER="${6:-false}"
CLEAN_OUTPUT="${7:-true}"
CHAR_SUBSPLIT_TEXT_KEY="${CHAR_SUBSPLIT_TEXT_KEY:-text}"
CHAR_SUBSPLIT_ID_KEY="${CHAR_SUBSPLIT_ID_KEY:-id}"
CHAR_SUBSPLIT_TRAIN_RATIO="${CHAR_SUBSPLIT_TRAIN_RATIO:-0.93}"
CHAR_SUBSPLIT_VAL_RATIO="${CHAR_SUBSPLIT_VAL_RATIO:-0.02}"
CHAR_SUBSPLIT_TEST_RATIO="${CHAR_SUBSPLIT_TEST_RATIO:-0.05}"
CHAR_SUBSPLIT_KEEP_SEED="${CHAR_SUBSPLIT_KEEP_SEED:-42}"
CHAR_SUBSPLIT_SPLIT_SEED="${CHAR_SUBSPLIT_SPLIT_SEED:-43}"
CHAR_SUBSPLIT_FIXED_TRAIN_GB="${CHAR_SUBSPLIT_FIXED_TRAIN_GB:-4.3}"
CHAR_SUBSPLIT_FIXED_VAL_GB="${CHAR_SUBSPLIT_FIXED_VAL_GB:-0.2}"
CHAR_SUBSPLIT_FIXED_TEST_GB="${CHAR_SUBSPLIT_FIXED_TEST_GB:-0.5}"
CHAR_SUBSPLIT_TARGET_TOTAL_GB="${CHAR_SUBSPLIT_TARGET_TOTAL_GB:-}"
USE_FIXED_SPLIT_BUDGET="false"
if [[ -n "$CHAR_SUBSPLIT_FIXED_VAL_GB" && -n "$CHAR_SUBSPLIT_FIXED_TEST_GB" ]]; then
  USE_FIXED_SPLIT_BUDGET="true"
fi

LANG_DIR="$SUBPROJECT_ROOT/data/languages/$LANG_CODE"
REF_DIR="$SUBPROJECT_ROOT/data/languages/$REFERENCE_LANG"
SOURCE_SPLITS_DIR="$LANG_DIR/$SOURCE_SPLITS_BASE_DIR"
REFERENCE_SPLITS_DIR="$REF_DIR/$SOURCE_SPLITS_BASE_DIR"
TARGET_SPLITS_DIR="$LANG_DIR/$TARGET_SPLITS_BASE_DIR"

# Backwards-compatibility with legacy "splits_doc" full split root.
if [[ ! -f "$SOURCE_SPLITS_DIR/train/docs.parquet" ]]; then
  if [[ "$SOURCE_SPLITS_BASE_DIR" == "splits/full" && -f "$LANG_DIR/splits_doc/train/docs.parquet" ]]; then
    SOURCE_SPLITS_BASE_DIR="splits_doc"
    SOURCE_SPLITS_DIR="$LANG_DIR/$SOURCE_SPLITS_BASE_DIR"
    echo "INFO: Falling back to legacy source split dir: $SOURCE_SPLITS_BASE_DIR"
  fi
fi

if [[ "$USE_FIXED_SPLIT_BUDGET" != "true" && -z "$CHAR_SUBSPLIT_TARGET_TOTAL_GB" ]]; then
  if [[ ! -f "$REFERENCE_SPLITS_DIR/train/docs.parquet" ]]; then
    if [[ "$SOURCE_SPLITS_BASE_DIR" == "splits_doc" && -f "$REF_DIR/splits_doc/train/docs.parquet" ]]; then
      REFERENCE_SPLITS_DIR="$REF_DIR/splits_doc"
      echo "INFO: Falling back to legacy reference split dir: splits_doc"
    elif [[ "$SOURCE_SPLITS_BASE_DIR" == "splits/full" && -f "$REF_DIR/splits_doc/train/docs.parquet" ]]; then
      REFERENCE_SPLITS_DIR="$REF_DIR/splits_doc"
      echo "INFO: Falling back to legacy reference split dir: splits_doc"
    fi
  fi
fi

if [[ ! -f "$SOURCE_SPLITS_DIR/train/docs.parquet" ]]; then
  echo "ERROR: Missing source split parquet: $SOURCE_SPLITS_DIR/train/docs.parquet" >&2
  exit 1
fi
if [[ "$USE_FIXED_SPLIT_BUDGET" != "true" && -z "$CHAR_SUBSPLIT_TARGET_TOTAL_GB" ]]; then
  if [[ ! -f "$REFERENCE_SPLITS_DIR/train/docs.parquet" ]]; then
    echo "ERROR: Missing reference split parquet: $REFERENCE_SPLITS_DIR/train/docs.parquet" >&2
    exit 1
  fi
fi

cd "$PROJECT_ROOT"

CMD=(
  "$PYTHON_BIN" cross_lingual_transfer_multilingual/scripts/data_prep/create_char_budget_subsplit.py
  --source-splits-dir "$SOURCE_SPLITS_DIR"
  --target-splits-dir "$TARGET_SPLITS_DIR"
  --text-key "$CHAR_SUBSPLIT_TEXT_KEY"
  --id-key "$CHAR_SUBSPLIT_ID_KEY"
  --train-ratio "$CHAR_SUBSPLIT_TRAIN_RATIO"
  --val-ratio "$CHAR_SUBSPLIT_VAL_RATIO"
  --test-ratio "$CHAR_SUBSPLIT_TEST_RATIO"
  --keep-seed "$CHAR_SUBSPLIT_KEEP_SEED"
  --split-seed "$CHAR_SUBSPLIT_SPLIT_SEED"
)

if [[ "$USE_FIXED_SPLIT_BUDGET" == "true" ]]; then
  if [[ -n "$CHAR_SUBSPLIT_FIXED_TRAIN_GB" ]]; then
    CMD+=(--fixed-train-gb "$CHAR_SUBSPLIT_FIXED_TRAIN_GB")
  fi
  CMD+=(--fixed-val-gb "$CHAR_SUBSPLIT_FIXED_VAL_GB")
  CMD+=(--fixed-test-gb "$CHAR_SUBSPLIT_FIXED_TEST_GB")
elif [[ -n "$CHAR_SUBSPLIT_TARGET_TOTAL_GB" ]]; then
  CMD+=(--target-total-gb "$CHAR_SUBSPLIT_TARGET_TOTAL_GB")
else
  CMD+=(--reference-splits-dir "$REFERENCE_SPLITS_DIR")
fi

if [[ "$ALLOW_SMALLER" == "true" ]]; then
  CMD+=(--allow-smaller)
fi
if [[ "$CLEAN_OUTPUT" == "true" ]]; then
  CMD+=(--clean-output)
fi

echo "=============================================="
echo "Create Char-Budget Subsplit"
echo "Language:         $LANG_CODE"
echo "Reference lang:   $REFERENCE_LANG"
echo "Subsplit name:    $SUBSPLIT_NAME"
echo "Source splits:    $SOURCE_SPLITS_BASE_DIR"
if [[ "$USE_FIXED_SPLIT_BUDGET" == "true" || -n "$CHAR_SUBSPLIT_TARGET_TOTAL_GB" ]]; then
  echo "Reference splits: (not used)"
else
  echo "Reference splits: ${REFERENCE_SPLITS_DIR#$SUBPROJECT_ROOT/data/languages/$REFERENCE_LANG/}"
fi
echo "Target splits:    $TARGET_SPLITS_BASE_DIR"
if [[ "$USE_FIXED_SPLIT_BUDGET" == "true" ]]; then
  echo "Fixed split GB:   train=${CHAR_SUBSPLIT_FIXED_TRAIN_GB:-<all-remaining>} val=$CHAR_SUBSPLIT_FIXED_VAL_GB test=$CHAR_SUBSPLIT_FIXED_TEST_GB"
else
  echo "Target total GB:  ${CHAR_SUBSPLIT_TARGET_TOTAL_GB:-<from reference>}"
fi
echo "Text key:         $CHAR_SUBSPLIT_TEXT_KEY"
echo "ID key:           $CHAR_SUBSPLIT_ID_KEY"
echo "Ratios:           train=$CHAR_SUBSPLIT_TRAIN_RATIO val=$CHAR_SUBSPLIT_VAL_RATIO test=$CHAR_SUBSPLIT_TEST_RATIO"
echo "Seeds:            keep=$CHAR_SUBSPLIT_KEEP_SEED split=$CHAR_SUBSPLIT_SPLIT_SEED"
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
