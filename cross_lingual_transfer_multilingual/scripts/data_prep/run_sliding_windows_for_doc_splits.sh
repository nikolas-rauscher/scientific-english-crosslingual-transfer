#!/usr/bin/env bash
set -euo pipefail

# Run sliding-window preprocessing separately for train/val/test doc splits.
# Before processing each split, the input docs parquet is sharded into N files
# to enable higher parallelism on multi-core jobs.
#
# Usage:
#   bash cross_lingual_transfer_multilingual/scripts/data_prep/run_sliding_windows_for_doc_splits.sh \
#     spa_Latn \
#     vgaraujov/t5-base-spanish \
#     8 \
#     8 \
#     75 \
#     true \
#     train,val,test \
#     splits_doc \
#     sliding_windows
#
# Simple mode switch via env:
#   DATA_MODE=full   (default): use full split roots
#   DATA_MODE=subset: auto-create doc subsplit (if enabled) and use sub roots
#   SUBSPLIT_MODE=rows  (default): fixed row-count subsplit (create_doc_subsplit.py)
#   SUBSPLIT_MODE=chars: char-budget subsplit (default fixed split GB caps, optional reference fallback)
#
# Example:
#   DATA_MODE=subset bash cross_lingual_transfer_multilingual/scripts/data_prep/run_sliding_windows_for_doc_splits.sh \
#     spa_Latn vgaraujov/t5-base-spanish 75 75 75 true train,val,test

PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
SUBPROJECT_ROOT="$PROJECT_ROOT/cross_lingual_transfer_multilingual"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv_pretraining/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_pretraining/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

LANG_CODE="${1:?Missing language code, e.g. spa_Latn}"
TOKENIZER_PATH="${2:?Missing tokenizer path, e.g. HF id or local path}"
TASKS="${3:-75}"
WORKERS="${4:-75}"
NUM_SHARDS="${5:-75}"
MERGE_WINDOWS="${6:-true}"
SPLITS_CSV="${7:-train,val,test}"
SPLITS_BASE_DIR="${8:-splits_doc}"
WINDOWS_BASE_DIR="${9:-sliding_windows}"
PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-60}"
WINDOW_TARGET_TOKENS="${WINDOW_TARGET_TOKENS:-568}"
WINDOW_MIN_ACTUAL_TOKENS="${WINDOW_MIN_ACTUAL_TOKENS:-530}"

# Optional simple mode: "full" | "subset"
DATA_MODE_RAW="${DATA_MODE:-full}"
SUBSPLIT_NAME="${SUBSPLIT_NAME:-sub_800k_17k_43k_seed42}"
SUBSPLIT_TRAIN_SIZE="${SUBSPLIT_TRAIN_SIZE:-800000}"
SUBSPLIT_VAL_SIZE="${SUBSPLIT_VAL_SIZE:-17000}"
SUBSPLIT_TEST_SIZE="${SUBSPLIT_TEST_SIZE:-43000}"
SUBSPLIT_SEED="${SUBSPLIT_SEED:-42}"
SUBSPLIT_ALLOW_SMALLER="${SUBSPLIT_ALLOW_SMALLER:-true}"
SUBSPLIT_CLEAN_OUTPUT="${SUBSPLIT_CLEAN_OUTPUT:-true}"
AUTO_CREATE_SUBSPLIT="${AUTO_CREATE_SUBSPLIT:-true}"
SUBSPLIT_MODE_RAW="${SUBSPLIT_MODE:-rows}"
CHAR_BUDGET_REFERENCE_LANG="${CHAR_BUDGET_REFERENCE_LANG:-pol_Latn}"
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

case "$DATA_MODE_RAW" in
  full)
    DATA_MODE="full"
    ;;
  subset|sub)
    DATA_MODE="subset"
    ;;
  *)
    echo "ERROR: Invalid DATA_MODE='$DATA_MODE_RAW'. Use: full | subset" >&2
    exit 1
    ;;
esac

case "$SUBSPLIT_MODE_RAW" in
  rows|row|count)
    SUBSPLIT_MODE="rows"
    ;;
  chars|char|char_budget)
    SUBSPLIT_MODE="chars"
    ;;
  *)
    echo "ERROR: Invalid SUBSPLIT_MODE='$SUBSPLIT_MODE_RAW'. Use: rows | chars" >&2
    exit 1
    ;;
esac

cd "$PROJECT_ROOT"

IFS=',' read -r -a SPLITS <<< "$SPLITS_CSV"

if [[ "$DATA_MODE" == "subset" ]]; then
  # If caller did not explicitly pass custom roots, route to default sub roots.
  if [[ -z "${8:-}" ]]; then
    SPLITS_BASE_DIR="splits/sub/$SUBSPLIT_NAME"
  fi
  if [[ -z "${9:-}" ]]; then
    WINDOWS_BASE_DIR="sliding_windows/sub/$SUBSPLIT_NAME"
  fi

  if [[ "$AUTO_CREATE_SUBSPLIT" == "true" ]]; then
    if [[ "$SUBSPLIT_MODE" == "chars" ]]; then
      CHAR_SUBSPLIT_TEXT_KEY="$CHAR_SUBSPLIT_TEXT_KEY" \
      CHAR_SUBSPLIT_ID_KEY="$CHAR_SUBSPLIT_ID_KEY" \
      CHAR_SUBSPLIT_TRAIN_RATIO="$CHAR_SUBSPLIT_TRAIN_RATIO" \
      CHAR_SUBSPLIT_VAL_RATIO="$CHAR_SUBSPLIT_VAL_RATIO" \
      CHAR_SUBSPLIT_TEST_RATIO="$CHAR_SUBSPLIT_TEST_RATIO" \
      CHAR_SUBSPLIT_KEEP_SEED="$CHAR_SUBSPLIT_KEEP_SEED" \
      CHAR_SUBSPLIT_SPLIT_SEED="$CHAR_SUBSPLIT_SPLIT_SEED" \
      CHAR_SUBSPLIT_FIXED_TRAIN_GB="$CHAR_SUBSPLIT_FIXED_TRAIN_GB" \
      CHAR_SUBSPLIT_FIXED_VAL_GB="$CHAR_SUBSPLIT_FIXED_VAL_GB" \
      CHAR_SUBSPLIT_FIXED_TEST_GB="$CHAR_SUBSPLIT_FIXED_TEST_GB" \
      CHAR_SUBSPLIT_TARGET_TOTAL_GB="$CHAR_SUBSPLIT_TARGET_TOTAL_GB" \
      bash cross_lingual_transfer_multilingual/scripts/data_prep/run_char_budget_subsplit_pipeline.sh \
        "$LANG_CODE" \
        "$SUBSPLIT_NAME" \
        "$CHAR_BUDGET_REFERENCE_LANG" \
        "splits/full" \
        "$SPLITS_BASE_DIR" \
        "$SUBSPLIT_ALLOW_SMALLER" \
        "$SUBSPLIT_CLEAN_OUTPUT"
    else
      bash cross_lingual_transfer_multilingual/scripts/data_prep/run_doc_subsplit_pipeline.sh \
        "$LANG_CODE" \
        "$SUBSPLIT_NAME" \
        "$SUBSPLIT_TRAIN_SIZE" \
        "$SUBSPLIT_VAL_SIZE" \
        "$SUBSPLIT_TEST_SIZE" \
        "$SUBSPLIT_SEED" \
        "splits/full" \
        "$SPLITS_BASE_DIR" \
        "$SUBSPLIT_ALLOW_SMALLER" \
        "$SUBSPLIT_CLEAN_OUTPUT"
    fi
  fi
fi

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
  else
    echo 0
  fi
}

dir_size_mb() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    du -sm "$dir" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

monitor_split_progress() {
  local split="$1"
  local log_dir="$2"
  local output_dir="$3"
  local tasks_target="$4"
  local interval="$5"
  local start_ts="$6"
  local sw_pid="$7"

  while kill -0 "$sw_pid" >/dev/null 2>&1; do
    local completed
    local task_logs
    local out_files
    local out_size_mb
    local elapsed_sec

    completed="$(count_files "$log_dir/completions" "*")"
    task_logs="$(count_files "$log_dir/logs" "task_*.log")"
    out_files="$(count_files "$output_dir" "*.parquet")"
    out_size_mb="$(dir_size_mb "$output_dir")"
    elapsed_sec=$(( $(date +%s) - start_ts ))

    echo "[progress][$split] elapsed=${elapsed_sec}s completed_tasks=${completed}/${tasks_target} task_logs=${task_logs} output_parquet=${out_files} output_size_mb=${out_size_mb}"
    sleep "$interval"
  done
}

for SPLIT in "${SPLITS[@]}"; do
  DOCS_FILE="$SUBPROJECT_ROOT/data/languages/$LANG_CODE/$SPLITS_BASE_DIR/$SPLIT/docs.parquet"
  SHARD_DIR="$SUBPROJECT_ROOT/data/languages/$LANG_CODE/$SPLITS_BASE_DIR/$SPLIT/shards_${NUM_SHARDS}"
  OUTPUT_DIR="$SUBPROJECT_ROOT/data/languages/$LANG_CODE/$WINDOWS_BASE_DIR/$SPLIT"
  LOG_DIR_SUFFIX="$(echo "$WINDOWS_BASE_DIR" | tr '/' '_')"
  LOG_DIR="$SUBPROJECT_ROOT/logs/$LANG_CODE/${LOG_DIR_SUFFIX}_${SPLIT}"

  if [[ ! -f "$DOCS_FILE" ]]; then
    echo "ERROR: Missing split docs file: $DOCS_FILE" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  find "$OUTPUT_DIR" -maxdepth 1 -name '*.parquet' -delete || true
  rm -rf "$LOG_DIR"/* || true

  "$PYTHON_BIN" cross_lingual_transfer_multilingual/scripts/data_prep/split_parquet_to_n_shards.py \
    --input "$DOCS_FILE" \
    --output-dir "$SHARD_DIR" \
    --num-shards "$NUM_SHARDS" \
    --clean-output

  echo "=============================================="
  echo "Mode:     $DATA_MODE"
  echo "Language: $LANG_CODE"
  echo "Split:    $SPLIT"
  echo "Split root: $SPLITS_BASE_DIR"
  echo "Input:    $DOCS_FILE"
  echo "Shards:   $SHARD_DIR (${NUM_SHARDS})"
  echo "Windows root: $WINDOWS_BASE_DIR"
  echo "Output:   $OUTPUT_DIR"
  echo "Tokenizer:$TOKENIZER_PATH"
  echo "Window target tokens: $WINDOW_TARGET_TOKENS"
  echo "Window min actual tokens: $WINDOW_MIN_ACTUAL_TOKENS"
  echo "Tasks:    $TASKS"
  echo "Workers:  $WORKERS"
  echo "Merge:    $MERGE_WINDOWS"
  if [[ "$DATA_MODE" == "subset" ]]; then
    echo "Subsplit mode: $SUBSPLIT_MODE"
    if [[ "$SUBSPLIT_MODE" == "chars" ]]; then
      echo "Char budget reference language: $CHAR_BUDGET_REFERENCE_LANG"
      echo "Char fixed split GB: train=${CHAR_SUBSPLIT_FIXED_TRAIN_GB:-<all-remaining>} val=${CHAR_SUBSPLIT_FIXED_VAL_GB:-unset} test=${CHAR_SUBSPLIT_FIXED_TEST_GB:-unset}"
      echo "Char total target GB (fallback): ${CHAR_SUBSPLIT_TARGET_TOTAL_GB:-<from reference>}"
      echo "Char-split ratios: train=$CHAR_SUBSPLIT_TRAIN_RATIO val=$CHAR_SUBSPLIT_VAL_RATIO test=$CHAR_SUBSPLIT_TEST_RATIO"
      echo "Char-split seeds:  keep=$CHAR_SUBSPLIT_KEEP_SEED split=$CHAR_SUBSPLIT_SPLIT_SEED"
    fi
  fi
  echo "=============================================="

  SPLIT_START_TS="$(date +%s)"
  LIMIT_DOCUMENTS="-1"

  set +e
  "$PYTHON_BIN" src/dataprep/pipelines/run_sliding_windows.py \
    sliding_windows.paths.input_folder="$SHARD_DIR" \
    sliding_windows.paths.input_pattern="shard-*.parquet" \
    sliding_windows.paths.output_folder="$OUTPUT_DIR" \
    sliding_windows.paths.logging_dir="$LOG_DIR" \
    sliding_windows.reader.text_key=text \
    sliding_windows.reader.id_key=id \
    sliding_windows.tokenizer.name_or_path="$TOKENIZER_PATH" \
    sliding_windows.window_config.target_tokens="$WINDOW_TARGET_TOKENS" \
    sliding_windows.window_config.overlap_ratio=0.5 \
    sliding_windows.window_config.output_format=text \
    sliding_windows.window_config.dynamic_last_window.enable=false \
    sliding_windows.normalization.enable=true \
    sliding_windows.filters.enable=true \
    sliding_windows.filters.min_actual_tokens="$WINDOW_MIN_ACTUAL_TOKENS" \
    sliding_windows.stats.enable=false \
    sliding_windows.execution.tasks="$TASKS" \
    sliding_windows.execution.workers="$WORKERS" \
    sliding_windows.limit_documents="$LIMIT_DOCUMENTS" \
    sliding_windows.log_to_wandb=false &
  SW_PID=$!

  monitor_split_progress "$SPLIT" "$LOG_DIR" "$OUTPUT_DIR" "$TASKS" "$PROGRESS_INTERVAL_SEC" "$SPLIT_START_TS" "$SW_PID" &
  MON_PID=$!

  wait "$SW_PID"
  SW_EXIT=$?
  kill "$MON_PID" >/dev/null 2>&1 || true
  wait "$MON_PID" 2>/dev/null || true
  set -e

  if [[ "$SW_EXIT" -ne 0 ]]; then
    echo "ERROR: sliding windows failed for $LANG_CODE/$SPLIT (exit=$SW_EXIT)" >&2
    echo "Hint: inspect task logs in $LOG_DIR/logs and slurm output." >&2
    exit "$SW_EXIT"
  fi

  SPLIT_END_TS="$(date +%s)"
  SPLIT_ELAPSED_SEC=$((SPLIT_END_TS - SPLIT_START_TS))
  FINAL_COMPLETED="$(count_files "$LOG_DIR/completions" "*")"
  FINAL_OUT_FILES="$(count_files "$OUTPUT_DIR" "*.parquet")"
  FINAL_OUT_MB="$(dir_size_mb "$OUTPUT_DIR")"
  echo "[done][$SPLIT] elapsed=${SPLIT_ELAPSED_SEC}s completed_tasks=${FINAL_COMPLETED}/${TASKS} output_parquet=${FINAL_OUT_FILES} output_size_mb=${FINAL_OUT_MB}"

  if [[ "$MERGE_WINDOWS" == "true" ]]; then
    "$PYTHON_BIN" cross_lingual_transfer_multilingual/scripts/data_prep/merge_parquet_dir.py \
      --input-dir "$OUTPUT_DIR" \
      --glob "*.parquet" \
      --output "$OUTPUT_DIR/merged/windows.parquet" \
      --clean-output
  fi

done
