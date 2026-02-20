#!/usr/bin/env bash
set -euo pipefail

# Paper-track continued pretraining for one language.
# Usage:
#   bash cross_lingual_transfer_multilingual/scripts/paper/run_continued_pretraining_single_lang_paper.sh <LANG_CODE>

PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
SUBPROJECT_ROOT="$PROJECT_ROOT/cross_lingual_transfer_multilingual"

LANG_CODE="${1:?Missing LANG_CODE, e.g. deu_Latn}"

INIT_MODEL_ROOT="${INIT_MODEL_ROOT:-$SUBPROJECT_ROOT/models/wechsel_init_paper_spm32k/$LANG_CODE}"
DATA_VARIANT="sub"
SUBSPLIT_NAME="sub_charcap43gb_seed42"
DATA_DIR="${DATA_DIR:-$SUBPROJECT_ROOT/data/languages/$LANG_CODE/sliding_windows/sub/$SUBSPLIT_NAME/train}"

MAX_STEPS="${MAX_STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
ACCUM_GRAD="${ACCUM_GRAD:-1}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-1500}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"
TRAINER_PROFILE="${TRAINER_PROFILE:-gpu}"
AUTO_ADJUST_LARGE_MODELS="${AUTO_ADJUST_LARGE_MODELS:-true}"

if [[ -f "$INIT_MODEL_ROOT/model/config.json" ]]; then
  MODEL_PATH="$INIT_MODEL_ROOT/model"
  DEFAULT_TOKENIZER_PATH="$INIT_MODEL_ROOT/tokenizer"
elif [[ -f "$INIT_MODEL_ROOT/config.json" ]]; then
  MODEL_PATH="$INIT_MODEL_ROOT"
  DEFAULT_TOKENIZER_PATH="$INIT_MODEL_ROOT"
else
  echo "ERROR: Could not resolve model path from INIT_MODEL_ROOT=$INIT_MODEL_ROOT" >&2
  echo "Expected either: $INIT_MODEL_ROOT/model/config.json or $INIT_MODEL_ROOT/config.json" >&2
  exit 1
fi

# Keep effective batch size while avoiding OOM for larger DE model.
if [[ "$AUTO_ADJUST_LARGE_MODELS" == "true" && "$LANG_CODE" == "deu_Latn" && "$BATCH_SIZE" == "48" && "$ACCUM_GRAD" == "1" ]]; then
  echo "Auto-adjust for large model ($LANG_CODE): BATCH_SIZE 48->24, ACCUM_GRAD 1->2"
  BATCH_SIZE=24
  ACCUM_GRAD=2
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Missing DATA_DIR: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -e "$TOKENIZER_PATH" ]]; then
  echo "ERROR: Missing TOKENIZER_PATH: $TOKENIZER_PATH" >&2
  exit 1
fi

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$SUBPROJECT_ROOT/logs/train_paper_spm32k/$LANG_CODE/runs/$RUN_TS"
mkdir -p "$RUN_DIR"

WANDB_PROJECT="${WANDB_PROJECT:-BA-CrossLingual-Multilingual}"
WANDB_GROUP="${WANDB_GROUP:-multilingual_wechsel_paper_spm32k_continued_pretraining}"
WANDB_NAME="${WANDB_NAME:-${LANG_CODE}-paper-spm32k-continued-${RUN_TS}}"
WANDB_TAGS="${WANDB_TAGS:-[clt-multilingual,paper-spm32k,$LANG_CODE,continued-pretraining,wechsel]}"

echo "=============================================="
echo "Paper Track Continued Pretraining"
echo "Language:      $LANG_CODE"
echo "Model:         $MODEL_PATH"
echo "Tokenizer:     $TOKENIZER_PATH"
echo "Subsplit:      $SUBSPLIT_NAME"
echo "Data:          $DATA_DIR"
echo "Run dir:       $RUN_DIR"
echo "Max steps:     $MAX_STEPS"
echo "Batch size:    $BATCH_SIZE"
echo "Acc grad:      $ACCUM_GRAD"
echo "Val interval:  $VAL_INTERVAL"
echo "LR:            $LR"
echo "Warmup:        $WARMUP_STEPS"
echo "Logger:        $LOGGER_NAME"
echo "Trainer:       $TRAINER_PROFILE"
if [[ "$LOGGER_NAME" == "wandb" ]]; then
  echo "W&B project:   $WANDB_PROJECT"
  echo "W&B group:     $WANDB_GROUP"
  echo "W&B name:      $WANDB_NAME"
fi
echo "=============================================="

cd "$PROJECT_ROOT"
TRAIN_CMD=(
  python src/train.py
  experiment=german_continued_pretraining
  "trainer=$TRAINER_PROFILE"
  "logger=$LOGGER_NAME"
  train=true
  test=false
  seed=42
  hydra.run.dir="$RUN_DIR"
  experiment_name="clt_multi_${LANG_CODE}_continued_pretraining_paper_spm32k"
  tags="$WANDB_TAGS"
  data.data_dir="$DATA_DIR"
  data.tokenizer_name_or_path="$TOKENIZER_PATH"
  data.batch_size="$BATCH_SIZE"
  trainer.max_steps="$MAX_STEPS"
  trainer.accumulate_grad_batches="$ACCUM_GRAD"
  trainer.val_check_interval="$VAL_INTERVAL"
  model.t5_model.pretrained_model_name_or_path="$MODEL_PATH"
  model.optimizer.lr="$LR"
  model.scheduler.num_warmup_steps="$WARMUP_STEPS"
  callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints/best"
  callbacks.step_checkpoint.dirpath="$RUN_DIR/checkpoints/steps"
)

if [[ "$LOGGER_NAME" == "wandb" ]]; then
  TRAIN_CMD+=(
    logger.wandb.project="$WANDB_PROJECT"
    logger.wandb.group="$WANDB_GROUP"
    logger.wandb.name="$WANDB_NAME"
    logger.wandb.tags="$WANDB_TAGS"
    logger.wandb.job_type=pretraining
  )
fi

if [[ "$TRAINER_PROFILE" == "cpu" ]]; then
  TRAIN_CMD+=(
    trainer.accelerator=cpu
    trainer.devices=1
    trainer.precision=32
  )
fi

"${TRAIN_CMD[@]}"
