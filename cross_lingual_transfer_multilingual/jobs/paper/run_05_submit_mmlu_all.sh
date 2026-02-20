#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
GEN_SCRIPT="$PROJECT_ROOT/cross_lingual_transfer_multilingual/scripts/paper/generate_mmlu_configs_all.py"
JOB_SCRIPT="$PROJECT_ROOT/cross_lingual_transfer_multilingual/jobs/paper/run_mmlu_eval_single_lang_paper.sbatch"
LANGS=(deu_Latn jpn_Jpan spa_Latn rus_Cyrl pol_Latn por_Latn)

cd "$PROJECT_ROOT"
source .venv_eval/bin/activate
python "$GEN_SCRIPT"

DEPENDENCY_ARG=""
if [[ -n "${SLURM_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARG="--dependency=${SLURM_DEPENDENCY}"
fi

job_ids=()
for lang in "${LANGS[@]}"; do
  echo "Submitting paper MMLU eval for $lang ..."
  if [[ -n "$DEPENDENCY_ARG" ]]; then
    out=$(sbatch $DEPENDENCY_ARG "$JOB_SCRIPT" "$lang")
  else
    out=$(sbatch "$JOB_SCRIPT" "$lang")
  fi
  jid=$(awk '{print $4}' <<<"$out")
  echo "  $out"
  job_ids+=("$jid")
done

joined=$(IFS=,; echo "${job_ids[*]}")
echo "JOB_IDS=$joined"
