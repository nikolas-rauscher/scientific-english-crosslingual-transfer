#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
JOB_SCRIPT="$PROJECT_ROOT/cross_lingual_transfer_multilingual/jobs/paper/run_continued_pretraining_single_lang_paper.sbatch"
LANGS=(deu_Latn jpn_Jpan spa_Latn rus_Cyrl pol_Latn por_Latn)

DEPENDENCY_ARG=""
if [[ -n "${SLURM_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARG="--dependency=${SLURM_DEPENDENCY}"
fi

job_ids=()
for lang in "${LANGS[@]}"; do
  echo "Submitting paper continued pretraining for $lang ..."
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
