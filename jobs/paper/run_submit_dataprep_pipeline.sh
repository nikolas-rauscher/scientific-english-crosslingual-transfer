#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DEP=""
if [[ -n "${SLURM_DEPENDENCY:-}" ]]; then
  BASE_DEP="--dependency=${SLURM_DEPENDENCY}"
fi

extract_job_id() {
  awk '/Submitted batch job/ {print $4}' <<<"$1" | tail -n1
}

out_clean=$(sbatch $BASE_DEP "$SCRIPT_DIR/run_en_cleaning.sbatch")
job_clean=$(extract_job_id "$out_clean")
echo "$out_clean"

out_stats=$(sbatch --dependency="afterok:${job_clean}" "$SCRIPT_DIR/run_en_stats.sbatch")
job_stats=$(extract_job_id "$out_stats")
echo "$out_stats"

out_spacy=$(sbatch --dependency="afterok:${job_clean}" "$SCRIPT_DIR/run_en_spacy_stats.sbatch")
job_spacy=$(extract_job_id "$out_spacy")
echo "$out_spacy"

out_fasttext=$(sbatch --dependency="afterok:${job_clean}" "$SCRIPT_DIR/run_en_fasttext_stats.sbatch")
job_fasttext=$(extract_job_id "$out_fasttext")
echo "$out_fasttext"

out_windows=$(sbatch --dependency="afterok:${job_stats}" "$SCRIPT_DIR/run_en_sliding_windows.sbatch")
job_windows=$(extract_job_id "$out_windows")
echo "$out_windows"

echo "CLEAN_JOB_ID=${job_clean}"
echo "STATS_JOB_ID=${job_stats}"
echo "SPACY_JOB_ID=${job_spacy}"
echo "FASTTEXT_JOB_ID=${job_fasttext}"
echo "WINDOWS_JOB_ID=${job_windows}"
