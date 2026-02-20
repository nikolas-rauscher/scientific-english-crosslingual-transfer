#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
JOBS_DIR="$PROJECT_ROOT/cross_lingual_transfer_multilingual/jobs/paper"
STATUS_DIR="$PROJECT_ROOT/cross_lingual_transfer_multilingual/logs/paper_pipeline"
mkdir -p "$STATUS_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="$STATUS_DIR/pipeline_status_${TS}.json"

extract_job_id() {
  awk '/Submitted batch job/ {print $4}' <<<"$1" | tail -n1
}

# 01 -> 02
out01=$(sbatch "$JOBS_DIR/run_01_tokenizers_all.sbatch")
job01=$(extract_job_id "$out01")

echo "$out01"

out02=$(sbatch --dependency=afterok:$job01 "$JOBS_DIR/run_02_wechsel_init_all.sbatch")
job02=$(extract_job_id "$out02")

echo "$out02"

# 03 submit all continued runs, each depends on 02
export SLURM_DEPENDENCY="afterok:$job02"
out03=$("$JOBS_DIR/run_03_submit_continued_all.sh")
unset SLURM_DEPENDENCY

echo "$out03"
job03_csv=$(grep '^JOB_IDS=' <<<"$out03" | cut -d'=' -f2)
if [[ -z "$job03_csv" ]]; then
  echo "Failed to collect stage-03 job ids" >&2
  exit 1
fi

# 04 depends on all stage-03 jobs
out04=$(sbatch --dependency=afterok:$job03_csv "$JOBS_DIR/run_04_prepare_permanent_all.sbatch")
job04=$(extract_job_id "$out04")

echo "$out04"

# 05 submit all mmlu runs, each depends on 04
export SLURM_DEPENDENCY="afterok:$job04"
out05=$("$JOBS_DIR/run_05_submit_mmlu_all.sh")
unset SLURM_DEPENDENCY

echo "$out05"
job05_csv=$(grep '^JOB_IDS=' <<<"$out05" | cut -d'=' -f2)
if [[ -z "$job05_csv" ]]; then
  echo "Failed to collect stage-05 job ids" >&2
  exit 1
fi

# 06 depends on all stage-05 jobs
out06=$(sbatch --dependency=afterok:$job05_csv "$JOBS_DIR/run_06_aggregate_all.sbatch")
job06=$(extract_job_id "$out06")

echo "$out06"

TS="$TS" \
job01="$job01" out01="$out01" \
job02="$job02" out02="$out02" \
job03_csv="$job03_csv" \
job04="$job04" out04="$out04" \
job05_csv="$job05_csv" \
job06="$job06" out06="$out06" \
STATUS_FILE="$STATUS_FILE" \
python - <<'PY'
import json
import os

payload = {
    "timestamp": os.environ["TS"],
    "stages": {
        "01_tokenizers_all": {
            "job_id": os.environ["job01"],
            "submit_output": os.environ["out01"],
        },
        "02_wechsel_init_all": {
            "job_id": os.environ["job02"],
            "submit_output": os.environ["out02"],
            "dependency": f"afterok:{os.environ['job01']}",
        },
        "03_submit_continued_all": {
            "job_ids": os.environ["job03_csv"],
            "dependency": f"afterok:{os.environ['job02']}",
        },
        "04_prepare_permanent_all": {
            "job_id": os.environ["job04"],
            "submit_output": os.environ["out04"],
            "dependency": f"afterok:{os.environ['job03_csv']}",
        },
        "05_submit_mmlu_all": {
            "job_ids": os.environ["job05_csv"],
            "dependency": f"afterok:{os.environ['job04']}",
        },
        "06_aggregate_all": {
            "job_id": os.environ["job06"],
            "submit_output": os.environ["out06"],
            "dependency": f"afterok:{os.environ['job05_csv']}",
        },
    },
}
json.dump(payload, fp=open(os.environ["STATUS_FILE"], "w", encoding="utf-8"), indent=2)
PY

echo "Pipeline status written to: $STATUS_FILE"
echo "Stage 01 job: $job01"
echo "Stage 02 job: $job02"
echo "Stage 03 jobs: $job03_csv"
echo "Stage 04 job: $job04"
echo "Stage 05 jobs: $job05_csv"
echo "Stage 06 job: $job06"
