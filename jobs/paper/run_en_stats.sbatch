#!/bin/bash
#SBATCH --job-name=paper_stats_en
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=76
#SBATCH --mem=200G
#SBATCH --partition=V100-16GB,V100-32GB,A100-80GB,A100-40GB,H100,H200,H100-PCI,H100-SLT
#SBATCH --gres=gpu:0
#SBATCH --output=logs/slurm-stats-en-%j.out
#SBATCH --error=logs/slurm-stats-en-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/netscratch/${USER}/containers/hydra_pytorch_25.05.25-final.sqsh}"
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  CONTAINER_IMAGE="/netscratch/anonymous_user/containers/hydra_pytorch_25.05.25-final.sqsh"
fi

MOUNTS="/netscratch:/netscratch,/home/$USER:/home/$USER"
[[ -d /fscratch ]] && MOUNTS="$MOUNTS,/fscratch:/fscratch"
[[ -d /ds-slt ]] && MOUNTS="$MOUNTS,/ds-slt:/ds-slt"

EXTRA_ARGS="$*"

echo "Starting EN stats at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "Container: $CONTAINER_IMAGE"

srun -K \
  --cpus-per-task=50 \
  --gres=gpu:0 \
  --container-mounts="$MOUNTS" \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc "source .venv_pretraining/bin/activate && python src/dataprep/pipelines/run_stats.py $EXTRA_ARGS"
