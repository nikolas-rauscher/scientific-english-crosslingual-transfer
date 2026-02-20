#!/bin/bash
#SBATCH --job-name=paper_clean_en
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=76
#SBATCH --mem=150G
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,V100-16GB,H100,H100-SLT,H100-PCI,H200
#SBATCH --gres=gpu:0
#SBATCH --output=logs/slurm-clean-data-%j.out
#SBATCH --error=logs/slurm-clean-data-%j.err

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

echo "Starting EN cleaning at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "Container: $CONTAINER_IMAGE"

srun -K \
  --cpus-per-task=75 \
  --gres=gpu:0 \
  --container-mounts="$MOUNTS" \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc "source .venv_pretraining/bin/activate && pip install -q wandb regex && python src/dataprep/pipelines/clean_data.py $EXTRA_ARGS"
