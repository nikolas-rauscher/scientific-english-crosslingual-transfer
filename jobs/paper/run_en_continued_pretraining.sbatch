#!/bin/bash
#SBATCH --job-name=paper_train_en
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j-paper-train-en.out
#SBATCH --error=logs/slurm-%j-paper-train-en.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/netscratch/anonymous_user/containers/hydra_pytorch_25.05.25-final.sqsh}"
MOUNTS="/netscratch:/netscratch,/fscratch:/fscratch"
EXTRA_ARGS="$*"

if [[ -z "$EXTRA_ARGS" ]]; then
  EXTRA_ARGS="experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart trainer.log_every_n_steps=1"
fi

echo "Starting EN continued pretraining at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Node: ${SLURM_NODELIST:-n/a}"
echo "Container: $CONTAINER_IMAGE"

srun -K \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc "source .venv_pretraining/bin/activate && python src/train.py $EXTRA_ARGS"
