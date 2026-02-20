#!/bin/bash
#SBATCH --job-name=paper_wechsel_init_all
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200,A100-80GB,A100-40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:0
#SBATCH --time=16:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
cd "$PROJECT_ROOT"

srun -K \
  --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch,/ds-slt:/ds-slt \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc '
    set -euo pipefail
    source .venv_crosslingual_transfer/bin/activate
    python cross_lingual_transfer_multilingual/scripts/paper/run_wechsel_transfer_all.py
  '
