#!/bin/bash
#SBATCH --job-name=paper_agg
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200,A100-80GB,A100-40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=4:00:00
#SBATCH --output=/netscratch/anonymous_user/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.out
#SBATCH --error=/netscratch/anonymous_user/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
cd "$PROJECT_ROOT"

srun -K \
  --container-image=/netscratch/anonymous_user/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc '
    set -euo pipefail
    source .venv_eval/bin/activate
    python cross_lingual_transfer_multilingual/scripts/paper/aggregate_paper_ab_results.py
  '
