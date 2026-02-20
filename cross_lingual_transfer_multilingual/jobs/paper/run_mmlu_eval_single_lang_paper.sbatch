#!/bin/bash
#SBATCH --job-name=paper_mmlu
#SBATCH --partition=H100,H100-PCI,H100-SLT,A100-80GB,A100-40GB,RTX3090,V100-32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/netscratch/anonymous_user/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.out
#SBATCH --error=/netscratch/anonymous_user/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
LANG_CODE="${1:?Usage: sbatch run_mmlu_eval_single_lang_paper.sbatch <LANG_CODE>}"
EXPERIMENT_NAME="clt_multi_mmlu_paper_${LANG_CODE}"
CONFIG_PATH="$PROJECT_ROOT/configs/experiment/${EXPERIMENT_NAME}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

srun -K \
  --container-image=/netscratch/anonymous_user/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc "
    set -euo pipefail
    source .venv_eval/bin/activate
    python src/eval_pipeline.py experiment='$EXPERIMENT_NAME'
  "
