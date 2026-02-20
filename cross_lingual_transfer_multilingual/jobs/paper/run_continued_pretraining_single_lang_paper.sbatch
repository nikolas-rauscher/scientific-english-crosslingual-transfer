#!/bin/bash
#SBATCH --job-name=paper_train
#SBATCH --partition=H100,H100-PCI,H100-SLT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer_multilingual/logs/slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
LANG_CODE="${1:?Usage: sbatch run_continued_pretraining_single_lang_paper.sbatch <LANG_CODE>}"

cd "$PROJECT_ROOT"

srun -K \
  --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc "
    set -euo pipefail
    source .venv_pretraining/bin/activate
    export INIT_MODEL_ROOT='$PROJECT_ROOT/cross_lingual_transfer_multilingual/models/wechsel_init_paper_spm32k/$LANG_CODE'
    export MAX_STEPS='15000'
    export DATA_VARIANT='sub'
    export SUBSPLIT_NAME='sub_charcap43gb_seed42'
    bash cross_lingual_transfer_multilingual/scripts/paper/run_continued_pretraining_single_lang_paper.sh '$LANG_CODE'
  "
