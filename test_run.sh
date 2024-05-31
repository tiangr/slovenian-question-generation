#!/bin/sh
#SBATCH --job-name=lora
#SBATCH --output=lora.log
#SBATCH --error=loraerr.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G

srun singularity exec --nv ./containers/final.sif python3 \
    "slovenian-question-generation/finetuning.py"

