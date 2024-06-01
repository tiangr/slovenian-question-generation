#!/bin/sh
#SBATCH --job-name=lorav2
#SBATCH --output=v2.log
#SBATCH --error=erv2.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20

python3 ul-fri-nlp-course-project-linguini/queryingmodel.py
