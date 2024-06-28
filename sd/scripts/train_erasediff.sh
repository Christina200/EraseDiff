#!/bin/bash

#SBATCH --job-name=train_ed
#SBATCH --output=train_ed%j.out
#SBATCH --time=14-0
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

python train_scripts/erasediff_concept.py --train_method 'noxattn' \
 --epochs 10 --K_steps 2 --lambda_bome 0.1 --lr 1e-5 --batch_size 16 --device '0' \
 --concept_to_forget beard \
 --forget_path /home/xw6956/Generative-Model-Unlearning-Fairness/data/beard_SD-v1-4 \
 --retain_path /home/xw6956/Generative-Model-Unlearning-Fairness/data/no_beard_SD-v1-4 \

# python check_cuda.py