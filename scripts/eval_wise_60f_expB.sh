#!/bin/bash
set -ex

source /home/lfeng/miniconda3/etc/profile.d/conda.sh
conda activate pmf

cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo
source .venv/bin/activate

export PYTHONPATH=/mnt/vita/scratch/vita-students/users/lfeng/DreamDojo:$PYTHONPATH
export HF_HOME=/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface
export IMAGINAIRE_OUTPUT_ROOT=/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs
export CUDA_MODULE_LOADING=LAZY

CKPT=/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs/dreamdojo/wise_hpsearch/wise_lr1e4_3k/model_consolidated_fixed.pt
OUTDIR=/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs/wise_eval_60frames_expB

python -m scripts.eval_wise_60frames \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --ckpt_path=$CKPT \
  --output_dir=$OUTDIR \
  --num_samples=20 \
  -- experiment=dreamdojo_2b_480_640_wise_eval job.wandb_mode=disabled ~dataloader_train.dataloaders
