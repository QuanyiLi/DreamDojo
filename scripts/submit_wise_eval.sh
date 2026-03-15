#!/bin/bash
# Eval WISE 60-frame autoregressive with action CFG
# Usage: bash scripts/submit_wise_eval.sh [job-name] [ckpt-path] [output-dir] [guidance]
JOB_NAME=${1:-wise-eval}
CKPT_PATH=${2:-/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs/dreamdojo/wise_hpsearch/wise_lr1e4_3k_actdrop/model_consolidated.pt}
OUTPUT_DIR=${3:-/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs/wise_eval_60frames}
GUIDANCE=${4:-5.0}

runai submit \
  --name "$JOB_NAME" \
  --run-as-uid 235990 \
  --run-as-user lfeng \
  -i fenglan18009/ipad:latest \
  -g 1 \
  --pvc vita-scratch:/mnt/vita/scratch \
  --pvc home:/home/lfeng \
  --node-pool h200 \
  --large-shm \
  --environment USER=lfeng \
  --environment HF_HOME=/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface \
  --environment ENTITY=alan_lanfeng \
  --environment WANDB_API_KEY=68e83e5382ab18276f55b5aa2a219f429c2850c3 \
  --image-pull-policy IfNotPresent \
  --command -- bash -c "source /home/lfeng/miniconda3/etc/profile.d/conda.sh && conda activate pmf && cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo && source .venv/bin/activate && export PYTHONPATH=/mnt/vita/scratch/vita-students/users/lfeng/DreamDojo:\$PYTHONPATH && export HF_HOME=/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface && export IMAGINAIRE_OUTPUT_ROOT=/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs && export CUDA_MODULE_LOADING=LAZY && torchrun --nproc_per_node=1 --master_port=12349 -m scripts.eval_wise_60frames --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py --ckpt_path=$CKPT_PATH --output_dir=$OUTPUT_DIR --num_samples=20 --guidance=$GUIDANCE -- experiment=dreamdojo_2b_480_640_wise_eval job.wandb_mode=disabled ~dataloader_train.dataloaders"
