#!/bin/bash
# Train WISE with action dropout (CFG-based action conditioning)
# Config: configs/2b_480_640_wise_c.yaml
# Usage: bash scripts/submit_wise_train.sh [job-name] [config-name]
JOB_NAME=${1:-wise-train}
CONFIG=${2:-dreamdojo_2b_480_640_wise_c}

runai submit \
  --name "$JOB_NAME" \
  --run-as-uid 235990 \
  --run-as-user lfeng \
  -i fenglan18009/ipad:latest \
  -g 8 \
  --pvc vita-scratch:/mnt/vita/scratch \
  --pvc home:/home/lfeng \
  --node-pool h200 \
  --large-shm \
  --environment USER=lfeng \
  --environment HF_HOME=/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface \
  --environment ENTITY=alan_lanfeng \
  --environment WANDB_API_KEY=68e83e5382ab18276f55b5aa2a219f429c2850c3 \
  --image-pull-policy IfNotPresent \
  --command -- bash -c "source /home/lfeng/miniconda3/etc/profile.d/conda.sh && conda activate pmf && cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo && NPROC=8 bash launch.sh $CONFIG"
