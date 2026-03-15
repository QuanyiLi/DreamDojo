#!/bin/bash
set -ex

# WISE HP search training script
# Usage: CONFIG_NAME=dreamdojo_2b_480_640_wise_a bash scripts/train_wise_hp.sh

source /home/lfeng/miniconda3/etc/profile.d/conda.sh
conda activate pmf

cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo

CONFIG_NAME=${CONFIG_NAME:-dreamdojo_2b_480_640_wise_a}
NPROC=${NPROC:-8} bash launch.sh ${CONFIG_NAME}
