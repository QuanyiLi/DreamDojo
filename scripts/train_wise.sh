#!/bin/bash
set -ex

# WISE dataset post-training script
# Usage: NPROC=8 bash scripts/train_wise.sh

source /home/lfeng/miniconda3/etc/profile.d/conda.sh
conda activate pmf

cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo

# Use launch.sh with WISE experiment config
NPROC=${NPROC:-8} bash launch.sh dreamdojo_2b_480_640_wise
