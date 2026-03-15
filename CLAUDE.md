# DreamDojo - Claude Code Environment

## Compute Environment

You are running inside a **Kubernetes pod** with:
- **Local GPU**: 1x NVIDIA A100

### RunAI Job Submission

Jobs can be submitted via `runai submit` with up to **8x H100 or H200** GPUs.

**User config:**
- UID: `235990`
- User: `lfeng`
- Docker image: `fenglan18009/ipad:latest`
- Conda env: `pmf` (at `/home/lfeng/miniconda3`)
- PVC mounts:
  - `vita-scratch` → `/mnt/vita/scratch`
  - `home` → `/home/lfeng`
- HF cache: `/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface`
- WandB entity: `alan_lanfeng`
- WandB API key: `68e83e5382ab18276f55b5aa2a219f429c2850c3`

**Sample job submission:**

```bash
runai submit \
  --name <job-name> \
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
  --command -- bash -c "
source /home/lfeng/miniconda3/etc/profile.d/conda.sh && \
conda activate pmf && \
cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo && \
bash scripts/train.sh"
```

**Node pool options:** `h200` (preferred), `h100`

**CRITICAL:** The `--command` must include `cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo` before running any scripts, because `launch.sh` uses relative paths (e.g., `.venv/bin/activate`). Without this `cd`, the job will fail with "No such file or directory".

**Useful RunAI commands:**
```bash
runai list jobs          # list all jobs
runai logs <job-name>    # view job logs
runai delete <job-name>  # delete a job
runai describe job <job-name>  # job details
```

## Python Environment

- Uses **uv** (at `/home/lfeng/.local/bin/uv`) with `.venv` in the project root
- Python 3.10 from conda `dreamdojo` env (for Python.h headers)
- Key packages: torch 2.7+cu128, transformer_engine, flash_attn, megatron, natten, pytorch3d
- Activate: `source .venv/bin/activate`
- `launch.sh` already configured to use `.venv`

## Training

- Launch: `NPROC=1 bash launch.sh dreamdojo_2b_480_640_gr1`
- Configs in `configs/` (yaml) and `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/`
- Checkpoints: `checkpoints/DreamDojo/` (LAM_400k.ckpt downloaded)
- Datasets:
  - GR1: `datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/`
  - WISE: `datasets/wise_dataset_0.3.2/` (48 configs × {no_noise_demo_1_round, noise_demo_5_round})
- Logs: `$IMAGINAIRE_OUTPUT_ROOT` = `/mnt/vita/scratch/vita-students/users/lfeng/dreamdojo_logs`

## Config System Architecture

YAML configs in `configs/` are loaded by `cosmos_predict2/experiments/base/action.py`:

1. Each `configs/<name>.yaml` becomes experiment `dreamdojo_<name>`
2. Base config is selected by name: `_default_groot_config` (default), `_default_wise_config` (for "wise"), `_default_groot_config_14b` (for "14b")
3. YAML overrides are merged into the base config, then registered in Hydra ConfigStore

**Key files:**
- `cosmos_predict2/experiments/base/action.py` — base configs + YAML loading loop
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py` — dataset/dataloader registration
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py` — Hydra config entry point

**IMPORTANT — Hydra data_train override vs YAML dataset_path:**
WISE experiments MUST use `_default_wise_config` (which has `data_train: wise_13frame_480_640_train`). The default `_default_groot_config` uses `data_train: dreamdojo_13frame_480_640_train` which provides a LazyCall dataloader with GR1 paths. Hydra's `_self_` merge CANNOT override `dataset_path` inside nested LazyCall objects, so YAML `dataloader_train.dataset.dataset_path` is silently ignored if the wrong base config is used.

## Evaluation

- 60-frame autoregressive eval: `scripts/eval_wise_60frames.py` (1 GPU)
- 13-frame single-step eval: `scripts/eval_wise.py` (1 GPU)
- Eval configs: `configs/2b_480_640_wise_eval.yaml`, `configs/2b_480_640_wise_eval_lr1e4_3k.yaml`
- Eval uses `--ckpt_path` to override checkpoint, test data is hardcoded in the script (test split)

## Gotchas

- **Stale checkpoints**: If `dreamdojo_logs/<project>/<group>/<name>/checkpoints/` already has a checkpoint at `iter >= max_iter`, training completes instantly. Delete the log directory before re-training.
- **RunAI `cd` required**: See CRITICAL note above. Without `cd`, jobs fail with "No such file or directory".

## Project Structure

Working directory: `/mnt/vita/scratch/vita-students/users/lfeng/DreamDojo`
