# WISE Dataset Post-Training & Evaluation

This document describes the full pipeline for post-training the Cosmos Predict2 world model on the WISE (Panda robot) dataset and evaluating the results.

## Dataset

**WISE dataset v0.3.2** — tabletop manipulation with a Franka Panda robot arm.

- Location: `datasets/wise_dataset_0.3.2/`
- Subsets: `no_noise_demo_1_round/` (clean demos), `noise_demo_5_round/` (noisy demos)
- 24 task configs (`config_0` to `config_23`), each split into `_train` and `_test`
- Format: LeRobot (parquet + videos)
- Action dim: **8** (Panda joint actions)
- Episode length: 120 frames at `timestep_interval=2` → 59 usable frames (58 actions)

## Action Conditioning Pipeline

### Action Representation

Actions go through the following pipeline:

1. **Delta action computation** (`groot_dreams/data/dataset.py:1057-1059`):
   ```python
   for t in range(1, len(actions) - 1, 4):
       delta_actions.append(actions[t:t+4] - actions[t-1])
   ```
   Each 4-frame chunk produces cumulative displacements relative to the chunk start frame. Within one chunk, values represent 1-step, 2-step, 3-step, and 4-step cumulative deltas.

2. **384-dim sparse vector** (`dataset.py:1061-1076`):
   The 8-dim Panda actions are placed at indices `[169:177]` in a 384-dim vector. The last 32 dims are initialized to 1.0 (reserved for LAM latent actions, multiplied in during forward pass). All other dims are zero.

   Embodiment-specific offsets in the 384-dim vector:
   | Embodiment | Indices | Dims |
   |-----------|---------|------|
   | GR1 | `[0:29]` | 29 |
   | G1 | `[58:101]` | 43 |
   | YAM | `[101:147]` | 46 |
   | AgiBot | `[147:169]` | 22 |
   | WISE/Panda | `[169:177]` | 8 |
   | LAM latent | `[352:384]` | 32 |

3. **Action embedding** (`action_conditioned_minimal_v1_lvg_dit.py:261-273`):
   - 12 actions are reshaped into 3 chunks of 4 (matching temporal_compression_ratio=4)
   - Each chunk is flattened: `[4, 384] → [1536]`
   - Two MLPs produce embeddings: `action_embedder_B_D` (→ model_channels) and `action_embedder_B_3D` (→ 3×model_channels)
   - A zero embedding is prepended for the conditioning frame → final shape `[B, 4, D]`

4. **Injection into transformer** (`action_conditioned_minimal_v1_lvg_dit.py:308-311`):
   Action embeddings are **added** to timestep embeddings (both `t_embedding` and `adaln_lora`), then flow through all DiT blocks via adaptive layer norm.

### Key Design Choices (inherited from official Cosmos)

- **No action dropout** (`conditioner.py:275`): `dropout_rate=0.0` for action. This means CFG (classifier-free guidance) cannot amplify the action signal — the unconditional path still sees full actions.
- **Additive injection**: Actions are added to timestep embeddings rather than using cross-attention or concatenation.
- **384-dim multi-embodiment vector**: Designed for joint training across robots. For WISE-only training, only 8/384 dims carry signal.

## Training

### Config System

YAML configs in `configs/` are loaded by `cosmos_predict2/experiments/base/action.py`:
- Configs with "wise" in the name use `_default_wise_config` (base config with WISE dataloader)
- YAML overrides are merged on top

### Training Configs

| Config | File | Description |
|--------|------|-------------|
| Initial experiment | `configs/2b_480_640_wise.yaml` | 1k iters, lr default |
| HP search (expB) | `configs/2b_480_640_wise_b.yaml` | 3k iters, lr=1e-4, cosine schedule |

Key parameters in `configs/2b_480_640_wise_b.yaml`:
```yaml
optimizer:
  lr: 1.0e-4
trainer:
  max_iter: 3000
  grad_accum_iter: 1
model:
  config:
    net:
      action_dim: 384          # Full multi-embodiment dim
      temporal_compression_ratio: 4
      num_action_per_chunk: 12
dataloader_train:
  batch_size: 4
  dataset:
    num_frames: 13             # 1 conditioning + 12 generated
    dataset_path: [...]        # 48 paths (24 configs × 2 subsets)
```

### Running Training

```bash
# Single command (from project dir)
NPROC=8 bash launch.sh dreamdojo_2b_480_640_wise_b

# Via RunAI job submission
runai submit --name wise-train \
  --run-as-uid 235990 --run-as-user lfeng \
  -i fenglan18009/ipad:latest -g 8 \
  --pvc vita-scratch:/mnt/vita/scratch --pvc home:/home/lfeng \
  --node-pool h200 --large-shm \
  --environment USER=lfeng \
  --environment HF_HOME=/mnt/vita/scratch/vita-students/users/lfeng/.cache/huggingface \
  --environment ENTITY=alan_lanfeng \
  --environment WANDB_API_KEY=<key> \
  --command -- bash -c "source /home/lfeng/miniconda3/etc/profile.d/conda.sh && \
    conda activate pmf && \
    cd /mnt/vita/scratch/vita-students/users/lfeng/DreamDojo && \
    bash scripts/train_wise_hp.sh"
```

### Checkpoint Consolidation

`scripts/train.py` monkey-patches the training loop to automatically consolidate the DCP (distributed checkpoint) into a single `.pt` file after training completes. The consolidated checkpoint is saved as `model_consolidated.pt` in the job output directory.

For manual consolidation:
```python
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save("path/to/iter_000003000/model", "output.pt")
```

## Evaluation

### 13-Frame Single-Step Eval

Script: `scripts/eval_wise.py`

Generates 13 frames (1 conditioning + 12 predicted) given GT first frame and 12 actions. Outputs side-by-side GT|Pred comparison videos.

```bash
torchrun --nproc_per_node=1 --master_port=12347 -m scripts.eval_wise \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --ckpt_path=<checkpoint.pt> \
  --output_dir=<output_dir> \
  -- experiment=dreamdojo_2b_480_640_wise_eval job.wandb_mode=disabled \
  ~dataloader_train.dataloaders
```

### 60-Frame Autoregressive Eval

Script: `scripts/eval_wise_60frames.py`

Generates 49 frames (4 autoregressive steps of 12 frames each) for long-horizon evaluation.

```bash
torchrun --nproc_per_node=1 --master_port=12349 -m scripts.eval_wise_60frames \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --ckpt_path=<checkpoint.pt> \
  --output_dir=<output_dir> \
  --num_samples=20 \
  -- experiment=dreamdojo_2b_480_640_wise_eval job.wandb_mode=disabled \
  ~dataloader_train.dataloaders
```

Autoregressive generation details:
- **Step 0**: Uses all 13 GT frames for clean VAE encoding; generates 13 frames
- **Steps 1-3**: Uses last predicted frame as conditioning; fills all 13 input slots with the conditioning frame (required by VAE 3D temporal convolution); generates 12 new frames (skip conditioning frame)
- Total output: 13 + 12 × 3 = 49 frames

### Eval Shell Script

`scripts/eval_wise_60f_expB.sh` — ready-to-run eval for the expB checkpoint:
```bash
bash scripts/eval_wise_60f_expB.sh
```

## Code Changes from Official Cosmos

Changes made to support WISE post-training (vs initial commit `2d56fd0`):

### Modified Files

1. **`groot_dreams/data/dataset.py`**:
   - Added WISE/Panda action mapping at indices `[169:177]` (via `config_` path detection)
   - Initialize last 32 dims to 1.0 for LAM latent action preservation
   - Simplified `inhouse_human` handling

2. **`cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`**:
   - Fixed `rearrange` dimension order: `(t b) → (b t)` for LAM latent action reshape

3. **`cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`**:
   - Added WISE dataset/dataloader registration in Hydra ConfigStore

4. **`cosmos_predict2/experiments/base/action.py`**:
   - Added `_default_wise_config` base config pointing to WISE dataloaders
   - Added "wise" keyword routing in YAML config loading

5. **`launch.sh`**: Updated paths from NVIDIA internal to local cluster

6. **`scripts/train.py`**: Added automatic DCP→.pt checkpoint consolidation after training

### New Files

- `configs/2b_480_640_wise.yaml` — WISE training config (1k iters)
- `configs/2b_480_640_wise_b.yaml` — WISE HP search config (3k iters, lr=1e-4)
- `configs/2b_480_640_wise_eval.yaml` — Eval config (pretrained baseline)
- `configs/2b_480_640_wise_eval_lr1e4_3k.yaml` — Eval config (expB checkpoint)
- `scripts/eval_wise.py` — 13-frame single-step evaluation
- `scripts/eval_wise_60frames.py` — 60-frame autoregressive evaluation
- `scripts/train_wise.sh`, `scripts/train_wise_hp.sh` — Training launch scripts
- `scripts/eval_wise_60f_expB.sh` — Eval launch script for expB

### Unchanged (Official Design)

The following are inherited from the official NVIDIA Cosmos codebase and were **not modified**:
- Action `dropout_rate=0.0` (no CFG amplification for actions)
- 384-dim multi-embodiment sparse action vector
- Cumulative delta action computation within 4-frame chunks
- Additive action injection into timestep embeddings
- Action embedding MLP architecture

## Known Limitations

1. **Action CFG not supported**: Since `action dropout_rate=0.0`, the unconditional path in CFG still sees full actions. Setting `guidance > 0` only amplifies text/video conditioning, not action adherence.

2. **Sparse action input**: Only 8/384 dims carry WISE action signal. The MLP processes 1536 input dims (384×4) but only 32 (8×4) are non-zero/non-constant. This may dilute the action signal.

3. **Autoregressive drift**: Over multiple AR steps, prediction errors accumulate. The conditioning frame quality degrades because it comes from the previous step's output rather than GT.
