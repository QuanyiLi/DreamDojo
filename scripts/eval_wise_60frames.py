"""
Autoregressive long-horizon evaluation on WISE dataset test set.
Generates GT vs Prediction comparison videos using multiple autoregressive steps of 12 frames each.

With panda timestep_interval=2 and 120-frame episodes, max num_frames=59 (58 actions).
This supports 4 full AR steps (48 actions, 49 frames).

Usage:
  torchrun --nproc_per_node=1 --master_port=12349 -m scripts.eval_wise_60frames \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=dreamdojo_2b_480_640_wise_eval job.wandb_mode=disabled ~dataloader_train.dataloaders
"""

import os
import argparse
import importlib
import glob

import mediapy
import numpy as np
import torch
from PIL import Image
from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cosmos_predict2._src.imaginaire.config import Config
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp
from cosmos_oss.init import init_environment, init_output_dir, is_rank0

from groot_dreams.dataloader import MultiVideoActionDataset

NUM_ACTIONS_PER_STEP = 12
# panda: timestep_interval=2, episode=120 frames → max num_frames=59 (58 actions)
# 4 full AR steps × 12 = 48 actions → need num_frames = 49
NUM_AR_STEPS = 4
DATASET_NUM_FRAMES = 49  # 49 video frames, 48 actions


def load_model(config):
    """Load model from config (pretrained or finetuned)."""
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        model = instantiate(config.model)
    return model


def build_first_step_data_batch(
    data,              # full dataset sample dict
    gt_13_frames,      # [C, 13, H, W] uint8 - first 13 GT frames
    actions_12,        # [12, 384] float
    text_embeddings,   # precomputed
):
    """Build data_batch for the first AR step using full GT frames.

    Uses all 13 GT frames so the VAE encoding is clean (matches eval_wise.py).
    """
    C, _, H, W = gt_13_frames.shape

    # Build data_batch from dataset sample like eval_wise.py does
    data_batch = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            t = v.unsqueeze(0).cuda()
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            data_batch[k] = t
        elif isinstance(v, (int, float)):
            data_batch[k] = torch.tensor([v]).to(torch.bfloat16).cuda()
        else:
            data_batch[k] = v

    # Override video with first 13 GT frames and action with first 12
    data_batch["video"] = gt_13_frames.unsqueeze(0).cuda()  # [1, C, 13, H, W] uint8
    data_batch["action"] = actions_12.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 12, 384]
    data_batch["num_conditional_frames"] = 1
    data_batch["t5_text_embeddings"] = text_embeddings
    data_batch["t5_text_mask"] = torch.ones(
        text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
    )

    return data_batch


def build_step_data_batch(
    cond_frame_uint8,  # [C, H, W] uint8
    actions_12,        # [12, 384] float
    text_embeddings,   # precomputed
):
    """Build a data_batch for a single 13-frame generation step.

    IMPORTANT: All 13 frames are filled with the conditioning frame (not zeros).
    The VAE uses 3D temporal convolutions, so zero-filled frames corrupt the
    encoding of the first latent frame used for conditioning.
    """
    C, H, W = cond_frame_uint8.shape

    # Fill all 13 frames with the conditioning frame to avoid VAE temporal corruption
    vid = cond_frame_uint8.unsqueeze(1).expand(C, 13, H, W).clone()
    vid = vid.unsqueeze(0)  # [1, C, 13, H, W]

    # Text mask (all ones = attend to all tokens)
    t5_mask = torch.ones(
        text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
    )

    data_batch = {
        "dataset_name": "video_data",
        "video": vid.cuda(),  # uint8, model normalizes internally
        "action": actions_12.unsqueeze(0).cuda().to(torch.bfloat16),  # [1, 12, 384]
        "fps": torch.tensor([4.0]).cuda().to(torch.bfloat16),
        "padding_mask": torch.zeros(1, 1, H, W).cuda().to(torch.bfloat16),
        "num_conditional_frames": 1,
        "t5_text_embeddings": text_embeddings,
        "t5_text_mask": t5_mask,
        "ai_caption": [""],
    }

    return data_batch


@torch.no_grad()
def run_eval_autoregressive(model, test_dataset, output_dir, num_samples=20, guidance=0, num_steps=35, sample_indices=None):
    """Run autoregressive evaluation on test samples."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    model.on_train_start()
    model.eval()

    # Precompute empty text embeddings once
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        caption_batch = {"ai_caption": [""], "images": None}
        text_embeddings = model.text_encoder.compute_text_embeddings_online(
            caption_batch, "ai_caption"
        )
    else:
        from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
        text_embeddings = get_text_embedding("")

    # Use provided sample indices or fallback to sequential
    if sample_indices is None:
        sample_indices = list(range(min(num_samples, len(test_dataset))))

    for sample_num, idx in enumerate(sample_indices):
        try:
            data = test_dataset[idx]
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            continue

        gt_video = data["video"]  # [C, T, H, W] uint8
        gt_actions = data["action"]  # [T-1, 384] float

        num_total_actions = gt_actions.shape[0]
        num_steps_possible = num_total_actions // NUM_ACTIONS_PER_STEP
        if num_steps_possible < 1:
            logger.warning(f"Sample {idx}: only {num_total_actions} actions, need at least {NUM_ACTIONS_PER_STEP}. Skipping.")
            continue

        total_pred_frames = 1 + num_steps_possible * NUM_ACTIONS_PER_STEP  # 49

        C, T, H, W = gt_video.shape
        logger.info(f"Sample {idx}: GT video {gt_video.shape}, actions {gt_actions.shape}, "
                     f"AR steps={num_steps_possible}, pred frames={total_pred_frames}")

        # Autoregressive generation
        all_pred_frames = []
        cond_frame = gt_video[:, 0, :, :]  # [C, H, W] uint8 - start with GT first frame

        for step in range(num_steps_possible):
            action_start = step * NUM_ACTIONS_PER_STEP
            action_end = action_start + NUM_ACTIONS_PER_STEP
            step_actions = gt_actions[action_start:action_end]  # [12, 384]

            if step == 0:
                # First step: use full 13 GT frames for clean VAE encoding
                # This matches eval_wise.py exactly
                gt_13 = gt_video[:, :13, :, :]  # [C, 13, H, W]
                data_batch = build_first_step_data_batch(
                    data, gt_13, step_actions, text_embeddings
                )
            else:
                data_batch = build_step_data_batch(cond_frame, step_actions, text_embeddings)

            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                n_sample=1,
                num_steps=num_steps,
            )

            video_out = model.decode(sample)  # [1, C, 13, H, W] in [-1, 1]

            video_norm = (1.0 + video_out.float().cpu().clamp(-1, 1)) / 2.0
            video_uint8 = (video_norm[0] * 255).to(torch.uint8)  # [C, 13, H, W]

            if step == 0:
                all_pred_frames.append(video_uint8)  # all 13 frames
            else:
                all_pred_frames.append(video_uint8[:, 1:, :, :])  # skip conditional frame

            cond_frame = video_uint8[:, -1, :, :]  # last frame as next condition

            logger.info(f"  Step {step+1}/{num_steps_possible} done")

        # Concatenate: 13 + 12*(N-1) frames
        pred_full = torch.cat(all_pred_frames, dim=1)

        # Trim GT to match pred length
        gt_trimmed = gt_video[:, :pred_full.shape[1]]

        # Convert to [T, H, W, C]
        pred_np = pred_full.permute(1, 2, 3, 0).numpy()
        gt_np = gt_trimmed.permute(1, 2, 3, 0).numpy()

        # Side-by-side: GT | Pred
        concat_video = np.concatenate([gt_np, pred_np], axis=2)

        video_path = os.path.join(output_dir, f"sample_{sample_num:04d}_idx{idx}_ar{num_steps_possible}.mp4")
        mediapy.write_video(video_path, concat_video, fps=5)

        # Save key frames at each AR step boundary
        frame_labels = [(0, "first")]
        for s in range(num_steps_possible):
            t = min(13 + s * 12 - 1, pred_full.shape[1] - 1)
            frame_labels.append((t, f"step{s+1}_end"))
        frame_labels.append((-1, "last"))
        for t_idx, t_label in frame_labels:
            frame = concat_video[t_idx]
            img = Image.fromarray(frame)
            img.save(os.path.join(output_dir, "frames", f"sample_{sample_num:04d}_{t_label}.png"))

        logger.info(f"Saved sample {sample_num} (dataset idx {idx}): {video_path} ({pred_full.shape[1]} frames)")
        torch.cuda.empty_cache()

    logger.info(f"Evaluation complete. Results saved to {output_dir}")


def main():
    init_environment()

    parser = argparse.ArgumentParser(description="Autoregressive Long-Horizon Evaluation")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test samples")
    parser.add_argument("--num_steps", type=int, default=35, help="Diffusion steps per AR step")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale for action CFG")
    parser.add_argument("--ckpt_path", default=None, help="Override checkpoint path")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    overrides = list(args.opts)
    config = override(config, overrides)

    if args.ckpt_path:
        config.checkpoint.load_path = args.ckpt_path

    config.model.config.fsdp_shard_size = 1
    config.validate()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "eval_output"),
            "wise_eval_autoregressive"
        )

    logger.info("Loading model...")
    model = load_model(config)
    config.freeze()

    # Load test dataset
    test_paths = sorted(glob.glob("datasets/wise_dataset_0.3.2/no_noise_demo_1_round/config_*_test/lerobot_data"))
    logger.info(f"Found {len(test_paths)} test dataset paths")

    test_dataset = MultiVideoActionDataset(
        dataset_path=test_paths,
        num_frames=DATASET_NUM_FRAMES,  # 49 frames → 48 actions → 4 AR steps
        data_split="full",
        single_base_index=True,  # One sample per episode to avoid near-duplicate starting frames
    )
    logger.info(f"Test dataset has {len(test_dataset)} samples")

    # Compute stride to spread samples across different configs/episodes
    total = len(test_dataset)
    num_samples = min(args.num_samples, total)
    stride = max(1, total // num_samples)
    sample_indices = [i * stride for i in range(num_samples) if i * stride < total]
    logger.info(f"Sampling {len(sample_indices)} indices with stride {stride} from {total} total samples")

    run_eval_autoregressive(
        model, test_dataset, args.output_dir,
        num_samples=args.num_samples,
        guidance=args.guidance,
        num_steps=args.num_steps,
        sample_indices=sample_indices,
    )


if __name__ == "__main__":
    main()
