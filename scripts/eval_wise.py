"""
Eval script for WISE dataset test set.
Generates GT vs Prediction comparison videos/images.

Usage:
  torchrun --nproc_per_node=1 --master_port=12347 -m scripts.eval_wise \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=dreamdojo_2b_480_640_wise job.wandb_mode=disabled ~dataloader_train.dataloaders
"""

import os
import argparse
import importlib
import glob

import mediapy
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cosmos_predict2._src.imaginaire.config import Config
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp
from cosmos_oss.init import init_environment, init_output_dir, is_rank0

from groot_dreams.dataloader import MultiVideoActionDataset


def load_model(config):
    """Load model from config (pretrained or finetuned)."""
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        model = instantiate(config.model)
    return model


@torch.no_grad()
def run_eval(model, test_dataset, output_dir, num_samples=20, guidance=0, num_steps=35, sample_indices=None):
    """Run evaluation on test samples."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    model.on_train_start()  # Convert model to bf16
    model.eval()

    # Use provided sample indices or fallback to sequential
    if sample_indices is None:
        sample_indices = list(range(min(num_samples, len(test_dataset))))

    for sample_num, idx in enumerate(sample_indices):
        try:
            data = test_dataset[idx]
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            continue

        # Build data batch (single sample)
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
        # Ensure num_conditional_frames is set
        if "num_conditional_frames" not in data_batch:
            data_batch["num_conditional_frames"] = 1

        # Ensure text embeddings exist
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            caption = data_batch.get("ai_caption", "")
            if isinstance(caption, str):
                caption = [caption]
            if not caption:
                caption = [""]
            caption_batch = {"ai_caption": caption, "images": None}
            text_embeddings = model.text_encoder.compute_text_embeddings_online(
                caption_batch, "ai_caption"
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        # Get GT data and condition
        raw_data, x0, condition = model.get_data_and_condition(data_batch)

        # Generate sample
        sample = model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=num_steps,
        )

        # Decode
        if hasattr(model, "decode"):
            sample = model.decode(sample)

        # Normalize to [0, 1]
        pred = (1.0 + sample.float().cpu().clamp(-1, 1)) / 2.0  # [B, C, T, H, W]
        gt = (1.0 + raw_data.float().cpu().clamp(-1, 1)) / 2.0

        # Convert to video format [T, H, W, C]
        pred_video = (pred[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).numpy()
        gt_video = (gt[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).numpy()

        # Side-by-side: GT | Pred
        concat_video = np.concatenate([gt_video, pred_video], axis=2)

        # Save video
        video_path = os.path.join(output_dir, f"sample_{sample_num:04d}_idx{idx}.mp4")
        mediapy.write_video(video_path, concat_video, fps=5)

        # Save key frames as images
        for t_idx, t_label in [(0, "first"), (len(concat_video)//2, "mid"), (-1, "last")]:
            frame = concat_video[t_idx]
            img = Image.fromarray(frame)
            img.save(os.path.join(output_dir, "frames", f"sample_{sample_num:04d}_{t_label}.png"))

        logger.info(f"Saved sample {sample_num} (dataset idx {idx}): {video_path}")

        torch.cuda.empty_cache()

    logger.info(f"Evaluation complete. Results saved to {output_dir}")


def main():
    init_environment()

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument("--output_dir", default=None, help="Output directory for eval results")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test samples to evaluate")
    parser.add_argument("--ckpt_path", default=None, help="Override checkpoint path for finetuned model")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Load config
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    overrides = list(args.opts)
    config = override(config, overrides)

    if args.ckpt_path:
        config.checkpoint.load_path = args.ckpt_path

    # Set fsdp_shard_size=1 for single-GPU eval to avoid DTensor issues
    config.model.config.fsdp_shard_size = 1

    config.validate()
    config.freeze()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "eval_output"),
            "wise_eval"
        )

    # Load model
    logger.info("Loading model...")
    model = load_model(config)

    # Load test dataset - use no_noise test configs
    test_paths = sorted(glob.glob("datasets/wise_dataset_0.3.2/no_noise_demo_1_round/config_*_test/lerobot_data"))
    logger.info(f"Found {len(test_paths)} test dataset paths")

    test_dataset = MultiVideoActionDataset(
        dataset_path=test_paths,
        num_frames=13,
        data_split="full",  # test sets are already split
        single_base_index=True,  # One sample per episode to avoid near-duplicate starting frames
    )
    logger.info(f"Test dataset has {len(test_dataset)} samples")

    # Compute stride to spread samples across different configs/episodes
    total = len(test_dataset)
    num_samples = min(args.num_samples, total)
    stride = max(1, total // num_samples)
    sample_indices = [i * stride for i in range(num_samples) if i * stride < total]
    logger.info(f"Sampling {len(sample_indices)} indices with stride {stride} from {total} total samples")

    # Run evaluation
    run_eval(model, test_dataset, args.output_dir, num_samples=args.num_samples, sample_indices=sample_indices)


if __name__ == "__main__":
    main()
