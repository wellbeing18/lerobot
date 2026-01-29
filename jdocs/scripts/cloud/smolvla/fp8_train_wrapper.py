#!/usr/bin/env python3
"""
FP8 Training Wrapper for SmolVLA using torchao direct API.

This script wraps lerobot_train to inject FP8 support using torchao's
convert_to_float8_training API directly, bypassing accelerate's FP8 backend
which has version compatibility issues.

Usage:
    python fp8_train_wrapper.py [lerobot_train args...]

Example:
    python fp8_train_wrapper.py \
        --dataset.repo_id=jasmine314342/picknplace-bimanual-464 \
        --policy.path=lerobot/smolvla_base \
        --batch_size=128 \
        --steps=40000
"""

import sys
import os
import logging
import warnings

# Suppress torchao warnings about cpp extensions
warnings.filterwarnings('ignore', message='.*cpp extensions.*')
os.environ['TORCHAO_DISABLE_CPP_EXTENSION_WARNING'] = '1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_fp8_support():
    """Check if FP8 training is supported on this system."""
    import torch

    # Check for H100/Hopper GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_name}")

    if "H100" not in gpu_name and "H200" not in gpu_name:
        logger.warning(f"FP8 is optimized for H100/H200 GPUs. Detected: {gpu_name}")
        logger.warning("FP8 may still work but performance gains may be limited.")

    # Check torchao
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        logger.info("torchao FP8 support: OK")
        return True
    except ImportError as e:
        logger.error(f"torchao FP8 not available: {e}")
        return False


def convert_model_to_fp8(model):
    """
    Convert model's linear layers to FP8 using torchao.

    Args:
        model: PyTorch model (SmolVLAPolicy)

    Returns:
        model with FP8 layers injected
    """
    import torch
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig

    logger.info("Converting model to FP8...")

    # Create FP8 config - use default which is most compatible
    try:
        # Try tensorwise recipe (fastest) if available
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        logger.info("Using tensorwise FP8 recipe")
    except (AttributeError, TypeError):
        # Fall back to default config
        config = Float8LinearConfig()
        logger.info("Using default FP8 config")

    # Count layers before conversion
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))

    # Filter function: skip layers with incompatible dimensions
    # FP8 requires dimensions divisible by 16
    skipped_layers = []

    def module_filter_fn(mod, fqn):
        if isinstance(mod, torch.nn.Linear):
            # Skip layers with dimensions not divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                skipped_layers.append(f"{fqn} ({mod.in_features}x{mod.out_features})")
                return False
        return True

    # Convert to FP8
    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

    # Count converted layers
    try:
        from torchao.float8 import Float8Linear
        fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    except ImportError:
        fp8_count = linear_count - len(skipped_layers)

    logger.info(f"Converted {fp8_count}/{linear_count} Linear layers to FP8")
    if skipped_layers:
        logger.info(f"Skipped {len(skipped_layers)} layers (dims not divisible by 16)")

    return model


def main():
    """Main entry point - runs lerobot training with FP8 injection."""
    import torch

    # Check FP8 support first
    if not check_fp8_support():
        logger.error("FP8 not supported. Falling back to BF16 training.")
        # Run standard training
        os.execvp("python", ["python", "-m", "lerobot.scripts.lerobot_train"] + sys.argv[1:])
        return

    # Import after check to avoid import errors if torchao is broken
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.datasets.utils import cycle
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy
    from lerobot.utils.logging_utils import MetricsTracker
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint
    from lerobot.utils.utils import init_logging
    from lerobot.scripts.lerobot_train import update_policy

    # Parse config using draccus (same as lerobot's parser does internally)
    import draccus
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=sys.argv[1:])
    cfg.validate()

    # Create accelerator with BF16 base (FP8 injected via torchao)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs]
    )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    device = accelerator.device

    # Enable performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logger.info("=" * 60)
        logger.info("SmolVLA FP8 Training (torchao direct injection)")
        logger.info("=" * 60)
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision base: bf16 (FP8 injected via torchao)")

    # Set seed
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Create dataset
    if is_main_process:
        logger.info("Creating dataset...")
    dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()

    # Create policy
    if is_main_process:
        logger.info("Creating policy...")
    policy = make_policy(cfg, device=device, ds_meta=dataset.meta)

    # === INJECT FP8 HERE ===
    if is_main_process:
        logger.info("=" * 60)
        logger.info("Injecting FP8 layers via torchao...")

    policy = convert_model_to_fp8(policy)

    # Optional: Apply torch.compile for better FP8 performance
    # Note: This can increase compilation time but improves throughput
    USE_TORCH_COMPILE = os.environ.get('USE_TORCH_COMPILE', 'false').lower() == 'true'
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        if is_main_process:
            logger.info("Applying torch.compile (this may take a few minutes)...")
        policy = torch.compile(policy)

    if is_main_process:
        logger.info("FP8 injection complete!")
        logger.info("=" * 60)

    # Create optimizer and scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create dataloader
    sampler = EpisodeAwareSampler(
        dataset.episode_data_index,
        drop_n_last_frames=cfg.policy.drop_n_last_frames if hasattr(cfg.policy, 'drop_n_last_frames') else 0,
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare with accelerator
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    # Create data iterator
    dl_iter = cycle(dataloader)

    # Training loop
    train_metrics = MetricsTracker()

    if is_main_process:
        logger.info(f"Starting training for {cfg.steps} steps...")
        logger.info(f"Batch size: {cfg.batch_size}")

    for step in range(cfg.steps):
        batch = next(dl_iter)

        train_metrics, output_dict = update_policy(
            train_metrics=train_metrics,
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Logging
        if step % cfg.log_freq == 0 and is_main_process:
            metrics = train_metrics.get_metrics()
            loss = metrics.get('loss', 0)
            lr = metrics.get('learning_rate', 0)
            logger.info(f"Step {step}/{cfg.steps} | Loss: {loss:.4f} | LR: {lr:.2e}")

        # Save checkpoint
        if step > 0 and step % cfg.save_freq == 0:
            if is_main_process:
                logger.info(f"Saving checkpoint at step {step}...")
                save_checkpoint(
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                update_last_checkpoint(cfg.output_dir, step)

    # Final checkpoint
    if is_main_process:
        logger.info("Training complete! Saving final checkpoint...")
        save_checkpoint(
            step=cfg.steps,
            cfg=cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        update_last_checkpoint(cfg.output_dir, cfg.steps)
        logger.info(f"Checkpoint saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
