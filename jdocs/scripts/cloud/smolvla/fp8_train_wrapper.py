#!/usr/bin/env python3
"""
FP8 Training Wrapper for SmolVLA using torchao direct API.

This script wraps lerobot_train to inject FP8 support using torchao's
convert_to_float8_training API directly.

Usage:
    python fp8_train_wrapper.py [lerobot_train args...]
"""

import sys
import os
import logging
import warnings

# Suppress warnings (including in DataLoader worker processes)
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
os.environ.setdefault('TORCHAO_DISABLE_CPP_EXTENSION_WARNING', '1')
warnings.filterwarnings('ignore', message='.*cpp extensions.*')  # torchao cpp extension warning
warnings.filterwarnings('ignore', message='.*video decoding.*deprecated.*')  # torchvision video deprecation
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')  # torchvision warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_fp8_support():
    """Check if FP8 training is supported."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_name}")

    if "H100" not in gpu_name and "H200" not in gpu_name:
        logger.warning(f"FP8 optimized for H100/H200. Detected: {gpu_name}")

    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        logger.info("torchao FP8 support: OK")
        return True
    except ImportError as e:
        logger.error(f"torchao FP8 not available: {e}")
        return False


def convert_model_to_fp8(model):
    """Convert model's linear layers to FP8 using torchao."""
    import torch
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig

    logger.info("Converting model to FP8...")

    try:
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        logger.info("Using tensorwise FP8 recipe")
    except (AttributeError, TypeError):
        config = Float8LinearConfig()
        logger.info("Using default FP8 config")

    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    skipped = []

    def module_filter_fn(mod, fqn):
        if isinstance(mod, torch.nn.Linear):
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                skipped.append(fqn)
                return False
        return True

    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

    try:
        from torchao.float8 import Float8Linear
        fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    except ImportError:
        fp8_count = linear_count - len(skipped)

    logger.info(f"Converted {fp8_count}/{linear_count} layers to FP8")
    if skipped:
        logger.info(f"Skipped {len(skipped)} layers (dims not divisible by 16)")

    return model


def main():
    """Main entry point."""
    import torch

    if not check_fp8_support():
        logger.error("FP8 not supported. Running standard training.")
        os.execvp("python", ["python", "-m", "lerobot.scripts.lerobot_train"] + sys.argv[1:])
        return

    # Monkey-patch make_policy to inject FP8
    from lerobot.policies import factory as policy_factory
    original_make_policy = policy_factory.make_policy

    def make_policy_with_fp8(*args, **kwargs):
        logger.info("=" * 60)
        logger.info("Creating policy...")
        policy = original_make_policy(*args, **kwargs)
        logger.info("Injecting FP8 layers via torchao...")
        policy = convert_model_to_fp8(policy)
        logger.info("FP8 injection complete!")
        logger.info("=" * 60)
        return policy

    policy_factory.make_policy = make_policy_with_fp8

    # Also patch it in the train module
    from lerobot.scripts import lerobot_train
    lerobot_train.make_policy = make_policy_with_fp8

    logger.info("=" * 60)
    logger.info("SmolVLA FP8 Training (torchao direct injection)")
    logger.info("=" * 60)

    # Import and call train - it will use our patched make_policy
    from lerobot.scripts.lerobot_train import train

    # Create accelerator with BF16 base
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs]
    )

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Call train with our accelerator
    # The @parser.wrap() decorator will parse sys.argv
    train(accelerator=accelerator)


if __name__ == "__main__":
    main()
