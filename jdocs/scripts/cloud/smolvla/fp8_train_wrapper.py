#!/usr/bin/env python3
"""
FP8 Training Wrapper for SmolVLA

This script wraps lerobot_train to inject FP8 support using torchao's
convert_to_float8_training API directly, bypassing accelerate's FP8 backend.

This is useful when:
- accelerate doesn't recognize the torchao backend
- TransformerEngine has version compatibility issues
- You want more control over FP8 conversion

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
    if "H100" not in gpu_name and "H200" not in gpu_name:
        logger.warning(f"FP8 is optimized for H100/H200 GPUs. Detected: {gpu_name}")
        logger.warning("Performance may not be optimal on this GPU.")

    # Check torchao
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        logger.info("torchao FP8 support: OK")
        return True
    except ImportError as e:
        logger.error(f"torchao FP8 not available: {e}")
        return False


def inject_fp8_into_model(model):
    """
    Convert model's linear layers to FP8 using torchao.

    Args:
        model: PyTorch model (SmolVLAPolicy)

    Returns:
        model with FP8 layers injected
    """
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    import torch

    logger.info("Injecting FP8 layers into model...")

    # Create FP8 config
    # Use tensorwise recipe for best speedup
    try:
        config = Float8LinearConfig.from_recipe_name("tensorwise")
    except AttributeError:
        # Older torchao versions
        config = Float8LinearConfig()

    # Filter function: skip layers with incompatible dimensions
    # FP8 requires dimensions divisible by 16
    def module_filter_fn(mod, fqn):
        if isinstance(mod, torch.nn.Linear):
            # Skip layers with dimensions not divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                logger.debug(f"Skipping {fqn}: dims not divisible by 16 "
                            f"({mod.in_features}x{mod.out_features})")
                return False
            # Optionally skip first/last layers for stability
            # (torchao recommends keeping these in higher precision)
            if 'lm_head' in fqn or 'embed' in fqn.lower():
                logger.debug(f"Skipping {fqn}: embedding/head layer")
                return False
        return True

    # Count layers before conversion
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info(f"Found {linear_count} Linear layers")

    # Convert to FP8
    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

    # Count converted layers
    from torchao.float8 import Float8Linear
    fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    logger.info(f"Converted {fp8_count}/{linear_count} layers to FP8")

    return model


def main():
    """Main entry point."""
    # Check FP8 support
    if not check_fp8_support():
        logger.error("FP8 not supported. Please use standard BF16 training.")
        sys.exit(1)

    # Import lerobot modules
    from lerobot.scripts.lerobot_train import train, parser
    from lerobot.common.utils.config_utils import parse_cfg
    import torch

    # Parse config from command line args
    args = sys.argv[1:]
    cfg = parse_cfg(args, parser)

    # Create accelerator with BF16 (FP8 will be handled by torchao directly)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",  # Use BF16 as base, FP8 injected separately
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs]
    )

    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")

    # We need to hook into the training to inject FP8 after model creation
    # This is done by monkey-patching the policy's post_init

    original_train = train

    def train_with_fp8(cfg, accelerator=None):
        """Wrapper that injects FP8 into the model after creation."""
        from lerobot.common.policies.factory import make_policy
        original_make_policy = make_policy

        def make_policy_with_fp8(*args, **kwargs):
            policy = original_make_policy(*args, **kwargs)
            # Inject FP8
            policy = inject_fp8_into_model(policy)
            # Compile for optimal performance
            if hasattr(torch, 'compile'):
                logger.info("Applying torch.compile for optimal FP8 performance...")
                policy = torch.compile(policy)
            return policy

        # Monkey-patch
        import lerobot.common.policies.factory as factory_module
        factory_module.make_policy = make_policy_with_fp8

        # Run training
        return original_train(cfg, accelerator=accelerator)

    # Run training with FP8
    logger.info("Starting FP8 training...")
    train_with_fp8(cfg, accelerator=accelerator)


if __name__ == "__main__":
    main()
