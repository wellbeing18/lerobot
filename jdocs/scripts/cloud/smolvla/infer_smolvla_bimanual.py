#!/usr/bin/env python3
"""
SmolVLA Bimanual Inference Script

This script runs inference with a trained SmolVLA model on bimanual tasks.
Can be used for testing checkpoints or generating action predictions.

Usage:
    # Test with a checkpoint
    python infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_xxx/checkpoints/last/pretrained_model

    # Test with HuggingFace Hub model
    python infer_smolvla_bimanual.py --checkpoint your-username/smolvla-bimanual

    # Test on specific task
    python infer_smolvla_bimanual.py --checkpoint path/to/model --task "Use left arm to pick up the banana"

    # Save predictions
    python infer_smolvla_bimanual.py --checkpoint path/to/model --output predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load SmolVLA model from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading model from: {checkpoint_path}")

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    print(f"Model loaded on {device}")
    print(f"  Chunk size: {policy.config.chunk_size}")
    print(f"  N action steps: {policy.config.n_action_steps}")
    print(f"  Max action dim: {policy.config.max_action_dim}")

    return policy


def create_dummy_observation(device: str = "cuda"):
    """Create dummy observation for testing."""
    # Create random images for 3 cameras (SmolVLA expects camera1, camera2, camera3)
    batch_size = 1
    img_size = (3, 256, 256)

    observation = {
        "observation.images.camera1": torch.randn(batch_size, *img_size, device=device),
        "observation.images.camera2": torch.randn(batch_size, *img_size, device=device),
        "observation.images.camera3": torch.randn(batch_size, *img_size, device=device),
        "observation.state": torch.zeros(batch_size, 12, device=device),  # 12 DOF bimanual
    }

    return observation


def run_inference(
    policy,
    observation: dict,
    task: str,
    device: str = "cuda",
):
    """Run inference with the policy."""
    from lerobot.policies.factory import make_pre_post_processors

    # Create preprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=None,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Add task to observation
    observation["task"] = task

    # Preprocess
    processed = preprocessor(observation)

    # Run inference
    policy.reset()
    with torch.inference_mode():
        action = policy.select_action(processed)

    # Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    return action


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Bimanual Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint or HuggingFace Hub model ID",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Use left arm to pick up the orange and place it on the plate",
        help="Task description for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (JSON)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of inference samples to run",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SmolVLA Bimanual Inference")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    policy = load_model(args.checkpoint, args.device)

    # Run inference samples
    results = []
    print(f"\nRunning {args.num_samples} inference samples...")

    for i in range(args.num_samples):
        # Create dummy observation
        observation = create_dummy_observation(args.device)

        # Run inference
        action = run_inference(policy, observation, args.task, args.device)

        # Store results
        result = {
            "sample": i,
            "task": args.task,
            "action_shape": list(action.shape),
            "action_mean": float(np.mean(action)),
            "action_std": float(np.std(action)),
            "action_min": float(np.min(action)),
            "action_max": float(np.max(action)),
        }
        results.append(result)

        print(f"  Sample {i+1}: action shape={action.shape}, "
              f"mean={result['action_mean']:.4f}, std={result['action_std']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Action shape: {results[0]['action_shape']}")

    avg_mean = np.mean([r["action_mean"] for r in results])
    avg_std = np.mean([r["action_std"] for r in results])
    print(f"  Average action mean: {avg_mean:.4f}")
    print(f"  Average action std: {avg_std:.4f}")

    # Check for NaN/Inf
    has_nan = any(np.isnan(r["action_mean"]) for r in results)
    has_inf = any(np.isinf(r["action_max"]) or np.isinf(r["action_min"]) for r in results)

    if has_nan or has_inf:
        print("\n  WARNING: NaN or Inf detected in actions!")
    else:
        print("\n  All actions are valid (no NaN/Inf)")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
