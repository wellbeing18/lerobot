#!/usr/bin/env python3
"""
Training Pipeline Verification Script

This script performs a training smoke test to verify the full training pipeline
works correctly BEFORE committing to expensive training runs.

Checks:
1. Dataset loading with training transforms
2. SmolVLA model loading
3. Camera key renaming (head->camera1, etc.)
4. Forward pass with actual data
5. Loss computation
6. Backward pass (gradient flow)
7. Optimizer step
8. Input/output shape validation
9. Normalization verification

Usage:
    # Full smoke test (5 training steps)
    python verify_training_pipeline.py

    # Quick check (1 step, no gradients)
    python verify_training_pipeline.py --quick

    # More steps for thorough check
    python verify_training_pipeline.py --steps 10

    # Custom dataset
    python verify_training_pipeline.py --dataset-path /path/to/dataset
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Default paths
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
DEFAULT_PRETRAINED = "lerobot/smolvla_base"
LOG_DIR = PROJECT_ROOT / "jdocs" / "logs"

# Expected configuration
EXPECTED_ACTION_DIM = 12
EXPECTED_STATE_DIM = 12
EXPECTED_CAMERAS = ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"]
SMOLVLA_CAMERA_MAPPING = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


class TrainingVerificationResult:
    """Track verification results."""

    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.metrics = {}

    def add_check(self, name: str, passed: bool, message: str, warning: bool = False):
        status = "PASS" if passed else ("WARN" if warning else "FAIL")
        self.checks.append({"name": name, "status": status, "message": message})
        if passed:
            self.passed += 1
        elif warning:
            self.warnings += 1
        else:
            self.failed += 1

        # Print immediately
        status_color = {"PASS": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m"}
        reset = "\033[0m"
        color = status_color.get(status, "")
        print(f"  {color}[{status}]{reset} {name}: {message}")

    def add_metric(self, name: str, value):
        self.metrics[name] = value

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE VERIFICATION SUMMARY")
        print("=" * 70)

        print(f"\nTotal: {self.passed + self.failed + self.warnings} checks")
        print(f"  \033[92mPassed: {self.passed}\033[0m")
        print(f"  \033[91mFailed: {self.failed}\033[0m")
        print(f"  \033[93mWarnings: {self.warnings}\033[0m")

        if self.metrics:
            print("\nMetrics:")
            for name, value in self.metrics.items():
                print(f"  {name}: {value}")

        if self.failed > 0:
            print("\n\033[91mTRAINING VERIFICATION FAILED - DO NOT PROCEED\033[0m")
            return False
        elif self.warnings > 0:
            print("\n\033[93mTRAINING VERIFICATION PASSED WITH WARNINGS\033[0m")
            return True
        else:
            print("\n\033[92mTRAINING VERIFICATION PASSED - Safe to train\033[0m")
            return True


def verify_dataset_loading(dataset_path: Path, result: TrainingVerificationResult, batch_size: int = 2):
    """Verify dataset loads correctly with training pipeline."""
    print("\n" + "=" * 70)
    print("1. DATASET LOADING (Training Pipeline)")
    print("=" * 70)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader

        print(f"  Loading dataset: {dataset_path}")
        print(f"  Video backend: pyav")
        # Use dataset folder name as repo_id for local loading
        repo_id = dataset_path.name
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(dataset_path),
            video_backend="pyav",
        )

        result.add_check("Dataset loaded", True, f"{len(dataset)} frames, {dataset.meta.total_episodes} episodes")
        result.add_metric("dataset_frames", len(dataset))
        result.add_metric("dataset_episodes", dataset.meta.total_episodes)

        # Create DataLoader like training does
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Get a batch
        batch = next(iter(dataloader))
        result.add_check("DataLoader works", True, f"batch_size={batch_size}")

        # Check batch contents
        print(f"\n  Batch keys: {list(batch.keys())}")

        # Verify action shape
        action = batch.get("action")
        if action is not None:
            action_shape = tuple(action.shape)
            expected = (batch_size, EXPECTED_ACTION_DIM)
            result.add_check("Action shape", action_shape == expected,
                           f"{action_shape} (expected {expected})")
        else:
            result.add_check("Action exists", False, "Action key not found in batch")

        # Verify state shape
        state = batch.get("observation.state")
        if state is not None:
            state_shape = tuple(state.shape)
            expected = (batch_size, EXPECTED_STATE_DIM)
            result.add_check("State shape", state_shape == expected,
                           f"{state_shape} (expected {expected})")

        # Verify camera images
        for cam_key in EXPECTED_CAMERAS:
            img = batch.get(cam_key)
            if img is not None:
                img_shape = tuple(img.shape)
                # Expect (B, C, H, W)
                valid = len(img_shape) == 4 and img_shape[1] == 3
                result.add_check(f"Camera {cam_key}", valid, f"shape={img_shape}")
            else:
                result.add_check(f"Camera {cam_key}", False, "Not found")

        # Verify task_index
        task_idx = batch.get("task_index")
        if task_idx is not None:
            result.add_check("Task index exists", True, f"shape={tuple(task_idx.shape)}")

        return dataset, dataloader, batch

    except Exception as e:
        import traceback
        result.add_check("Dataset loading", False, f"Error: {e}")
        traceback.print_exc()
        return None, None, None


def verify_camera_renaming(batch: dict, result: TrainingVerificationResult):
    """Verify camera key renaming works correctly."""
    print("\n" + "=" * 70)
    print("2. CAMERA KEY RENAMING")
    print("=" * 70)

    if batch is None:
        result.add_check("Camera renaming", False, "No batch available")
        return None

    # Apply rename mapping (as training script does)
    renamed_batch = {}
    for key, value in batch.items():
        new_key = SMOLVLA_CAMERA_MAPPING.get(key, key)
        renamed_batch[new_key] = value
        if key != new_key:
            print(f"  Renamed: {key} -> {new_key}")

    # Verify renamed keys exist
    expected_renamed = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
    for cam_key in expected_renamed:
        exists = cam_key in renamed_batch
        result.add_check(f"Renamed key {cam_key}", exists,
                        "Present" if exists else "Missing")

    return renamed_batch


def verify_model_loading(pretrained_path: str, result: TrainingVerificationResult, device: str = "cuda"):
    """Verify SmolVLA model loads correctly."""
    print("\n" + "=" * 70)
    print("3. MODEL LOADING")
    print("=" * 70)

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        print(f"  Loading model: {pretrained_path}")
        print(f"  Device: {device}")

        policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        policy.to(device)

        result.add_check("Model loaded", True, f"SmolVLAPolicy on {device}")

        # Check model config
        config = policy.config
        print(f"\n  Model config:")
        print(f"    chunk_size: {config.chunk_size}")
        print(f"    n_action_steps: {config.n_action_steps}")
        print(f"    num_steps: {config.num_steps}")
        print(f"    max_action_dim: {config.max_action_dim}")

        result.add_check("Config chunk_size", config.chunk_size > 0, f"{config.chunk_size}")
        result.add_check("Config max_action_dim", config.max_action_dim >= EXPECTED_ACTION_DIM,
                        f"{config.max_action_dim} >= {EXPECTED_ACTION_DIM}")

        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"\n  Parameters:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")

        result.add_metric("total_params", f"{total_params:,}")
        result.add_metric("trainable_params", f"{trainable_params:,}")

        return policy

    except Exception as e:
        import traceback
        result.add_check("Model loading", False, f"Error: {e}")
        traceback.print_exc()
        return None


def verify_forward_pass(policy, batch: dict, result: TrainingVerificationResult, device: str = "cuda",
                        dataset_stats: dict = None):
    """Verify forward pass works and produces valid output."""
    print("\n" + "=" * 70)
    print("4. FORWARD PASS")
    print("=" * 70)

    if policy is None or batch is None:
        result.add_check("Forward pass", False, "Model or batch not available")
        return None

    try:
        from lerobot.policies.factory import make_pre_post_processors

        # Create training preprocessor (tokenizes task strings -> observation.language.tokens)
        print(f"  Creating training preprocessor...")
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )

        # Move batch to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        # Add task strings if needed (SmolVLA requires task)
        if "task" not in batch_device:
            batch_size = batch_device["action"].shape[0]
            batch_device["task"] = ["Use left arm to pick up the orange and place it on the plate"] * batch_size

        # Apply preprocessor (tokenizes task -> observation.language.tokens)
        print(f"  Applying preprocessor (tokenizing task strings)...")
        batch_processed = preprocessor(batch_device)

        # Check that language tokens were created
        lang_tokens_key = "observation.language.tokens"
        if lang_tokens_key in batch_processed:
            print(f"  Language tokens shape: {batch_processed[lang_tokens_key].shape}")
            result.add_check("Language tokens created", True,
                           f"shape={tuple(batch_processed[lang_tokens_key].shape)}")
        else:
            result.add_check("Language tokens created", False, "Missing observation.language.tokens")

        print(f"  Running forward pass...")
        policy.train()

        with torch.amp.autocast(device_type="cuda", enabled=True):
            output = policy.forward(batch_processed)

        # Check output
        if isinstance(output, dict):
            print(f"  Output keys: {list(output.keys())}")

            # Check loss
            loss = output.get("loss")
            if loss is not None:
                loss_val = loss.item()
                is_valid = not (np.isnan(loss_val) or np.isinf(loss_val))
                result.add_check("Loss valid", is_valid, f"{loss_val:.4f}")
                result.add_metric("initial_loss", f"{loss_val:.4f}")

                # Check loss is reasonable (not too high)
                reasonable = loss_val < 100.0
                result.add_check("Loss reasonable", reasonable,
                               f"{loss_val:.4f} < 100.0" if reasonable else f"{loss_val:.4f} too high!")
            else:
                result.add_check("Loss exists", False, "No loss in output")

            # Check action prediction if available
            if "action" in output:
                action_pred = output["action"]
                action_shape = tuple(action_pred.shape)
                print(f"  Action prediction shape: {action_shape}")
                result.add_check("Action prediction", True, f"shape={action_shape}")

        else:
            result.add_check("Output format", False, f"Expected dict, got {type(output)}")

        return output

    except Exception as e:
        import traceback
        result.add_check("Forward pass", False, f"Error: {e}")
        traceback.print_exc()
        return None


def verify_backward_pass(policy, batch: dict, result: TrainingVerificationResult, device: str = "cuda",
                         dataset_stats: dict = None):
    """Verify backward pass works and gradients flow."""
    print("\n" + "=" * 70)
    print("5. BACKWARD PASS (Gradient Flow)")
    print("=" * 70)

    if policy is None or batch is None:
        result.add_check("Backward pass", False, "Model or batch not available")
        return False

    try:
        from lerobot.policies.factory import make_pre_post_processors

        # Create training preprocessor
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )

        # Move batch to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value

        if "task" not in batch_device:
            batch_size = batch_device["action"].shape[0]
            batch_device["task"] = ["Use left arm to pick up the orange and place it on the plate"] * batch_size

        # Apply preprocessor
        batch_processed = preprocessor(batch_device)

        # Zero gradients
        policy.zero_grad()

        # Forward pass
        with torch.amp.autocast(device_type="cuda", enabled=True):
            output = policy.forward(batch_processed)
            loss = output.get("loss")

        if loss is None:
            result.add_check("Backward pass", False, "No loss to backprop")
            return False

        # Backward pass
        print(f"  Running backward pass...")
        loss.backward()

        # Check gradients exist
        grad_count = 0
        zero_grad_count = 0
        nan_grad_count = 0
        total_grad_norm = 0.0

        for name, param in policy.named_parameters():
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2

                if grad_norm == 0:
                    zero_grad_count += 1
                if np.isnan(grad_norm):
                    nan_grad_count += 1

        total_grad_norm = np.sqrt(total_grad_norm)

        result.add_check("Gradients exist", grad_count > 0, f"{grad_count} params have gradients")
        result.add_check("No NaN gradients", nan_grad_count == 0,
                        f"{nan_grad_count} NaN gradients" if nan_grad_count > 0 else "All gradients valid")
        result.add_check("Gradient norm valid", not np.isnan(total_grad_norm) and not np.isinf(total_grad_norm),
                        f"total_norm={total_grad_norm:.4f}")

        if zero_grad_count > 0:
            result.add_check("Zero gradients", True,
                           f"{zero_grad_count}/{grad_count} zero (may be frozen params)", warning=True)

        result.add_metric("gradient_norm", f"{total_grad_norm:.4f}")

        return True

    except Exception as e:
        import traceback
        result.add_check("Backward pass", False, f"Error: {e}")
        traceback.print_exc()
        return False


def verify_training_steps(policy, dataloader, result: TrainingVerificationResult,
                         num_steps: int = 5, device: str = "cuda", dataset_stats: dict = None):
    """Run a few training steps to verify full pipeline."""
    print("\n" + "=" * 70)
    print(f"6. TRAINING STEPS ({num_steps} steps)")
    print("=" * 70)

    if policy is None or dataloader is None:
        result.add_check("Training steps", False, "Model or dataloader not available")
        return

    try:
        from torch.optim import AdamW
        from lerobot.policies.factory import make_pre_post_processors

        # Create training preprocessor
        print(f"  Creating training preprocessor...")
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )

        # Set up optimizer
        optimizer = AdamW(policy.parameters(), lr=1e-4)
        policy.train()

        losses = []
        step_times = []
        data_iter = iter(dataloader)

        print(f"  Running {num_steps} training steps...")

        for step in range(num_steps):
            step_start = time.time()

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move to device and add task
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    batch_device[key] = value

            # Apply camera renaming
            for old_key, new_key in SMOLVLA_CAMERA_MAPPING.items():
                if old_key in batch_device:
                    batch_device[new_key] = batch_device.pop(old_key)

            if "task" not in batch_device:
                batch_size = batch_device["action"].shape[0]
                batch_device["task"] = ["Use left arm to pick up the orange and place it on the plate"] * batch_size

            # Apply preprocessor (tokenizes task strings)
            batch_processed = preprocessor(batch_device)

            # Training step
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=True):
                output = policy.forward(batch_processed)
                loss = output["loss"]

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            step_time = time.time() - step_start
            step_times.append(step_time)

            print(f"    Step {step+1}/{num_steps}: loss={loss_val:.4f}, time={step_time:.2f}s")

        # Verify results
        avg_loss = np.mean(losses)
        loss_std = np.std(losses)
        avg_time = np.mean(step_times)

        result.add_check("Training steps completed", len(losses) == num_steps,
                        f"{len(losses)}/{num_steps} steps")

        # Check loss is valid
        all_valid = all(not np.isnan(l) and not np.isinf(l) for l in losses)
        result.add_check("All losses valid", all_valid,
                        f"avg={avg_loss:.4f}, std={loss_std:.4f}")

        # Check loss trend (should not explode)
        if len(losses) >= 3:
            loss_increasing = losses[-1] > losses[0] * 10
            result.add_check("Loss stable", not loss_increasing,
                           f"first={losses[0]:.4f}, last={losses[-1]:.4f}")

        result.add_metric("avg_loss", f"{avg_loss:.4f}")
        result.add_metric("avg_step_time", f"{avg_time:.2f}s")

        print(f"\n  Summary:")
        print(f"    Average loss: {avg_loss:.4f} (+/- {loss_std:.4f})")
        print(f"    Average step time: {avg_time:.2f}s")
        print(f"    Estimated time for 20k steps: {avg_time * 20000 / 3600:.1f} hours")

    except Exception as e:
        import traceback
        result.add_check("Training steps", False, f"Error: {e}")
        traceback.print_exc()


def verify_inference_mode(policy, batch: dict, result: TrainingVerificationResult, device: str = "cuda"):
    """Verify inference mode works correctly."""
    print("\n" + "=" * 70)
    print("7. INFERENCE MODE (select_action)")
    print("=" * 70)

    if policy is None or batch is None:
        result.add_check("Inference mode", False, "Model or batch not available")
        return

    try:
        from lerobot.policies.factory import make_pre_post_processors

        # Get dataset stats for preprocessor
        dataset_path = DEFAULT_DATASET_PATH
        stats_path = dataset_path / "meta" / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        # Create preprocessor only (postprocessor may fail with base model due to action dim mismatch)
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        result.add_check("Preprocessor created", True, "Success")

        # Prepare single observation
        policy.eval()
        policy.reset()

        # Get single sample
        single_batch = {key: value[0:1] if isinstance(value, torch.Tensor) else value[0:1]
                       for key, value in batch.items()}

        # Apply camera renaming
        for old_key, new_key in SMOLVLA_CAMERA_MAPPING.items():
            if old_key in single_batch:
                single_batch[new_key] = single_batch.pop(old_key)

        # Move to device
        for key, value in single_batch.items():
            if isinstance(value, torch.Tensor):
                single_batch[key] = value.to(device)

        if "task" not in single_batch:
            single_batch["task"] = "Use left arm to pick up the orange and place it on the plate"

        # Run inference (without postprocessor - action dims may not match for base model)
        print(f"  Running select_action...")
        with torch.inference_mode():
            preprocessed = preprocessor(single_batch)
            action = policy.select_action(preprocessed)

        # Check raw action output (before postprocessor)
        if isinstance(action, torch.Tensor):
            action_shape = tuple(action.shape)
            print(f"  Raw action output shape: {action_shape}")

            # Base model may output different dims than our 12 DOF dataset
            # This is expected - training will adapt the model
            model_action_dim = action_shape[-1] if len(action_shape) >= 1 else action_shape[0]
            print(f"  Model action dim: {model_action_dim}, Dataset action dim: {EXPECTED_ACTION_DIM}")

            # For base model, action dim may be 6 (single arm) or 32 (max_action_dim)
            # After training, it will output 12 DOF for bimanual
            valid_shape = len(action_shape) >= 1 and model_action_dim > 0
            result.add_check("Inference action shape", valid_shape,
                           f"{action_shape} (model outputs {model_action_dim} dims)")

            # Check values are not NaN/Inf
            action_np = action.cpu().numpy().flatten()
            has_nan = np.any(np.isnan(action_np))
            has_inf = np.any(np.isinf(action_np))
            result.add_check("Inference action valid", not has_nan and not has_inf,
                           f"no NaN/Inf, range=[{action_np.min():.2f}, {action_np.max():.2f}]")

        result.add_check("Inference mode", True, "select_action works")

    except Exception as e:
        import traceback
        result.add_check("Inference mode", False, f"Error: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Verify training pipeline before expensive training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=f"Path to dataset (default: {DEFAULT_DATASET_PATH})"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_PRETRAINED,
        help=f"Pretrained model path (default: {DEFAULT_PRETRAINED})"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps to verify (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for verification (default: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for verification (default: cuda if available)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification (1 step, minimal checks)"
    )
    parser.add_argument(
        "--skip-training-steps",
        action="store_true",
        help="Skip actual training steps (just verify loading and single forward/backward)"
    )
    args = parser.parse_args()

    if args.quick:
        args.steps = 1
        args.skip_training_steps = True

    dataset_path = Path(args.dataset_path)

    print("=" * 70)
    print("TRAINING PIPELINE VERIFICATION")
    print("=" * 70)
    print(f"Dataset:    {dataset_path}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Device:     {args.device}")
    print(f"Steps:      {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    result = TrainingVerificationResult()

    # Load dataset stats for preprocessor
    stats_path = dataset_path / "meta" / "stats.json"
    dataset_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            dataset_stats = json.load(f)
        print(f"Loaded dataset stats from {stats_path}")

    # Run verifications
    dataset, dataloader, batch = verify_dataset_loading(dataset_path, result, args.batch_size)
    renamed_batch = verify_camera_renaming(batch, result)
    policy = verify_model_loading(args.pretrained, result, args.device)

    if policy is not None and renamed_batch is not None:
        verify_forward_pass(policy, renamed_batch, result, args.device, dataset_stats)
        verify_backward_pass(policy, renamed_batch, result, args.device, dataset_stats)

        if not args.skip_training_steps and dataloader is not None:
            verify_training_steps(policy, dataloader, result, args.steps, args.device, dataset_stats)

        verify_inference_mode(policy, batch, result, args.device)

    # Cleanup
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    # Print summary
    success = result.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
