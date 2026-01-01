#!/usr/bin/env python3
"""
Inference Pipeline Diagnostic Script.

This script traces values through the ENTIRE inference pipeline to identify
where state/action mismatches occur:

1. Robot state (raw degrees from hardware)
2. Preprocessor normalization (what the policy sees)
3. Policy output (raw action from model)
4. Postprocessor denormalization (what gets sent to robot)

Compares each step with training dataset statistics to identify issues.

Supports both SmolVLA (MEAN_STD normalization) and xVLA (IDENTITY normalization).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import yaml

# Dataset path
DATASET_PATH = PROJECT_ROOT / "datasets_bimanuel" / "bimanual" / "combined_pick_and_place"
HARDWARE_CONFIG = PROJECT_ROOT / "jdocs/configs/hardware/xlerobot_bimanual.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/smolvla_bimanual/checkpoints/020000/pretrained_model"

# Motor names for bimanual
MOTOR_NAMES_LEFT = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                    "left_wrist_flex", "left_wrist_roll", "left_gripper"]
MOTOR_NAMES_RIGHT = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                     "right_wrist_flex", "right_wrist_roll", "right_gripper"]
MOTOR_NAMES = MOTOR_NAMES_LEFT + MOTOR_NAMES_RIGHT
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_dataset_stats() -> dict:
    """Load dataset statistics."""
    stats_path = DATASET_PATH / "meta" / "stats.json"
    with open(stats_path) as f:
        return json.load(f)


def load_hardware_config() -> dict:
    """Load hardware configuration."""
    with open(HARDWARE_CONFIG) as f:
        return yaml.safe_load(f)


def get_robot_state(hw_config: dict) -> np.ndarray:
    """Read current robot state from both arms."""
    from lerobot.robots.bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
    from lerobot.robots.bi_so101_follower.bi_so101_follower import BiSO101Follower

    robot_cfg = hw_config.get("robot", {})
    left_cfg = robot_cfg.get("left_arm", {})
    right_cfg = robot_cfg.get("right_arm", {})

    config = BiSO101FollowerConfig(
        id=robot_cfg.get("id", "xlerobot_bimanual"),
        left_arm_port=left_cfg.get("port"),
        right_arm_port=right_cfg.get("port"),
        left_arm_id=left_cfg.get("id"),
        right_arm_id=right_cfg.get("id"),
        left_arm_use_degrees=left_cfg.get("use_degrees", True),
        right_arm_use_degrees=right_cfg.get("use_degrees", True),
    )

    robot = BiSO101Follower(config)
    robot.connect(calibrate=False)

    obs = robot.get_observation()
    state = np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float32)

    robot.disconnect()
    return state


def apply_normalization(state: np.ndarray, stats: dict, mode: str = "min_max") -> np.ndarray:
    """Apply normalization to state."""
    state_stats = stats.get("observation.state", {})

    if mode == "min_max":
        min_val = np.array(state_stats["min"])
        max_val = np.array(state_stats["max"])
        denom = max_val - min_val
        denom = np.where(denom == 0, 1e-8, denom)
        return 2 * (state - min_val) / denom - 1

    elif mode == "mean_std":
        mean = np.array(state_stats["mean"])
        std = np.array(state_stats["std"])
        std = np.where(std == 0, 1e-8, std)
        return (state - mean) / std

    else:  # identity
        return state


def apply_denormalization(action: np.ndarray, stats: dict, mode: str = "min_max") -> np.ndarray:
    """Apply denormalization to action."""
    action_stats = stats.get("action", {})

    if mode == "min_max":
        min_val = np.array(action_stats["min"])
        max_val = np.array(action_stats["max"])
        denom = max_val - min_val
        return (action + 1) / 2 * denom + min_val

    elif mode == "mean_std":
        mean = np.array(action_stats["mean"])
        std = np.array(action_stats["std"])
        return action * std + mean

    else:  # identity
        return action


def load_preprocessor_config(checkpoint_path: Path) -> dict:
    """Load preprocessor config from checkpoint."""
    config_path = checkpoint_path / "preprocessor_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def load_postprocessor_config(checkpoint_path: Path) -> dict:
    """Load postprocessor config from checkpoint."""
    config_path = checkpoint_path / "postprocessor_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def print_comparison_table(title: str, left_values: list, right_values: list,
                           ref_left: list = None, ref_right: list = None):
    """Print a comparison table for left/right arms."""
    print(f"\n{title}")
    print("-" * 90)

    header = f"{'Joint':<15} {'Left':>10} {'Right':>10}"
    if ref_left is not None:
        header += f" {'RefL':>10} {'RefR':>10} {'DiffL':>8} {'DiffR':>8}"
    print(header)
    print("-" * 90)

    for i, name in enumerate(JOINT_NAMES):
        left = left_values[i]
        right = right_values[i]
        row = f"{name:<15} {left:>10.2f} {right:>10.2f}"

        if ref_left is not None:
            ref_l = ref_left[i]
            ref_r = ref_right[i]
            diff_l = left - ref_l
            diff_r = right - ref_r
            row += f" {ref_l:>10.2f} {ref_r:>10.2f} {diff_l:>+8.1f} {diff_r:>+8.1f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Diagnose inference pipeline")
    parser.add_argument("--checkpoint", "-c", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to checkpoint")
    parser.add_argument("--no-robot", action="store_true",
                        help="Skip robot reading, use dataset first frame instead")
    parser.add_argument("--simulate-action", action="store_true",
                        help="Simulate what action output would look like")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    print("=" * 90)
    print("INFERENCE PIPELINE DIAGNOSTIC")
    print("=" * 90)

    # Load configs
    print("\n[1] Loading configurations...")
    hw_config = load_hardware_config()
    stats = load_dataset_stats()

    print(f"  Hardware config: {HARDWARE_CONFIG}")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Load processor configs
    pre_config = load_preprocessor_config(checkpoint_path)
    post_config = load_postprocessor_config(checkpoint_path)

    print(f"\n  Preprocessor config exists: {bool(pre_config)}")
    print(f"  Postprocessor config exists: {bool(post_config)}")

    # Check normalization mode from config
    # SmolVLA default: MEAN_STD, xVLA default: IDENTITY
    norm_mode = "mean_std"  # default for SmolVLA
    if pre_config:
        steps = pre_config.get("steps", [])
        for step in steps:
            if step.get("name") == "normalizer_processor":
                norm_map = step.get("config", {}).get("norm_map", {})
                state_norm = norm_map.get("STATE", norm_map.get("state", "MEAN_STD"))
                action_norm = norm_map.get("ACTION", norm_map.get("action", "MEAN_STD"))
                print(f"\n  State normalization mode: {state_norm}")
                print(f"  Action normalization mode: {action_norm}")

                if state_norm.upper() == "MIN_MAX":
                    norm_mode = "min_max"
                elif state_norm.upper() == "MEAN_STD":
                    norm_mode = "mean_std"
                elif state_norm.upper() == "IDENTITY":
                    norm_mode = "identity"
    else:
        # No config found, assume SmolVLA default
        print(f"\n  No preprocessor config found, assuming SmolVLA defaults:")
        print(f"  State normalization mode: MEAN_STD")
        print(f"  Action normalization mode: MEAN_STD")

    # Dataset statistics
    print("\n" + "=" * 90)
    print("[2] DATASET STATISTICS (observation.state)")
    print("=" * 90)

    state_stats = stats.get("observation.state", {})
    print_comparison_table(
        "State Stats - Mean (expected typical values)",
        state_stats["mean"][:6], state_stats["mean"][6:12]
    )
    print_comparison_table(
        "State Stats - Min",
        state_stats["min"][:6], state_stats["min"][6:12]
    )
    print_comparison_table(
        "State Stats - Max",
        state_stats["max"][:6], state_stats["max"][6:12]
    )

    # Get robot state
    print("\n" + "=" * 90)
    print("[3] CURRENT ROBOT STATE (degrees)")
    print("=" * 90)

    if args.no_robot:
        # Use dataset mean as simulated state
        state = np.array(state_stats["mean"], dtype=np.float32)
        print("  (Using dataset mean as simulated state)")
    else:
        print("  Reading from robot...")
        state = get_robot_state(hw_config)

    left_state = state[:6]
    right_state = state[6:12]

    print_comparison_table(
        "Current Robot State vs Dataset Mean",
        left_state.tolist(), right_state.tolist(),
        state_stats["mean"][:6], state_stats["mean"][6:12]
    )

    # Check if state is within training range
    print("\n  Range Check (is current state within training data range?):")
    state_min = np.array(state_stats["min"])
    state_max = np.array(state_stats["max"])
    for i, name in enumerate(MOTOR_NAMES):
        val = state[i]
        min_v = state_min[i]
        max_v = state_max[i]
        in_range = min_v <= val <= max_v
        status = "OK" if in_range else "OUT OF RANGE!"
        if not in_range:
            print(f"    {name:<25}: {val:>8.1f}° range=[{min_v:.1f}, {max_v:.1f}] {status}")

    # Apply normalization
    print("\n" + "=" * 90)
    print(f"[4] NORMALIZED STATE (mode: {norm_mode})")
    print("=" * 90)

    normalized_state = apply_normalization(state, stats, norm_mode)
    left_norm = normalized_state[:6]
    right_norm = normalized_state[6:12]

    print_comparison_table(
        "Normalized State (what policy sees)",
        left_norm.tolist(), right_norm.tolist()
    )

    if norm_mode == "min_max":
        print("\n  Expected range: [-1, 1]")
        out_of_range = (normalized_state < -1.5) | (normalized_state > 1.5)
        if out_of_range.any():
            print("  WARNING: Some normalized values are outside [-1.5, 1.5]!")
            for i, name in enumerate(MOTOR_NAMES):
                if out_of_range[i]:
                    print(f"    {name}: {normalized_state[i]:.3f}")

    # Simulate action output
    if args.simulate_action:
        print("\n" + "=" * 90)
        print("[5] SIMULATED ACTION DENORMALIZATION")
        print("=" * 90)

        # Simulate policy outputting zeros (should map to mean action)
        simulated_policy_output = np.zeros(12, dtype=np.float32)
        denorm_action = apply_denormalization(simulated_policy_output, stats, norm_mode)

        action_stats = stats.get("action", {})
        print_comparison_table(
            "If policy outputs zeros → denormalized action",
            denorm_action[:6].tolist(), denorm_action[6:12].tolist(),
            action_stats["mean"][:6], action_stats["mean"][6:12]
        )

        # Simulate policy outputting current normalized state (should stay in place)
        print("\n  If policy outputs current normalized state as action:")
        denorm_action2 = apply_denormalization(normalized_state, stats, norm_mode)
        print_comparison_table(
            "Denormalized action (should ≈ current state)",
            denorm_action2[:6].tolist(), denorm_action2[6:12].tolist(),
            left_state.tolist(), right_state.tolist()
        )

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Calculate key diagnostics
    left_diff = np.abs(left_state - np.array(state_stats["mean"][:6]))
    right_diff = np.abs(right_state - np.array(state_stats["mean"][6:12]))

    print(f"\n  Left arm avg deviation from training mean:  {np.mean(left_diff):.1f}°")
    print(f"  Right arm avg deviation from training mean: {np.mean(right_diff):.1f}°")

    if np.mean(left_diff) > 30:
        print("\n  WARNING: Left arm position differs significantly from training data!")
        print("  This could cause the policy to output unexpected actions.")

    if np.mean(right_diff) > 30:
        print("\n  WARNING: Right arm position differs significantly from training data!")
        print("  This could cause the policy to output unexpected actions.")

    # Check for arm asymmetry
    arm_diff = np.abs(left_state - right_state)
    print(f"\n  Left-Right arm difference (current): {np.mean(arm_diff):.1f}° avg")

    train_left_mean = np.array(state_stats["mean"][:6])
    train_right_mean = np.array(state_stats["mean"][6:12])
    train_arm_diff = np.abs(train_left_mean - train_right_mean)
    print(f"  Left-Right arm difference (training): {np.mean(train_arm_diff):.1f}° avg")

    if np.mean(arm_diff) > np.mean(train_arm_diff) + 20:
        print("\n  WARNING: Arms are more asymmetric than during training!")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
