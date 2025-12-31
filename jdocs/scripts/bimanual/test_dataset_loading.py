#!/home/jrobot/anaconda3/envs/lerobot/bin/python
"""
Runtime test for bimanual dataset loading.

This script actually loads the dataset with LeRobotDataset (same as training)
and verifies the data format at runtime.

Usage:
    python test_dataset_loading.py
    python test_dataset_loading.py --dataset path/to/dataset
    python test_dataset_loading.py --dataset path/to/combined_dataset --merged
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DATASET = PROJECT_ROOT / "datasets_bimanuel" / "bimanual" / "combined_pick_and_place"
SNAPSHOT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "verification_snapshots"


def load_task_strings(dataset_path: Path) -> dict:
    """Load task index to string mapping from tasks.parquet.

    The correct LeRobotDataset format has:
    - Task string as DataFrame INDEX
    - task_index as column

    Wrong format (causes Task cannot be None error):
    - task_index as column, task string as another column
    """
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        df = pd.read_parquet(tasks_path)
        # Correct format: index is task string, column is task_index
        # df.index = ['Left arm pick up...', 'Right arm pick up...']
        # df['task_index'] = [0, 1]
        return {int(row['task_index']): idx for idx, row in df.iterrows()}
    return {}


def verify_tasks_parquet_format(dataset_path: Path) -> tuple[bool, str]:
    """Verify tasks.parquet has correct format for LeRobotDataset.

    CRITICAL: LeRobotDataset expects task string as DataFrame index.
    If task string is stored as a column, training will fail with:
        ValueError: Task cannot be None

    Returns (passed, error_message).
    """
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return False, "tasks.parquet not found"

    df = pd.read_parquet(tasks_path)

    # Check 1: Index should be task strings (not integers)
    if len(df.index) == 0:
        return False, "tasks.parquet is empty"

    first_index = df.index[0]
    if not isinstance(first_index, str):
        return False, f"Index should be task STRING, got {type(first_index).__name__}: {first_index}"

    # Check 2: Should have 'task_index' column
    if 'task_index' not in df.columns:
        return False, "Missing 'task_index' column"

    # Check 3: Should NOT have 'task' column (that means wrong format)
    if 'task' in df.columns:
        return False, "Found 'task' column - task string should be INDEX, not column"

    # Check 4: Verify task strings are meaningful (not empty or numeric)
    for task_str in df.index:
        if not task_str or len(task_str) < 5:
            return False, f"Task string too short or empty: '{task_str}'"
        if task_str.isdigit():
            return False, f"Task string should not be numeric: '{task_str}'"

    return True, f"Format correct: {len(df)} tasks with strings as index"


def verify_stats_completeness(dataset_path: Path) -> tuple[bool, list[str]]:
    """Verify stats.json has all required keys for training.

    Training requires stats (mean/std) for normalization of all features.
    Returns (passed, list of missing keys).
    """
    import json

    stats_path = dataset_path / "meta" / "stats.json"
    info_path = dataset_path / "meta" / "info.json"

    if not stats_path.exists():
        return False, ["stats.json not found"]
    if not info_path.exists():
        return False, ["info.json not found"]

    with open(stats_path) as f:
        stats = json.load(f)
    with open(info_path) as f:
        info = json.load(f)

    missing = []
    features = info.get("features", {})

    for feature_name, feature_info in features.items():
        # Training needs stats for action, state, and image features
        if feature_name in ["action", "observation.state"] or feature_name.startswith("observation.images."):
            if feature_name not in stats:
                missing.append(feature_name)
            elif "mean" not in stats[feature_name] or "std" not in stats[feature_name]:
                missing.append(f"{feature_name} (missing mean/std)")

    return len(missing) == 0, missing


def save_camera_snapshot(sample: dict, camera_key: str, output_path: Path) -> None:
    """Save a camera image from a sample to disk."""
    img_tensor = sample[camera_key]  # (C, H, W) in [0, 1]
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test bimanual dataset loading")
    parser.add_argument("--dataset", "-d", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-samples", "-n", type=int, default=3)
    parser.add_argument("--merged", "-m", action="store_true",
                        help="Run additional tests for merged dataset (left+right arms)")
    parser.add_argument("--save-snapshots", "-s", action="store_true",
                        help="Save camera snapshots for visual verification")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_OUTPUT_DIR,
                        help="Directory to save snapshots")
    args = parser.parse_args()

    # Auto-detect merged dataset
    if "combined" in args.dataset.name:
        args.merged = True

    print("=" * 70)
    print("RUNTIME DATASET LOADING TEST")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Merged dataset: {args.merged}")
    print()

    # Load task strings
    task_strings = load_task_strings(args.dataset)
    print(f"Tasks found: {len(task_strings)}")
    for idx, task_str in task_strings.items():
        print(f"  [{idx}] {task_str}")
    print()

    # Verify tasks.parquet format (CRITICAL - wrong format causes "Task cannot be None")
    print("Verifying tasks.parquet format...")
    tasks_format_ok, tasks_format_msg = verify_tasks_parquet_format(args.dataset)
    if tasks_format_ok:
        print(f"✓ {tasks_format_msg}")
    else:
        print(f"✗ {tasks_format_msg}")
        print("\nCRITICAL: Training will fail with 'Task cannot be None' error!")
    print()

    # Verify stats completeness (required for training normalization)
    print("Verifying stats.json completeness...")
    stats_ok, missing_stats = verify_stats_completeness(args.dataset)
    if stats_ok:
        print("✓ All required stats present (action, state, images)")
    else:
        print("✗ Missing stats for training:")
        for m in missing_stats:
            print(f"    - {m}")
        print("\nWARNING: Training will fail without these stats!")
    print()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset with pyav backend (same as training)...")
    dataset = LeRobotDataset(
        repo_id=args.dataset.name,
        root=str(args.dataset),
        video_backend='pyav',
    )
    print(f"✓ Loaded {len(dataset)} frames")
    print()

    # Camera rename map (what training applies)
    rename_map = {
        'observation.images.head': 'observation.images.camera1',
        'observation.images.left_wrist': 'observation.images.camera2',
        'observation.images.right_wrist': 'observation.images.camera3',
    }

    print("=" * 70)
    print(f"SAMPLE DATA (first {args.num_samples} frames)")
    print("=" * 70)

    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Frame {i} ---")

        # Check action
        action = sample['action']
        print(f"action: shape={action.shape}, dtype={action.dtype}")
        print(f"  left arm  [0:6]:  [{', '.join([f'{v:.1f}' for v in action[:6]])}]")
        print(f"  right arm [6:12]: [{', '.join([f'{v:.1f}' for v in action[6:]])}]")

        # Check state
        state = sample['observation.state']
        print(f"state: shape={state.shape}, dtype={state.dtype}")
        print(f"  left arm  [0:6]:  [{', '.join([f'{v:.1f}' for v in state[:6]])}]")
        print(f"  right arm [6:12]: [{', '.join([f'{v:.1f}' for v in state[6:]])}]")

        # Check images
        for orig_key, new_key in rename_map.items():
            if orig_key in sample:
                img = sample[orig_key]
                new_name = new_key.replace('observation.images.', '')
                orig_name = orig_key.replace('observation.images.', '')
                print(f"{orig_name} -> {new_name}: shape={img.shape}, range=[{img.min():.2f}, {img.max():.2f}]")
            else:
                print(f"✗ MISSING: {orig_key}")

        # Check task
        task_idx = int(sample['task_index']) if 'task_index' in sample else int(sample['task'])
        task_str = task_strings.get(task_idx, f"Unknown task {task_idx}")
        print(f"task_index: {task_idx} -> \"{task_str}\"")
        print(f"episode_index: {sample['episode_index']}")

    # If merged dataset, also sample from second half
    if args.merged:
        print()
        print("=" * 70)
        print("MERGED DATASET: SAMPLING FROM RIGHT ARM EPISODES")
        print("=" * 70)

        # Find frame from right arm (episode 50+)
        mid_point = len(dataset) // 2
        for i in range(mid_point, len(dataset)):
            sample = dataset[i]
            ep_idx = int(sample['episode_index'])
            if ep_idx >= 50:  # Right arm episodes start at 50
                print(f"\n--- Frame {i} (episode {ep_idx}) ---")

                action = sample['action']
                print(f"action: shape={action.shape}")
                print(f"  left arm  [0:6]:  [{', '.join([f'{v:.1f}' for v in action[:6]])}]")
                print(f"  right arm [6:12]: [{', '.join([f'{v:.1f}' for v in action[6:]])}]")

                task_idx = int(sample['task_index']) if 'task_index' in sample else int(sample['task'])
                task_str = task_strings.get(task_idx, f"Unknown task {task_idx}")
                print(f"task_index: {task_idx} -> \"{task_str}\"")

                # Check images load correctly
                for orig_key in rename_map.keys():
                    if orig_key in sample:
                        img = sample[orig_key]
                        print(f"✓ {orig_key}: shape={img.shape}")
                break

    # Save camera snapshots if requested
    if args.save_snapshots:
        print()
        print("=" * 70)
        print("SAVING CAMERA SNAPSHOTS")
        print("=" * 70)

        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.snapshot_dir}")

        # Save from left arm episode (frame 0)
        left_sample = dataset[0]
        left_ep = int(left_sample['episode_index'])
        print(f"\nLeft arm episode {left_ep}:")
        for cam_name in ['head', 'left_wrist', 'right_wrist']:
            key = f'observation.images.{cam_name}'
            if key in left_sample:
                save_camera_snapshot(
                    left_sample, key,
                    args.snapshot_dir / f'left_arm_ep{left_ep}_{cam_name}.png'
                )

        # Save from right arm episode if merged
        if args.merged:
            # Find first right arm episode
            for i in range(len(dataset) // 2, len(dataset)):
                right_sample = dataset[i]
                right_ep = int(right_sample['episode_index'])
                if right_ep >= 50:
                    print(f"\nRight arm episode {right_ep}:")
                    for cam_name in ['head', 'left_wrist', 'right_wrist']:
                        key = f'observation.images.{cam_name}'
                        if key in right_sample:
                            save_camera_snapshot(
                                right_sample, key,
                                args.snapshot_dir / f'right_arm_ep{right_ep}_{cam_name}.png'
                            )
                    break

        print(f"\n✓ Snapshots saved to: {args.snapshot_dir}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sample = dataset[0]
    checks_passed = True

    # Verify action dimension
    if sample['action'].shape[0] == 12:
        print("✓ Action dimension: 12 (correct for bimanual)")
    else:
        print(f"✗ Action dimension: {sample['action'].shape[0]} (expected 12)")
        checks_passed = False

    # Verify state dimension
    if sample['observation.state'].shape[0] == 12:
        print("✓ State dimension: 12 (correct for bimanual)")
    else:
        print(f"✗ State dimension: {sample['observation.state'].shape[0]} (expected 12)")
        checks_passed = False

    # Verify cameras
    for orig_key, new_key in rename_map.items():
        if orig_key in sample:
            img = sample[orig_key]
            if img.shape == (3, 480, 640):
                print(f"✓ {orig_key}: (3, 480, 640)")
            else:
                print(f"✗ {orig_key}: {img.shape} (expected (3, 480, 640))")
                checks_passed = False
        else:
            print(f"✗ {orig_key}: MISSING")
            checks_passed = False

    # Verify task field returns STRING (CRITICAL for SmolVLA/xVLA tokenizer)
    task_value = sample.get('task')
    if isinstance(task_value, str) and len(task_value) > 5:
        print(f"✓ Task field is STRING: \"{task_value[:50]}...\"")
    else:
        print(f"✗ Task field should be STRING, got {type(task_value).__name__}: {task_value}")
        print("  CRITICAL: SmolVLA/xVLA tokenizer will fail!")
        checks_passed = False

    # Verify task_index (check task_index is valid)
    task_idx = int(sample['task_index']) if 'task_index' in sample else -1
    if task_idx in task_strings:
        print(f"✓ Task index {task_idx} maps correctly")
    else:
        print(f"✗ Task index {task_idx}: not found in task mapping")
        checks_passed = False

    # Verify tasks.parquet format (critical for training)
    if tasks_format_ok:
        print("✓ Tasks.parquet format correct (task string as index)")
    else:
        print(f"✗ Tasks.parquet format wrong: {tasks_format_msg}")
        checks_passed = False

    # Verify stats completeness (critical for training)
    if stats_ok:
        print("✓ Stats complete for training (action, state, images)")
    else:
        print(f"✗ Stats incomplete - missing: {', '.join(missing_stats)}")
        checks_passed = False

    # For merged datasets, verify both tasks are present
    if args.merged:
        print()
        print("--- Merged Dataset Verification ---")

        # Verify we have 2 tasks
        if len(task_strings) == 2:
            print(f"✓ Two tasks found (left arm + right arm)")
        else:
            print(f"✗ Expected 2 tasks, found {len(task_strings)}")
            checks_passed = False

        # Verify left arm task (should have "Left" in it)
        left_task = task_strings.get(0, "")
        if "Left" in left_task or "left" in left_task:
            print(f"✓ Task 0 is left arm: \"{left_task[:50]}...\"")
        else:
            print(f"✗ Task 0 doesn't appear to be left arm: \"{left_task[:50]}\"")
            checks_passed = False

        # Verify right arm task
        right_task = task_strings.get(1, "")
        if "Right" in right_task or "right" in right_task:
            print(f"✓ Task 1 is right arm: \"{right_task[:50]}...\"")
        else:
            print(f"✗ Task 1 doesn't appear to be right arm: \"{right_task[:50]}\"")
            checks_passed = False

        # Sample from both halves to verify video loading works
        print()
        print("--- Video Loading Verification ---")
        try:
            left_sample = dataset[0]  # Left arm
            right_sample = dataset[len(dataset) - 1]  # Right arm

            # Check images are different (not just black/zeros)
            left_head = left_sample['observation.images.head']
            right_head = right_sample['observation.images.head']

            left_mean = float(left_head.mean())
            right_mean = float(right_head.mean())

            if left_mean > 0.01 and right_mean > 0.01:
                print(f"✓ Left arm video loads (mean={left_mean:.3f})")
                print(f"✓ Right arm video loads (mean={right_mean:.3f})")
            else:
                print(f"✗ Video data appears empty (left mean={left_mean:.3f}, right mean={right_mean:.3f})")
                checks_passed = False

            # Verify task indices are different
            left_task_idx = int(left_sample['task_index']) if 'task_index' in left_sample else int(left_sample['task'])
            right_task_idx = int(right_sample['task_index']) if 'task_index' in right_sample else int(right_sample['task'])

            if left_task_idx == 0 and right_task_idx == 1:
                print(f"✓ Task indices correct: left={left_task_idx}, right={right_task_idx}")
            else:
                print(f"✗ Task indices unexpected: left={left_task_idx}, right={right_task_idx}")
                checks_passed = False

        except Exception as e:
            print(f"✗ Error loading samples: {e}")
            checks_passed = False

    print()
    if checks_passed:
        print("=" * 70)
        print("✓ ALL CHECKS PASSED - Dataset ready for training!")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("✗ SOME CHECKS FAILED - Review issues above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
