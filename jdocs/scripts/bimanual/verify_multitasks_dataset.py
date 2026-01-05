#!/usr/bin/env python3
"""
Multitasks Dataset Verification Script

This script performs comprehensive runtime verification of the multitasks dataset
BEFORE expensive training runs. It checks:

1. Dataset loading and structure
2. Camera image mapping (saves samples for visual verification)
3. Task index and task string correctness
4. Action/state dimensions and statistics
5. Data loader pipeline
6. SmolVLA camera renaming compatibility

Usage:
    python verify_multitasks_dataset.py
    python verify_multitasks_dataset.py --dataset-path /path/to/dataset
    python verify_multitasks_dataset.py --save-images --num-samples 5
    python verify_multitasks_dataset.py --quick  # Fast check without images

Output:
    - Console report with pass/fail status
    - Sample camera images saved to jdocs/verification_images/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Default paths
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "verification_images"

# Expected configuration for multitasks dataset
EXPECTED_CONFIG = {
    "codebase_version": "v3.0",
    "robot_type": "bi_so101_follower",
    "fps": 30,
    "action_dim": 12,
    "state_dim": 12,
    "cameras": ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"],
    "smolvla_camera_mapping": {
        "observation.images.head": "observation.images.camera1",
        "observation.images.left_wrist": "observation.images.camera2",
        "observation.images.right_wrist": "observation.images.camera3",
    },
}

# Expected task strings (from data collection script template)
EXPECTED_TASK_PATTERNS = [
    "Use left arm to pick up",
    "Use right arm to pick up",
    "place it on the plate",
    "place it in the bin",
]


class VerificationResult:
    """Track verification results."""

    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_check(self, name: str, passed: bool, message: str, warning: bool = False):
        status = "PASS" if passed else ("WARN" if warning else "FAIL")
        self.checks.append({"name": name, "status": status, "message": message})
        if passed:
            self.passed += 1
        elif warning:
            self.warnings += 1
        else:
            self.failed += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        for check in self.checks:
            status_color = {
                "PASS": "\033[92m",  # Green
                "FAIL": "\033[91m",  # Red
                "WARN": "\033[93m",  # Yellow
            }
            reset = "\033[0m"
            color = status_color.get(check["status"], "")
            print(f"{color}[{check['status']}]{reset} {check['name']}: {check['message']}")

        print("\n" + "-" * 70)
        total = self.passed + self.failed + self.warnings
        print(f"Total: {total} checks | Passed: {self.passed} | Failed: {self.failed} | Warnings: {self.warnings}")

        if self.failed > 0:
            print("\n\033[91mVERIFICATION FAILED - DO NOT PROCEED WITH TRAINING\033[0m")
            return False
        elif self.warnings > 0:
            print("\n\033[93mVERIFICATION PASSED WITH WARNINGS - Review before training\033[0m")
            return True
        else:
            print("\n\033[92mVERIFICATION PASSED - Safe to proceed with training\033[0m")
            return True


def verify_dataset_structure(dataset_path: Path, result: VerificationResult):
    """Verify dataset directory structure and metadata files."""
    print("\n" + "=" * 70)
    print("1. DATASET STRUCTURE VERIFICATION")
    print("=" * 70)

    # Check dataset exists
    if not dataset_path.exists():
        result.add_check("Dataset exists", False, f"Path not found: {dataset_path}")
        return None
    result.add_check("Dataset exists", True, str(dataset_path))

    # Check required files
    required_files = [
        "meta/info.json",
        "meta/stats.json",
    ]

    for file_path in required_files:
        full_path = dataset_path / file_path
        exists = full_path.exists()
        result.add_check(f"File: {file_path}", exists,
                        "Found" if exists else "Missing")

    # Load and verify info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None

    with open(info_path) as f:
        info = json.load(f)

    # Verify codebase version
    version = info.get("codebase_version", "unknown")
    result.add_check("Codebase version", version == EXPECTED_CONFIG["codebase_version"],
                    f"{version} (expected: {EXPECTED_CONFIG['codebase_version']})")

    # Verify robot type
    robot_type = info.get("robot_type", "unknown")
    result.add_check("Robot type", robot_type == EXPECTED_CONFIG["robot_type"],
                    f"{robot_type} (expected: {EXPECTED_CONFIG['robot_type']})")

    # Verify FPS
    fps = info.get("fps", 0)
    result.add_check("FPS", fps == EXPECTED_CONFIG["fps"],
                    f"{fps} (expected: {EXPECTED_CONFIG['fps']})")

    # Verify episode count
    total_episodes = info.get("total_episodes", 0)
    result.add_check("Total episodes", total_episodes > 0, f"{total_episodes}")

    # Verify task count
    total_tasks = info.get("total_tasks", 0)
    result.add_check("Total tasks", total_tasks > 0, f"{total_tasks}")

    # Verify action dimension
    action_shape = info.get("features", {}).get("action", {}).get("shape", [])
    action_dim = action_shape[0] if action_shape else 0
    result.add_check("Action dimension", action_dim == EXPECTED_CONFIG["action_dim"],
                    f"{action_dim} (expected: {EXPECTED_CONFIG['action_dim']})")

    # Verify state dimension
    state_shape = info.get("features", {}).get("observation.state", {}).get("shape", [])
    state_dim = state_shape[0] if state_shape else 0
    result.add_check("State dimension", state_dim == EXPECTED_CONFIG["state_dim"],
                    f"{state_dim} (expected: {EXPECTED_CONFIG['state_dim']})")

    # Verify cameras exist
    features = info.get("features", {})
    for cam in EXPECTED_CONFIG["cameras"]:
        exists = cam in features
        result.add_check(f"Camera: {cam}", exists, "Found" if exists else "Missing")

    return info


def verify_tasks(dataset_path: Path, result: VerificationResult):
    """Verify task index and task strings."""
    print("\n" + "=" * 70)
    print("2. TASK VERIFICATION")
    print("=" * 70)

    # Try to load tasks from tasks.parquet
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    tasks_list = []

    try:
        import pyarrow.parquet as pq
        tasks_table = pq.read_table(tasks_path)
        tasks_df = tasks_table.to_pandas()
        print(f"  Tasks parquet loaded: {len(tasks_df)} tasks")
        result.add_check("Tasks parquet readable", True, f"{len(tasks_df)} tasks")

        # Get task strings
        if "__index_level_0__" in tasks_df.columns:
            tasks_list = tasks_df["__index_level_0__"].tolist()
        elif "task" in tasks_df.columns:
            tasks_list = tasks_df["task"].tolist()
        else:
            # Try index
            tasks_list = tasks_df.index.tolist()

    except Exception as e:
        result.add_check("Tasks parquet readable", False, f"Error: {e}", warning=True)
        print(f"  Warning: Cannot read tasks.parquet: {e}")
        print("  Will verify tasks from data files instead...")

    # If tasks not from parquet, try to get from data
    if not tasks_list:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(
                repo_id="multitasks",
                root=str(dataset_path),
            )
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'tasks'):
                tasks_list = list(dataset.meta.tasks.keys()) if isinstance(dataset.meta.tasks, dict) else dataset.meta.tasks
            elif hasattr(dataset, 'tasks'):
                tasks_list = dataset.tasks
        except Exception as e:
            print(f"  Warning: Cannot get tasks from LeRobotDataset: {e}")

    if tasks_list:
        print(f"\n  Found {len(tasks_list)} tasks:")
        for i, task in enumerate(tasks_list):
            print(f"    [{i}] {task}")

        # Verify task string patterns
        valid_patterns = 0
        for task in tasks_list:
            task_str = str(task)
            has_valid_pattern = any(pattern in task_str for pattern in EXPECTED_TASK_PATTERNS)
            if has_valid_pattern:
                valid_patterns += 1

        result.add_check("Task string format", valid_patterns == len(tasks_list),
                        f"{valid_patterns}/{len(tasks_list)} tasks have expected patterns")
    else:
        result.add_check("Task strings", False, "Could not retrieve task strings", warning=True)

    return tasks_list


def verify_data_loading(dataset_path: Path, result: VerificationResult, num_samples: int = 3,
                        video_backend: str = "pyav"):
    """Verify data can be loaded correctly via LeRobotDataset."""
    print("\n" + "=" * 70)
    print("3. DATA LOADING VERIFICATION")
    print("=" * 70)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print(f"  Loading dataset from {dataset_path}...")
        print(f"  Video backend: {video_backend}")
        dataset = LeRobotDataset(
            repo_id="multitasks",
            root=str(dataset_path),
            video_backend=video_backend,
        )

        result.add_check("LeRobotDataset loads", True, f"{len(dataset)} frames")

        # Verify dataset length
        print(f"  Dataset length: {len(dataset)}")

        # Sample some data points
        print(f"\n  Sampling {num_samples} data points...")

        for i in range(min(num_samples, len(dataset))):
            idx = i * (len(dataset) // num_samples)  # Spread samples across dataset
            sample = dataset[idx]

            print(f"\n  Sample {i+1} (index {idx}):")

            # Check action shape
            action = sample.get("action")
            if action is not None:
                action_shape = tuple(action.shape) if hasattr(action, 'shape') else len(action)
                print(f"    action shape: {action_shape}")
                if i == 0:
                    expected = (EXPECTED_CONFIG["action_dim"],)
                    result.add_check("Action shape", action_shape == expected,
                                    f"{action_shape} (expected: {expected})")

            # Check state shape
            state = sample.get("observation.state")
            if state is not None:
                state_shape = tuple(state.shape) if hasattr(state, 'shape') else len(state)
                print(f"    observation.state shape: {state_shape}")
                if i == 0:
                    expected = (EXPECTED_CONFIG["state_dim"],)
                    result.add_check("State shape", state_shape == expected,
                                    f"{state_shape} (expected: {expected})")

            # Check task_index
            task_index = sample.get("task_index")
            if task_index is not None:
                task_idx_val = task_index.item() if hasattr(task_index, 'item') else task_index
                print(f"    task_index: {task_idx_val}")

            # Check episode_index
            episode_index = sample.get("episode_index")
            if episode_index is not None:
                ep_idx_val = episode_index.item() if hasattr(episode_index, 'item') else episode_index
                print(f"    episode_index: {ep_idx_val}")

            # Check camera images
            for cam_key in EXPECTED_CONFIG["cameras"]:
                img = sample.get(cam_key)
                if img is not None:
                    img_shape = tuple(img.shape) if hasattr(img, 'shape') else "unknown"
                    print(f"    {cam_key} shape: {img_shape}")
                    if i == 0:
                        # Expect (C, H, W) format
                        valid_shape = len(img_shape) == 3 and img_shape[0] == 3
                        result.add_check(f"Image shape: {cam_key}", valid_shape,
                                        f"{img_shape}")

        return dataset

    except Exception as e:
        import traceback
        result.add_check("LeRobotDataset loads", False, f"Error: {e}")
        traceback.print_exc()
        return None


def verify_camera_mapping_and_save_images(dataset, result: VerificationResult,
                                          output_dir: Path, num_samples: int = 3):
    """Verify camera mapping by saving sample images for visual inspection."""
    print("\n" + "=" * 70)
    print("4. CAMERA MAPPING VERIFICATION (Saving Images)")
    print("=" * 70)

    if dataset is None:
        result.add_check("Camera images", False, "Dataset not loaded")
        return

    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"verify_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving sample images to: {session_dir}")

    saved_count = 0

    for i in range(min(num_samples, len(dataset))):
        idx = i * (len(dataset) // max(num_samples, 1))
        sample = dataset[idx]

        # Get metadata for filename
        episode_idx = sample.get("episode_index", torch.tensor([-1]))
        episode_idx = episode_idx.item() if hasattr(episode_idx, 'item') else episode_idx
        frame_idx = sample.get("frame_index", torch.tensor([-1]))
        frame_idx = frame_idx.item() if hasattr(frame_idx, 'item') else frame_idx
        task_idx = sample.get("task_index", torch.tensor([-1]))
        task_idx = task_idx.item() if hasattr(task_idx, 'item') else task_idx

        print(f"\n  Sample {i+1}: episode={episode_idx}, frame={frame_idx}, task={task_idx}")

        for cam_key in EXPECTED_CONFIG["cameras"]:
            img_tensor = sample.get(cam_key)
            if img_tensor is None:
                print(f"    {cam_key}: NOT FOUND")
                continue

            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img_tensor)

            # Convert from float [0,1] to uint8 [0,255] if needed
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                img_np = (img_np * 255).astype(np.uint8)

            # Get short camera name
            cam_short = cam_key.replace("observation.images.", "")
            smolvla_name = EXPECTED_CONFIG["smolvla_camera_mapping"].get(cam_key, "unknown")
            smolvla_short = smolvla_name.replace("observation.images.", "")

            # Save image with descriptive filename
            filename = f"ep{episode_idx:03d}_frame{frame_idx:04d}_task{task_idx}_{cam_short}_maps_to_{smolvla_short}.png"
            filepath = session_dir / filename

            img = Image.fromarray(img_np)
            img.save(filepath)
            saved_count += 1

            print(f"    {cam_key} ({img_np.shape}) -> {smolvla_name}")
            print(f"      Saved: {filename}")

    result.add_check("Camera images saved", saved_count > 0,
                    f"{saved_count} images saved to {session_dir}")

    # Create a summary file with camera mapping info
    summary_path = session_dir / "CAMERA_MAPPING_INFO.txt"
    with open(summary_path, "w") as f:
        f.write("CAMERA MAPPING FOR SMOLVLA TRAINING\n")
        f.write("=" * 50 + "\n\n")
        f.write("The training script renames cameras as follows:\n\n")
        for orig, mapped in EXPECTED_CONFIG["smolvla_camera_mapping"].items():
            f.write(f"  {orig}\n")
            f.write(f"    -> {mapped}\n\n")
        f.write("\nVISUAL VERIFICATION:\n")
        f.write("1. Open the saved images\n")
        f.write("2. Verify 'head' camera shows the overhead/front view\n")
        f.write("3. Verify 'left_wrist' shows the left arm's wrist camera view\n")
        f.write("4. Verify 'right_wrist' shows the right arm's wrist camera view\n")
        f.write("\nIf cameras are swapped, check:\n")
        f.write("  - Camera index assignments in hardware config\n")
        f.write("  - Physical camera USB connections\n")

    print(f"\n  Camera mapping info saved to: {summary_path}")
    print(f"\n  \033[93mIMPORTANT: Visually verify the saved images!\033[0m")
    print(f"  Open: {session_dir}")


def verify_statistics(dataset_path: Path, result: VerificationResult):
    """Verify dataset statistics are reasonable."""
    print("\n" + "=" * 70)
    print("5. STATISTICS VERIFICATION")
    print("=" * 70)

    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        result.add_check("Stats file exists", False, "stats.json not found")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Check action statistics
    action_stats = stats.get("action", {})
    if action_stats:
        action_mean = action_stats.get("mean", [])
        action_min = action_stats.get("min", [])
        action_max = action_stats.get("max", [])

        print(f"  Action statistics:")
        print(f"    mean: [{', '.join([f'{v:.1f}' for v in action_mean[:6]])}] (left arm)")
        print(f"          [{', '.join([f'{v:.1f}' for v in action_mean[6:]])}] (right arm)")
        print(f"    min:  [{', '.join([f'{v:.1f}' for v in action_min[:6]])}] (left arm)")
        print(f"          [{', '.join([f'{v:.1f}' for v in action_min[6:]])}] (right arm)")
        print(f"    max:  [{', '.join([f'{v:.1f}' for v in action_max[:6]])}] (left arm)")
        print(f"          [{', '.join([f'{v:.1f}' for v in action_max[6:]])}] (right arm)")

        # Check for reasonable ranges (joint angles typically -180 to 180)
        all_in_range = all(-200 <= v <= 200 for v in action_min + action_max)
        result.add_check("Action range reasonable", all_in_range,
                        f"min={min(action_min):.1f}, max={max(action_max):.1f}")
    else:
        result.add_check("Action statistics", False, "Not found in stats.json")

    # Check state statistics
    state_stats = stats.get("observation.state", {})
    if state_stats:
        state_mean = state_stats.get("mean", [])
        print(f"\n  State statistics:")
        print(f"    mean: [{', '.join([f'{v:.1f}' for v in state_mean[:6]])}] (left arm)")
        print(f"          [{', '.join([f'{v:.1f}' for v in state_mean[6:]])}] (right arm)")
        result.add_check("State statistics", True, f"Found with dim={len(state_mean)}")
    else:
        result.add_check("State statistics", False, "Not found in stats.json")


def verify_dataloader(dataset, result: VerificationResult, batch_size: int = 4):
    """Verify PyTorch DataLoader works correctly."""
    print("\n" + "=" * 70)
    print("6. DATALOADER VERIFICATION")
    print("=" * 70)

    if dataset is None:
        result.add_check("DataLoader", False, "Dataset not loaded")
        return

    try:
        from torch.utils.data import DataLoader

        print(f"  Creating DataLoader with batch_size={batch_size}...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        # Get one batch
        print("  Fetching first batch...")
        batch = next(iter(dataloader))

        print(f"  Batch keys: {list(batch.keys())}")

        # Check batch dimensions
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: {value.shape}")

        # Verify action batch shape
        action = batch.get("action")
        if action is not None:
            expected_shape = (batch_size, EXPECTED_CONFIG["action_dim"])
            actual_shape = tuple(action.shape)
            result.add_check("DataLoader batch action", actual_shape == expected_shape,
                            f"{actual_shape} (expected: {expected_shape})")

        # Verify we can iterate a few batches
        batch_count = 0
        for i, batch in enumerate(dataloader):
            batch_count += 1
            if batch_count >= 3:
                break

        result.add_check("DataLoader iteration", batch_count >= 3,
                        f"Successfully iterated {batch_count} batches")

    except Exception as e:
        import traceback
        result.add_check("DataLoader", False, f"Error: {e}")
        traceback.print_exc()


def verify_smolvla_compatibility(dataset_path: Path, result: VerificationResult):
    """Verify dataset is compatible with SmolVLA training script."""
    print("\n" + "=" * 70)
    print("7. SMOLVLA TRAINING COMPATIBILITY")
    print("=" * 70)

    # Check training script exists
    train_script = PROJECT_ROOT / "jdocs" / "scripts" / "bimanual" / "train_smolvla_bimanual.sh"
    result.add_check("Training script exists", train_script.exists(), str(train_script))

    if train_script.exists():
        # Check it references the correct dataset
        with open(train_script) as f:
            content = f.read()

        uses_multitasks = "multitasks" in content
        result.add_check("Training script uses multitasks", uses_multitasks,
                        "Default dataset is multitasks" if uses_multitasks else "WARNING: Not using multitasks")

    # Check camera rename map is correct
    expected_rename = '{"observation.images.head":"observation.images.camera1"'
    if train_script.exists():
        with open(train_script) as f:
            content = f.read()
        has_rename = "rename_map" in content and "camera1" in content
        result.add_check("Camera rename map", has_rename,
                        "Found rename_map in training script")

    # Verify info.json has correct camera keys
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        features = info.get("features", {})
        all_cameras_present = all(cam in features for cam in EXPECTED_CONFIG["cameras"])
        result.add_check("All cameras in dataset", all_cameras_present,
                        f"{len(EXPECTED_CONFIG['cameras'])} cameras expected")


def main():
    parser = argparse.ArgumentParser(
        description="Verify multitasks dataset before training",
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
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory for verification images (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to check/save (default: 3)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save sample camera images for visual verification (default: True)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip saving images (faster verification)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification (skip image saving and some checks)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader test (default: 4)"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Video backend for decoding (default: pyav)"
    )
    args = parser.parse_args()

    if args.no_images or args.quick:
        args.save_images = False

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("MULTITASKS DATASET VERIFICATION")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Images:  {'Yes' if args.save_images else 'No'}")
    print("=" * 70)

    result = VerificationResult()

    # Run verifications
    info = verify_dataset_structure(dataset_path, result)
    tasks = verify_tasks(dataset_path, result)
    dataset = verify_data_loading(dataset_path, result, num_samples=args.num_samples,
                                  video_backend=args.video_backend)

    if args.save_images:
        verify_camera_mapping_and_save_images(dataset, result, output_dir, num_samples=args.num_samples)

    verify_statistics(dataset_path, result)

    if not args.quick:
        verify_dataloader(dataset, result, batch_size=args.batch_size)

    verify_smolvla_compatibility(dataset_path, result)

    # Print summary
    success = result.print_summary()

    if args.save_images:
        print(f"\n\033[93mREMINDER: Visually verify images in {output_dir}\033[0m")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
