#!/usr/bin/env python3
"""
Upload Local LeRobot Dataset to HuggingFace Hub

This script uploads a locally collected LeRobot dataset to HuggingFace Hub,
making it available for training, sharing, and collaboration.

Prerequisites:
    1. HuggingFace account: https://huggingface.co/join
    2. HuggingFace CLI login:
       $ huggingface-cli login
       # Or set HF_TOKEN environment variable
    3. Dataset in LeRobot v3.0 format (see Dataset Structure below)

Dataset Structure (LeRobot v3.0):
    your_dataset/
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet    # Action, state, timestamps
    ├── meta/
    │   ├── info.json               # Dataset metadata
    │   ├── episodes.jsonl          # Episode information
    │   ├── stats.json              # Normalization statistics
    │   └── tasks.jsonl             # Task descriptions
    └── videos/
        ├── observation.images.head/
        │   └── chunk-000/
        │       └── file-000.mp4
        └── observation.images.*/
            └── ...

Usage:
    # Basic upload (public dataset)
    python upload_dataset_to_hub.py \\
        --local-path ./datasets/my_dataset \\
        --repo-id your-username/my-dataset

    # Private dataset
    python upload_dataset_to_hub.py \\
        --local-path ./datasets/my_dataset \\
        --repo-id your-username/my-dataset \\
        --private

    # Upload without videos (faster, smaller)
    python upload_dataset_to_hub.py \\
        --local-path ./datasets/my_dataset \\
        --repo-id your-username/my-dataset \\
        --no-videos

    # Dry run (validate only, no upload)
    python upload_dataset_to_hub.py \\
        --local-path ./datasets/my_dataset \\
        --repo-id your-username/my-dataset \\
        --dry-run

    # Upload to specific branch
    python upload_dataset_to_hub.py \\
        --local-path ./datasets/my_dataset \\
        --repo-id your-username/my-dataset \\
        --branch v1.0

Examples:
    # Upload bimanual multitasks dataset
    python upload_dataset_to_hub.py \\
        --local-path ./datasets_bimanuel/multitasks \\
        --repo-id jrobot/bimanual-multitasks \\
        --license apache-2.0 \\
        --tags bimanual,so101,manipulation

Author: Auto-generated for LeRobot bimanual project
Date: 2026-01-05
"""

import argparse
import json
import sys
from pathlib import Path

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def validate_dataset(dataset_path: Path) -> dict:
    """
    Validate that the dataset has the correct LeRobot v3.0 structure.

    Returns:
        dict: Dataset info if valid

    Raises:
        ValueError: If dataset structure is invalid
    """
    errors = []
    warnings = []

    # Check required directories
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).is_dir():
            errors.append(f"Missing required directory: {dir_name}/")

    # Check required meta files
    required_meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl"]
    for file_name in required_meta_files:
        if not (dataset_path / "meta" / file_name).exists():
            errors.append(f"Missing required file: meta/{file_name}")

    # Validate info.json
    info_path = dataset_path / "meta" / "info.json"
    info = {}
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)

            # Check required fields
            required_fields = ["codebase_version", "total_episodes", "total_frames", "fps"]
            for field in required_fields:
                if field not in info:
                    errors.append(f"Missing required field in info.json: {field}")

            # Check codebase version
            version = info.get("codebase_version", "")
            if not version.startswith("v3"):
                warnings.append(f"Dataset version {version} may not be compatible (expected v3.x)")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in info.json: {e}")

    # Check for data files
    data_dir = dataset_path / "data"
    if data_dir.is_dir():
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            errors.append("No .parquet files found in data/")
        else:
            print(f"  Found {len(parquet_files)} parquet file(s)")

    # Check for video files
    videos_dir = dataset_path / "videos"
    if videos_dir.is_dir():
        video_files = list(videos_dir.rglob("*.mp4"))
        if video_files:
            print(f"  Found {len(video_files)} video file(s)")
        else:
            warnings.append("No video files found in videos/")
    else:
        warnings.append("No videos/ directory found")

    # Report results
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError(f"Dataset validation failed with {len(errors)} error(s)")

    return info


def get_dataset_size(dataset_path: Path) -> dict:
    """Calculate dataset size breakdown."""
    sizes = {
        "data_mb": 0,
        "videos_mb": 0,
        "meta_mb": 0,
        "total_mb": 0,
    }

    for subdir, key in [("data", "data_mb"), ("videos", "videos_mb"), ("meta", "meta_mb")]:
        path = dataset_path / subdir
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            sizes[key] = size / (1024 * 1024)

    sizes["total_mb"] = sizes["data_mb"] + sizes["videos_mb"] + sizes["meta_mb"]
    return sizes


def upload_dataset(
    local_path: Path,
    repo_id: str,
    private: bool = False,
    push_videos: bool = True,
    branch: str = None,
    license: str = "apache-2.0",
    tags: list = None,
    dry_run: bool = False,
):
    """
    Upload a local LeRobot dataset to HuggingFace Hub.

    Args:
        local_path: Path to local dataset directory
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        private: Whether to create a private repository
        push_videos: Whether to upload video files
        branch: Optional branch name to push to
        license: License for the dataset
        tags: Optional list of tags for the dataset
        dry_run: If True, validate only without uploading
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("=" * 60)
    print("LeRobot Dataset Upload to HuggingFace Hub")
    print("=" * 60)
    print(f"\nLocal Path:  {local_path}")
    print(f"Repo ID:     {repo_id}")
    print(f"Private:     {private}")
    print(f"Push Videos: {push_videos}")
    print(f"Branch:      {branch or 'main'}")
    print(f"License:     {license}")
    print(f"Tags:        {tags or []}")
    print(f"Dry Run:     {dry_run}")

    # Step 1: Validate dataset structure
    print("\n" + "-" * 40)
    print("Step 1: Validating dataset structure...")
    print("-" * 40)

    info = validate_dataset(local_path)
    print(f"\n  Dataset:   {local_path.name}")
    print(f"  Episodes:  {info.get('total_episodes', 'N/A')}")
    print(f"  Frames:    {info.get('total_frames', 'N/A')}")
    print(f"  FPS:       {info.get('fps', 'N/A')}")
    print(f"  Robot:     {info.get('robot_type', 'N/A')}")
    print(f"  Tasks:     {info.get('total_tasks', 'N/A')}")

    # Step 2: Calculate sizes
    print("\n" + "-" * 40)
    print("Step 2: Calculating dataset size...")
    print("-" * 40)

    sizes = get_dataset_size(local_path)
    print(f"\n  Data:      {sizes['data_mb']:.1f} MB")
    print(f"  Videos:    {sizes['videos_mb']:.1f} MB")
    print(f"  Meta:      {sizes['meta_mb']:.1f} MB")
    print(f"  Total:     {sizes['total_mb']:.1f} MB")

    if not push_videos:
        upload_size = sizes['data_mb'] + sizes['meta_mb']
        print(f"\n  Upload size (no videos): {upload_size:.1f} MB")

    # Step 3: Load dataset
    print("\n" + "-" * 40)
    print("Step 3: Loading dataset...")
    print("-" * 40)

    # Extract dataset name from path for local loading
    dataset_name = local_path.name

    dataset = LeRobotDataset(
        repo_id=repo_id,  # Use target repo_id
        root=str(local_path.parent),  # Parent directory
        local_files_only=True,
    )

    print(f"\n  Loaded {len(dataset)} frames")
    print(f"  Features: {list(dataset.meta.features.keys())}")

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE - No upload performed")
        print("=" * 60)
        print("\nDataset is valid and ready to upload.")
        print(f"Run without --dry-run to upload to: {repo_id}")
        return

    # Step 4: Upload to HuggingFace
    print("\n" + "-" * 40)
    print("Step 4: Uploading to HuggingFace Hub...")
    print("-" * 40)

    # Check HuggingFace authentication
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"\n  Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"\n  ERROR: HuggingFace authentication failed: {e}")
        print("  Please run: huggingface-cli login")
        sys.exit(1)

    # Prepare tags
    upload_tags = tags or []
    if "LeRobot" not in upload_tags:
        upload_tags.append("LeRobot")

    print(f"\n  Creating/updating repository: {repo_id}")
    print(f"  This may take a while for large datasets...")

    # Determine if we need upload_large_folder based on size
    use_large_folder = sizes['total_mb'] > 5000  # > 5GB

    dataset.push_to_hub(
        branch=branch,
        tags=upload_tags,
        license=license,
        push_videos=push_videos,
        private=private,
        upload_large_folder=use_large_folder,
    )

    # Step 5: Success
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)

    privacy_str = "private" if private else "public"
    print(f"\n  Dataset uploaded successfully!")
    print(f"  URL: https://huggingface.co/datasets/{repo_id}")
    print(f"  Visibility: {privacy_str}")

    print("\n  To use this dataset in training:")
    print(f"    --dataset.repo_id={repo_id}")

    if private:
        print("\n  Note: Private datasets require authentication:")
        print("    huggingface-cli login")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local LeRobot dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic upload
  %(prog)s --local-path ./my_dataset --repo-id user/my-dataset

  # Private dataset without videos
  %(prog)s --local-path ./my_dataset --repo-id user/my-dataset --private --no-videos

  # Validate only (dry run)
  %(prog)s --local-path ./my_dataset --repo-id user/my-dataset --dry-run
        """
    )

    parser.add_argument(
        "--local-path",
        type=str,
        required=True,
        help="Path to local dataset directory"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/dataset-name')"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)"
    )

    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip uploading video files (faster, smaller)"
    )

    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch to push to (default: main)"
    )

    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="License for the dataset (default: apache-2.0)"
    )

    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags (e.g., 'bimanual,so101,manipulation')"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset without uploading"
    )

    args = parser.parse_args()

    # Parse local path
    local_path = Path(args.local_path).resolve()
    if not local_path.is_dir():
        print(f"ERROR: Dataset directory not found: {local_path}")
        sys.exit(1)

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",")]

    # Run upload
    try:
        upload_dataset(
            local_path=local_path,
            repo_id=args.repo_id,
            private=args.private,
            push_videos=not args.no_videos,
            branch=args.branch,
            license=args.license,
            tags=tags,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
