#!/home/jrobot/anaconda3/envs/lerobot/bin/python
"""
Merge Multiple LeRobot Datasets for Bimanual Training.

This script merges multiple LeRobot datasets into a single dataset for training.
Useful for combining left-arm-only and right-arm-only datasets to train a
bimanual model that can differentiate between arms based on task descriptions.

Key features:
- Combines parquet data files with proper episode re-indexing
- Merges video files (copies or symlinks)
- Combines statistics (recomputes mean/std/min/max across all data)
- Preserves task descriptions from each source dataset

Usage:
    # Merge left and right arm datasets
    python merge_bimanual_datasets.py \
        --datasets datasets_bimanuel/bimanual/left_arm_pick_and_place \
                   datasets_bimanuel/bimanual/right_arm_pick_and_place \
        --output datasets_bimanuel/bimanual/combined_pick_and_place

    # Dry run (show what would be done)
    python merge_bimanual_datasets.py \
        --datasets path/to/dataset1 path/to/dataset2 \
        --output path/to/merged \
        --dry-run

    # Use symlinks for videos (saves disk space)
    python merge_bimanual_datasets.py \
        --datasets path/to/dataset1 path/to/dataset2 \
        --output path/to/merged \
        --symlink-videos
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project src to path
# Script is at jdocs/scripts/bimanual/ -> go up 3 levels to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_dataset_info(dataset_path: Path) -> dict:
    """Load dataset info.json."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")

    with open(info_path) as f:
        return json.load(f)


def load_dataset_stats(dataset_path: Path) -> dict:
    """Load dataset stats.json."""
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")

    with open(stats_path) as f:
        return json.load(f)


def load_episodes_metadata(dataset_path: Path) -> pd.DataFrame:
    """Load episodes metadata from parquet files."""
    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.exists():
        logger.warning(f"Episodes directory not found: {episodes_dir}")
        return None

    # Find all parquet files in episodes directory
    parquet_files = list(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        return None

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def load_data_files(dataset_path: Path) -> pd.DataFrame:
    """Load all data parquet files from a dataset."""
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {pf}")
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")

    if not dfs:
        raise ValueError(f"Failed to load any data from {dataset_path}")

    return pd.concat(dfs, ignore_index=True)


def load_tasks(dataset_path: Path) -> list[dict]:
    """Load task descriptions from dataset."""
    # Try tasks.parquet first
    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        try:
            df = pd.read_parquet(tasks_parquet)
            # Handle case where task string is in the index
            if 'task' not in df.columns and df.index.name is None:
                # Task string might be the index
                df = df.reset_index()
                df.columns = ['task', 'task_index']
            elif 'task' not in df.columns:
                # Try to get task from index name or first column
                df = df.reset_index()
                if len(df.columns) >= 2:
                    df.columns = ['task'] + list(df.columns[1:])
            return df.to_dict('records')
        except Exception as e:
            logger.warning(f"Failed to read tasks.parquet: {e}")

    # Fall back to config.yaml
    config_path = dataset_path / "meta" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        task_info = config.get("data_collection", {}).get("task", {})
        if task_info:
            return [{"task_index": 0, "task": task_info.get("task_string", "unknown task")}]

    return [{"task_index": 0, "task": "unknown task"}]


def merge_datasets(
    dataset_paths: list[Path],
    output_path: Path,
    symlink_videos: bool = False,
    dry_run: bool = False,
) -> None:
    """Merge multiple datasets into one."""

    logger.info(f"Merging {len(dataset_paths)} datasets into {output_path}")

    # Validate all datasets exist
    for dp in dataset_paths:
        if not dp.exists():
            raise FileNotFoundError(f"Dataset not found: {dp}")
        if not (dp / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Invalid dataset (missing info.json): {dp}")

    # Load info from all datasets
    infos = [load_dataset_info(dp) for dp in dataset_paths]

    # Verify datasets are compatible
    first_features = infos[0]["features"]
    for i, info in enumerate(infos[1:], 1):
        if info["features"].keys() != first_features.keys():
            logger.warning(f"Dataset {i} has different features than dataset 0")
        if info.get("robot_type") != infos[0].get("robot_type"):
            logger.warning(f"Dataset {i} has different robot_type")

    # Load and merge data
    logger.info("Loading data from all datasets...")
    all_data = []
    all_tasks = []
    episode_offset = 0
    task_offset = 0
    frame_offset = 0

    dataset_episode_counts = []

    for i, (dp, info) in enumerate(zip(dataset_paths, infos)):
        logger.info(f"  Dataset {i}: {dp.name}")
        logger.info(f"    Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")

        # Load data
        df = load_data_files(dp)
        original_episodes = df['episode_index'].nunique()
        original_frames = len(df)

        # Re-index
        df['episode_index'] = df['episode_index'] + episode_offset
        df['index'] = df['index'] + frame_offset
        df['task_index'] = df['task_index'] + task_offset

        all_data.append(df)
        dataset_episode_counts.append(info['total_episodes'])

        # Load tasks
        tasks = load_tasks(dp)
        for task in tasks:
            task['task_index'] = task.get('task_index', 0) + task_offset
            all_tasks.append(task)

        # Update offsets
        episode_offset += info['total_episodes']
        frame_offset += info['total_frames']
        task_offset += len(tasks)

        logger.info(f"    -> Reindexed to episodes {episode_offset - info['total_episodes']}-{episode_offset - 1}")

    # Combine data
    logger.info("Combining data...")
    merged_df = pd.concat(all_data, ignore_index=True)

    # Verify merged data
    total_episodes = merged_df['episode_index'].nunique()
    total_frames = len(merged_df)
    logger.info(f"Merged dataset: {total_episodes} episodes, {total_frames} frames")

    if dry_run:
        logger.info("\n[DRY RUN] Would create the following structure:")
        logger.info(f"  {output_path}/")
        logger.info(f"  ├── data/chunk-000/")
        logger.info(f"  │   └── file-000.parquet ({total_frames} rows)")
        logger.info(f"  ├── meta/")
        logger.info(f"  │   ├── info.json")
        logger.info(f"  │   ├── stats.json")
        logger.info(f"  │   ├── tasks.parquet ({len(all_tasks)} tasks)")
        logger.info(f"  │   └── episodes/")
        logger.info(f"  └── videos/ (from {len(dataset_paths)} source datasets)")
        logger.info(f"\nTasks:")
        for task in all_tasks:
            logger.info(f"  [{task['task_index']}] {task.get('task', 'unknown')}")
        return

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos").mkdir(parents=True, exist_ok=True)

    # Save merged data
    logger.info("Saving merged data...")
    data_file = output_path / "data" / "chunk-000" / "file-000.parquet"
    merged_df.to_parquet(data_file, index=False)

    # Calculate merged statistics
    logger.info("Computing statistics...")
    stats = compute_merged_stats(merged_df, infos[0]["features"])

    # Copy image stats from first source dataset (videos not in dataframe)
    first_stats = load_dataset_stats(dataset_paths[0])
    for feature_name, feature_info in infos[0]["features"].items():
        if feature_info["dtype"] == "video" and feature_name in first_stats:
            stats[feature_name] = first_stats[feature_name]
            logger.debug(f"Copied image stats for {feature_name}")

    # Create merged info.json
    merged_info = {
        "codebase_version": infos[0].get("codebase_version", "v3.0"),
        "robot_type": infos[0].get("robot_type", "bi_so101_follower"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": round(data_file.stat().st_size / (1024 * 1024)),
        "video_files_size_in_mb": sum(info.get("video_files_size_in_mb", 0) for info in infos),
        "fps": infos[0].get("fps", 30),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": infos[0]["features"],
    }

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)

    # Save stats
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Save tasks (format: task string as index, task_index as column)
    # This matches LeRobotDataset expected format
    logger.info("Saving tasks...")
    tasks_df = pd.DataFrame(
        {'task_index': [t['task_index'] for t in all_tasks]},
        index=[t['task'] for t in all_tasks]
    )
    tasks_df.to_parquet(output_path / "meta" / "tasks.parquet")

    # Copy/link videos FIRST (before creating episodes metadata)
    # so we know the video file offsets
    logger.info("Processing videos...")
    copy_videos(dataset_paths, output_path, dataset_episode_counts, symlink_videos)

    # Create episodes metadata (needs infos for offsets)
    logger.info("Creating episodes metadata...")
    create_episodes_metadata(dataset_paths, output_path, infos, all_tasks)

    # Create config.yaml
    create_merged_config(dataset_paths, output_path, all_tasks)

    logger.info(f"\nMerged dataset created successfully at: {output_path}")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Total tasks: {len(all_tasks)}")


def compute_merged_stats(df: pd.DataFrame, features: dict) -> dict:
    """Compute statistics for merged dataset."""
    stats = {}

    for feature_name, feature_info in features.items():
        if feature_info["dtype"] == "video":
            continue  # Skip video features

        if feature_name in df.columns:
            col_data = df[feature_name]

            # Handle array columns
            if feature_info.get("shape") and len(feature_info["shape"]) > 0:
                # Stack arrays
                try:
                    arr = np.stack(col_data.values)
                    stats[feature_name] = {
                        "mean": arr.mean(axis=0).tolist(),
                        "std": arr.std(axis=0).tolist(),
                        "min": arr.min(axis=0).tolist(),
                        "max": arr.max(axis=0).tolist(),
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute stats for {feature_name}: {e}")
            else:
                # Scalar columns
                stats[feature_name] = {
                    "mean": [float(col_data.mean())],
                    "std": [float(col_data.std())],
                    "min": [float(col_data.min())],
                    "max": [float(col_data.max())],
                }

    return stats


def create_episodes_metadata(
    dataset_paths: list[Path],
    output_path: Path,
    infos: list[dict],
    all_tasks: list[dict],
) -> None:
    """Create episodes metadata by merging from source datasets.

    This function properly copies all episode metadata columns including:
    - Video file indices and timestamps for all cameras
    - Data file indices
    - Per-episode statistics

    It adjusts indices appropriately for the merged dataset.
    """
    all_episodes = []
    episode_offset = 0
    frame_offset = 0
    video_file_offset = 0
    task_offset = 0

    for ds_idx, (ds_path, info) in enumerate(zip(dataset_paths, infos)):
        episodes_dir = ds_path / "meta" / "episodes"
        if not episodes_dir.exists():
            logger.warning(f"No episodes directory in {ds_path}")
            continue

        # Load all episode parquet files from this dataset
        parquet_files = sorted(episodes_dir.rglob("*.parquet"))

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)

                # Adjust episode indices
                df['episode_index'] = df['episode_index'] + episode_offset

                # Adjust data indices
                if 'dataset_from_index' in df.columns:
                    df['dataset_from_index'] = df['dataset_from_index'] + frame_offset
                if 'dataset_to_index' in df.columns:
                    df['dataset_to_index'] = df['dataset_to_index'] + frame_offset

                # Adjust video file indices for all cameras
                # Video files are being copied with offset, so we need to update file indices
                video_columns = [c for c in df.columns if c.startswith('videos/') and '/file_index' in c]
                for vc in video_columns:
                    df[vc] = df[vc] + video_file_offset

                # Update task strings based on new task indices
                if 'tasks' in df.columns:
                    # Tasks column contains list of task strings - update from merged task list
                    def update_task(row):
                        old_task_idx = row.get('task_index', 0) if 'task_index' in df.columns else 0
                        # For first row of episode, get task index
                        new_task_idx = old_task_idx + task_offset
                        # Find task string from merged task list
                        for t in all_tasks:
                            if t['task_index'] == new_task_idx:
                                return [t.get('task', 'unknown task')]
                        return row['tasks']
                    # Don't modify tasks column - keep original task strings

                all_episodes.append(df)
                logger.debug(f"Loaded {len(df)} episodes from {pf}")

            except Exception as e:
                logger.warning(f"Failed to read {pf}: {e}")

        # Count video files in this dataset to set offset for next dataset
        videos_dir = ds_path / "videos"
        if videos_dir.exists():
            # Count unique video files across all cameras
            max_file_idx = 0
            for camera_dir in videos_dir.iterdir():
                if camera_dir.is_dir():
                    for chunk_dir in camera_dir.iterdir():
                        if chunk_dir.is_dir():
                            video_files = list(chunk_dir.glob("*.mp4"))
                            for vf in video_files:
                                # Extract file index from filename (file-XXX.mp4)
                                try:
                                    file_idx = int(vf.stem.split('-')[1])
                                    max_file_idx = max(max_file_idx, file_idx + 1)
                                except (IndexError, ValueError):
                                    pass
            video_file_offset += max_file_idx

        # Update offsets for next dataset
        episode_offset += info['total_episodes']
        frame_offset += info['total_frames']
        task_offset += len(load_tasks(ds_path))

    if not all_episodes:
        logger.error("No episode metadata found in any dataset!")
        return

    # Concatenate all episodes
    merged_episodes = pd.concat(all_episodes, ignore_index=True)

    # Update meta/episodes file indices
    if 'meta/episodes/chunk_index' in merged_episodes.columns:
        merged_episodes['meta/episodes/chunk_index'] = 0
    if 'meta/episodes/file_index' in merged_episodes.columns:
        merged_episodes['meta/episodes/file_index'] = 0

    # Save merged episodes metadata
    merged_episodes.to_parquet(
        output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        index=False
    )

    logger.info(f"Created merged episodes metadata with {len(merged_episodes)} episodes")


def copy_videos(
    dataset_paths: list[Path],
    output_path: Path,
    episode_counts: list[int],
    use_symlinks: bool = False,
) -> dict[int, int]:
    """Copy or symlink video files from source datasets.

    Returns a dict mapping dataset index to the video file offset used.
    Video files from second dataset onwards are renamed with offset indices
    to avoid collisions (e.g., file-050.mp4, file-051.mp4, ...).
    """
    # Get video keys from first dataset
    first_videos_dir = dataset_paths[0] / "videos"
    if not first_videos_dir.exists():
        logger.warning("No videos directory found in first dataset")
        return {}

    video_keys = [d.name for d in first_videos_dir.iterdir() if d.is_dir()]
    video_file_offsets = {}
    current_file_offset = 0

    for ds_idx, (ds_path, ep_count) in enumerate(zip(dataset_paths, episode_counts)):
        logger.info(f"  Processing videos from {ds_path.name} (file offset: {current_file_offset})...")
        video_file_offsets[ds_idx] = current_file_offset

        videos_dir = ds_path / "videos"
        if not videos_dir.exists():
            logger.warning(f"  No videos directory in {ds_path}")
            continue

        max_file_idx_in_dataset = 0

        for video_key in video_keys:
            video_key_dir = videos_dir / video_key
            if not video_key_dir.exists():
                continue

            # Create output video key directory
            out_video_key_dir = output_path / "videos" / video_key
            out_video_key_dir.mkdir(parents=True, exist_ok=True)

            # Process each chunk directory
            for chunk_dir in sorted(video_key_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue

                # Create output chunk directory
                out_chunk_dir = out_video_key_dir / chunk_dir.name
                out_chunk_dir.mkdir(parents=True, exist_ok=True)

                # Process each video file
                for video_file in sorted(chunk_dir.glob("*.mp4")):
                    # Parse original file index from filename (file-XXX.mp4)
                    try:
                        orig_file_idx = int(video_file.stem.split('-')[1])
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse file index from {video_file.name}")
                        continue

                    # Track max file index in this dataset
                    max_file_idx_in_dataset = max(max_file_idx_in_dataset, orig_file_idx + 1)

                    # Calculate new file index with offset
                    new_file_idx = orig_file_idx + current_file_offset
                    new_filename = f"file-{new_file_idx:03d}.mp4"
                    out_video = out_chunk_dir / new_filename

                    if out_video.exists():
                        continue

                    if use_symlinks:
                        out_video.symlink_to(video_file.resolve())
                    else:
                        shutil.copy2(video_file, out_video)

        # Update offset for next dataset
        current_file_offset += max_file_idx_in_dataset

    return video_file_offsets


def create_merged_config(dataset_paths: list[Path], output_path: Path, tasks: list[dict]) -> None:
    """Create a config.yaml for the merged dataset."""
    import yaml
    from datetime import datetime

    config = {
        "experiment": {
            "name": output_path.name,
            "date": datetime.now().isoformat(),
            "merged_from": [str(dp) for dp in dataset_paths],
        },
        "data_collection": {
            "tasks": [
                {"task_index": t["task_index"], "task_string": t.get("task", "unknown")}
                for t in tasks
            ],
        },
        "_metadata": {
            "saved_at": datetime.now().isoformat(),
            "script": "merge_bimanual_datasets.py",
        },
    }

    with open(output_path / "meta" / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets for bimanual training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to datasets to merge"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for merged dataset"
    )
    parser.add_argument(
        "--symlink-videos",
        action="store_true",
        help="Use symlinks for videos instead of copying (saves disk space)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually merging"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert to absolute paths
    dataset_paths = [Path(dp).resolve() for dp in args.datasets]
    output_path = Path(args.output).resolve()

    # Check output doesn't exist
    if output_path.exists() and not args.dry_run:
        logger.error(f"Output path already exists: {output_path}")
        logger.error("Please remove it or choose a different output path")
        sys.exit(1)

    try:
        merge_datasets(
            dataset_paths=dataset_paths,
            output_path=output_path,
            symlink_videos=args.symlink_videos,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
