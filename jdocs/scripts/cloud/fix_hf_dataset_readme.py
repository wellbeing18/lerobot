#!/usr/bin/env python3
"""
Fix HuggingFace dataset by uploading a proper README.md.

This script fixes datasets that were uploaded without a README.md,
causing HuggingFace to auto-detect the wrong parquet files.

Usage:
    python fix_hf_dataset_readme.py \
        --local-path ./datasets_bimanuel/picknplace_300_144 \
        --repo-id jasmine314342/picknplace-bimanual-464
"""

import argparse
import json
import sys
from pathlib import Path


def generate_readme(dataset_path: Path, info: dict, license: str = "apache-2.0", tags: list = None) -> str:
    """Generate a README.md for HuggingFace datasets."""
    all_tags = ["LeRobot", "robotics"]
    if tags:
        all_tags.extend(tags)
    tags_yaml = "\n".join(f"  - {tag}" for tag in all_tags)

    features = info.get("features", {})
    feature_lines = []
    for name, feat in features.items():
        dtype = feat.get("dtype", "unknown")
        shape = feat.get("shape", [])
        if dtype == "video":
            video_info = feat.get("info", {})
            h, w = video_info.get("video.height", "?"), video_info.get("video.width", "?")
            codec = video_info.get("video.codec", "?")
            feature_lines.append(f"- `{name}`: video ({h}x{w}, {codec})")
        else:
            feature_lines.append(f"- `{name}`: {dtype} {shape}")
    features_md = "\n".join(feature_lines)

    tasks_md = ""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = []
        with open(tasks_path) as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks.append(f"- {task.get('task', task.get('task_index', 'unknown'))}")
        if tasks:
            tasks_md = "\n## Tasks\n\n" + "\n".join(tasks[:20])
            if len(tasks) > 20:
                tasks_md += f"\n- ... and {len(tasks) - 20} more"

    readme = f"""---
license: {license}
task_categories:
  - robotics
tags:
{tags_yaml}
configs:
  - config_name: default
    data_files: data/*/*.parquet
---

# {dataset_path.name}

LeRobot dataset for robot manipulation.

## Dataset Info

| Property | Value |
|----------|-------|
| Codebase Version | {info.get('codebase_version', 'N/A')} |
| Robot Type | {info.get('robot_type', 'N/A')} |
| Total Episodes | {info.get('total_episodes', 'N/A')} |
| Total Frames | {info.get('total_frames', 'N/A')} |
| Total Tasks | {info.get('total_tasks', 'N/A')} |
| FPS | {info.get('fps', 'N/A')} |

## Features

{features_md}
{tasks_md}

## Usage

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{dataset_path.name}")
```
"""
    return readme


def main():
    parser = argparse.ArgumentParser(description="Fix HuggingFace dataset by uploading README.md")
    parser.add_argument("--local-path", type=str, required=True, help="Path to local dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License")
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--dry-run", action="store_true", help="Show README without uploading")

    args = parser.parse_args()

    local_path = Path(args.local_path).resolve()
    if not local_path.is_dir():
        print(f"ERROR: Dataset directory not found: {local_path}")
        sys.exit(1)

    # Load info.json
    info_path = local_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: info.json not found at: {info_path}")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",")]

    # Generate README
    readme_content = generate_readme(local_path, info, args.license, tags)

    print("=" * 60)
    print("Generated README.md:")
    print("=" * 60)
    print(readme_content)
    print("=" * 60)

    if args.dry_run:
        print("\nDry run - no upload performed")
        return

    # Upload README to HuggingFace
    from huggingface_hub import HfApi
    import tempfile

    api = HfApi()

    try:
        user_info = api.whoami()
        print(f"\nAuthenticated as: {user_info['name']}")
    except Exception as e:
        print(f"\nERROR: HuggingFace authentication failed: {e}")
        print("Please run: huggingface-cli login")
        sys.exit(1)

    # Write README to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        temp_path = f.name

    print(f"\nUploading README.md to {args.repo_id}...")

    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    # Clean up
    Path(temp_path).unlink()

    print(f"\nSUCCESS! README.md uploaded to: https://huggingface.co/datasets/{args.repo_id}")
    print("\nThe dataset should now display correctly on HuggingFace.")
    print("Note: It may take a few minutes for HuggingFace to re-index the dataset.")


if __name__ == "__main__":
    main()
