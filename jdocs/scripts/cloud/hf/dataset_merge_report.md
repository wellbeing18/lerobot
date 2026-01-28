# LeRobot Dataset Merge Report: `picknplace_300_144`

**Date:** January 28, 2026
**Author:** Claude (with fixes from Gemini and review from GPT)
**Output Dataset:** `datasets_bimanuel/picknplace_300_144`

## Executive Summary

Successfully merged two bimanual robot datasets into a single combined dataset for cloud training and HuggingFace upload. The merge process encountered two critical bugs that were identified and fixed.

| Metric | Value |
|--------|-------|
| Total Episodes | 464 (320 + 144) |
| Total Frames | 118,772 |
| Unique Tasks | 16 |
| Data Files | 24 parquet files |
| Video Files | 24 per camera (72 total) |

---

## 1. Source Datasets

### Dataset A: `multitasks`
```
Episodes:    320 (indices 0-319)
Frames:      76,597
Tasks:       16 unique tasks
Structure:   16 data files (file-000 to file-015)
             16 video files per camera
```

### Dataset B: `pairs20260126`
```
Episodes:    144 (indices 0-143)
Frames:      42,175
Tasks:       12 unique tasks (subset of multitasks)
Structure:   144 data files (one per episode)
             144 video files per camera (one per episode)
```

---

## 2. LeRobot v3.0 Dataset Structure

Understanding the structure is critical for proper merging:

```
dataset/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet    # Episodes 0-19 (20 episodes per file)
│       ├── file-001.parquet    # Episodes 20-39
│       └── ...
├── meta/
│   ├── info.json               # Dataset metadata
│   ├── stats.json              # Normalization statistics
│   ├── tasks.parquet           # Task descriptions & indices
│   ├── tasks.jsonl             # Task descriptions (HF upload)
│   ├── episodes.jsonl          # Episode metadata (HF upload)
│   ├── config.yaml             # Recording configuration
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # Episode metadata with file pointers
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       ├── file-000.mp4    # Videos for episodes 0-19
    │       ├── file-001.mp4    # Videos for episodes 20-39
    │       └── ...
    ├── observation.images.left_wrist/
    └── observation.images.right_wrist/
```

### Key Concept: File Indexing

Each episode maps to a specific file via integer division:

```
file_index = episode_index // episodes_per_file

Example (episodes_per_file = 20):
  Episode 0   -> file-000  (0 // 20 = 0)
  Episode 19  -> file-000  (19 // 20 = 0)
  Episode 20  -> file-001  (20 // 20 = 1)
  Episode 319 -> file-015  (319 // 20 = 15)
  Episode 320 -> file-016  (320 // 20 = 16)
```

---

## 3. The Merge Process

### Step 1: Task Unification

Both datasets have overlapping tasks. We create a unified task list:

```
multitasks tasks (16):              pairs tasks (12):
┌────┬─────────────────────┐        ┌────┬─────────────────────┐
│ 0  │ orange -> plate (L) │        │ 0  │ banana -> plate (L) │
│ 1  │ orange -> plate (R) │        │ 1  │ corn -> plate (L)   │
│ 2  │ bread -> plate (L)  │        │ 2  │ banana -> plate (R) │
│ 3  │ bread -> plate (R)  │        │ 3  │ corn -> plate (R)   │
│ 4  │ corn -> plate (L)   │        │ 4  │ bread -> plate (L)  │
│ 5  │ corn -> plate (R)   │        │ 5  │ bread -> plate (R)  │
│ 6  │ banana -> plate (L) │◄───────│ 6  │ yogurt -> bin (L)   │
│ 7  │ banana -> plate (R) │        │ 7  │ tissue -> bin (L)   │
│ ...│ ...                 │        │ ...│ ...                 │
└────┴─────────────────────┘        └────┴─────────────────────┘

Task Remapping for pairs:
  pairs task 0 (banana L)  -> combined task 6
  pairs task 1 (corn L)    -> combined task 4
  pairs task 2 (banana R)  -> combined task 7
  pairs task 4 (bread L)   -> combined task 2
  ...
```

### Step 2: Episode Index Remapping

```
Source datasets:                    Combined dataset:
┌─────────────┐                     ┌─────────────────────────┐
│ multitasks  │                     │ picknplace_300_144      │
│ ep 0-319    │────────────────────►│ ep 0-319 (unchanged)    │
└─────────────┘                     │                         │
┌─────────────┐                     │                         │
│ pairs       │                     │                         │
│ ep 0-143    │───── +320 ─────────►│ ep 320-463 (remapped)   │
└─────────────┘                     └─────────────────────────┘
```

### Step 3: Data File Structure

```
Combined data/chunk-000/:

file-000.parquet  ─┐
file-001.parquet   │
...                ├── From multitasks (episodes 0-319)
file-014.parquet   │
file-015.parquet  ─┘
file-016.parquet  ─┐
file-017.parquet   │
...                ├── From pairs (episodes 320-463)
file-022.parquet   │
file-023.parquet  ─┘

Total: 24 files
```

### Step 4: Episode Metadata Pointers

The `meta/episodes/` parquet contains pointers to data and video files:

```
Episode metadata structure:
┌───────────────┬─────────────────┬─────────────────┬─────────────────┐
│ episode_index │ data/file_index │ data/chunk_index│ videos/.../file │
├───────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 0             │ 0               │ 0               │ 0               │
│ 19            │ 0               │ 0               │ 0               │
│ 20            │ 1               │ 0               │ 1               │
│ ...           │ ...             │ ...             │ ...             │
│ 319           │ 15              │ 0               │ 15              │
│ 320           │ 16              │ 0               │ 16              │ ◄── Critical!
│ 321           │ 16              │ 0               │ 16              │
│ ...           │ ...             │ ...             │ ...             │
│ 463           │ 23              │ 0               │ 23              │
└───────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## 4. Bugs Encountered and Fixes

### Bug #1: Episode Data File Pointers (CRITICAL)

**Problem:** Episodes 320-463 retained their original file indices (0-143) instead of being recalculated (16-23).

```
BEFORE FIX (WRONG):
┌───────────────┬─────────────────┐
│ episode_index │ data/file_index │
├───────────────┼─────────────────┤
│ 319           │ 15              │ ✓ Correct
│ 320           │ 0               │ ✗ WRONG! Points to file-000
│ 321           │ 1               │ ✗ WRONG! Points to file-001
│ 322           │ 2               │ ✗ WRONG! Points to file-002
│ ...           │ ...             │
│ 463           │ 143             │ ✗ WRONG! file-143 doesn't exist!
└───────────────┴─────────────────┘

AFTER FIX (CORRECT):
┌───────────────┬─────────────────┐
│ episode_index │ data/file_index │
├───────────────┼─────────────────┤
│ 319           │ 15              │ ✓ Correct
│ 320           │ 16              │ ✓ 320 // 20 = 16
│ 321           │ 16              │ ✓ 321 // 20 = 16
│ 322           │ 16              │ ✓ 322 // 20 = 16
│ ...           │ ...             │
│ 463           │ 23              │ ✓ 463 // 20 = 23
└───────────────┴─────────────────┘
```

**Root Cause:** The original merge script only updated `episode_index` but didn't recalculate `data/file_index`.

**Fix:** Added explicit recalculation after merging:
```python
def calc_file_index(ep_idx):
    return ep_idx // episodes_per_file

merged_episodes["data/file_index"] = merged_episodes["episode_index"].apply(calc_file_index)
```

---

### Bug #2: Video Files Not Chunked (CRITICAL)

**Problem:** Videos were copied as individual files with global episode indices instead of being concatenated into chunked files.

```
BEFORE FIX (WRONG):
videos/observation.images.head/chunk-000/
├── file-000.mp4   ─┐
├── file-001.mp4    │ From multitasks (chunked, 20 eps each)
├── ...             │
├── file-015.mp4   ─┘
├── file-320.mp4   ─┐
├── file-321.mp4    │ From pairs (individual files!)
├── file-322.mp4    │
├── ...             │ These don't match the expected
├── file-463.mp4   ─┘ file-016 to file-023 pattern!

AFTER FIX (CORRECT):
videos/observation.images.head/chunk-000/
├── file-000.mp4   ─┐
├── file-001.mp4    │ From multitasks
├── ...             │
├── file-015.mp4   ─┘
├── file-016.mp4   ─┐ Concatenated from pairs episodes 320-339
├── file-017.mp4    │ Concatenated from pairs episodes 340-359
├── ...             │
├── file-023.mp4   ─┘ Concatenated from pairs episodes 460-463
```

**Root Cause:** The merge script copied videos individually instead of concatenating them into chunked files.

**Fix:** Used ffmpeg to concatenate videos:
```python
def concatenate_videos(video_files: list[Path], output_path: Path):
    # Create file list for ffmpeg
    with open("concat_list.txt", "w") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")

    # Concatenate without re-encoding
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt",
        "-c", "copy",  # No re-encoding
        str(output_path)
    ])
```

---

### Bug #3: Stale JSONL Files

**Problem:** After Gemini fixed the parquet files, the JSONL files (used for HuggingFace upload) still had old incorrect values.

```
episodes.jsonl BEFORE regeneration:
{"episode_index": 320, "data/file_index": 0, ...}   ✗ WRONG

episodes.jsonl AFTER regeneration:
{"episode_index": 320, "data/file_index": 16, ...}  ✓ CORRECT
```

**Fix:** Regenerated JSONL files from the corrected parquet:
```python
# Read corrected parquet
ep_df = pd.read_parquet("meta/episodes/chunk-000/file-000.parquet")

# Write to JSONL
with open("meta/episodes.jsonl", "w") as f:
    for _, row in ep_df.iterrows():
        f.write(json.dumps(dict(row)) + "\n")
```

---

### Bug #4: Episode Video File Pointers (CRITICAL)

**Problem:** Episodes 320-463 also had wrong `videos/.../file_index` pointers (0-143 instead of 16-23). This was separate from Bug #1 which only fixed data file indices.

```
BEFORE FIX (WRONG):
  Episode 320: videos/head/file_index = 0   ✗ Should be 16
  Episode 463: videos/head/file_index = 143 ✗ Should be 23 (file-143 doesn't exist!)

AFTER FIX (CORRECT):
  Episode 320: videos/head/file_index = 16  ✓
  Episode 463: videos/head/file_index = 23  ✓
```

**Root Cause:** The initial fix only corrected `data/file_index` but not `videos/.../file_index` columns.

**Fix:** Applied same formula to all video file index columns:
```python
video_cols = [col for col in df.columns if col.startswith('videos/') and col.endswith('/file_index')]
for col in video_cols:
    df[col] = df['episode_index'] // episodes_per_file
```

---

### Bug #5: Missing Image Statistics (CRITICAL)

**Problem:** The merge script only computed stats for action, state, and scalar columns, not for image observations. Training requires stats for all features.

```
stats.json BEFORE FIX:
  Keys: ['action', 'observation.state', 'timestamp', ...]
  Missing: observation.images.head, observation.images.left_wrist, observation.images.right_wrist

stats.json AFTER FIX:
  Keys: ['action', 'observation.state', ..., 'observation.images.head', ...]
```

**Root Cause:** The merge script didn't compute or copy image statistics.

**Fix:** Copied image stats from source dataset (same image format):
```python
image_keys = [k for k in mt_stats.keys() if 'images' in k]
for key in image_keys:
    pk_stats[key] = mt_stats[key]
```

---

## 5. Verification Results

### Sanity Check: Boundary Episodes

| Episode | Role | file_index | Data Exists | Videos Exist | Task |
|---------|------|------------|-------------|--------------|------|
| 0 | First overall | 0 | ✓ | ✓ | orange (task 0) |
| 319 | Last multitasks | 15 | ✓ | ✓ | tissue R (task 15) |
| 320 | First pairs | 16 | ✓ | ✓ | banana L (task 6) |
| 463 | Last overall | 23 | ✓ | ✓ | ketchup R (task 11) |

### Task Remapping Verification

```
Original pairs episode 0:
  task_index: 0 (banana in pairs)

Combined episode 320:
  task_index: 6 (banana in combined) ✓

Mapping: pairs_task_0 -> combined_task_6 ✓
```

---

## 6. Final Dataset Structure

```
datasets_bimanuel/picknplace_300_144/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet    # Episodes 0-19
│       ├── file-001.parquet    # Episodes 20-39
│       ├── ...
│       ├── file-015.parquet    # Episodes 300-319
│       ├── file-016.parquet    # Episodes 320-339
│       ├── ...
│       └── file-023.parquet    # Episodes 460-463
├── meta/
│   ├── info.json               # 464 episodes, 118772 frames
│   ├── stats.json              # Combined statistics
│   ├── tasks.parquet           # 16 tasks
│   ├── tasks.jsonl             # For HF upload
│   ├── episodes.jsonl          # For HF upload (regenerated)
│   ├── config.yaml
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # Fixed file pointers
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       ├── file-000.mp4 ... file-023.mp4
    ├── observation.images.left_wrist/
    │   └── chunk-000/
    │       ├── file-000.mp4 ... file-023.mp4
    └── observation.images.right_wrist/
        └── chunk-000/
            ├── file-000.mp4 ... file-023.mp4
```

---

## 7. Lessons Learned

### For Future Dataset Merges

1. **Always recalculate file indices** - Don't preserve original indices; compute them from the new episode index.

2. **Match video structure to data structure** - If data files are chunked (20 episodes per file), videos must also be chunked.

3. **Regenerate all derived files** - After fixing parquet, also regenerate JSONL and any other derived formats.

4. **Verify boundary episodes** - Always check the first/last episodes and the merge boundary (e.g., 319/320).

5. **Verify task mapping** - Confirm task indices in data match the combined task list.

### Verification Checklist

```
□ Episode count matches sum of sources
□ Frame count matches sum of sources
□ File indices are sequential (0, 1, 2, ... N)
□ Boundary episodes (N-1, N) have correct file indices
□ Last episode points to existing file
□ Task indices correctly remapped
□ Videos exist for all file indices
□ JSONL files regenerated from corrected parquet
```

---

## 8. Commands Reference

### Merge Datasets
```bash
python jdocs/scripts/data/merge_datasets.py \
    --datasets ./datasets_bimanuel/multitasks ./datasets_bimanuel/pairs20260126 \
    --output ./datasets_bimanuel/picknplace_300_144
```

### Regenerate JSONL from Parquet
```bash
python -c "
import json, pandas as pd
from pathlib import Path

path = Path('./datasets_bimanuel/picknplace_300_144')
df = pd.read_parquet(path / 'meta/episodes/chunk-000/file-000.parquet')
with open(path / 'meta/episodes.jsonl', 'w') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')
"
```

### Upload to HuggingFace
```bash
python jdocs/scripts/cloud/upload_dataset_to_hub.py \
    --local-path ./datasets_bimanuel/picknplace_300_144 \
    --repo-id jasmine314342/picknplace-bimanual-464
```

---

## 9. Acknowledgments

- **Gemini**: Identified and fixed the episode metadata file pointers and video chunking issues
- **GPT**: Provided thorough review identifying task remapping concerns and verification checklist
- **Claude**: Initial merge implementation and final report

---

## 10. Training Verification

Successfully ran 2 training steps on the merged dataset:

```
INFO dataset.num_frames=118772 (119K)
INFO dataset.num_episodes=464
INFO Start offline training on a fixed dataset
INFO step:1 smpl:2 ep:0 epch:0.00 loss:0.630 grdn:7.115 lr:2.0e-07
INFO step:2 smpl:4 ep:0 epch:0.00 loss:1.651 grdn:24.236 lr:3.0e-07
INFO End of training

BIMANUAL Training completed successfully!
```

All 5 bugs were fixed:
1. ✓ Data file indices (Gemini)
2. ✓ Video chunking (Gemini)
3. ✓ JSONL regeneration (Claude)
4. ✓ Video file indices (Claude)
5. ✓ Image statistics (Claude)

---

**Status: READY for HuggingFace upload and cloud training**
