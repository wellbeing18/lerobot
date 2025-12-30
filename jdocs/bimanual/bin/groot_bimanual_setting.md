# Detailed Analysis Report: Isaac-GR00T-UIUC Fork Changes

## Table of Contents

- [Repository Overview](#repository-overview)
- [Key Commits Analyzed](#key-commits-analyzed)
- [Major Changes Summary](#major-changes-summary)
  - [1. SO-101 Bimanual Robot Support (NEW)](#1-so-101-bimanual-robot-support-new)
  - [2. LeRobot Data Format Conversion (v3.0 -> v2.1)](#2-lerobot-data-format-conversion-v30---v21)
  - [3. Automatic HuggingFace Upload Integration](#3-automatic-huggingface-upload-integration)
- [Comparison with Upstream NVIDIA GR00T](#comparison-with-upstream-nvidia-groot)
  - [DATA_CONFIG_MAP Comparison](#data_config_map-comparison)
  - [New Files in Fork](#new-files-in-fork-not-in-upstream)
- [Steps to Reproduce the Modifications](#steps-to-reproduce-the-modifications)
  - [Step 1: Add SO-101 Bimanual Support](#step-1-add-so-101-bimanual-support)
  - [Step 2: Add Data Format Conversion](#step-2-add-data-format-conversion)
  - [Step 3: Add HuggingFace Upload](#step-3-add-huggingface-upload)
- [Code Quality Notes](#code-quality-notes)
- [Usage Examples](#usage-examples)
  - [Fine-tuning with SO-101 Bimanual](#fine-tuning-with-so-101-bimanual)
  - [Converting LeRobot v3.0 to v2.1](#converting-lerobot-v30-to-v21)
  - [Training with Auto-Upload to HuggingFace](#training-with-auto-upload-to-huggingface)
- [Recommendations](#recommendations)

---

## Repository Overview

- **Fork**: `SIGRobotics-UIUC/Isaac-GR00T-UIUC`
- **Original**: `NVIDIA/Isaac-GR00T` (GR00T N1.5/N1.6)
- **Current HEAD**: `main` branch at commit `959e569`
- **Key changes are on**: `origin/master` branch

## Key Commits Analyzed

| Commit | Description | Author |
|--------|-------------|--------|
| `959e569` | Added lerobot bimanual support step 1 | Leo Lin |
| `e2997c8` | 3.0 -> 2.1 data conversion + custom modality + bimanual so101 support | Keshav Badrinath |
| `6cd6e20` | Auto upload to HuggingFace | Keshav Badrinath |

---

## Major Changes Summary

### 1. SO-101 Bimanual Robot Support (NEW)

The fork adds support for the **SO-101 bimanual robot** which is not present in the upstream NVIDIA GR00T repository.

**Key files added/modified:**

#### `gr00t/experiment/data_config.py`
- Added `So101ArmsDataConfig` class
- Added `BimanualSo101DataConfig` class
- Added entries to `DATA_CONFIG_MAP`:
  - `"so101": So101ArmsDataConfig()`
  - `"bimanual_so101_arms": BimanualSo101DataConfig()`

**Configuration details for SO-101 Bimanual:**
```python
video_keys = ["video.right", "video.left", "video.top_depth"]
state_keys = [
    "state.left_arm",     # 4 joint positions (indices 1-5)
    "state.gripper1",     # 1 gripper position (index 5-6)
    "state.right_arm",    # 4 joint positions (indices 1-5)
    "state.gripper2",     # 1 gripper position (index 5-6)
]
action_keys = [
    "action.left_arm",
    "action.gripper1",
    "action.right_arm",
    "action.gripper2",
]
language_keys = ["annotation.human.task_description"]
```

#### `examples/so101_dualcam__modality.json` (on main branch)
Modality mapping configuration for LeRobot data format:
- Maps `observation.images.right`, `observation.images.left`, `observation.images.top_depth` to video keys
- Maps state/action indices for bimanual arm and gripper control
- Uses `task_index` for task description annotation

#### `examples/SO-101/` directory (on master branch)
Contains:
- `custom_data_config.py`: `So101BimanualDataConfig` class
- `so101_bimanual__modality.json`: Modality mapping
- `eval_gr00t_so101.py`: Evaluation script for SO-101
- `README.md`: Usage documentation

---

### 2. LeRobot Data Format Conversion (v3.0 -> v2.1)

The fork adds a **data conversion utility** to convert LeRobot datasets from v3.0 format back to v2.1 format.

**File added:** `scripts/conversion.py`

**Purpose:** The newer LeRobot v3.0 uses a consolidated file layout, while GR00T N1.5 expects the legacy v2.1 per-episode structure.

**Key conversions performed:**
1. **Parquet data**: Converts consolidated parquet files back to per-episode parquets
2. **Video files**: Splits concatenated MP4 files into per-episode videos using ffmpeg
3. **Metadata**: Reconstructs `meta/episodes.jsonl` and `meta/episodes_stats.json`
4. **Tasks**: Converts `meta/tasks` parquet to legacy JSONL format
5. **Info**: Updates `codebase_version` from `v3.0` to `v2.1`

**Usage:**
```bash
python scripts/conversion.py \
    --repo-id lerobot/your-dataset \
    --root /path/to/datasets
```

---

### 3. Automatic HuggingFace Upload Integration

The fork adds automatic model upload to HuggingFace Hub during/after training.

**Files added:**
- `gr00t/utils/huggingface_upload.py`: Core upload functionality
- `gr00t/utils/huggingface_callback.py`: Training integration callback
- `docs/huggingface_upload.md`: Documentation
- `HUGGINGFACE_UPLOAD.md`: Quick start guide

**Files modified:**
- `gr00t/experiment/runner.py`: Added HF callback integration
- `scripts/gr00t_finetune.py`: Added HF upload after training

**Features:**
- Automatic model upload after training completion
- Optional checkpoint uploads during training
- Auto-generated model cards with training metadata
- Support for public/private repositories
- Environment variable configuration

**Environment variables:**
```bash
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID="your-username/gr00t-model-name"
export HF_TOKEN="hf_your_token_here"
export HF_PRIVATE=true  # Optional
export HF_UPLOAD_ON_SAVE=true  # Optional: upload checkpoints
```

---

## Comparison with Upstream NVIDIA GR00T

### DATA_CONFIG_MAP Comparison

| Config | Upstream | Fork |
|--------|----------|------|
| `fourier_gr1_arms_waist` | Yes | Yes |
| `fourier_gr1_arms_only` | Yes | Yes |
| `fourier_gr1_full_upper_body` | Yes | Yes |
| `bimanual_panda_gripper` | Yes | Yes |
| `bimanual_panda_hand` | Yes | Yes |
| `single_panda_gripper` | Yes | Yes |
| `so100` | Yes | Yes |
| `so100_dualcam` | Yes | Yes |
| `so101` | **No** | **Yes (NEW)** |
| `bimanual_so101_arms` | **No** | **Yes (NEW)** |
| `unitree_g1` | Yes | Yes |
| `unitree_g1_full_body` | Yes | Yes |
| `oxe_droid` | Yes | Yes |
| `agibot_genie1` | Yes | Yes |

### New Files in Fork (not in upstream)

1. `scripts/conversion.py` - v3.0 -> v2.1 data conversion
2. `gr00t/utils/huggingface_upload.py` - HF upload utility
3. `gr00t/utils/huggingface_callback.py` - Training callback
4. `examples/SO-101/` directory - SO-101 robot support
5. `examples/so101_dualcam__modality.json` - Modality config
6. `docs/huggingface_upload.md` - Upload documentation
7. `HUGGINGFACE_UPLOAD.md` - Quick start guide

---

## Steps to Reproduce the Modifications

### Step 1: Add SO-101 Bimanual Support

1. **Create modality configuration** (`examples/so101_dualcam__modality.json`):
```json
{
    "state": {
        "left_arm": { "start": 1, "end": 5 },
        "gripper1": { "start": 5, "end": 6 },
        "right_arm": { "start": 1, "end": 5 },
        "gripper2": { "start": 5, "end": 6 }
    },
    "action": {
        "left_arm": { "start": 1, "end": 5 },
        "gripper1": { "start": 5, "end": 6 },
        "right_arm": { "start": 1, "end": 5 },
        "gripper2": { "start": 5, "end": 6 }
    },
    "video": {
        "right": { "original_key": "observation.images.right" },
        "left": { "original_key": "observation.images.left" },
        "top_depth": { "original_key": "observation.images.top_depth" }
    },
    "annotation": {
        "human.task_description": { "original_key": "task_index" }
    }
}
```

2. **Add data config classes** to `gr00t/experiment/data_config.py`:
   - Create `So101ArmsDataConfig(BaseDataConfig)` with bimanual video/state/action keys
   - Create `BimanualSo101DataConfig(So101ArmsDataConfig)`
   - Add to `DATA_CONFIG_MAP`

3. **Create external config** (`examples/SO-101/custom_data_config.py`):
   - `So101BimanualDataConfig` class with proper transforms
   - Use for finetuning: `--data_config examples.SO-101.custom_data_config:So101BimanualDataConfig`

### Step 2: Add Data Format Conversion

1. **Create** `scripts/conversion.py` with:
   - `convert_dataset()` function for v3.0 -> v2.1
   - Parquet conversion, video splitting, metadata reconstruction
   - Command line interface

2. **Dependencies**: Add lerobot, jsonlines, pyarrow, pandas

### Step 3: Add HuggingFace Upload

1. **Create** `gr00t/utils/huggingface_upload.py`:
   - `HuggingFaceUploader` class
   - `create_uploader_from_env()` function
   - `extract_training_info()`, `extract_dataset_info()` helpers

2. **Create** `gr00t/utils/huggingface_callback.py`:
   - `HuggingFaceUploadCallback(TrainerCallback)` class
   - Handles `on_save` and `on_train_end` events

3. **Modify** `gr00t/experiment/runner.py`:
   - Import HF utilities
   - Add callback in `create_trainer()` method

4. **Modify** `scripts/gr00t_finetune.py`:
   - Print HF upload configuration
   - Call `upload_model_to_hf()` after training

---

## Code Quality Notes

**Issues found in the fork:**

1. **Indentation problems** in `data_config.py`:
   - `So101ArmsDataConfig` and `BimanualSo101DataConfig` have mixed/incorrect indentation
   - The `modality_config()` method in `So101ArmsDataConfig` is incomplete (truncated)

2. **Missing files in main branch**:
   - The key changes are on `origin/master`, not `main`
   - Users need to checkout master or merge to get full functionality

---

## Usage Examples

### Fine-tuning with SO-101 Bimanual

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/so101_dataset/ \
    --data_config examples.SO-101.custom_data_config:So101BimanualDataConfig \
    --num-gpus 8 \
    --batch-size 90 \
    --output-dir /tmp/so101-checkpoints \
    --max-steps 60000
```

### Converting LeRobot v3.0 to v2.1

```bash
python scripts/conversion.py \
    --repo-id your-org/your-dataset \
    --root /path/to/datasets
```

### Training with Auto-Upload to HuggingFace

```bash
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID="your-username/gr00t-so101"
export HF_TOKEN="hf_xxxxx"

python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --data_config bimanual_so101_arms \
    --num-gpus 4
```

---

## Recommendations

1. **Merge master into main** to consolidate changes
2. **Fix indentation issues** in `data_config.py`
3. **Complete the truncated** `So101ArmsDataConfig.modality_config()` method
4. **Add tests** for the conversion script and HF upload
5. **Document the differences** from upstream for maintainability
