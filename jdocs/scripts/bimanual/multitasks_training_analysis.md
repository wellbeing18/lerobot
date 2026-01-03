# Multitasks Dataset Training & Inference Analysis

## Overview

This document analyzes the changes needed to support the new `multitasks` dataset for SmolVLA bimanual training and inference.

## Dataset Comparison

| Aspect | Old (combined_pick_and_place) | New (multitasks) |
|--------|------------------------------|------------------|
| Location | `datasets_bimanuel/bimanual/combined_pick_and_place` | `datasets_bimanuel/multitasks` |
| Episodes | 100 | 266 |
| Frames | 26,221 | 64,489 |
| Tasks | 2 | 14 |
| Format | v3.0 LeRobotDataset | v3.0 LeRobotDataset |
| Collection | Merged from 2 single-task datasets | Unified multi-task collection |

## Task String Format

The multitasks dataset uses a new task string format:

**Old format:**
```
"Left arm pick up the tissue packet and place it on the plate"
```

**New format:**
```
"Use left arm to pick up the {object} and place it {in/on} the {target}"
```

### Available Tasks (14 total)

**Plate tasks (8):**
- Use left/right arm to pick up the orange and place it on the plate
- Use left/right arm to pick up the bread and place it on the plate
- Use left/right arm to pick up the corn and place it on the plate
- Use left/right arm to pick up the banana and place it on the plate

**Bin tasks (6):**
- Use left/right arm to pick up the ice cream and place it in the bin
- Use left/right arm to pick up the ketchup bottle and place it in the bin
- Use left/right arm to pick up the yogurt bottle and place it in the bin

## Script Changes Made

### 1. Training Script (`train_smolvla_bimanual.sh`)

**Updated lines 66-74:**
- Changed default `DATASET_PATH` to `multitasks`
- Updated comments to describe 14-task structure

**Usage:**
```bash
# Train on multitasks (default)
bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh

# Train on old combined dataset (override)
DATASET_PATH=datasets_bimanuel/bimanual/combined_pick_and_place \
DATASET_NAME=combined_pick_and_place \
bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh
```

### 2. Inference Script (`infer_smolvla_bimanual.py`)

**Changes:**
1. Updated default task strings to new format (lines 71-73)
2. Added `TASK_EXAMPLES` dictionary with all 14 tasks (lines 75-94)
3. Updated default dataset path to `multitasks` (line 146)
4. Added `--task-key` / `-k` argument for convenient task selection (lines 840-846)
5. Updated task determination logic with priority handling (lines 933-946)

**Usage:**
```bash
# Default task (left arm, orange, plate)
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py -c <checkpoint>

# Using task key
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py -c <checkpoint> -k right_ketchup_bin

# Custom task string
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py -c <checkpoint> -t "your custom task"

# List available task keys
python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --help | grep -A20 "task-key"
```

### Task Key Reference

| Key | Task String |
|-----|-------------|
| `left_orange_plate` | Use left arm to pick up the orange and place it on the plate |
| `right_orange_plate` | Use right arm to pick up the orange and place it on the plate |
| `left_bread_plate` | Use left arm to pick up the bread and place it on the plate |
| `right_bread_plate` | Use right arm to pick up the bread and place it on the plate |
| `left_corn_plate` | Use left arm to pick up the corn and place it on the plate |
| `right_corn_plate` | Use right arm to pick up the corn and place it on the plate |
| `left_banana_plate` | Use left arm to pick up the banana and place it on the plate |
| `right_banana_plate` | Use right arm to pick up the banana and place it on the plate |
| `left_icecream_bin` | Use left arm to pick up the ice cream and place it in the bin |
| `right_icecream_bin` | Use right arm to pick up the ice cream and place it in the bin |
| `left_ketchup_bin` | Use left arm to pick up the ketchup bottle and place it in the bin |
| `right_ketchup_bin` | Use right arm to pick up the ketchup bottle and place it in the bin |
| `left_yogurt_bin` | Use left arm to pick up the yogurt bottle and place it in the bin |
| `right_yogurt_bin` | Use right arm to pick up the yogurt bottle and place it in the bin |

## What Is NOT Needed

1. **No merge script** - The multitasks dataset is already unified during collection
2. **No new training script** - Existing script works with env var overrides
3. **No new inference script** - Minor modifications sufficient
4. **No data format conversion** - Dataset is already v3.0 format

## Backward Compatibility

All changes maintain backward compatibility:

```bash
# Old dataset still works
DATASET_PATH=datasets_bimanuel/bimanual/combined_pick_and_place \
bash train_smolvla_bimanual.sh

# Custom task strings still work
python infer_smolvla_bimanual.py --task "your custom task"
```

## Data Collection Script Changes

The data collection script (`jdocs/bimanual/jassy/scripts/collect_bimanuel_xlerobot_data.py`) was modified to support multi-task collection:

1. Added `--multitask` / `-m` flag for unified dataset mode
2. Added `--dataset-name` for custom naming
3. Auto-resume existing multitask datasets
4. Changed task string template from "Left arm..." to "Use left arm to..."

See `collect_bimanuel_xlerobot_data_backup.py` for the original single-task version.
