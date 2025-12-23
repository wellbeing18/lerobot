# SmolVLA Scripts Review

## Overview

This document compares the generated SmolVLA scripts (`jdocs/scripts/train_smolvla_pickplace.sh` and `jdocs/scripts/infer_smolvla_so101.py`) against the official [LeRobot SmolVLA Tutorial](https://huggingface.co/docs/lerobot/smolvla).

## Analysis of `train_smolvla_pickplace.sh`

The script is a comprehensive wrapper around `lerobot-train` for fine-tuning the `smolvla_base` model.

### Alignment with Tutorial
*   **Base Model:** Correctly uses `lerobot/smolvla_base` as the starting point (`--policy.path`).
*   **Hyperparameters:**
    *   **Batch Size & Steps:** Defaults to `BATCH_SIZE=64` and `MAX_STEPS=20000`, matching the tutorial's recommendation for ~50 episodes.
    *   **Device:** Explicitly sets `cuda`, consistent with the tutorial.
*   **Dataset:** Points to the local `pick_and_place` dataset.

### Extensions & Best Practices
The script adds several advanced configurations not explicitly detailed in the minimal tutorial command but valuable for robust training:
*   **Scheduler:** Explicitly configures a cosine decay scheduler with warmup (`--policy.scheduler_warmup_steps`, etc.). This is generally good practice for transformers/VLAs.
*   **Optimization:** Adds `train_expert_only=true` and `freeze_vision_encoder=true`. This indicates a strategy of fine-tuning only specific parts of the model (likely the action head/expert), which preserves the pretrained VLM capabilities and speeds up training.
*   **Resuming:** Includes logic to resume from checkpoints, which is critical for long training runs.
*   **Logging:** sets up dual logging to file and console.

### Verdict
The training script is **excellent**. It follows the tutorial's core instructions while adding production-grade features (resuming, logging, scheduler control).

## Analysis of `infer_smolvla_so101.py`

This script implements a custom inference loop for the SO-101 robot using the fine-tuned SmolVLA policy.

### Alignment with Tutorial
*   **Task Conditioning:** The tutorial emphasizes that SmolVLA requires a natural language instruction.
    *   *Implementation:* The script correctly accepts a `--task` argument and injects it into the observation: `observation["task"] = task`.
*   **Hardware:** Reuses the `so101_hardware.yaml` and `RobotController` logic, maintaining consistency with the ACT setup.
*   **Policy Loading:** Uses `SmolVLAPolicy.from_pretrained`, which is the correct API.

### Comparison: Custom Script vs. `lerobot-record`
The tutorial suggests using `lerobot-record` for evaluation.
*   **Tutorial approach:**
    ```bash
    lerobot-record --dataset.single_task="Your task..." ...
    ```
*   **Current Script approach:**
    Manually constructs the loop.
    *   **Pros:** Allows full control over the `action_interval`, custom logging, and dry-run capabilities without writing to a dataset.
    *   **Cons:** Re-implements the standard recording loop.

### Key Observation: Camera Names
SmolVLA is flexible with camera inputs, but they must match what was seen during fine-tuning.
*   The script uses `head` and `left_wrist`.
*   **Validation:** Ensure the `pick_and_place` dataset used for fine-tuning actually has these keys. If the dataset uses different keys (e.g., `front`, `wrist`), the inference script must match them, or the policy will fail to find the expected images.

### Verdict
The inference script is **good** and functionally correct. It properly handles the unique requirement of SmolVLA (text conditioning).

## Summary

Both scripts are well-written and align with the LeRobot SmolVLA documentation. They provide a more robust and "production-ready" wrapper than the minimal command-line examples in the tutorial.

## Fixes Applied (2025-12-22)

Upon further inspection, the following issues were identified and fixed:

1.  **Camera Mapping Inconsistency**:
    *   **Issue**: The training script explicitly renames `head` → `camera1` and `left_wrist` → `camera2` (standardizing inputs for SmolVLA). However, the inference script was passing the original names `head` and `left_wrist` to the policy, which would cause a key mismatch during execution.
    *   **Fix**: Updated `infer_smolvla_so101.py` to automatically remap the camera keys in the observation dictionary to match the training configuration (`head` → `camera1`, `left_wrist` → `camera2`).

2.  **Dataset Path Mismatch**:
    *   **Issue**: The scripts defaulted to `${PROJECT_ROOT}/datasets/pick_and_place`, but the user's collected dataset is located at `/home/jrobot/project/XLeRobot/datasets/left/pick_and_place`.
    *   **Fix**: Updated `DATASET_PATH` in both `train_smolvla_pickplace.sh` and `infer_smolvla_so101.py` to point to the correct XLeRobot dataset location.

3.  **Task Name**:
    *   **Verification**: Verified that the `task_string` in `meta/config.yaml` matches the default task string in the inference script. No changes needed.

## Further Analysis (Questions)

### 1. Explicit Task Name in Training Script?
**Question**: Do we need to explicitly add a task name argument to the training script, or is it covered implicitly?
**Analysis**:
*   The LeRobot dataset format (v2.0+) includes a `task_index` in `info.json` which maps to task descriptions stored in `meta/tasks.parquet`.
*   During training, the `LeRobotDataset` class automatically retrieves the text instruction associated with each frame/episode using this index.
*   **Conclusion**: No explicit task name argument is required in the training script. The model will automatically receive the correct text conditioning ("pick up the block and place it on the plate") from the dataset metadata.

### 2. Training Hyperparameters (Batch Size & Steps)
**Question**: Is `BATCH_SIZE=32` and `MAX_STEPS=30000` good enough for the `pick_and_place` dataset (50 episodes)?
**Analysis**:
*   **Tutorial Baseline**: The tutorial recommends 20,000 steps for ~50 episodes. In their example, they use `batch_size=64`.
    *   Total Samples = 20,000 steps * 64 batch = **1,280,000 samples**.
*   **Proposed Config**: `BATCH_SIZE=32`, `MAX_STEPS=30000`.
    *   Total Samples = 30,000 steps * 32 batch = **960,000 samples**.
*   **Comparison**: The proposed configuration exposes the model to about **75%** of the total training volume compared to the tutorial's high-batch baseline.
*   **Assessment**:
    *   32 is a safer batch size for standard GPUs (preventing OOM).
    *   30,000 steps is a significant training duration and likely sufficient for convergence on a 50-episode dataset.
    *   While slightly less than the tutorial's total volume, it is well within the "good enough" range.
*   **Recommendation**: **Yes, this is good enough.** If you observe underfitting (high loss or poor robot performance), consider extending training to 40,000 steps (which would equal ~1.28M samples).
