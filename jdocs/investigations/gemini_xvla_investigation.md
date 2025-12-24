# X-VLA Scripts Review

## Overview

This document reviews the generated X-VLA scripts (`jdocs/scripts/train_xvla_pickplace.sh` and `jdocs/scripts/infer_xvla_so101.py`) against the official [LeRobot X-VLA Documentation](https://huggingface.co/docs/lerobot/xvla).

## Analysis of `train_xvla_pickplace.sh`

The training script wraps `lerobot-train` for fine-tuning `xvla-base`.

### Alignment with Documentation

1.  **Model Configuration**:
    *   **Action Mode**: `ACTION_MODE="auto"` is correctly set, following the "Best Practice" in the docs for handling diverse action dimensions.
    *   **Precision**: `DTYPE="bfloat16"` is correctly set, which is **critical** for X-VLA to avoid OOM errors as noted in the documentation.
    *   **Base Model**: Points to `lerobot/xvla-base`, the recommended starting point.

2.  **Fine-Tuning Strategy**:
    *   **Full Fine-Tuning**: The script defaults to unfreezing everything (`FREEZE_VISION=false`, `TRAIN_POLICY_TRANSFORMER=true`, etc.). This matches the documentation's recommendation for "Best Performance" (Phase II adaptation).
    *   **Learning Rate**: The script uses `1e-4`, which is standard. The docs mention that the VLM backbone is automatically trained at 1/10th this rate by the training logic, so no manual adjustment is needed here.

3.  **Observation & Domain**:
    *   **Camera Renaming**: The script correctly includes `--rename_map` to map `head` -> `camera1` and `left_wrist` -> `camera2`, satisfying the X-VLA requirement for standard camera names.
    *   **Domain ID**: It sets `DOMAIN_ID=20` (arbitrary ID for custom robot), which is a valid approach for a new embodiment. The documentation notes that domain IDs guide the soft prompts.

### Suggestions

*   **Batch Size**: The script uses `BATCH_SIZE=16`. Given the 0.9B model size, this might be aggressive for a single GPU (depending on VRAM). The documentation example uses implicit defaults but warns about OOM. If OOM occurs, reducing this to 8 or 4 is the first fix.

## Analysis of `infer_xvla_so101.py`

The inference script implements the control loop for the SO-101 robot with X-VLA.

### Alignment with Documentation

1.  **Inputs**:
    *   **Task**: X-VLA is language-conditioned. The script correctly injects the `task` string into the observation.
    *   **Domain ID**: The script adds `observation["domain_id"] = domain_id`. This is crucial because X-VLA uses this ID to select the correct soft prompts learned during training.
    *   **Camera Mapping**: It correctly renames cameras (`head` -> `camera1`) to match what the model expects, mirroring the training script's logic.

2.  **Processing**:
    *   **Normalization**: The script comments mention "ImageNet normalization". This is handled by the `make_pre_post_processors` factory which loads the config from the checkpoint. This is correct; manual normalization in the script body is not needed if the preprocessor is loaded correctly.

3.  **Action Handling**:
    *   **Padding**: X-VLA often outputs actions padded to `MAX_ACTION_DIM` (e.g., 20). The script includes logic to trim the action vector back to the robot's DOF (`action[:len(joint_names)]`). This is robust and handles the `auto` action mode behavior described in the docs.

### Suggestions

1.  **Domain ID Consistency**: Ensure the `DEFAULT_DOMAIN_ID` in the inference script (20) matches whatever was used during training. The script defaults to 20, which matches the training script default.
2.  **Action Mode**: The inference script logs the `policy.config.action_mode`. If `auto` was used, the model will output 20-dim actions. The script handles this correctly.

## Verdict

Both scripts are **excellent**. They adhere strictly to the specific requirements of X-VLA (bfloat16, domain IDs, specific action modes, and unfreezing strategies) that differ from smaller models like ACT or SmolVLA.

**Status:** Ready to run.









