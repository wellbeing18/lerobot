# Pi0.5 Scripts Review

## Overview

This document reviews the generated Pi0.5 scripts (`jdocs/scripts/train_pi05_pickplace.sh`, `jdocs/scripts/infer_pi05_so101.py`, and `jdocs/scripts/infer_pi05_rtc.py`) against LeRobot standards for the Pi0.5 model.

## Analysis of `train_pi05_pickplace.sh`

The script configures training for the large Pi0.5 model (PaliGemma-based).

### Best Practices & Alignment
1.  **Memory Management (Critical)**:
    *   **Gradient Checkpointing**: `GRADIENT_CHECKPOINTING=true` is correctly set. This is essential for fine-tuning a 2B+ parameter model on consumer/prosumer GPUs (e.g., RTX 3090/4090 with 24GB VRAM).
    *   **Precision**: `DTYPE="bfloat16"` is used, which is required for training these models efficiently without NaN issues.
    *   **Batch Size**: Defaults to `8`, which is a safe starting point for 24GB VRAM with this model size.

2.  **Model Configuration**:
    *   **Policy Type**: Correctly uses `--policy.type=pi05` and `--policy.pretrained_path=lerobot/pi05_base`.
    *   **Normalization**: Defaults to `QUANTILES`, which is the standard for Pi0.5 (unlike ACT/SmolVLA which often use Mean/Std). It includes a helpful warning check for `stats.json`.

3.  **Camera Mapping**:
    *   Renames `head` -> `base_0_rgb` and `left_wrist` -> `left_wrist_0_rgb`. This follows the Physical Intelligence / Pi0 convention of naming cameras.

### Suggestions
*   **Learning Rate**: Defaults to `2.5e-5`. This is lower than SmolVLA (1e-4), which is appropriate for a larger, pre-trained backbone to avoid catastrophic forgetting.
*   **Steps**: Defaults to 3000. Since Pi0.5 is a strong foundation model, it typically requires fewer steps to adapt than training from scratch, so 3k-6k is a reasonable range for a small dataset (50 episodes).

## Analysis of `infer_pi05_so101.py`

Standard inference script for Pi0.5.

### key Observations
1.  **Task Conditioning**: Correctly passes `observation["task"] = [task]`. Note that Pi0.5 (like SmolVLA) expects a list of strings for the task, which the script handles correctly.
2.  **Camera Mapping**:
    *   The script maps `head` -> `base_0_rgb` and `left_wrist` -> `left_wrist_0_rgb`.
    *   **Verification**: This matches the training script's `--rename_map`. This consistency is crucial.
3.  **Image Processing**: The script relies on the preprocessor (loaded from checkpoint) to handle resizing (typically to 224x224 for PaliGemma) and normalization. This is the correct approach.

## Analysis of `infer_pi05_rtc.py`

Real-Time Chunking (RTC) inference for Pi0.5.

### Best Practices & Alignment
1.  **RTC Configuration**:
    *   Uses `RTCConfig` with `execution_horizon=10` and `max_guidance_weight=10.0`. These are the standard "safe" defaults for flow-matching policies like Pi0.5.
    *   Enables RTC via `policy.init_rtc_processor()`.
2.  **Threading**:
    *   Implements the standard `ActionQueue` pattern with separate `get_actions` and `execute_actions` threads.
    *   Includes `LatencyTracker` to dynamically adjust `inference_delay`. This is superior to a fixed delay parameter.
3.  **Safety**:
    *   Uses `ThreadSafeRobot` and `ThreadSafeCameras` wrappers with `Lock`s, ensuring that concurrent access (RTC thread capturing images vs Main thread logging/monitoring) doesn't cause race conditions or segfaults in underlying C++ drivers (OpenCV/Serial).

### Suggestions
*   **Dataset Path**: The script correctly points to `DATASET_PATH` for loading statistics. Ensure `PROJECT_ROOT` resolves correctly in your environment (it uses `Path(__file__).parents[1]`, which should be `lerobot/`).

## Conclusion

All three scripts are high-quality and "production-ready" for LeRobot.
*   **Training**: robustly handles memory constraints for the large model.
*   **Inference**: Correctly handles the specific input requirements (camera names, task text) of Pi0.5.
*   **RTC**: Implements the advanced real-time chunking logic correctly for smoother control.

**Verdict**: The scripts are good to go. No changes recommended.









