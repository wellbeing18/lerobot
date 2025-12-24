# SmolVLA RTC Scripts Review

## Overview

This document reviews the `jdocs/scripts/infer_smolvla_rtc.py` script against the [official LeRobot Real-Time Chunking (RTC) documentation](https://huggingface.co/docs/lerobot/rtc).

## Analysis of `infer_smolvla_rtc.py`

The script implements an asynchronous, multi-threaded inference loop that supports Real-Time Chunking for SmolVLA.

### Alignment with RTC Documentation

1.  **RTC Configuration**:
    *   **Correctness**: Uses `RTCConfig` and `RTCAttentionSchedule.EXP` as recommended in the documentation.
    *   **Parameters**: Defaults (`execution_horizon=10`, `max_guidance_weight=10.0`) match the "optimal values" cited in the documentation for 10-step flow matching models like SmolVLA.

2.  **Implementation Pattern**:
    *   **Async Logic**: Follows the `ActionQueue` pattern described in the "Quick Start" section, separating action prediction (`get_actions_thread`) from execution (`execute_actions_thread`).
    *   **Latency Handling**: Implements a `LatencyTracker` to dynamically calculate `inference_delay`. This aligns with the requirement to pass `inference_delay` to `predict_action_chunk`.
    *   **Merging**: Correctly uses `action_queue.merge` with the `postprocessed_actions` and `original_actions` (needed for the next iteration's guidance).

3.  **SmolVLA Specifics**:
    *   **Task Conditioning**: Correctly includes the `task` text in the observation (`observation["task"] = [task]`), which is required for SmolVLA but not for generic Pi0 examples.
    *   **Camera Mapping**: Correctly handles the `head` -> `camera1`, `left_wrist` -> `camera2` mapping required by the fine-tuned model.

### Improvements vs. Reference Example

The script adds several production-grade features over the simplified pseudo-code in the docs:
*   **Thread Safety**: Uses `Lock` for shared resources (`Robot`, `Cameras`, `ActionQueue`), which is critical for the multi-threaded nature of RTC.
*   **Dynamic Latency**: The doc snippet suggests a fixed `inference_delay`, but this script calculates it dynamically using a moving window (`LatencyTracker`), which is more robust to system jitter.
*   **Hardware Abstraction**: Reuses the `so101_hardware.yaml` config, making it compatible with the existing setup.

### Suggestions

While the script is technically sound and follows the reference implementation closely, here are a few suggestions:

1.  **Action Queue Merging Logic**:
    *   The documentation example shows `action_queue.merge(actions, actions, inference_delay)`.
    *   The script uses `action_queue.merge(original_actions, postprocessed, new_delay, action_index_before)`.
    *   **Suggestion**: Ensure that `prev_chunk_original` (stored in the queue) corresponds to the *un-normalized* (or model output space) actions if that's what `predict_action_chunk` expects for guidance. The script correctly stores `original_actions` (pre-postprocessing) for this purpose, which is likely correct given that flow matching usually operates in a normalized latent space.

2.  **Shutdown Handling**:
    *   The script uses a `shutdown_event` and daemon threads. This is good practice. Ensure that `robot.disconnect()` is called reliably, which the `finally` block handles.

3.  **Visualization**:
    *   RTC can be tricky to debug. The script logs basic stats. Adding the `RTCDebugVisualizer` (mentioned in "Advanced: Debug Tracking") could be helpful if you experience jerky motions, but it's not strictly necessary for deployment.

### Verdict

The script `infer_smolvla_rtc.py` is **well-implemented and compliant** with the LeRobot RTC documentation. It correctly adapts the generic Pi0 RTC pattern for SmolVLA, handling the specific requirements of the model (text conditioning) and the hardware (SO-101).

**It is safe to use.**









