# Analysis Report: SmolVLA and Pi05 Inference Traces

**Date:** 2025-12-24
**Subject:** Analysis of Pick-and-Place Failure Modes (Spatial Miss & Early Release)
**Traces Analyzed:**
1. `outputs/inference_traces/trace_smolvla_20251224_123946` (SmolVLA)
2. `outputs/inference_traces/trace_pi05_20251224_124451` (Pi05)

## Executive Summary
The investigation into the reported symptoms—"picking in space next to the block" and "grasping but releasing immediately"—identifies **high execution latency** (Observation Age ~800ms-1000ms) as the primary root cause for spatial errors, and **action instability (chatter)** combined with latency as the cause for grasp failures.

## 1. Trace Overview

### Trace 1: SmolVLA
- **Model:** SmolVLA (Checkpoint: `smolvla_pickplace_20251223_001238`)
- **Rate:** 30Hz
- **Inference Latency:** Mean 163ms, P99 442ms.
- **Queue Size (RTC):** Mean ~24 steps.
- **Observation Freshness:** Mean **816ms** (approx. 24 frames lag).
- **Symptoms Observed in Trace:** Gripper command fluctuations (e.g., Step 402: 24% -> 9%).

### Trace 2: Pi05
- **Model:** Pi05 (Checkpoint: `pi05_pickplace_20251223_144340`)
- **Rate:** 30Hz
- **Inference Latency:** Mean 313ms, P99 616ms.
- **Queue Size (RTC):** Mean ~32 steps.
- **Observation Freshness:** Estimated **~1000ms** (based on queue size).
- **Symptoms Observed in Trace:** Significant high-frequency gripper oscillation (chatter).

## 2. Root Cause Analysis

### Symptom A: "Picking in space next to the block"
**Diagnosis:** Systematic spatial error due to stale observations (High Latency).

**Evidence:**
- In `trace_smolvla`, the `observation_age_ms` is consistently around **800ms**.
- This means the action executed by the robot at time $T$ is based on an image captured at $T - 0.8s$.
- Since the camera is wrist-mounted (indicated by `left_wrist.jpg` in file logs), the camera pose changes as the robot moves.
- **Mechanism:**
    1. The robot moves towards the block.
    2. An image is captured.
    3. The model predicts the block's location in the *camera frame* of that image.
    4. Due to the large action queue (24 steps), this action is executed 0.8s later.
    5. By the time the action executes, the robot arm (and camera) has moved. The "forward" vector from 0.8s ago is no longer the correct vector to reach the block from the *current* position.
    6. Result: The robot reaches for where the block *was* relative to the camera 0.8s ago, which manifests as picking "next to" the block in world coordinates.

### Symptom B: "Grasp but release right after"
**Diagnosis:** Action instability (Chatter) exacerbated by execution delays.

**Evidence:**
- `trace_pi05` shows severe gripper command oscillation.
    - Step 161-162: Command jumps 22 -> 31 -> 21 (Close -> Open) in consecutive steps.
    - Step 238-239: Command jumps 24 -> 30 -> 23.
- `trace_smolvla` shows similar but less frequent drops (Step 402: 24 -> 9).
- **Mechanism:**
    1. The model is uncertain or the temporal aggregation (chunking) is producing inconsistent boundaries.
    2. One inference chunk commands "Close".
    3. The next inference chunk (or a specific step within it) commands "Open" or a lower value.
    4. Because of the 1-second lag, these conflicting commands play out on the robot long after the visual event that triggered them.
    5. If the robot successfully grasps (based on the "Close" command), the subsequent "Open" spike (potentially due to noise or the model thinking it missed in a stale image) causes an immediate release.

## 3. Detailed Data Evidence

### Gripper Command Instability (Pi05 Example)
The following sequence demonstrates the "Grasp and Release" chatter:
```
Step 161: Command 22.12 -> 31.04 (Close/Peak)
Step 162: Command 31.04 -> 21.22 (Open/Release)
Step 188: Command 24.10 -> 32.63 (Close/Peak)
Step 189: Command 32.63 -> 26.32 (Release)
```
This rapid fluctuation prevents a stable grasp.

### Latency/Queue Statistics (SmolVLA)
```json
"rtc": {
  "queue_size": {
    "mean": 24.2,
    "p50": 25.0
  }
},
"observation_freshness": {
  "observation_age_ms": {
    "mean": 816.59,
    "p50": 801.00
  }
}
```
A 25-step queue at 30Hz is $25 / 30 = 0.833$ seconds of delay.

## 4. Conclusion & Recommendations

The "Pick in Space" issue is a direct consequence of **latency in the Real-Time Control (RTC) loop**, specifically the large queue size causing stale observations in a moving-camera setup. The "Early Release" is caused by **model prediction noise (chatter)** which might also be a symptom of the model struggling with stale observations (seeing the hand in an unexpected location relative to the previous plan).

**Recommendations:**
1.  **Reduce Queue Size:** Tune `action_queue_threshold` to be lower (closer to the inference time equivalent). For SmolVLA (160ms inference), a buffer of 6-10 steps is sufficient. Current buffer is ~24 steps.
2.  **Increase Inference Speed:** If possible, optimize the model or use a faster GPU/quantization to reduce inference time, allowing for a tighter control loop.
3.  **Check Aggregation:** Investigate why `pi05` produces such oscillatory gripper commands. Increasing `temporal_ensemble` smoothing or using a hysteresis filter on the gripper action could help.

