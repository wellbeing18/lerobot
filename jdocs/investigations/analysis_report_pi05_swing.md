# Analysis Report: Pi05 "Crazy Swing" Behavior

**Date:** 2025-12-24
**Subject:** Investigation of "Arm Swinging Up and Down" Symptom
**Trace Analyzed:** `outputs/inference_traces/trace_pi05_20251224_124451`

## Executive Summary
The reported symptom of the robot arm "swinging crazy in the air" is confirmed by trace analysis. The `shoulder_lift` and `elbow_flex` joints exhibit **severe, high-frequency oscillations**, with command deltas exceeding **40 degrees per step** (30Hz). This behavior is likely caused by a combination of **model prediction instability** (chatter) and **high inference latency** (313ms) creating an unstable feedback loop.

## 1. Trace Observations

### Joint Trajectory Instability
Analysis of the `trace.jsonl` reveals massive step-to-step fluctuations in the action commands sent to the robot.

**Example: Elbow Flex Joint (Step 161-163)**
- **Step 161:** Command jumps **+41.16°** (17.94 -> 59.10)
- **Step 162:** Command jumps **-45.68°** (59.10 -> 13.42)
- **Step 163:** Command jumps **+18.82°** (13.42 -> 32.24)

**Example: Shoulder Lift Joint (Step 96-99)**
- **Step 96:** Command jumps **-27.03°** (-34.02 -> -61.05)
- **Step 97:** Command jumps **+17.44°** (-61.05 -> -43.62)
- **Step 98:** Command jumps **-22.35°** (-43.62 -> -65.96)

These are physically aggressive movements that would cause the robot to shake violently. The fact that they alternate sign (+/-) in consecutive steps indicates a classic oscillation.

### Latency Context
- **Inference Time:** Mean 313ms (approx. 9-10 steps at 30Hz).
- **Queue Size:** Mean ~32 steps.
- **Observation Age:** Estimated >1000ms.

## 2. Root Cause Analysis

### Primary Cause: Model Instability (Chatter)
The Pi05 model appears to be outputting erratic action chunks.
- If the oscillation exists *within* a single inference chunk, the model itself is predicting a "shake" motion.
- If the oscillation occurs *between* chunks, the temporal aggregation (averaging of overlapping chunks) is failing to smooth out disagreements between consecutive inferences.
- Given the magnitude (40 degrees), this is not just noise; it is a fundamental disagreement in the model's output about where the arm should be.

### Secondary Cause: Latency-Induced Feedback Loop
The 1-second observation lag creates a "Pilot Induced Oscillation" effect:
1.  The arm is slightly too low.
2.  The camera sees this (but the image is processed 1s later).
3.  The model commands a strong "UP".
4.  By the time "UP" executes, the arm might have drifted or been corrected by a previous command.
5.  The violent "UP" motion causes the camera to see a new view (sky/ceiling), which 1s later triggers a violent "DOWN".
6.  The cycle repeats and amplifies.

## 3. Conclusion & Recommendations

The "crazy swing" is a result of the model issuing high-amplitude, alternating commands, exacerbated by a slow control loop.

**Recommendations:**
1.  **Aggressive Smoothing:** Implement a stronger **Exponential Moving Average (EMA)** or low-pass filter on the actions before sending them to the robot. A 40-degree jump in 33ms is physically impossible/dangerous and should be clamped.
2.  **Action Delta Limiting:** Enforce a safety limit on the maximum change in joint angle per step (e.g., max 5 degrees/step).
3.  **Investigate Model Training:** Check if the Pi05 training data contains similar jitter or if the model is over-fitting to noise.
4.  **Reduce Latency:** The 300ms inference time is marginal for dynamic control. Optimization is needed.

