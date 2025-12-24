# Root Cause Analysis: Pi0.5 "Crazy Swing" & Instability

**Date:** 2025-12-24
**Subject:** Investigation of Training Configuration and Root Cause of Pi0.5 Instability
**Reference:** `jdocs/scripts/train_pi05_pickplace.sh`

## Executive Summary
The "crazy swing" behavior observed in the Pi0.5 model is **not due to a syntax bug** in the training script, but rather a combination of **Normalization Sensitivity**, **High Latency**, and potentially **Visual Domain Gap** in the fine-tuning configuration.

## 1. Finding: Normalization Strategy Risk
The training script defaults to **Quantile Normalization**:
```bash
NORMALIZATION_MODE="${NORMALIZATION_MODE:-QUANTILES}"
```
Analysis of the dataset (`datasets/pick_and_place/meta/stats.json`) reveals:
-   **Extreme Action Ranges:** `shoulder_lift` and `elbow_flex` span approx `[-100, 100]` degrees.
-   **Quantile Sensitivity:** Quantile normalization maps the data distribution to a uniform/Gaussian range. If the model makes a small error in the normalized space (e.g., predicting 0.95 instead of 0.90), this can map to a **massive jump** in the denormalized physical space (e.g., 20 degrees difference) at the edges of the distribution.
-   **Symptom Match:** The trace shows jumps of +/- 40 degrees. This is consistent with the model outputting values that fluctuate near the extremes of the normalization range, which get magnified upon denormalization.

## 2. Finding: Observation Latency
The Pi0.5 model has an inference latency of **313ms** (mean) and up to **600ms** (p99).
-   At 30Hz, this is a **10-20 frame delay**.
-   The robot executes an action based on where it *was* 0.3s - 1.0s ago.
-   If the arm starts to swing, the model sees the swing 0.5s later and commands a correction. By the time the correction executes, the arm is already somewhere else. This creates a classic **Pilot-Induced Oscillation (PIO)**.

## 3. Finding: Visual Feature Mapping
The script relies on implicit camera ordering:
```bash
# NOTE: Pi0.5 uses the dataset's original camera names directly
```
-   Dataset has: `head`, `left_wrist`.
-   Pi0.5 Pre-trained Base likely expects: `base_rgb`, `wrist_rgb` (or similar).
-   While LeRobot handles the plumbing, if the visual perspective of `head` is significantly different from the pre-training `base` view, the **frozen Vision Encoder** (standard in LoRA fine-tuning) may produce "out of distribution" embeddings.
-   Garbage visual embeddings -> Garbage/Noisy action predictions.

## 4. Recommendations for Mitigation

To fix the swinging issue, we recommend the following changes to the training and inference pipeline:

### A. Change Normalization to Mean/Std
Mean/Std normalization is linear and generally more robust to outliers and noise than Quantiles for continuous control tasks where smoothness is key.
**Action:** Modify `train_pi05_pickplace.sh` or set env var:
```bash
export NORMALIZATION_MODE=MEAN_STD
```
*Note: This requires re-training.*

### B. Increase Damping / Smoothing
Since the model is noisy, we must filter the output.
**Action:** In `infer_pi05_rtc.py` (or the inference config), increase temporal smoothing or add a low-pass filter.

### C. Reduce Latency (Critical)
The 300ms latency is the primary destabilizer.
**Action:**
-   Ensure `COMPILE_MODEL=true` is used if VRAM permits (might fail on 24GB).
-   Reduce `NUM_INFERENCE_STEPS` from 10 to 4 or 6 (trade-off quality for speed).
-   Reduce `CHUNK_SIZE` from 50 to 16 or 32.

### D. Verify Camera Mapping
Ensure the `head` camera is actually being treated as the primary global view. The implicit ordering usually works (alphabetical or definition order), but explicitly renaming to `camera_0` and `camera_1` via `--rename_map` might ensure better alignment with LeRobot conventions, though Pi0.5 policy claims to handle native names.

