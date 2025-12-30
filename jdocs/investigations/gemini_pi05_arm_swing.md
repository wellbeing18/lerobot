# Investigation Report: Pi0.5 Erratic Arm Movement during Inference

**Date:** December 25, 2025  
**Subject:** Analysis of erratic "swinging" behavior in Pi0.5 VLA model after fine-tuning.  
**Investigated Trace:** `outputs/inference_traces/trace_pi05_20251225_105101`  
**Investigated Scripts:** `jdocs/scripts/train_pi05_pickplace.sh`, `jdocs/scripts/infer_pi05_rtc_trace.py`

---

## 1. Executive Summary

The fine-tuned Pi0.5 model exhibits erratic "swinging" behavior during inference. While initial suspicion focused on normalization (`MEAN_STD` vs. `QUANTILES`), technical analysis of the inference trace and RTC (Real-Time Chunking) logic reveals that the primary cause is a **mathematical bug in the action queue management**. This bug causes the robot to "jump" to future points in a trajectory while it is still physically at its starting position, resulting in high-velocity, unstable movements.

---

## 2. Detailed Trace Analysis

### 2.1 The "Jump" Bug (Action Index Mismatch)

The most critical issue was found in the `ActionQueue.merge` method within `jdocs/scripts/infer_pi05_rtc_trace.py`.

**How it works (intended):**
When an inference finishes after a delay (e.g., 300ms), the script should skip the first few actions of the new prediction because the robot has already moved during that 300ms.

**The Bug (Actual):**
```python
# jdocs/scripts/infer_pi05_rtc_trace.py (Line 425-428)
consumed = self.action_index - action_index_before
skip = max(0, consumed + inference_delay)
```
1.  `self.action_index` only increments when an action is **successfully retrieved** from the queue and executed.
2.  At the start of the trace (Step 0 to 30), the queue is empty because the first inference is still running.
3.  Because the queue is empty, `self.get()` returns `None`, and **`self.action_index` stays at 0**.
4.  When the first inference (Step 31) finishes, `consumed` is calculated as `0 - 0 = 0`.
5.  The `inference_delay` is calculated as `31` steps (based on the ~1.5s initial latency).
6.  The script skips **31 actions** from the new chunk.

**Result:** The robot is physically at the start position (Step 0), but it suddenly attempts to execute Action #31 from the trajectory. This creates a massive discontinuity in target positions, causing the arm to swing violently to catch up to the "future" target.

### 2.2 Cumulative Latency (Queue Append Issue)

**The Logic Error:**
```python
# jdocs/scripts/infer_pi05_rtc_trace.py (Line 432)
self.queue.extend(new_actions.unbind(0))
```
The script uses `.extend()` to add new actions to the back of the queue. In a Real-Time Chunking (RTC) setup, new inferences should **replace** outdated future actions, not append to them. 

**Result:** Every time a new inference finishes, the robot is forced to finish executing whatever was left in the old queue before it can start the new, more up-to-date trajectory. This leads to a "laggy" robot that is always reacting to what it saw 0.5s–1.0s ago, which often manifests as oscillations or overshooting.

---

## 3. Hardware and Latency Observations

### 3.1 Initial "Cold Start" Latency
The trace shows that the very first inference took **1523ms**, while subsequent ones averaged **~300ms**.
*   **Camera Capture:** The first capture took **880ms**. This suggests the OpenCV camera buffers or the USB bus were not fully primed.
*   **Inference:** The first model pass took **612ms**, likely due to CUDA kernels being compiled/loaded into the GPU cache.

This massive initial delay, combined with the "Jump Bug" described in Section 2.1, is the primary trigger for the initial erratic swing.

---

## 4. Training and Normalization Analysis

### 4.1 Normalization Mode
The model was switched from `QUANTILES` to `MEAN_STD`. While `QUANTILES` is the default and recommended mode for Pi0.5 (as it handles outliers in joint space better), `MEAN_STD` is technically valid as long as the dataset stats are correct. 

However, the "crazy swinging" was reported in both modes. This confirms that **the issue is not the normalization**, but the inference execution logic.

### 4.2 Training Progress
Training for 7k-10k steps on a 50-episode dataset is appropriate for a LoRA fine-tune of Pi0.5. The model should have a basic understanding of the task. The bad performance observed is likely the robot's hardware interface failing to follow the model's intended smooth path.

---

## 5. Design Thinking: How to Fix

To resolve the erratic behavior, the inference pipeline should be restructured following these steps:

1.  **Synchronize Action Index with Time:** Instead of counting "actions executed," the queue should be indexed by a global timer or a step counter that increments every loop iteration (e.g., 30 times per second), regardless of whether the queue was empty or not.
2.  **Implement Queue Replacement:** Use a slicing/replacement strategy for the action queue. When a new chunk arrives, it should overwrite the future indices of the existing queue.
3.  **Hardware Pre-warming:** Before entering the main control loop, the script should run 1-2 "dummy" cycles (Capture -> Format -> Predict) to clear camera buffers and warm up the GPU.
4.  **Match FPS:** Ensure the control loop runs at the same FPS as the training data (30 FPS). The trace showed the loop running at 20 FPS, which creates a mismatch in the "speed" of the predicted actions.

---

## 6. Conclusion

The Pi0.5 model is likely trained correctly, but the **`infer_pi05_rtc_trace.py` script contains a mathematical error** in how it compensates for inference delay. By skipping actions based on a stalled index counter, it forces the robot to make impossible physical jumps, resulting in the observed "crazy" swinging behavior. Fixing the `ActionQueue` logic is the highest priority.





