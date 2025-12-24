# SmolVLA RTC Inference Investigation

**Date**: 2024-12-24
**Trace Analyzed**: `outputs/inference_traces/trace_smolvla_20251224_123946`

## Reported Symptoms

1. **Model picks in space next to the block** - arm doesn't locate the block correctly
2. **Arm grasps block but releases immediately** - gripper closes then opens mid-task

## Executive Summary

Investigation reveals **two critical issues** in the RTC inference pipeline:

| Issue | Impact | Evidence |
|-------|--------|----------|
| High observation staleness | Actions based on 660-816ms old observations | Mean staleness: 660-816ms |
| Gripper oscillation | Gripper repeatedly opens/closes instead of completing grasp | 22.6→31.0 jump at step 323; command drops (24%→9%) |

## Detailed Analysis

### 1. Observation Staleness Problem

**Finding**: Actions are executed based on observations that are 660-1400ms old.

| Metric | Value |
|--------|-------|
| Mean staleness | 660-816 ms |
| Max staleness | 1397 ms |
| P95 staleness | 1264 ms |
| Mean queue size | 24 steps |

**Calculation**: A 24-step queue at 30Hz = 24/30 = **800ms** of delay.

**Impact**: At 30 FPS, the robot moves ~20-40 steps between when an observation was captured and when the resulting action is executed. This creates a temporal mismatch:
- The model sees the block at position X
- By the time actions execute, the arm has already moved
- Result: "picking in space" where the block was, not where it is

**Wrist Camera Factor** (from Gemini's analysis): Since the camera is wrist-mounted (`left_wrist.jpg`), the camera pose changes as the robot moves. The "forward" vector from 800ms ago no longer points to the block from the current arm position.

**Root Cause**: The RTC dual-threaded architecture creates inherent latency:
```
Capture (t=0) → Inference (t=163ms) → Queue (t=800ms) → Execute (t+staleness)
```

### 2. Gripper Behavior Analysis: Oscillating Gripper

**Observation**: Gripper state oscillates instead of completing close sequence.

| Step | State | Action | Delta | Event |
|------|-------|--------|-------|-------|
| 323 | 22.6 | 31.0 | +8.3 | Inference opens gripper mid-close |
| 402 | ~24% | ~9% | -15% | Command drops mid-grasp |

**Evidence**: At step 323, an inference was triggered while gripper was at 22.6 (partially closed). The new inference predicted action 31.0 (opening), causing a +8.3 degree reversal.

**Why this happens** (combined analysis):
1. Model trained on demonstrations where gripper was consistently open or closed
2. RTC inference triggers on queue threshold, not gripper state
3. New inference can predict "open" when gripper is mid-close because:
   - Observation is stale (800ms old)
   - Model sees gripper as "open" in stale frame
   - Predicts more "open" actions despite gripper currently closing
4. **Chatter mechanism** (Gemini): Model uncertainty or temporal aggregation (chunking) produces inconsistent boundaries between inference chunks

### 3. Inference Timing Analysis

| Metric | Value |
|--------|-------|
| Total inferences | 23 |
| Mean inference duration | 163 ms |
| P99 inference duration | 442 ms |
| Mean latency (queue wait) | 216 ms |
| Mean queue size | 24 steps |

### 4. Action Buffer Discontinuities

When new inference completes, the action buffer is replaced. This can cause:
- Spatial discontinuities (arm jumps)
- Gripper state reversals

**Evidence**: Large action jumps (>5 degrees in any joint) detected:
- 4 large jumps total, 1 at inference boundary

## Root Cause Summary

### Issue 1: Picking in Space

**Primary Cause**: Observation staleness (~800ms)

The observation used by the model is significantly older than the current robot state. With the arm moving continuously, actions based on stale observations target where the block *was*, not where it *is*.

**Contributing Factors**:
1. Inference latency: ~163ms mean for forward pass
2. Queue depth: ~24 actions queued before execution
3. At 30 FPS: 24 queued actions = 800ms delay
4. Wrist-mounted camera compounds the issue - camera pose changes during delay

### Issue 2: Grasp and Release

**Primary Cause**: Action instability (chatter) + inference interrupting gripper sequences

New inferences are triggered by queue depth threshold, not by gripper state. When an inference happens mid-close:
1. Model sees stale observation with gripper open
2. Predicts action sequence assuming gripper is open
3. New "open" actions override in-progress close sequence
4. Gripper reverses direction → block drops

**Evidence**:
- Step 323: gripper at 22.6 (closing), inference triggers, new action is 31.0 (opening)
- Step 402: command drops from 24% to 9%

## RTC Parameter Tuning Experiments

### Experiment Results (2024-12-24)

| Config | threshold | horizon | guidance | Mean Staleness | Max Staleness | Result |
|--------|-----------|---------|----------|----------------|---------------|--------|
| Original | 10 | 10 | 10 | 660ms | 1397ms | Baseline |
| Attempt 1 | 5 | 8 | 10 | 739ms | 1506ms | **Worse** (-12%) |
| Attempt 2 | 15 | 5 | 15 | **572ms** | **1202ms** | **Better** (+13%) |

**Key Insight**: Lower `action_queue_threshold` alone doesn't help because chunk_size=50. Need **higher threshold + lower horizon** to trigger inference earlier and transition to fresh chunks faster.

### Best Parameters So Far
```bash
python jdocs/scripts/infer_smolvla_rtc_trace.py \
    --checkpoint outputs/smolvla_pickplace_20251223_001238/checkpoints/checkpoints/last/pretrained_model \
    --action-queue-threshold 15 \
    --execution-horizon 5 \
    --max-guidance-weight 15.0 \
    --duration 30
```

**Results**:
- Observation staleness: 660ms → 572ms (13% improvement)
- Inferences: 21 → 26 (more frequent)
- Max action index: 44 → 35 (fresher observations)
- **Robot still had issues locating gripper to pick up block**

### Next Experiment: More Aggressive Settings
```bash
python jdocs/scripts/infer_smolvla_rtc_trace.py \
    --checkpoint outputs/smolvla_pickplace_20251223_001238/checkpoints/checkpoints/last/pretrained_model \
    --action-queue-threshold 20 \
    --execution-horizon 3 \
    --max-guidance-weight 20.0 \
    --duration 30
```

**Expected**: Further reduce staleness to ~400ms (20 actions × 33ms/action)

## Recommendations

### Immediate Fixes (RTC Parameter Tuning)

1. **Use optimized RTC parameters**:
   ```bash
   --action-queue-threshold 20  # Trigger inference earlier
   --execution-horizon 3        # Faster transition to fresh chunk
   --max-guidance-weight 20.0   # Tighter blending
   ```

2. **Gripper State Guard**: Add logic to prevent gripper reversal mid-sequence:
   ```python
   if gripper_closing and new_action > current_state:
       new_action = current_state - 1  # Continue closing
   ```

3. **Hysteresis Filter on Gripper**: Apply temporal smoothing to gripper commands to reduce chatter:
   ```python
   gripper_action = alpha * new_action + (1 - alpha) * prev_action
   ```

### Short-term Fixes (Requires Retraining)

1. **Reduce Chunk Size**: Train with smaller chunks for inherently lower staleness:
   ```bash
   CHUNK_SIZE=15 N_ACTION_STEPS=15 bash jdocs/scripts/train_smolvla_pickplace.sh
   ```

### Long-term Fixes

1. **State-Conditioned Inference**: Only trigger inference when:
   - Queue is low AND
   - No critical motion in progress (gripper close, approach phase)

2. **Observation Freshness Tracking**: Inject observation age as model input to let model account for temporal offset.

3. **Action Blending Improvements**: Instead of replacing action buffer, blend old and new predictions with weighted average.

4. **Increase temporal_ensemble smoothing**: Investigate chunking boundary issues causing inconsistent predictions.

## Appendix: Trace Configuration

### SmolVLA Trace Config
```json
{
  "checkpoint": "outputs/smolvla_pickplace_20251223_001238/checkpoints/checkpoints/last/pretrained_model",
  "task": "pick up the block and place it on the plate",
  "rtc_enabled": true,
  "execution_horizon": 5,
  "max_guidance_weight": 7.0,
  "action_queue_threshold": 10,
  "fps": 30,
  "duration": 30.0
}
```

### Trace Statistics
```json
{
  "total_steps": 896,
  "total_time_s": 30.0,
  "effective_rate_hz": 29.87,
  "num_inferences": 23,
  "timing": {
    "inference_ms": {"mean": 163, "p99": 442},
    "latency_ms": {"mean": 216, "p99": 1173}
  },
  "rtc": {
    "queue_size": {"mean": 24.2, "p50": 25.0}
  },
  "observation_freshness": {
    "observation_age_ms": {"mean": 816, "p50": 801}
  }
}
```

## Files Referenced

- SmolVLA trace (original): `outputs/inference_traces/trace_smolvla_20251224_123946/`
- SmolVLA trace (threshold=15, horizon=5): `outputs/inference_traces/trace_smolvla_20251224_143041/`
- Trace analysis script: `jdocs/scripts/analyze_inference_trace.py`
- SmolVLA RTC inference: `jdocs/scripts/infer_smolvla_rtc_trace.py`
- Gemini analysis: `jdocs/investigations/analysis_report_smolvla_traces.md`
