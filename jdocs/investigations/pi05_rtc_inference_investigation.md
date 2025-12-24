# Pi0.5 RTC Inference Investigation

**Date**: 2024-12-24
**Trace Analyzed**: `outputs/inference_traces/trace_pi05_20251224_124451`
**Checkpoint**: `outputs/pi05_pickplace_20251223_144340/checkpoints/last/pretrained_model`

## Current Status

| Status | Action |
|--------|--------|
| RTC parameter tuning | Tested - **Did NOT fix** the issue |
| Retraining with MEAN_STD | **In Progress** - LORA_RANK=64, BATCH_SIZE=16 |

**Next step**: Test new checkpoint after retraining completes (~10K steps)

## Reported Symptoms

**"The arm goes crazy swinging in the air up and down without actually doing anything meaningful."**

## Executive Summary

Investigation reveals the Pi0.5 model is producing **chaotic, oscillatory action predictions** that do not form coherent motion trajectories. The model generates action buffers with 32-36 direction reversals per 50-action chunk, causing the robot arm to oscillate wildly instead of performing purposeful movements.

| Issue | Evidence |
|-------|----------|
| Chaotic action buffers | 32-36 sign changes in 49 steps (shoulder_lift/elbow) |
| Massive within-buffer oscillations | Velocity swings up to 45-48 degrees between consecutive actions |
| State-action mismatch | First predicted action is +61 degrees off from actual state |
| All joints affected | 732 large oscillations (>5 deg) detected in 392 action steps |

**Root Causes** (Combined Analysis with Gemini):
1. **Quantile Normalization Sensitivity**: Small errors in normalized space map to massive physical jumps (~40 deg) at distribution edges
2. **Pilot-Induced Oscillation (PIO)**: 313ms latency creates feedback instability - corrections arrive when arm has already moved
3. **Insufficient LoRA Adaptation**: Only 1% of parameters trained, frozen weights dominate behavior
4. **Visual Domain Gap**: Frozen Vision Encoder may produce out-of-distribution embeddings for different camera views

## Detailed Analysis

### 1. Action Oscillation Statistics

**Finding**: The model predicts highly oscillatory action sequences that reverse direction every 1-2 steps.

| Joint | Sign Changes (49 steps) | Max Jump | Mean Velocity | Std Velocity |
|-------|------------------------|----------|---------------|--------------|
| shoulder_pan | 151 | 10.0 deg | 0.00 | 3.33 |
| shoulder_lift | 151 | **45.0 deg** | -0.03 | **14.47** |
| elbow | 150 | **52.3 deg** | +0.01 | **14.55** |
| wrist_1 | 179 | 15.5 deg | 0.00 | 4.98 |
| wrist_2 | 173 | 10.4 deg | 0.00 | 2.30 |
| gripper | 165 | 12.1 deg | 0.00 | 3.37 |

**Key Observation**:
- Sign changes in shoulder_lift/elbow (151/150) in 392 steps = direction reversal every ~2.6 steps
- This means the arm is constantly changing direction, creating the "swinging" behavior

### 2. Within-Buffer Oscillation Analysis

**Finding**: Each 50-action prediction buffer contains severe internal oscillations.

Sample from Step 49 inference buffer (shoulder_lift velocities):
```
[-20.8, +5.1, +10.9, -1.1, +14.7, -31.0, +19.8, +21.6, -21.4, -0.7,
 -7.1, +13.1, -6.9, +2.2, +12.2, -11.3, -6.3, +4.7, +17.5, -14.4, ...]
```

| Joint | Sign Changes in 49-step Buffer | Max Velocity in Buffer |
|-------|-------------------------------|----------------------|
| shoulder_lift | **32** | 31.0 deg |
| elbow | **36** | 48.0 deg |

**Implication**: The model's flow matching inference is not producing smooth trajectories - it's generating quasi-random noise-like predictions.

### 3. State-Action Mismatch at Inference

**Finding**: The model's first predicted action often dramatically differs from the current robot state.

| Inference Step | State→First Action Delta |
|----------------|-------------------------|
| Step 49 | shoulder_pan: +2.4, **shoulder_lift: +61.0**, **elbow: -76.4**, wrist_1: +6.6, wrist_2: +2.7 |
| Step 58 | shoulder_pan: -2.8, shoulder_lift: +17.6, elbow: -29.6, wrist_1: +1.2, wrist_2: +0.6 |
| Step 77 | shoulder_pan: +0.9, shoulder_lift: +10.5, elbow: +2.7, wrist_1: -2.1, wrist_2: -1.0 |
| Step 105 | shoulder_pan: -4.1, shoulder_lift: +13.7, elbow: -2.1, wrist_1: +3.3, wrist_2: +1.5 |

**At Step 49**: The model predicts shoulder_lift at -37.6 when actual state is -98.6 (a +61 degree error!). This indicates the model is not conditioning on the current state properly.

### 4. Large Oscillation Count

**Finding**: 732 large action changes (>5 degrees in any joint) detected across 392 action steps.

```
Step   58: elbow           change= +12.1
Step   59: shoulder_lift   change= -15.0
Step   59: elbow           change= -11.3
Step   60: shoulder_lift   change= +19.4
Step   60: elbow           change= +11.8
Step   61: shoulder_lift   change=  -8.8
Step   61: elbow           change= -16.4
Step   62: shoulder_lift   change= +22.7
Step   62: elbow           change=  +9.3
Step   63: shoulder_lift   change= -17.2
...
```

**Pattern**: Large oscillations cluster around inference boundaries but occur throughout execution, confirming the model itself produces chaotic predictions.

### 5. Training Analysis

**Training Configuration**:
```
- Model: Pi0.5 (PaliGemma 2B + Gemma Expert 300M)
- LoRA: rank=32, alpha=64, dropout=0.1
- Training steps: 10,000 (completed)
- Learning rate: 2.5e-5 → 2.5e-6 (cosine decay)
- Chunk size: 50 actions
- Normalization: QUANTILES
```

**Training Log (final steps)**:
```
step:10K loss:0.027-0.030 grdn:0.9-1.0 lr:2.5e-06
```

**Observation**: Training loss appears reasonable, but low MSE loss in flow matching doesn't guarantee coherent action sequences. The model may have learned to match noise statistics without learning trajectory structure.

### 6. Comparison with SmolVLA

| Metric | Pi0.5 | SmolVLA |
|--------|-------|---------|
| Model size | ~4B params | ~0.5B params |
| LoRA | Yes (1% trainable) | No (full fine-tune) |
| Inference time | 314ms | 163ms |
| Sign changes per step | 151 | Much lower |
| Behavior | Chaotic swinging | Oscillating but directed |

**Key Difference**: Pi0.5 uses LoRA which only updates ~1% of parameters, while SmolVLA does full fine-tuning. This may explain why Pi0.5 hasn't adapted to the action distribution.

## Root Cause Analysis

### Root Cause 1: Quantile Normalization Sensitivity (from Gemini)

**Finding**: Quantile normalization amplifies small model errors into massive physical movements.

**Dataset Action Ranges**:
- `shoulder_lift`: [-100, +70] degrees
- `elbow`: [-82, +100] degrees

**Mechanism**:
- Quantile normalization maps data to uniform/Gaussian range [-1, 1]
- Small prediction errors (e.g., 0.95 vs 0.90 in normalized space) near distribution edges
- These map to **massive jumps** (20-40 degrees) in denormalized physical space
- The observed +/-40 degree jumps match this hypothesis exactly

**Evidence**: Trace shows velocity swings of 31-48 degrees, consistent with edge-of-distribution errors being magnified.

### Root Cause 2: Pilot-Induced Oscillation (PIO) from Latency (from Gemini)

**Finding**: High inference latency (313ms mean, 600ms p99) creates feedback instability.

**Mechanism**:
1. At 30Hz, 313ms latency = **10-20 frame delay**
2. Robot executes action based on where it *was* 0.3-1.0s ago
3. When arm starts swinging, model sees the swing 0.5s later
4. Commands correction, but arm is already somewhere else
5. Classic **feedback oscillation** pattern

**Evidence**: Oscillations persist and amplify over time rather than damping out.

### Root Cause 3: Insufficient LoRA Adaptation

**Evidence**:
1. Model produces chaotic predictions despite low training loss
2. First actions are massively offset from current state (+61 deg)
3. Action buffers contain 32-36 reversals in 49 steps (nearly random)

**Mechanism**:
- LoRA only trains ~1% of parameters (40M out of 4B)
- The frozen pretrained weights dominate inference behavior
- The model hasn't learned the specific action distribution for this task
- Flow matching produces multi-modal outputs that appear as oscillations

### Root Cause 4: Visual Domain Gap (from Gemini)

**Finding**: Frozen Vision Encoder may produce garbage embeddings for different camera views.

**Configuration Issue**:
- Dataset has: `head`, `left_wrist` cameras
- Pi0.5 pretrained on: `base_rgb`, `wrist_rgb` (different perspectives)
- LoRA fine-tuning keeps Vision Encoder **frozen**

**Mechanism**:
- If `head` camera perspective differs significantly from pre-training `base` view
- Frozen Vision Encoder produces "out of distribution" embeddings
- Garbage visual embeddings → Garbage/noisy action predictions

### Contributing Factors

1. **Flow Matching Multi-modality**: The denoising process may converge to different modes at each step, creating discontinuous predictions

2. **Chunk Size Too Large**: 50-action chunks require long-horizon coherent prediction which the model hasn't learned

## RTC Parameter Tuning Experiments

### Experiment Results (2024-12-24)

**Attempted short-term RTC fixes** - adjusting parameters without retraining:

| Config | threshold | horizon | guidance | Result |
|--------|-----------|---------|----------|--------|
| Original | 30 | 10 | 10 | Baseline - crazy swinging |
| Attempt 1 | 20 | 5 | 15 | **No improvement** - still swinging |

**Conclusion**: RTC parameter tuning does NOT fix the Pi0.5 swinging issue.

**Reason**: The problem is not observation staleness (as with SmolVLA) - it's that the model itself produces chaotic predictions with 32-36 reversals per action buffer. No amount of RTC tuning can fix predictions that are fundamentally oscillatory.

**Required fix**: Retrain with MEAN_STD normalization to eliminate quantile sensitivity.

## Recommendations

### Immediate Fixes (No Retraining)

1. **Reduce Latency** (from Gemini - Critical):
   ```bash
   # Reduce inference steps (quality vs speed trade-off)
   NUM_INFERENCE_STEPS=4  # Down from 10

   # Reduce chunk size for faster inference
   CHUNK_SIZE=16  # Down from 50
   ```

2. **Add Action Smoothing/Damping** (from Gemini):
   ```python
   # In inference script, add low-pass filter
   smoothed = alpha * new_action + (1 - alpha) * prev_action
   ```

### Short-term Fixes (Requires Retraining)

1. **Use MEAN_STD Normalization with Increased LoRA Rank** (Recommended - In Progress):
   ```bash
   nohup bash -c 'NORMALIZATION_MODE=MEAN_STD LORA_RANK=64 LORA_ALPHA=128 BATCH_SIZE=16 \
       bash jdocs/scripts/train_pi05_pickplace.sh' > nohup_pi05_meanstd.out 2>&1 &
   ```

   **Why this combination**:
   - MEAN_STD normalization is linear and more robust than Quantiles for continuous control
   - LORA_RANK=64 (up from 32) = more trainable parameters for better adaptation
   - BATCH_SIZE=16 (up from 8) = more stable gradients, fits in ~20GB VRAM

   **Resume command** (if needed):
   ```bash
   nohup bash -c 'NORMALIZATION_MODE=MEAN_STD LORA_RANK=64 LORA_ALPHA=128 BATCH_SIZE=16 RESUME=true \
       bash jdocs/scripts/train_pi05_pickplace.sh' > nohup_pi05_meanstd.out 2>&1 &
   ```

   **Check progress**:
   ```bash
   tail -f nohup_pi05_meanstd.out
   ```

2. **Alternative: Reduce Chunk Size** (if MEAN_STD doesn't fully fix):
   ```bash
   CHUNK_SIZE=16 N_ACTION_STEPS=16 NORMALIZATION_MODE=MEAN_STD bash jdocs/scripts/train_pi05_pickplace.sh
   ```

3. **Alternative: Try Full Fine-tuning** (requires more VRAM):
   ```bash
   USE_LORA=false BATCH_SIZE=4 bash jdocs/scripts/train_pi05_pickplace.sh
   ```
   *Note*: Requires ~48GB VRAM or gradient accumulation

### Medium-term Fixes

1. **Verify Camera Mapping** (from Gemini):
   - Ensure `head` camera is treated as primary global view
   - Consider explicit `--rename_map` to `camera_0`, `camera_1`

2. **Increase Training Steps**: The model may need more training to adapt
   ```bash
   MAX_STEPS=20000 bash jdocs/scripts/train_pi05_pickplace.sh
   ```

3. **Verify Dataset Quality**: Ensure training demonstrations have smooth, coherent trajectories

### Long-term Fixes

1. **Use Different Base Model**: Consider SmolVLA which achieved better results with full fine-tuning

2. **Action Chunking Curriculum**: Start with small chunks (5-10) and gradually increase

3. **Unfreeze Vision Encoder**: Train with vision encoder unfrozen to adapt to camera views

## Appendix: Trace Configuration

### Pi0.5 Trace Config
```json
{
  "checkpoint": "outputs/pi05_pickplace_20251223_144340/checkpoints/last/pretrained_model",
  "task": "pick up the block and place it on the plate",
  "rtc_enabled": true,
  "execution_horizon": 10,
  "max_guidance_weight": 10.0,
  "action_queue_threshold": 30,
  "fps": 30,
  "duration": 15.0
}
```

### Model Configuration
```json
{
  "type": "pi05",
  "chunk_size": 50,
  "n_action_steps": 50,
  "num_inference_steps": 10,
  "use_lora": true,
  "lora_rank": 32,
  "lora_alpha": 64,
  "normalization_mapping": {
    "ACTION": "QUANTILES",
    "STATE": "QUANTILES"
  }
}
```

### Trace Statistics
```json
{
  "total_steps": 449,
  "total_time_s": 15.02,
  "effective_rate_hz": 29.89,
  "num_inferences": 14,
  "timing": {
    "inference_ms": {"mean": 314, "max": 661},
    "latency_ms": {"mean": 393, "max": 1611}
  }
}
```

## Files Referenced

- Pi0.5 trace: `outputs/inference_traces/trace_pi05_20251224_124451/`
- Pi0.5 checkpoint: `outputs/pi05_pickplace_20251223_144340/checkpoints/last/pretrained_model`
- Training script: `jdocs/scripts/train_pi05_pickplace.sh`
- Training log: `jdocs/logs/train_pi05_pickplace_20251223_144340.log`
- Inference script: `jdocs/scripts/infer_pi05_rtc_trace.py`
- Pi0.5 policy: `src/lerobot/policies/pi05/modeling_pi05.py`
- Gemini analysis: `jdocs/investigations/analysis_report_pi05_root_cause.md`
