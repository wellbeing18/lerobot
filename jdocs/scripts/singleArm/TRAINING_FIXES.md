# Pi0.5 Training Script Fixes

## Summary

This document explains the fixes applied to `train_pi05_pickplace.sh` to resolve the issue where LeRobot Pi0.5 LoRA finetuning produced non-functional behavior (robot swinging aimlessly despite training loss decreasing).

## Root Cause

The training failure was caused by **LoRA hyperparameter mismatch** between LeRobot's defaults and OpenPI's tested production values.

### The Problem

| Setting | LeRobot Default | OpenPI JAX Default | Effect |
|---------|-----------------|---------------------|--------|
| `lora_alpha` | **32** | **16** | 2x scaling difference |
| `lora_dropout` | **0.1** | **0.0** | Added noise to updates |
| LoRA Scaling | `32/16 = 2.0` | `16/16 = 1.0` | LeRobot updates 2x stronger |

**Symptom**: Training loss decreased normally, but the model produced degenerate behavior at inference time.

**Diagnosis**: The 2x LoRA scaling factor caused catastrophic forgetting - the model was overwriting pretrained knowledge too aggressively.

---

## Fixes Applied

### Fix 1: LoRA Alpha (Critical)

**Before:**
```bash
LORA_ALPHA="${LORA_ALPHA:-32}"      # Typically 2x rank
```

**After:**
```bash
LORA_ALPHA="${LORA_ALPHA:-16}"      # Match OpenPI: alpha = rank (scaling = 1.0)
```

**Why**: LoRA scaling factor is `alpha/rank`. OpenPI uses `alpha=rank` for a 1.0 scaling factor. LeRobot's default of `alpha=2*rank` created 2x stronger updates, leading to catastrophic forgetting of pretrained knowledge.

### Fix 2: LoRA Dropout (Medium Priority)

**Before:**
```bash
LORA_DROPOUT="${LORA_DROPOUT:-0.1}" # Regularization
```

**After:**
```bash
LORA_DROPOUT="${LORA_DROPOUT:-0.0}" # Match OpenPI: no dropout
```

**Why**: OpenPI's LoRA implementation uses no dropout. Adding 10% dropout to already small LoRA updates can:
- Add noise that destabilizes training
- Slow convergence for small datasets
- Create train/inference mismatch

### Fix 3: Documentation Updates

Added clear comments in the script explaining:
- Why `alpha=rank` is important (lines 35-39)
- The scaling factor calculation
- Reference to OpenPI's production-tested values

### Fix 4: Logging Improvements

Added logging to show LoRA configuration during training:
```bash
log "  LoRA Alpha:            ${LORA_ALPHA} (scaling=${LORA_ALPHA}/${LORA_RANK})"
log "  LoRA Dropout:          ${LORA_DROPOUT}"
```

---

## Technical Background

### What is LoRA Scaling?

LoRA (Low-Rank Adaptation) adds low-rank matrices A and B to pretrained weights:
```
W' = W + (alpha/rank) * BA
```

The `alpha/rank` term controls how much the LoRA updates affect the final weights:
- `alpha=rank` → scaling = 1.0 (standard)
- `alpha=2*rank` → scaling = 2.0 (updates are 2x stronger)

### Why OpenPI's Defaults Work

OpenPI's JAX implementation uses `alpha=16, rank=16` which has been tested in production for Pi0.5 finetuning. LeRobot's PEFT-based implementation chose different defaults (`alpha=32`) without testing against the OpenPI baseline.

---

## Verification

After applying these fixes, retrain and verify:

1. **Training loss should decrease** (same as before)
2. **Inference should produce functional behavior** (fixed)
3. **Robot should perform the trained task** rather than swinging aimlessly

---

## Additional Recommendations

If training still fails after these fixes, check:

1. **Quantile statistics**: Ensure `datasets/pick_and_place/meta/stats.json` contains `q01` and `q99` fields
2. **Action dimensions**: Your dataset action dim should match or be smaller than `max_action_dim=32`
3. **Camera names**: Dataset should use `observation.images.*` format

For detailed comparison between OpenPI and LeRobot implementations, see:
- `/home/jrobot/project/refs/openpi/LEROBOT_VS_OPENPI_INVESTIGATION.md`
