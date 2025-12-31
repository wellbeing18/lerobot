# Pi0.5 LoRA Training Fix Documentation

## Problem Summary

LeRobot Pi0.5 LoRA fine-tuning was producing non-functional behavior:
- **Symptom**: Robot swings aimlessly during inference
- **Training metrics**: Loss decreased normally (model appeared to learn)
- **Inference result**: Completely wrong actions

This pattern of "training looks good, inference fails" is characteristic of **catastrophic forgetting** caused by LoRA misconfiguration.

---

## Root Cause Analysis

### Investigation Method

Compared LeRobot's Pi0.5 LoRA implementation against OpenPI JAX's production-tested implementation. Key differences found in hyperparameters:

### Issue 1: LoRA Scaling Factor Mismatch (PRIMARY CAUSE)

| Parameter | LeRobot Default | OpenPI Default | Effect |
|-----------|-----------------|----------------|--------|
| `lora_rank` | 16 | 16 | Same |
| `lora_alpha` | **32** | **16** | 2x difference |
| **Scaling Factor** | `32/16 = 2.0` | `16/16 = 1.0` | **LeRobot 2x stronger** |

**How LoRA scaling works:**
```
actual_update = base_update * (alpha / rank)
```

With `alpha=32, rank=16`, every LoRA update was **2x stronger** than intended. This caused:
- Pretrained knowledge overwritten too quickly
- Training instability
- Mode collapse (robot learns a single degenerate behavior)

**Code locations:**
- LeRobot: `lerobot/src/lerobot/policies/pi05/configuration_pi05.py:81-82`
- OpenPI: `openpi/src/openpi/models/gemma.py:105`

### Issue 2: LoRA Dropout (SECONDARY CAUSE)

| Parameter | LeRobot Default | OpenPI Default |
|-----------|-----------------|----------------|
| `lora_dropout` | **0.1** | **0.0** |

10% dropout during training:
- Added noise to already small LoRA updates
- Made learning slower and less stable for small datasets
- OpenPI's tested configuration uses no dropout

---

## Fixes Applied

### File Modified
`/home/jrobot/project/lerobot/jdocs/scripts/singleArm/train_pi05_pickplace.sh`

### Change 1: LoRA Alpha (Line 134)

**Before:**
```bash
LORA_ALPHA="${LORA_ALPHA:-32}"      # Typically 2x rank
```

**After:**
```bash
# IMPORTANT: alpha=rank matches OpenPI's tested defaults (scaling factor = 1.0)
# Previous alpha=32 caused 2x scaling, leading to catastrophic forgetting
LORA_ALPHA="${LORA_ALPHA:-16}"      # Match OpenPI: alpha = rank (scaling = 1.0)
```

### Change 2: LoRA Dropout (Line 136)

**Before:**
```bash
LORA_DROPOUT="${LORA_DROPOUT:-0.1}" # Regularization
```

**After:**
```bash
# IMPORTANT: OpenPI uses no dropout in LoRA - dropout can destabilize small LoRA updates
LORA_DROPOUT="${LORA_DROPOUT:-0.0}" # Match OpenPI: no dropout
```

### Change 3: Header Documentation (Lines 35-39)

Added clear documentation about the LoRA hyperparameters:
```bash
# LoRA Hyperparameters (IMPORTANT - matches OpenPI's tested defaults):
#   - lora_rank=16, lora_alpha=16 -> scaling factor = 1.0
#   - lora_dropout=0.0 (no dropout)
#   - These match OpenPI JAX's production-tested values
#   - Previous defaults (alpha=32, dropout=0.1) caused training failures
```

### Change 4: Logging Enhancement (Lines 236-237)

Updated logging to show LoRA scaling calculation:
```bash
log "  LoRA Alpha:            ${LORA_ALPHA} (scaling=${LORA_ALPHA}/${LORA_RANK})"
log "  LoRA Dropout:          ${LORA_DROPOUT}"
```

---

## How to Use

### Fresh Training (Recommended)

```bash
# Navigate to lerobot project
cd /home/jrobot/project/lerobot

# Run with fixed defaults
bash jdocs/scripts/singleArm/train_pi05_pickplace.sh
```

### Custom Configuration

```bash
# Override any parameter via environment variables
MAX_STEPS=6000 BATCH_SIZE=4 bash jdocs/scripts/singleArm/train_pi05_pickplace.sh
```

### Verify Fixes Are Active

Check the training log output for:
```
Configuration:
  ...
  Use LoRA:              true
  LoRA Rank:             16
  LoRA Alpha:            16 (scaling=16/16)
  LoRA Dropout:          0.0
```

The scaling should show `16/16 = 1.0`, not `32/16 = 2.0`.

---

## Additional Recommendations

### 1. Verify Quantile Statistics

Pi0.5 requires quantile stats for state tokenization:

```bash
# Check if stats exist
cat datasets/pick_and_place/meta/stats.json | jq 'keys'
# Should include: q01, q99, mean, std

# If missing, compute them:
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=pick_and_place
```

### 2. Consider Longer Training

With the corrected (lower) scaling factor, you may need more steps:

```bash
MAX_STEPS=6000 bash jdocs/scripts/singleArm/train_pi05_pickplace.sh
```

### 3. Monitor for Overfitting

LeRobot lacks training augmentations that OpenPI has (crop, rotation, color jitter). Watch for:
- Training loss continuing to decrease
- Validation/inference performance degrading

If overfitting occurs, try:
- Fewer training steps
- Lower learning rate: `LEARNING_RATE=1.0e-5`

---

## Technical Background

### Why LoRA Scaling Matters

LoRA (Low-Rank Adaptation) works by adding small trainable matrices to frozen pretrained weights:

```
output = pretrained_weight(x) + (alpha/rank) * lora_B(lora_A(x))
```

The `alpha/rank` ratio controls how much influence the LoRA adaptation has:
- Too high (2.0): Overwrites pretrained knowledge too fast
- Too low (0.5): Learning too slow, may not adapt
- Just right (1.0): Balanced learning, preserves base capabilities

### OpenPI vs LeRobot LoRA Comparison

| Aspect | OpenPI JAX | LeRobot PyTorch |
|--------|------------|-----------------|
| Implementation | Native custom | HuggingFace PEFT |
| Default alpha | 16 (= rank) | 32 (= 2x rank) |
| Default dropout | 0.0 | 0.1 |
| Tested in production | Yes | No (community defaults) |

Note: OpenPI PyTorch does NOT support LoRA (only full fine-tuning). Only OpenPI JAX has native LoRA support.

---

## References

- Investigation report: `/home/jrobot/project/refs/openpi/LEROBOT_VS_OPENPI_INVESTIGATION.md`
- OpenPI LoRA config: `openpi/src/openpi/models/gemma.py:55-109`
- LeRobot LoRA config: `lerobot/src/lerobot/policies/pi05/configuration_pi05.py:81-83`
- PEFT LoRA documentation: https://huggingface.co/docs/peft/conceptual_guides/lora
