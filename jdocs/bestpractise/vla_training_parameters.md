# VLA Training Parameter Best Practices

This document summarizes research findings and best practices for training Vision-Language-Action (VLA) models, specifically XVLA and SmolVLA on bimanual robot tasks.

## Quick Reference

| Parameter | Pretraining | Fine-tuning | Notes |
|-----------|-------------|-------------|-------|
| weight_decay | 0.01 | 0.0 | Preserve pretrained weights during fine-tuning |
| learning_rate | 1e-4 | 1e-4 | Standard for VLA models |
| LR scheduler | cosine decay | cosine decay | With warmup |
| warmup_steps | 1000-2000 | 1000 | Gradual LR ramp-up |
| dtype | bfloat16 | bfloat16 | Required to avoid OOM |

---

## Weight Decay

### Key Insight: Pretraining vs Fine-tuning

**Pretraining** (training from scratch):
- Use `weight_decay=0.01` for regularization
- Helps prevent overfitting when learning new representations
- Source: X-VLA paper (arXiv:2510.10274, Appendix G)

**Fine-tuning** (adapting pretrained model):
- Use `weight_decay=0.0` to preserve pretrained weights
- Lower weight decay avoids over-regularizing learned representations
- Only increase if overfitting is observed

### Evidence from LeRobot Policies

| Policy | weight_decay | Type |
|--------|-------------|------|
| XVLA | 0.0 | VLA fine-tuning |
| SmolVLA | 1e-10 | VLA fine-tuning |
| OpenVLA | 0.0 | VLA fine-tuning |
| Pi0 | 0.01 | Full training |
| Pi05 | 0.01 | Full training |
| ACT | 1e-4 | Imitation learning |
| Diffusion | 1e-6 | Diffusion policy |
| VQ-BeT | 1e-6 | Behavior transformer |
| GROOT | 1e-5 | Foundation model |

### Recommendation

```bash
# Fine-tuning pretrained VLA (default)
WEIGHT_DECAY=0.0

# If overfitting observed (validation loss increases while training loss decreases)
WEIGHT_DECAY=1e-4  # Start small

# Full training from scratch
WEIGHT_DECAY=0.01
```

---

## Learning Rate

### Standard Settings

| Model | Learning Rate | Source |
|-------|--------------|--------|
| X-VLA | 1e-4 | Paper Appendix G |
| SmolVLA | 1e-4 | LeRobot default |
| OpenVLA | 2e-5 to 5e-4 | Depends on phase |

### Differential Learning Rates (XVLA)

XVLA uses differential LR for different model components:
- **VLM parameters**: `lr * 0.1` (1/10 of base LR)
- **Soft prompts**: `lr * 1.0` (full LR)
- **Policy transformer**: `lr * 1.0` (full LR)

This is handled automatically by `XVLAAdamWConfig`.

### LR Scheduling

**Cosine decay with warmup** is the industry standard:

```
LR
^
|    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
|   /                          \
|  /                            \
| /                              \___
|/________________________________\___ steps
  warmup    peak LR      decay    min_lr
```

Settings:
```bash
WARMUP_STEPS=1000          # Gradual ramp-up
DECAY_STEPS=${MAX_STEPS}   # Decay throughout training
DECAY_LR=2.5e-6            # Minimum LR at end
```

**Important**: Set `DECAY_STEPS` equal to `MAX_STEPS` for proper cosine decay throughout training.

---

## Batch Size and Training Steps

### Epoch Calculation

```
epochs = (steps × batch_size) / total_frames
```

Example with multitasks dataset (76,597 frames):
| Batch Size | Steps | Epochs |
|------------|-------|--------|
| 12 | 120,000 | ~19 |
| 16 | 120,000 | ~25 |
| 32 | 80,000 | ~33 |

### Memory Considerations

| Batch Size | GPU Memory | Notes |
|------------|------------|-------|
| 64 | OOM | Too large for single GPU |
| 32 | ~75GB | Borderline on 80GB A100 |
| 16 | ~50GB | May still OOM with full VLM training |
| 12 | ~40GB | Safe for XVLA bimanual |

### Recommendations by Dataset Size

| Episodes | Recommended Epochs | Strategy |
|----------|-------------------|----------|
| < 50 | 50-100 | Freeze vision encoder |
| 50-200 | 30-50 | Train vision, freeze language |
| > 200 | 15-30 | Full VLM training |

---

## Freezing Strategy

### XVLA Fine-tuning Modes

```bash
# Mode 1: Policy Only (< 50 episodes)
FREEZE_VISION=true
FREEZE_LANGUAGE=true

# Mode 2: Vision + Policy (50-200 episodes)
FREEZE_VISION=false
FREEZE_LANGUAGE=true

# Mode 3: Full VLM Training (> 200 episodes) - Recommended
FREEZE_VISION=false
FREEZE_LANGUAGE=false
```

### HuggingFace X-VLA Documentation Recommendation

> "When fine-tuning X-VLA for a new embodiment or task, we recommend **not freezing the VLM**, and also setting `policy.dtype=bfloat16` to not hit OOM errors."

---

## Optimizer Settings

### AdamW Configuration

| Parameter | XVLA Default | X-VLA Paper | Notes |
|-----------|-------------|-------------|-------|
| lr | 1e-4 | 1e-4 | Standard |
| betas | (0.9, 0.99) | (0.9, 0.95) | Minor difference |
| eps | 1e-8 | - | Standard |
| weight_decay | 0.0 | 0.01 (pretrain) | See above |
| grad_clip_norm | 10.0 | - | Gradient clipping |

---

## Precision (dtype)

**Always use `bfloat16` for VLA training:**

```bash
DTYPE=bfloat16
```

Reasons:
1. Prevents OOM errors
2. Faster training
3. Sufficient precision for VLA models
4. Native support on modern GPUs (A100, H100)

---

## Camera Configuration

### XVLA vs SmolVLA Camera Naming

| Original (Dataset) | XVLA | SmolVLA |
|-------------------|------|---------|
| observation.images.head | image | camera1 |
| observation.images.left_wrist | image2 | camera2 |
| observation.images.right_wrist | image3 | camera3 |

### XVLA 3-Camera Setup

```bash
# Camera rename mapping
--rename_map='{"observation.images.head":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'

# Input features for 3 cameras
--policy.input_features='{"observation.images.image":{"type":"VISUAL","shape":[3,256,256]},"observation.images.image2":{"type":"VISUAL","shape":[3,256,256]},"observation.images.image3":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[12]}}'

# Disable empty cameras
--policy.empty_cameras=0
--policy.num_image_views=3
```

---

## Domain ID and Soft Prompts

### Understanding Domain IDs

XVLA uses **soft prompts** indexed by `domain_id` to encode embodiment-specific variations:
- 30 domain slots available (0-29)
- Default training uses `domain_id=0`
- Different domain IDs can distinguish robot configurations

### Pretrained Domain IDs (xvla-base)

| Dataset | Domain ID |
|---------|-----------|
| Bridge | 0 |
| RT1 | 1 |
| Calvin | 2 |
| LIBERO | 3 |
| WidowX-Air | 4 |
| ... | ... |

### Recommendation

For custom fine-tuning, use `domain_id=0` (default) unless:
- You're fine-tuning multiple distinct embodiments
- You want to leverage multi-domain generalization

---

## Complete Training Command Example

```bash
# XVLA Bimanual Training (120k steps, batch 12)
nohup env \
    FREEZE_VISION=false \
    FREEZE_LANGUAGE=false \
    MAX_STEPS=120000 \
    BATCH_SIZE=12 \
    LEARNING_RATE=1e-4 \
    WEIGHT_DECAY=0.0 \
    WARMUP_STEPS=1000 \
    DTYPE=bfloat16 \
    bash jdocs/scripts/bimanual/train_xvla_bimanual.sh \
    > outputs/xvla_bimanual_training.log 2>&1 &
```

---

## Monitoring Training

### Key Metrics to Watch

| Metric | Healthy Range | Warning Signs |
|--------|--------------|---------------|
| loss | Decreasing | Sudden spikes, plateaus |
| grdn (gradient norm) | 20-60 | > 100 (instability) |
| lr | Should follow schedule | Stuck at warmup |
| updt_s | ~0.8s (XVLA) | > 2s (memory issues) |

### Training Progress Example

```
step:1K  loss:1.170  grdn:44.137  lr:9.5e-06  # Warmup phase
step:2K  loss:0.482  grdn:26.089  lr:1.0e-05  # Approaching peak LR
step:5K  loss:0.250  grdn:20.000  lr:1.0e-04  # Peak LR
step:50K loss:0.150  grdn:15.000  lr:5.0e-05  # Decay phase
```

---

## References

- [X-VLA Paper (arXiv:2510.10274)](https://arxiv.org/abs/2510.10274)
- [X-VLA GitHub](https://github.com/2toinf/X-VLA)
- [LeRobot X-VLA Documentation](https://huggingface.co/docs/lerobot/xvla)
- [OpenVLA-OFT Fine-Tuning](https://openvla-oft.github.io/)
- [OpenVLA GitHub](https://github.com/openvla/openvla)

---

## Changelog

- 2026-01-05: Initial version based on XVLA bimanual training research
