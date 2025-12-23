# SmolVLA Training and Inference Guide

This guide covers fine-tuning and deploying SmolVLA (Small Vision-Language-Action Model) on the SO-101 robot for pick-and-place tasks.

## Overview

SmolVLA is a lightweight Vision-Language-Action foundation model that:
- Uses a pretrained VLM backbone for visual understanding
- Accepts natural language task descriptions
- Predicts action chunks via flow matching
- Supports efficient fine-tuning with frozen vision encoder

### Key Differences from ACT

| Feature | ACT | SmolVLA |
|---------|-----|---------|
| Architecture | Transformer encoder-decoder + VAE | Pretrained VLM + flow matching |
| Language | Not required | Required (task description) |
| Action generation | VAE sampling | Flow matching denoising |
| LR schedule | Constant | Cosine decay with warmup |
| Fine-tuning | Train all parameters | Freeze vision, train expert only |
| Default steps | 100k | 20k |
| Default LR | 1e-5 | 1e-4 |

## Prerequisites

```bash
# Ensure LeRobot is installed with SmolVLA dependencies
pip install lerobot[smolvla]

# Verify dataset exists
ls datasets/pick_and_place/meta/info.json
```

## Training

### Quick Start

```bash
# Train with defaults (20k steps)
bash jdocs/scripts/train_smolvla_pickplace.sh

# Custom steps and batch size
MAX_STEPS=30000 BATCH_SIZE=32 bash jdocs/scripts/train_smolvla_pickplace.sh
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STEPS` | 20000 | Total training steps |
| `BATCH_SIZE` | 64 | Batch size (reduce if OOM) |
| `LEARNING_RATE` | 1e-4 | Peak learning rate |
| `CHUNK_SIZE` | 50 | Action chunk size |
| `WARMUP_STEPS` | 1000 | LR warmup steps |
| `DECAY_STEPS` | 30000 | Steps for cosine decay |

### Fine-tuning Strategy

SmolVLA uses a specialized fine-tuning approach:

```bash
# Default settings (recommended)
FREEZE_VISION=true      # Freeze vision encoder
TRAIN_EXPERT_ONLY=true  # Only train expert projection
TRAIN_STATE_PROJ=true   # Train state projector
```

This approach:
- Preserves pretrained visual representations
- Reduces memory usage significantly
- Trains faster while maintaining quality

### Learning Rate Schedule

SmolVLA uses cosine decay with linear warmup:

```
LR
  ^
  |   /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |  /                    \
  | /                      \
  |/                        \___
  +----------------------------> steps
    |     |                |
    0   1000             30000
      warmup           decay_steps
```

Parameters:
- `WARMUP_STEPS=1000`: Linear warmup from 0 to LR
- `DECAY_STEPS=30000`: Cosine decay to `DECAY_LR`
- `DECAY_LR=2.5e-6`: Final learning rate

### Resume Training

```bash
# Resume from checkpoint
RESUME_FROM=outputs/smolvla_pickplace_*/checkpoints/010000/pretrained_model \
  bash jdocs/scripts/train_smolvla_pickplace.sh
```

Note: When resuming, the script uses the original output directory to continue training.

### Recommended Settings by Dataset Size

| Episodes | Steps | Batch Size | GPU Memory |
|----------|-------|------------|------------|
| 20-30 | 15k | 64 | ~12GB |
| 40-60 | 20k | 64 | ~12GB |
| 80-100 | 30k | 64 | ~12GB |

### Memory Optimization

If you encounter OOM errors:

```bash
# Reduce batch size
BATCH_SIZE=32 bash jdocs/scripts/train_smolvla_pickplace.sh

# Or use gradient accumulation (if supported)
BATCH_SIZE=16 bash jdocs/scripts/train_smolvla_pickplace.sh
```

## Inference

### Running on Real Robot

```bash
# Basic inference (60 seconds, default task)
python jdocs/scripts/infer_smolvla_so101.py \
  --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model

# Custom task description
python jdocs/scripts/infer_smolvla_so101.py \
  --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \
  --task "pick up the red block"

# Extended duration
python jdocs/scripts/infer_smolvla_so101.py \
  --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \
  --duration 120
```

### Dry Run Mode

Test the pipeline without sending commands to the robot:

```bash
python jdocs/scripts/infer_smolvla_so101.py \
  --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \
  --dry-run
```

### Recording Inference

Record images during inference for debugging:

```bash
python jdocs/scripts/infer_smolvla_so101.py \
  --checkpoint outputs/smolvla_pickplace_*/checkpoints/020000/pretrained_model \
  --record
```

Images are saved to `jdocs/eval_images/inference_<timestamp>/`.

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint, -c` | Required | Path to checkpoint |
| `--task, -t` | "pick up the block and place it on the plate" | Task description |
| `--duration` | 60 | Max run duration (seconds) |
| `--dry-run` | False | Test mode without robot commands |
| `--record` | False | Save images during inference |
| `--hw-config` | `jdocs/scripts/so101_hardware.yaml` | Hardware config path |
| `--device` | cuda | Inference device |

## Troubleshooting

### Training Issues

**OOM Error**
```bash
# Reduce batch size
BATCH_SIZE=32 bash jdocs/scripts/train_smolvla_pickplace.sh
```

**Slow Training**
```bash
# Increase workers
NUM_WORKERS=8 bash jdocs/scripts/train_smolvla_pickplace.sh
```

**High Loss / Poor Performance**
- Ensure dataset has valid task descriptions
- Check that images are properly formatted
- Try increasing training steps
- Verify robot state normalization

### Inference Issues

**Model Not Found**
```bash
# Verify checkpoint path exists
ls outputs/smolvla_pickplace_*/checkpoints/
```

**Camera Not Opening**
- Check camera indices in `so101_hardware.yaml`
- Verify camera permissions: `ls -la /dev/video*`
- Test cameras: `v4l2-ctl --list-devices`

**Robot Connection Failed**
- Check serial port: `ls /dev/ttyACM*`
- Verify permissions: `sudo chmod 666 /dev/ttyACM1`
- Test with dry-run first

**Action Quality Poor**
- Ensure task description matches training data
- Check that cameras are positioned correctly
- Verify robot is in expected starting position

## Architecture Details

### SmolVLA Model Structure

```
Input:
├── Images (head, wrist) → Vision Encoder (frozen)
├── Robot State (6-DOF) → State Projector
└── Task Description → Language Encoder

Processing:
├── Vision features + State → Cross-attention
├── Language conditioning → Expert projection
└── Flow matching → Action prediction

Output:
└── Action chunk (50 steps, 6-DOF each)
```

### Flow Matching vs VAE

SmolVLA uses flow matching instead of VAE:
- **Training**: Learn a velocity field that transforms noise to actions
- **Inference**: Iteratively denoise for `num_steps` iterations
- **Advantage**: Better action quality, more stable training

### Fine-tuning Parameters

By default, SmolVLA freezes most of the model:

| Component | Trainable | Purpose |
|-----------|-----------|---------|
| Vision Encoder | No | Preserve pretrained features |
| Language Encoder | No | Preserve language understanding |
| Expert Projection | Yes | Adapt to robot actions |
| State Projector | Yes | Map robot state to embedding |

## Log Files

Training and inference logs are saved to `jdocs/logs/`:

```bash
# Training logs
jdocs/logs/train_smolvla_pickplace_YYYYMMDD_HHMMSS.log

# Inference logs
jdocs/logs/inference_smolvla_YYYYMMDD_HHMMSS.log
```

## Checkpoints

Checkpoints are saved during training:

```
outputs/smolvla_pickplace_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── 005000/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── train_config.json
│   ├── 010000/
│   │   └── pretrained_model/
│   ├── 015000/
│   │   └── pretrained_model/
│   └── 020000/
│       └── pretrained_model/
└── eval/
    └── ... (tensorboard logs)
```

## References

- [SmolVLA HuggingFace Documentation](https://huggingface.co/docs/lerobot/smolvla)
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
- [LeRobot Training Guide](https://huggingface.co/docs/lerobot/train)
- [Flow Matching for Action Prediction](https://arxiv.org/abs/2210.02747)
