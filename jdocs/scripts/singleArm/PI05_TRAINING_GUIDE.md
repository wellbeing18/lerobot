# Pi0.5 Training and Inference Guide

This guide covers fine-tuning and deploying Pi0.5 (Physical Intelligence's Vision-Language-Action Model) on the SO-101 robot for pick-and-place tasks.

## Overview

Pi0.5 is a powerful VLA model with open-world generalization capabilities:
- Uses PaliGemma (2B) for vision-language understanding
- Uses Gemma Expert (300M) for action prediction
- Accepts natural language task descriptions
- Predicts action chunks via flow matching
- Supports Real-Time Chunking (RTC) for smooth motion

### Key Differences from SmolVLA

| Feature | SmolVLA | Pi0.5 |
|---------|---------|-------|
| Model Size | ~0.5B params | ~3.5-4B params |
| Architecture | Small VLM + flow matching | PaliGemma 2B + Gemma Expert |
| Default LR | 1e-4 | 2.5e-5 |
| Default Steps | 20k | 3k |
| Normalization | Custom | QUANTILES (or MEAN_STD) |
| Camera Names | camera1, camera2 | base_0_rgb, left_wrist_0_rgb |
| Memory Optimization | Not required | gradient_checkpointing required |
| VRAM (24GB) | batch_size=32-64 | batch_size=8-16 |

## Prerequisites

```bash
# Install LeRobot with Pi0.5 dependencies
pip install -e ".[pi]"

# Verify dataset exists
ls datasets/pick_and_place/meta/info.json
```

## Training

### Quick Start

```bash
# Train with defaults (3k steps, optimized for 24GB VRAM)
bash jdocs/scripts/train_pi05_pickplace.sh

# Custom steps and batch size
MAX_STEPS=6000 BATCH_SIZE=16 bash jdocs/scripts/train_pi05_pickplace.sh
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STEPS` | 3000 | Total training steps |
| `BATCH_SIZE` | 8 | Batch size (8-16 for 24GB VRAM) |
| `LEARNING_RATE` | 2.5e-5 | Peak learning rate |
| `CHUNK_SIZE` | 50 | Action chunk size |
| `WARMUP_STEPS` | 100 | LR warmup steps |
| `DECAY_STEPS` | 3000 | Steps for cosine decay |

### 24GB VRAM Optimization

Pi0.5 is a larger model requiring memory optimization:

```bash
# Memory-critical settings (enabled by default)
GRADIENT_CHECKPOINTING=true  # Reduces VRAM by ~30-50%
DTYPE=bfloat16               # Half precision (~50% reduction)
COMPILE_MODEL=true           # Better memory efficiency
BATCH_SIZE=8                 # Safe for 24GB VRAM
```

**Estimated VRAM Usage:**

| Configuration | VRAM Usage |
|---------------|------------|
| batch_size=8, gradient_checkpointing=true | ~18-20GB |
| batch_size=16, gradient_checkpointing=true | ~22-24GB |
| batch_size=8, gradient_checkpointing=false | ~28-32GB (OOM) |

### Normalization

Pi0.5 uses QUANTILES normalization by default. If your dataset lacks quantile stats:

```bash
# Option 1: Generate quantile stats
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=pick_and_place

# Option 2: Use MEAN_STD normalization instead
NORMALIZATION_MODE=MEAN_STD bash jdocs/scripts/train_pi05_pickplace.sh
```

### Learning Rate Schedule

Pi0.5 uses cosine decay with linear warmup (auto-scaled):

```
LR
  ^
  |   /‾‾‾‾‾‾‾‾‾‾\
  |  /            \
  | /              \
  |/                \___
  +-------------------> steps
    |    |          |
    0   100       3000
      warmup    decay_steps
```

Parameters:
- `WARMUP_STEPS=100`: Linear warmup from 0 to LR
- `DECAY_STEPS=3000`: Cosine decay to `DECAY_LR`
- `DECAY_LR=2.5e-6`: Final learning rate (10x lower than peak)

### Resume Training

```bash
# Resume from checkpoint
RESUME_FROM=outputs/pi05_pickplace_*/checkpoints/002000/pretrained_model \
  bash jdocs/scripts/train_pi05_pickplace.sh
```

### Recommended Settings by Dataset Size

| Episodes | Steps | Batch Size | Notes |
|----------|-------|------------|-------|
| 20-30 | 3k | 8 | Default, safe for 24GB |
| 40-60 | 4-5k | 8-16 | Monitor VRAM |
| 80-100 | 6k | 8-16 | May need longer training |

## Inference

### Basic Inference

```bash
# Basic inference (60 seconds)
python jdocs/scripts/infer_pi05_so101.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model

# Custom task description
python jdocs/scripts/infer_pi05_so101.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --task "pick up the red block"

# Extended duration
python jdocs/scripts/infer_pi05_so101.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --duration 120
```

### RTC Inference (Smooth Motion)

Real-Time Chunking (RTC) provides smoother, more reactive control:

```bash
# RTC inference (recommended for real robot)
python jdocs/scripts/infer_pi05_rtc.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --task "pick up the block and place it on the plate"

# Custom RTC parameters
python jdocs/scripts/infer_pi05_rtc.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --execution-horizon 15 \
    --max-guidance-weight 10.0 \
    --fps 30 \
    --duration 120
```

### RTC vs Standard Inference

| Aspect | Standard | RTC |
|--------|----------|-----|
| Motion | Jerky transitions between chunks | Smooth, continuous |
| Latency | Pauses during inference | No pauses (async) |
| Reactivity | Limited | High |
| Complexity | Simple | Multi-threaded |

### RTC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--execution-horizon` | 10 | Steps to blend with previous chunk (8-15 typical) |
| `--max-guidance-weight` | 10.0 | How strongly to enforce smooth transitions |
| `--action-queue-threshold` | 30 | Request new chunk when queue size <= this |
| `--fps` | 30 | Control loop frequency |
| `--no-rtc` | False | Disable RTC (use standard chunking) |

### Dry Run Mode

Test the pipeline without sending commands to the robot:

```bash
# Basic inference dry run
python jdocs/scripts/infer_pi05_so101.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --dry-run

# RTC inference dry run
python jdocs/scripts/infer_pi05_rtc.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
    --dry-run
```

### Recording Inference

Record images during inference for debugging:

```bash
python jdocs/scripts/infer_pi05_so101.py \
    --checkpoint outputs/pi05_pickplace_*/checkpoints/003000/pretrained_model \
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
# Reduce batch size (most effective)
BATCH_SIZE=4 bash jdocs/scripts/train_pi05_pickplace.sh

# Ensure gradient checkpointing is enabled
GRADIENT_CHECKPOINTING=true bash jdocs/scripts/train_pi05_pickplace.sh
```

**Quantile Stats Missing**
```bash
# Option 1: Generate quantile stats
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=pick_and_place

# Option 2: Use MEAN_STD normalization
NORMALIZATION_MODE=MEAN_STD bash jdocs/scripts/train_pi05_pickplace.sh
```

**Slow Training**
```bash
# Enable model compilation
COMPILE_MODEL=true bash jdocs/scripts/train_pi05_pickplace.sh

# Reduce workers if CPU bound
NUM_WORKERS=2 bash jdocs/scripts/train_pi05_pickplace.sh
```

**High Loss / Poor Performance**
- Ensure dataset has valid task descriptions
- Check that images are properly formatted
- Try increasing training steps
- Verify camera mapping is correct

### Inference Issues

**Model Not Found**
```bash
# Verify checkpoint path exists
ls outputs/pi05_pickplace_*/checkpoints/
```

**Camera Not Opening**
- Check camera indices in `so101_hardware.yaml`
- Verify camera permissions: `ls -la /dev/video*`
- Test cameras: `v4l2-ctl --list-devices`

**Robot Connection Failed**
- Check serial port: `ls /dev/ttyACM*`
- Verify permissions: `sudo chmod 666 /dev/ttyACM1`
- Test with dry-run first

**RTC Action Queue Empty**
- Increase `--action-queue-threshold`
- Model inference may be too slow for FPS
- Try reducing `--fps` to 20

## Architecture Details

### Pi0.5 Model Structure

```
Input:
├── Images (head, wrist) → SigLIP Vision Encoder → PaliGemma
├── Robot State (6-DOF) → Discretization → Token embedding
└── Task Description → Tokenizer → PaliGemma

Processing:
├── PaliGemma (2B) → Vision-language understanding
├── Gemma Expert (300M) → Action prediction
└── Flow matching → Iterative denoising

Output:
└── Action chunk (50 steps, 6-DOF each)
```

### Camera Name Mapping

Pi0.5 uses flexible camera names. Training maps:

| Dataset Camera | Pi0.5 Name |
|----------------|------------|
| head | base_0_rgb |
| left_wrist | left_wrist_0_rgb |

This is configured via `--rename_map` in training and hardcoded in inference scripts.

### Flow Matching

Pi0.5 uses flow matching for action generation:
- **Training**: Learn a velocity field that transforms noise to actions
- **Inference**: Iteratively denoise for `num_inference_steps` iterations (default: 10)
- **Advantage**: Better action quality, more stable training

### Real-Time Chunking (RTC)

RTC improves real-time inference by:
1. Asynchronously generating the next chunk while executing current one
2. Guiding new chunks to blend smoothly with already-executed actions
3. Using prefix attention to enforce consistency in overlap region

```
Previous Chunk:  [==executed==][--remaining--]
                              ↓ (blend region)
New Chunk:       [--guided---][====new====]
                 ↓
Result:          Smooth, continuous motion
```

## Log Files

Training and inference logs are saved to `jdocs/logs/`:

```bash
# Training logs
jdocs/logs/train_pi05_pickplace_YYYYMMDD_HHMMSS.log

# Inference logs
jdocs/logs/inference_pi05_YYYYMMDD_HHMMSS.log
jdocs/logs/inference_pi05_rtc_YYYYMMDD_HHMMSS.log
```

## Checkpoints

Checkpoints are saved during training:

```
outputs/pi05_pickplace_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── 001000/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── train_config.json
│   ├── 002000/
│   │   └── pretrained_model/
│   └── 003000/
│       └── pretrained_model/
└── eval/
    └── ... (tensorboard logs)
```

## References

- [Pi0.5 HuggingFace Documentation](https://huggingface.co/docs/lerobot/pi05)
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi05)
- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- [RTC Documentation](https://huggingface.co/docs/lerobot/rtc)
- [LeRobot Training Guide](https://huggingface.co/docs/lerobot/train)
