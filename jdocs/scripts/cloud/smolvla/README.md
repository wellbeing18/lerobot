# SmolVLA Cloud Training for Bimanual SO-101

This directory contains scripts for training SmolVLA on bimanual tasks using cloud GPU instances.

## Quick Start

```bash
# 1. SSH to cloud instance
ssh user@cloud-instance

# 2. Clone and setup LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"

# 3. Login to HuggingFace (for dataset access)
huggingface-cli login

# 4. Run training
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

## Training Modes

SmolVLA supports different fine-tuning strategies:

| Mode | Params | Command | Use Case |
|------|--------|---------|----------|
| **Vision + Expert** | ~185M | `FREEZE_VISION=false TRAIN_EXPERT_ONLY=true` | Recommended for bimanual |
| Expert Only | ~100M | `FREEZE_VISION=true TRAIN_EXPERT_ONLY=true` | Fast, less memory |
| Full VLM | ~450M | `FREEZE_VISION=false TRAIN_EXPERT_ONLY=false` | Large datasets only |

**Recommended for bimanual: Vision + Expert mode** (default)

- Unfrozen vision encoder allows visual-spatial learning important for bimanual coordination
- Frozen language model preserves task understanding capabilities
- ~185M trainable parameters (~41% of total)

## GPU Requirements

| GPU | VRAM | Batch Size | Est. Time (50k steps) |
|-----|------|------------|----------------------|
| RTX 4090 | 24GB | 8-16 | ~15-20 hours |
| A100 40GB | 40GB | 32-64 | ~6-10 hours |
| A100 80GB | 80GB | 64-128 | ~4-6 hours |
| H100 | 80GB | 64-128 | ~3-5 hours |

## Usage Examples

### Basic Training (HuggingFace Hub Dataset)

```bash
# Uses default dataset: jasmine314342/picknplace-bimanual-464
bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Custom HuggingFace Dataset

```bash
DATASET_REPO_ID=your-username/your-dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Local Dataset

```bash
DATASET_PATH=/data/my_dataset \
DATASET_NAME=my_dataset \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Adjust for GPU Memory

```bash
# For A100 40GB
BATCH_SIZE=32 MAX_STEPS=50000 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# For RTX 4090 (24GB)
BATCH_SIZE=8 MAX_STEPS=50000 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh

# For H100 / A100 80GB
BATCH_SIZE=64 MAX_STEPS=50000 bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

### Enable W&B Logging

```bash
WANDB_ENABLE=true WANDB_PROJECT=smolvla-bimanual \
    bash jdocs/scripts/cloud/smolvla/train_smolvla_bimanual.sh
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_REPO_ID` | `jasmine314342/picknplace-bimanual-464` | HuggingFace dataset |
| `DATASET_PATH` | (none) | Local dataset path |
| `DATASET_NAME` | (none) | Local dataset name |
| `PRETRAINED_MODEL` | `lerobot/smolvla_base` | Base model |
| `BATCH_SIZE` | 32 | Batch size |
| `MAX_STEPS` | 50000 | Training steps |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `WARMUP_STEPS` | 1000 | LR warmup steps |
| `FREEZE_VISION` | false | Freeze vision encoder |
| `TRAIN_EXPERT_ONLY` | true | Only train expert (not language) |
| `GRADIENT_CHECKPOINTING` | true | Memory optimization |
| `SAVE_STEPS` | 5000 | Checkpoint frequency |
| `NUM_WORKERS` | 8 | Data loader workers |
| `WANDB_ENABLE` | false | W&B logging |

## Post-Training

### 1. Download Checkpoints

```bash
# From cloud to local
rsync -avz user@cloud:/path/to/outputs/smolvla_bimanual_xxx/checkpoints ./checkpoints/
```

### 2. Run Inference Test

```bash
python jdocs/scripts/cloud/smolvla/infer_smolvla_bimanual.py \
    --checkpoint ./checkpoints/last/pretrained_model \
    --task "Use left arm to pick up the banana and place it on the plate"
```

### 3. Push to HuggingFace Hub

```bash
huggingface-cli upload your-username/smolvla-bimanual \
    ./checkpoints/last/pretrained_model
```

### 4. Deploy on Robot

```bash
# On robot machine
python -m lerobot.scripts.control_robot \
    --policy.path=your-username/smolvla-bimanual \
    --robot.type=so101_bimanual
```

## Important Notes

1. **SmolVLA has NO native bimanual support** - Actions are treated as a flat 12D vector without arm-specific handling.

2. **Camera naming** - The script automatically maps:
   - `observation.images.head` → `observation.images.camera1`
   - `observation.images.left_wrist` → `observation.images.camera2`
   - `observation.images.right_wrist` → `observation.images.camera3`

3. **Action dimensions** - SmolVLA's base model outputs 6 DOF. During training, it adapts to output 12 DOF for bimanual (6 per arm).

4. **Alternative** - For better bimanual coordination, consider using xVLA which has native bimanual support.

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
BATCH_SIZE=8 bash train_smolvla_bimanual.sh

# Or enable gradient checkpointing (default: true)
GRADIENT_CHECKPOINTING=true bash train_smolvla_bimanual.sh
```

### Dataset Not Found

```bash
# Make sure you're logged in to HuggingFace
huggingface-cli login

# Or use a local dataset
DATASET_PATH=/path/to/local/dataset DATASET_NAME=dataset_name bash train_smolvla_bimanual.sh
```

### Training Diverges

```bash
# Lower learning rate
LEARNING_RATE=5e-5 bash train_smolvla_bimanual.sh

# Increase warmup
WARMUP_STEPS=2000 bash train_smolvla_bimanual.sh
```
