#!/bin/bash
# =============================================================================
# LeRobot GROOT N1.5 Training Script for SO-101
# =============================================================================
#
# This script trains GROOT N1.5 foundation model on SO-101 pick-and-place data
# using the LeRobot framework.
#
# Prerequisites:
#   - CUDA-enabled GPU (24GB+ VRAM recommended)
#   - Flash Attention 2.5+ installed
#   - LeRobot with groot extra: pip install -e ".[groot]"
#   - Dataset in LeRobot v3.0 format
#
# Usage:
#   ./train_groot_so101_n15.sh
#
# Key differences from Isaac-GR00T training:
#   - Uses lerobot-train command instead of custom trainer
#   - Dataset stays in LeRobot v3.0 format (no conversion needed)
#   - Normalization handled automatically by preprocessor
#   - Simpler configuration (no modality.json needed)
#
# =============================================================================

set -e

# =============================================================================
# Configuration - Modify these for your setup
# =============================================================================

# Dataset
DATASET_REPO_ID="/home/jrobot/project/lerobot/datasets/pick_and_place"

# Output
OUTPUT_DIR="outputs/groot_so101_n15_$(date +%Y%m%d_%H%M%S)"

# Training parameters
MAX_STEPS=10000          # 10k-20k recommended for pick-and-place
BATCH_SIZE=8             # Reduce to 4 for 16GB GPUs
SAVE_STEPS=1000
EVAL_STEPS=500

# Learning rate
LEARNING_RATE=1e-4       # NVIDIA recommended

# Model configuration
BASE_MODEL="nvidia/GR00T-N1.5-3B"
EMBODIMENT_TAG="new_embodiment"

# Fine-tuning strategy (freeze backbone, train projector + diffusion)
TUNE_LLM=false
TUNE_VISUAL=false
TUNE_PROJECTOR=true
TUNE_DIFFUSION=true

# LoRA settings (set LORA_RANK=32 for memory-constrained GPUs)
LORA_RANK=0              # 0 = no LoRA, 32 = enable LoRA
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Chunk and action settings
CHUNK_SIZE=50
N_ACTION_STEPS=50

# Data processing
VIDEO_BACKEND="decord"   # or "torchvision_av"
NUM_WORKERS=4

# =============================================================================
# Environment Setup
# =============================================================================

# Set memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure we're in the lerobot directory
cd /home/jrobot/project/lerobot

# Activate conda environment if not already active
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "lerobot" ]]; then
    echo "Activating lerobot conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate lerobot
fi

# =============================================================================
# Logging
# =============================================================================

echo "=============================================="
echo "LeRobot GROOT N1.5 Training"
echo "=============================================="
echo "Dataset:        $DATASET_REPO_ID"
echo "Output:         $OUTPUT_DIR"
echo "Max steps:      $MAX_STEPS"
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Base model:     $BASE_MODEL"
echo "Tune LLM:       $TUNE_LLM"
echo "Tune Visual:    $TUNE_VISUAL"
echo "Tune Projector: $TUNE_PROJECTOR"
echo "Tune Diffusion: $TUNE_DIFFUSION"
echo "LoRA rank:      $LORA_RANK"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "dataset_repo_id": "$DATASET_REPO_ID",
    "base_model": "$BASE_MODEL",
    "max_steps": $MAX_STEPS,
    "batch_size": $BATCH_SIZE,
    "learning_rate": $LEARNING_RATE,
    "tune_llm": $TUNE_LLM,
    "tune_visual": $TUNE_VISUAL,
    "tune_projector": $TUNE_PROJECTOR,
    "tune_diffusion": $TUNE_DIFFUSION,
    "lora_rank": $LORA_RANK,
    "chunk_size": $CHUNK_SIZE,
    "n_action_steps": $N_ACTION_STEPS,
    "embodiment_tag": "$EMBODIMENT_TAG",
    "timestamp": "$(date -Iseconds)"
}
EOF

# =============================================================================
# Training Command
# =============================================================================

# Build training command
TRAIN_CMD="lerobot-train"

# Policy configuration
TRAIN_CMD+=" --policy.type=groot"
TRAIN_CMD+=" --policy.base_model_path=$BASE_MODEL"
TRAIN_CMD+=" --policy.embodiment_tag=$EMBODIMENT_TAG"
TRAIN_CMD+=" --policy.tune_llm=$TUNE_LLM"
TRAIN_CMD+=" --policy.tune_visual=$TUNE_VISUAL"
TRAIN_CMD+=" --policy.tune_projector=$TUNE_PROJECTOR"
TRAIN_CMD+=" --policy.tune_diffusion_model=$TUNE_DIFFUSION"
TRAIN_CMD+=" --policy.optimizer_lr=$LEARNING_RATE"
TRAIN_CMD+=" --policy.use_bf16=true"
TRAIN_CMD+=" --policy.chunk_size=$CHUNK_SIZE"
TRAIN_CMD+=" --policy.n_action_steps=$N_ACTION_STEPS"

# LoRA configuration
if [ "$LORA_RANK" -gt 0 ]; then
    TRAIN_CMD+=" --policy.lora_rank=$LORA_RANK"
    TRAIN_CMD+=" --policy.lora_alpha=$LORA_ALPHA"
    TRAIN_CMD+=" --policy.lora_dropout=$LORA_DROPOUT"
fi

# Dataset configuration
TRAIN_CMD+=" --dataset.repo_id=$DATASET_REPO_ID"
TRAIN_CMD+=" --policy.video_backend=$VIDEO_BACKEND"

# Training configuration
TRAIN_CMD+=" --training.batch_size=$BATCH_SIZE"
TRAIN_CMD+=" --training.max_steps=$MAX_STEPS"
TRAIN_CMD+=" --training.save_steps=$SAVE_STEPS"
TRAIN_CMD+=" --training.eval_steps=$EVAL_STEPS"
TRAIN_CMD+=" --training.dataloader_num_workers=$NUM_WORKERS"

# Output configuration
TRAIN_CMD+=" --output_dir=$OUTPUT_DIR"

# Logging
TRAIN_CMD+=" --wandb.enable=true"
TRAIN_CMD+=" --wandb.project=groot-so101"
TRAIN_CMD+=" --wandb.run_name=groot_n15_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Execute Training
# =============================================================================

echo ""
echo "Executing training command:"
echo "$TRAIN_CMD"
echo ""

# Run training
$TRAIN_CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

# =============================================================================
# Post-Training
# =============================================================================

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints:      $OUTPUT_DIR/checkpoints/"
echo "Training log:     $OUTPUT_DIR/training.log"
echo ""
echo "To run inference:"
echo "  python jdocs/scripts/infer_groot_so101_n15.py \\"
echo "    --checkpoint $OUTPUT_DIR/checkpoints/last \\"
echo "    --task 'Pick up the cube and place it in the box'"
echo "=============================================="
