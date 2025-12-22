#!/bin/bash
# =============================================================================
# ACT Training Script for SO101 Pick and Place
# =============================================================================
#
# This script trains an ACT (Action Chunking with Transformers) policy on the
# pick_and_place dataset using LeRobot's training infrastructure.
#
# Usage:
#   # Fresh training with defaults
#   bash train_act_pickplace.sh
#
#   # Custom training steps and batch size
#   MAX_STEPS=50000 BATCH_SIZE=16 bash train_act_pickplace.sh
#
#   # Resume from checkpoint (same max_steps)
#   RESUME_FROM=outputs/act_pickplace/checkpoints/010000/pretrained_model bash train_act_pickplace.sh
#
#   # Extended training beyond original max_steps (use constant LR)
#   RESUME_FROM=outputs/act_pickplace/checkpoints/010000/pretrained_model \
#     MAX_STEPS=150000 bash train_act_pickplace.sh
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Environment Setup
# =============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Add lerobot src to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Change to project root (required for relative paths)
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# =============================================================================
# Configuration (Override via environment variables)
# =============================================================================

# Dataset Configuration
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/pick_and_place}"
DATASET_NAME="${DATASET_NAME:-pick_and_place}"

# Training Hyperparameters
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# ACT Architecture
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
DIM_MODEL="${DIM_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-3200}"
N_ENCODER_LAYERS="${N_ENCODER_LAYERS:-4}"
N_DECODER_LAYERS="${N_DECODER_LAYERS:-1}"
LATENT_DIM="${LATENT_DIM:-32}"
DROPOUT="${DROPOUT:-0.1}"

# VAE Configuration
USE_VAE="${USE_VAE:-true}"
KL_WEIGHT="${KL_WEIGHT:-10.0}"

# Vision Backbone
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-ResNet18_Weights.IMAGENET1K_V1}"

# Optimizer
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/act_pickplace_${TIMESTAMP}}"

# Device
DEVICE="${DEVICE:-cuda}"

# Resume Configuration
RESUME_FROM="${RESUME_FROM:-}"

# =============================================================================
# Validation and Setup
# =============================================================================

echo "=============================================="
echo "ACT Training for Pick and Place"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Dataset:        ${DATASET_PATH}"
echo "  Max Steps:      ${MAX_STEPS}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Learning Rate:  ${LEARNING_RATE}"
echo "  Chunk Size:     ${CHUNK_SIZE}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo ""

# Validate dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}/meta/info.json" ]; then
    echo "ERROR: Invalid dataset - missing meta/info.json"
    exit 1
fi

# =============================================================================
# Resume Logic
# =============================================================================

RESUME_FLAG=""
if [ -n "${RESUME_FROM}" ]; then
    if [ ! -d "${RESUME_FROM}" ]; then
        echo "ERROR: Resume checkpoint not found: ${RESUME_FROM}"
        exit 1
    fi

    echo "Resuming from: ${RESUME_FROM}"
    RESUME_FLAG="--resume --config_path=${RESUME_FROM}/train_config.json"

    # Use the checkpoint's output directory
    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
    echo "Using output directory: ${OUTPUT_DIR}"
fi

# =============================================================================
# Build Training Command
# =============================================================================

# Create logs directory (separate from output to avoid conflict with lerobot)
LOG_DIR="${PROJECT_ROOT}/jdocs/logs"
mkdir -p "${LOG_DIR}"

# Log file (named after the output dir for easy matching)
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

echo "Training started at $(date)"
echo "Log file: ${LOG_FILE}"
echo ""

# Build the training command
CMD="python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.root=${DATASET_PATH} \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=${DEVICE} \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.dim_model=${DIM_MODEL} \
    --policy.n_heads=${N_HEADS} \
    --policy.dim_feedforward=${DIM_FEEDFORWARD} \
    --policy.n_encoder_layers=${N_ENCODER_LAYERS} \
    --policy.n_decoder_layers=${N_DECODER_LAYERS} \
    --policy.latent_dim=${LATENT_DIM} \
    --policy.dropout=${DROPOUT} \
    --policy.use_vae=${USE_VAE} \
    --policy.kl_weight=${KL_WEIGHT} \
    --policy.vision_backbone=${VISION_BACKBONE} \
    --policy.pretrained_backbone_weights=${PRETRAINED_BACKBONE} \
    --policy.optimizer_lr=${LEARNING_RATE} \
    --policy.optimizer_lr_backbone=${LR_BACKBONE} \
    --policy.optimizer_weight_decay=${WEIGHT_DECAY} \
    --policy.push_to_hub=false \
    --batch_size=${BATCH_SIZE} \
    --steps=${MAX_STEPS} \
    --save_freq=${SAVE_STEPS} \
    --log_freq=${LOG_FREQ} \
    --num_workers=${NUM_WORKERS} \
    --seed=${SEED} \
    --output_dir=${OUTPUT_DIR} \
    --job_name=act_pickplace \
    --wandb.enable=false \
    ${RESUME_FLAG}"

# =============================================================================
# Run Training
# =============================================================================

echo "Running command:"
echo "${CMD}"
echo ""
echo "=============================================="

# Run training and log output
${CMD} 2>&1 | tee -a "${LOG_FILE}"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Training failed with exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
    echo "=============================================="
    exit ${EXIT_CODE}
fi
