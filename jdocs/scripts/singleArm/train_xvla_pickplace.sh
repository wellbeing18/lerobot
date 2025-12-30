#!/bin/bash
# =============================================================================
# xVLA Training Script for SO101 Pick and Place
# =============================================================================
#
# This script fine-tunes xVLA (Cross-Embodiment Vision-Language-Action Model)
# on the pick_and_place dataset using LeRobot's training infrastructure.
#
# xVLA is a 0.9B parameter VLA that uses soft prompts to handle different
# robot embodiments. Key features:
#   - Soft Prompts: Learnable embeddings for each robot/camera configuration
#   - Action Modes: Registry system for different action spaces (auto recommended)
#   - Domain IDs: Learnable identifiers for robot/camera configurations
#   - Florence2 VLM: Vision-language backbone with ImageNet normalization
#
# Key differences from SmolVLA:
#   - Full fine-tuning recommended (don't freeze VLM encoders)
#   - Uses bfloat16 precision to avoid OOM
#   - Uses action_mode=auto for automatic action dimension handling
#   - Uses domain_id for different embodiment configurations
#   - No flow matching num_steps (uses diffusion denoising instead)
#
# Usage:
#   # Fresh training with defaults (20k steps)
#   bash train_xvla_pickplace.sh
#
#   # Custom training steps and batch size
#   MAX_STEPS=30000 BATCH_SIZE=16 bash train_xvla_pickplace.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=outputs/xvla_pickplace_*/checkpoints/010000/pretrained_model \
#     bash train_xvla_pickplace.sh
#
# References:
#   - https://huggingface.co/docs/lerobot/xvla
#   - https://huggingface.co/lerobot/xvla-base
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

# Suppress all UserWarning, FutureWarning, DeprecationWarning
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# Disable tokenizers parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Change to project root (required for relative paths)
cd "${PROJECT_ROOT}"

# =============================================================================
# Configuration (Override via environment variables)
# =============================================================================

# Dataset Configuration
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/pick_and_place}"
DATASET_NAME="${DATASET_NAME:-pick_and_place}"

# Camera name mapping (dataset -> xVLA expected names)
# xVLA base trained with: camera1, camera2, camera3
# Our dataset has: head, left_wrist

# Pretrained Model
# Use HuggingFace model ID or local path
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/xvla-base}"

# Training Hyperparameters
# xVLA paper recommends ~20k steps for fine-tuning
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"           # Lower than SmolVLA due to larger model (0.9B)
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# xVLA Architecture
CHUNK_SIZE="${CHUNK_SIZE:-32}"            # xVLA default is 32 (SmolVLA uses 50)
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"    # Match chunk_size
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"  # Diffusion denoising steps

# Action Mode Configuration
# Options: auto (recommended), ee6d, joint, so101_bimanual
# "auto" automatically detects action dimension from dataset
ACTION_MODE="${ACTION_MODE:-auto}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-20}"    # Model's max action dim for padding

# Domain ID Configuration
# Choose domain_id that resembles your robot/camera setup
# See https://huggingface.co/docs/lerobot/xvla for domain ID table
# 0=Bridge, 1=RT1, 2=Calvin, 3=LIBERO, etc.
# For custom robot, use any unused ID (e.g., 20+)
DOMAIN_ID="${DOMAIN_ID:-20}"              # Custom domain for SO-101

# Precision (bfloat16 strongly recommended for xVLA to avoid OOM)
DTYPE="${DTYPE:-bfloat16}"

# Fine-tuning Strategy (xVLA recommends NOT freezing for best results)
# VLM trained at 1/10 of base LR automatically
FREEZE_VISION="${FREEZE_VISION:-false}"
FREEZE_LANGUAGE="${FREEZE_LANGUAGE:-false}"
TRAIN_POLICY_TRANSFORMER="${TRAIN_POLICY_TRANSFORMER:-true}"
TRAIN_SOFT_PROMPTS="${TRAIN_SOFT_PROMPTS:-true}"

# Optimizer (xVLA uses similar defaults to SmolVLA)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"       # xVLA default is 0
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Scheduler (cosine decay with warmup)
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-30000}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/xvla_pickplace_${TIMESTAMP}}"

# Device
DEVICE="${DEVICE:-cuda}"

# Resume Configuration
RESUME_FROM="${RESUME_FROM:-}"

# =============================================================================
# Validation and Setup
# =============================================================================

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

    # RESUME_FROM points to pretrained_model dir, checkpoint_path needs parent
    CHECKPOINT_PATH=$(dirname "${RESUME_FROM}")
    RESUME_FLAG="--resume=true --config_path=${RESUME_FROM}/train_config.json --checkpoint_path=${CHECKPOINT_PATH}"

    # Use the checkpoint's output directory
    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
fi

# =============================================================================
# Build Training Command
# =============================================================================

# Create logs directory
LOG_DIR="${PROJECT_ROOT}/jdocs/logs"
mkdir -p "${LOG_DIR}"

# Log file (named after the output dir for easy matching)
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

# Function to log to both terminal and file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Start logging
log "=============================================="
log "xVLA Training for Pick and Place"
log "=============================================="
log ""
log "Configuration:"
log "  Pretrained:        ${PRETRAINED_MODEL}"
log "  Dataset:           ${DATASET_PATH}"
log "  Max Steps:         ${MAX_STEPS}"
log "  Batch Size:        ${BATCH_SIZE}"
log "  Learning Rate:     ${LEARNING_RATE}"
log "  Chunk Size:        ${CHUNK_SIZE}"
log "  Action Mode:       ${ACTION_MODE}"
log "  Domain ID:         ${DOMAIN_ID}"
log "  Dtype:             ${DTYPE}"
log "  Warmup Steps:      ${WARMUP_STEPS}"
log "  Output Dir:        ${OUTPUT_DIR}"
log ""
log "Fine-tuning Strategy:"
log "  Freeze Vision:     ${FREEZE_VISION}"
log "  Freeze Language:   ${FREEZE_LANGUAGE}"
log "  Train Transformer: ${TRAIN_POLICY_TRANSFORMER}"
log "  Train Soft Prompts:${TRAIN_SOFT_PROMPTS}"
log ""
log "Training started at $(date)"
log "Log file: ${LOG_FILE}"
log ""

# Build the training command
CMD="python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.root=${DATASET_PATH} \
    --dataset.video_backend=pyav \
    --policy.path=${PRETRAINED_MODEL} \
    --policy.device=${DEVICE} \
    --policy.dtype=${DTYPE} \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_denoising_steps=${NUM_DENOISING_STEPS} \
    --policy.action_mode=${ACTION_MODE} \
    --policy.max_action_dim=${MAX_ACTION_DIM} \
    --policy.freeze_vision_encoder=${FREEZE_VISION} \
    --policy.freeze_language_encoder=${FREEZE_LANGUAGE} \
    --policy.train_policy_transformer=${TRAIN_POLICY_TRANSFORMER} \
    --policy.train_soft_prompts=${TRAIN_SOFT_PROMPTS} \
    --policy.optimizer_lr=${LEARNING_RATE} \
    --policy.optimizer_weight_decay=${WEIGHT_DECAY} \
    --policy.optimizer_grad_clip_norm=${GRAD_CLIP_NORM} \
    --policy.scheduler_warmup_steps=${WARMUP_STEPS} \
    --policy.scheduler_decay_steps=${DECAY_STEPS} \
    --policy.scheduler_decay_lr=${DECAY_LR} \
    --policy.push_to_hub=false \
    --batch_size=${BATCH_SIZE} \
    --steps=${MAX_STEPS} \
    --save_freq=${SAVE_STEPS} \
    --log_freq=${LOG_FREQ} \
    --num_workers=${NUM_WORKERS} \
    --seed=${SEED} \
    --output_dir=${OUTPUT_DIR} \
    --job_name=xvla_pickplace \
    --wandb.enable=false \
    ${RESUME_FLAG}"

# Add rename_map for camera name mapping (no spaces, no extra quotes)
# xVLA expects camera1, camera2; our dataset has head, left_wrist
CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\"}"

# =============================================================================
# Run Training
# =============================================================================

log "Running command:"
log "${CMD}"
log ""
log "=============================================="

# Run training and log output to both terminal and file
${CMD} 2>&1 | tee -a "${LOG_FILE}"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ ${EXIT_CODE} -eq 0 ]; then
    log ""
    log "=============================================="
    log "Training completed successfully!"
    log "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
    log "=============================================="
else
    log ""
    log "=============================================="
    log "Training failed with exit code: ${EXIT_CODE}"
    log "Check log file: ${LOG_FILE}"
    log "=============================================="
    exit ${EXIT_CODE}
fi
