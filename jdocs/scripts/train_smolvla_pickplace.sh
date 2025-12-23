#!/bin/bash
# =============================================================================
# SmolVLA Training Script for SO101 Pick and Place
# =============================================================================
#
# This script fine-tunes SmolVLA (Small Vision-Language-Action Model) on the
# pick_and_place dataset using LeRobot's training infrastructure.
#
# SmolVLA is a lightweight VLA foundation model that takes images, robot state,
# and language instructions to predict actions via flow matching.
#
# Usage:
#   # Fresh training with defaults (20k steps)
#   bash train_smolvla_pickplace.sh
#
#   # Custom training steps and batch size
#   MAX_STEPS=30000 BATCH_SIZE=32 bash train_smolvla_pickplace.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=outputs/smolvla_pickplace_*/checkpoints/010000/pretrained_model \
#     bash train_smolvla_pickplace.sh
#
# Key differences from ACT:
#   - Uses pretrained VLM backbone (smolvla_base)
#   - Requires task/language description in dataset
#   - Uses cosine decay with warmup scheduler
#   - Flow matching instead of VAE for action generation
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

# Suppress all UserWarning, FutureWarning, DeprecationWarning (propagates to worker processes)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# Change to project root (required for relative paths)
cd "${PROJECT_ROOT}"

# =============================================================================
# Configuration (Override via environment variables)
# =============================================================================

# Dataset Configuration
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets/pick_and_place}"
DATASET_NAME="${DATASET_NAME:-pick_and_place}"

# Camera name mapping (dataset → SmolVLA expected names)
# SmolVLA base expects: camera1, camera2, camera3
# Our dataset has: head, left_wrist

# Pretrained Model
# Use HuggingFace model ID or local path
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/smolvla_base}"

# Training Hyperparameters
# SmolVLA paper recommends 20k steps for ~50 episodes
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# SmolVLA Architecture (most should stay at defaults)
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"  # Flow matching denoising steps

# Fine-tuning Strategy (recommended to keep defaults)
FREEZE_VISION="${FREEZE_VISION:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"

# Optimizer (SmolVLA uses different defaults than ACT)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-10}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Scheduler (cosine decay with warmup)
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-30000}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_pickplace_${TIMESTAMP}}"

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

    RESUME_FLAG="--resume --config_path=${RESUME_FROM}/train_config.json"

    # Use the checkpoint's output directory
    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
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

# Function to log to both terminal and file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Start logging
log "=============================================="
log "SmolVLA Training for Pick and Place"
log "=============================================="
log ""
log "Configuration:"
log "  Pretrained:     ${PRETRAINED_MODEL}"
log "  Dataset:        ${DATASET_PATH}"
log "  Max Steps:      ${MAX_STEPS}"
log "  Batch Size:     ${BATCH_SIZE}"
log "  Learning Rate:  ${LEARNING_RATE}"
log "  Chunk Size:     ${CHUNK_SIZE}"
log "  Warmup Steps:   ${WARMUP_STEPS}"
log "  Output Dir:     ${OUTPUT_DIR}"
log ""
log "Training started at $(date)"
log "Log file: ${LOG_FILE}"
log ""

# Build the training command
# Note: SmolVLA uses --policy.path for pretrained model instead of --policy.type
# Use -W flags to suppress all UserWarnings and FutureWarnings
CMD="python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.root=${DATASET_PATH} \
    --dataset.video_backend=pyav \
    --policy.path=${PRETRAINED_MODEL} \
    --policy.device=${DEVICE} \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_steps=${NUM_STEPS} \
    --policy.freeze_vision_encoder=${FREEZE_VISION} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --policy.train_state_proj=${TRAIN_STATE_PROJ} \
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
    --job_name=smolvla_pickplace \
    --wandb.enable=false \
    ${RESUME_FLAG}"

# Add rename_map for camera name mapping (no spaces, no extra quotes)
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
