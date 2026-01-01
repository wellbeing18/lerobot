#!/bin/bash
# =============================================================================
# xVLA Bimanual Training Script for SO101
# =============================================================================
#
# This script fine-tunes xVLA (Cross-Embodiment Vision-Language-Action Model)
# on bimanual tasks using LeRobot's training infrastructure.
#
# xVLA has NATIVE BIMANUAL SUPPORT via BimanualSO101ActionSpace:
#   - 12D real action dim (6 DOF per arm) padded to 20D for model
#   - Separate loss computation for left/right arms
#   - Gripper sigmoid applied to indices 5 and 11
#   - Proper arm coordination learning
#
# Key differences from single arm:
#   - ACTION_MODE=so101_bimanual (native bimanual support)
#   - 12 DOF action space (6 per arm)
#   - Domain ID for bimanual configuration
#   - Optional 3-camera setup
#
# Usage:
#   # Fresh training with defaults (20k steps)
#   bash train_xvla_bimanual.sh
#
#   # Custom training steps and batch size
#   MAX_STEPS=30000 BATCH_SIZE=12 bash train_xvla_bimanual.sh
#
#   # Train with frozen vision (for small datasets < 50 episodes)
#   FREEZE_VISION=true bash train_xvla_bimanual.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=outputs/xvla_bimanual_*/checkpoints/010000/pretrained_model \
#     bash train_xvla_bimanual.sh
#
# References:
#   - https://huggingface.co/docs/lerobot/xvla
#   - jdocs/bimanual/bin/claude_bimanual_plans.md
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Environment Setup
# =============================================================================

# Get script directory and project root
# Script is at: jdocs/scripts/bimanual/ -> go up 3 levels to reach project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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
# Default to the combined left+right arm dataset for proof-of-concept bimanual training
# Override DATASET_PATH for other datasets (e.g., handover, bimanual_pick)
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets_bimanuel/bimanual/combined_pick_and_place}"
DATASET_NAME="${DATASET_NAME:-combined_pick_and_place}"

# Pretrained Model
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/xvla-base}"

# Training Hyperparameters
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-12}"          # Lower than single arm (more memory for bimanual)
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# xVLA Architecture
CHUNK_SIZE="${CHUNK_SIZE:-32}"           # xVLA default
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"   # Match chunk_size
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"

# =============================================================================
# BIMANUAL-SPECIFIC CONFIGURATION
# =============================================================================
# ACTION_MODE=so101_bimanual enables:
#   - BimanualSO101ActionSpace with 12D real -> 20D model padding
#   - Separate loss computation for left/right arms
#   - Gripper sigmoid at indices (5, 11)
#   - Proper arm coordination learning
ACTION_MODE="${ACTION_MODE:-so101_bimanual}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-20}"

# Domain ID for bimanual configuration
# Use a different domain ID than single arm (e.g., 21 for bimanual)
DOMAIN_ID="${DOMAIN_ID:-21}"

# Precision (bfloat16 REQUIRED for bimanual to avoid OOM)
DTYPE="${DTYPE:-bfloat16}"

# =============================================================================
# Fine-tuning Strategy
# =============================================================================
# xVLA supports module-level freezing similar to SmolVLA
#
# RECOMMENDED FOR BIMANUAL BY DATASET SIZE:
# - < 50 episodes:  FREEZE_VISION=true, FREEZE_LANGUAGE=true
# - 50-200 episodes: FREEZE_VISION=false, FREEZE_LANGUAGE=true
# - > 200 episodes: FREEZE_VISION=false, FREEZE_LANGUAGE=false (full training)
#
# For bimanual tasks, visual-spatial learning is important, so we default
# to NOT freezing vision encoder.
# =============================================================================

FREEZE_VISION="${FREEZE_VISION:-false}"
FREEZE_LANGUAGE="${FREEZE_LANGUAGE:-false}"
TRAIN_POLICY_TRANSFORMER="${TRAIN_POLICY_TRANSFORMER:-true}"
TRAIN_SOFT_PROMPTS="${TRAIN_SOFT_PROMPTS:-true}"

# Print fine-tuning mode
if [ "${FREEZE_VISION}" = "true" ] && [ "${FREEZE_LANGUAGE}" = "true" ]; then
    echo ">>> BIMANUAL MODE: Policy Only (frozen VLM) <<<"
    echo "    Recommended for: < 50 episodes"
elif [ "${FREEZE_VISION}" = "false" ] && [ "${FREEZE_LANGUAGE}" = "true" ]; then
    echo ">>> BIMANUAL MODE: Vision + Policy (frozen language) <<<"
    echo "    Recommended for: 50-200 episodes"
else
    echo ">>> BIMANUAL MODE: Full VLM Training <<<"
    echo "    Recommended for: > 200 episodes"
fi

# Optimizer
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Scheduler (cosine decay with warmup)
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-30000}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/xvla_bimanual_${TIMESTAMP}}"

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
    echo "Please set DATASET_PATH to your bimanual dataset location"
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

    CHECKPOINT_PATH=$(dirname "${RESUME_FROM}")
    RESUME_FLAG="--resume=true --config_path=${RESUME_FROM}/train_config.json --checkpoint_path=${CHECKPOINT_PATH}"

    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
fi

# =============================================================================
# Build Training Command
# =============================================================================

# Create logs directory
LOG_DIR="${PROJECT_ROOT}/jdocs/logs"
mkdir -p "${LOG_DIR}"

# Log file
OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

# Function to log to both terminal and file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Start logging
log "=============================================="
log "xVLA BIMANUAL Training"
log "=============================================="
log ""
log "Configuration:"
log "  Pretrained:        ${PRETRAINED_MODEL}"
log "  Dataset:           ${DATASET_PATH}"
log "  Max Steps:         ${MAX_STEPS}"
log "  Batch Size:        ${BATCH_SIZE}"
log "  Learning Rate:     ${LEARNING_RATE}"
log "  Chunk Size:        ${CHUNK_SIZE}"
log "  Action Mode:       ${ACTION_MODE} (BIMANUAL: 12D->20D)"
log "  Domain ID:         ${DOMAIN_ID}"
log "  Dtype:             ${DTYPE}"
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
    --job_name=xvla_bimanual \
    --wandb.enable=false \
    ${RESUME_FLAG}"

# Camera name mapping for bimanual (3 cameras)
# Maps: head->camera1, left_wrist->camera2, right_wrist->camera3
# xVLA expects camera1, camera2, camera3 naming
CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\",\"observation.images.right_wrist\":\"observation.images.camera3\"}"

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
    log "BIMANUAL Training completed successfully!"
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
