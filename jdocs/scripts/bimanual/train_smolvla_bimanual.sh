#!/bin/bash
# =============================================================================
# SmolVLA Bimanual Training Script for SO101
# =============================================================================
#
# WARNING: SmolVLA has NO NATIVE BIMANUAL SUPPORT. This script is EXPERIMENTAL.
#
# SmolVLA treats all actions as a flat vector, without:
#   - Arm-specific loss computation
#   - Separate gripper handling per arm
#   - Coordination-aware training
#
# For better bimanual results, consider using xVLA with so101_bimanual action mode.
#
# This script fine-tunes SmolVLA on bimanual tasks, treating the 12 DOF action
# space as a flat vector (indices 0-5: left arm, indices 6-11: right arm).
#
# Key differences from single arm:
#   - 12 DOF action space (6 per arm) as flat vector
#   - FREEZE_VISION=false recommended (bimanual needs visual-spatial learning)
#   - GRADIENT_CHECKPOINTING=true recommended (memory efficiency)
#   - Optional 3-camera setup
#
# Usage:
#   # Fresh training with defaults (20k steps)
#   bash train_smolvla_bimanual.sh
#
#   # Custom training
#   MAX_STEPS=30000 BATCH_SIZE=24 bash train_smolvla_bimanual.sh
#
#   # Mode 1: Expert only (frozen VLM)
#   FREEZE_VISION=true TRAIN_EXPERT_ONLY=true bash train_smolvla_bimanual.sh
#
#   # Mode 2: Vision + Expert (recommended for bimanual)
#   FREEZE_VISION=false TRAIN_EXPERT_ONLY=true bash train_smolvla_bimanual.sh
#
#   # Mode 3: Full VLM (for large datasets)
#   FREEZE_VISION=false TRAIN_EXPERT_ONLY=false bash train_smolvla_bimanual.sh
#
# References:
#   - https://huggingface.co/docs/lerobot/smolvla
#   - jdocs/bimanual/bin/claude_bimanual_plans.md
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Environment Setup
# =============================================================================

# Script is at: jdocs/scripts/bimanual/ -> go up 3 levels to reach project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_ROOT}"

# =============================================================================
# Configuration (Override via environment variables)
# =============================================================================

# Dataset Configuration
# Default to left_arm_pick_and_place for proof-of-concept bimanual training
# This dataset has 12D actions (both arms recorded) with 2 tasks:
#   Task 0: "Left arm pick up the tissue packet..."
#   Task 1: "Right arm pick up the tissue packet..."
# Override DATASET_PATH for other datasets
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/datasets_bimanuel/bimanual/combined_pick_and_place}"
DATASET_NAME="${DATASET_NAME:-combined_pick_and_place}"

# Pretrained Model
PRETRAINED_MODEL="${PRETRAINED_MODEL:-lerobot/smolvla_base}"

# Training Hyperparameters
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# SmolVLA Architecture
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"  # Flow matching denoising steps

# =============================================================================
# Fine-tuning Strategy for Bimanual
# =============================================================================
#
# SmolVLA Components:
#   1. Vision Encoder (SigLIP) - ~86M params
#   2. Language Model (SmolLM2) - ~260M params
#   3. Action Expert - ~100M params
#   Total: ~450M params
#
# WARNING: SmolVLA has NO native bimanual support!
# - Treats 12D action as flat vector (no arm separation)
# - No arm-specific loss computation
# - May have coordination challenges
#
# RECOMMENDED FOR BIMANUAL:
#   - FREEZE_VISION=false (visual-spatial learning important for bimanual)
#   - GRADIENT_CHECKPOINTING=true (memory efficiency for 12D actions)
#   - MAX_STEPS=20000+ (bimanual coordination is harder)
#
# Mode Recommendations:
#   Mode 2 (Vision + Expert) - RECOMMENDED for bimanual
#     FREEZE_VISION=false, TRAIN_EXPERT_ONLY=true
#     ~185M trainable params, batch_size=24, 20k steps
#
#   Mode 3 (Full VLM) - For large bimanual datasets (500+ episodes)
#     FREEZE_VISION=false, TRAIN_EXPERT_ONLY=false
#     ~450M params, risk of catastrophic forgetting
# =============================================================================

# Default to Mode 2 for bimanual (unfrozen vision)
FREEZE_VISION="${FREEZE_VISION:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
TRAIN_STATE_PROJ="${TRAIN_STATE_PROJ:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# Auto-adjust settings based on fine-tuning mode
if [ "${FREEZE_VISION}" = "false" ]; then
    BATCH_SIZE="${BATCH_SIZE:-32}"          # Lower than single arm (bimanual uses more memory)
    MAX_STEPS="${MAX_STEPS:-20000}"
    SAVE_STEPS="${SAVE_STEPS:-2000}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"

    if [ "${TRAIN_EXPERT_ONLY}" = "true" ]; then
        echo ">>> BIMANUAL MODE 2: VISION + EXPERT (Recommended) <<<"
        echo "    Trainable: Vision Encoder (~86M) + Action Expert (~100M) = ~185M params"
        echo "    WARNING: SmolVLA has NO native bimanual support!"
    else
        echo ">>> BIMANUAL MODE 3: FULL VLM (For large datasets) <<<"
        echo "    Trainable: Vision + Language + Expert = ~450M params"
        echo "    WARNING: Risk of catastrophic forgetting + no native bimanual support!"
    fi
else
    BATCH_SIZE="${BATCH_SIZE:-32}"
    MAX_STEPS="${MAX_STEPS:-20000}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"

    if [ "${TRAIN_EXPERT_ONLY}" = "true" ]; then
        echo ">>> BIMANUAL MODE 1: EXPERT ONLY (Not recommended for bimanual) <<<"
        echo "    Trainable: Action Expert only (~100M params)"
        echo "    WARNING: Frozen vision may hurt bimanual performance!"
    else
        echo ">>> BIMANUAL MODE 4: LANGUAGE + EXPERT <<<"
        echo "    Trainable: Language Model (~260M) + Action Expert (~100M) = ~360M params"
    fi
fi
echo "    batch_size=${BATCH_SIZE}, max_steps=${MAX_STEPS}, gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

# Optimizer (SmolVLA defaults)
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-10}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# Scheduler (cosine decay with warmup)
DECAY_STEPS="${DECAY_STEPS:-${MAX_STEPS}}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

# Checkpointing
LOG_FREQ="${LOG_FREQ:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_bimanual_${TIMESTAMP}}"

# Device
DEVICE="${DEVICE:-cuda}"

# Resume Configuration
RESUME_FROM="${RESUME_FROM:-}"

# =============================================================================
# Validation and Setup
# =============================================================================

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

    RESUME_FLAG="--resume=true --config_path=${RESUME_FROM}/train_config.json"
    CHECKPOINT_DIR=$(dirname "$(dirname "${RESUME_FROM}")")
    OUTPUT_DIR="${CHECKPOINT_DIR}"
fi

# =============================================================================
# Build Training Command
# =============================================================================

LOG_DIR="${PROJECT_ROOT}/jdocs/logs"
mkdir -p "${LOG_DIR}"

OUTPUT_NAME=$(basename "${OUTPUT_DIR}")
LOG_FILE="${LOG_DIR}/train_${OUTPUT_NAME}.log"

log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "=============================================="
log "SmolVLA BIMANUAL Training (EXPERIMENTAL)"
log "=============================================="
log ""
log "WARNING: SmolVLA has NO native bimanual support!"
log "         Consider using xVLA for better bimanual results."
log ""
log "Configuration:"
log "  Pretrained:           ${PRETRAINED_MODEL}"
log "  Dataset:              ${DATASET_PATH}"
log "  Max Steps:            ${MAX_STEPS}"
log "  Batch Size:           ${BATCH_SIZE}"
log "  Learning Rate:        ${LEARNING_RATE}"
log "  Chunk Size:           ${CHUNK_SIZE}"
log "  Action Dim:           12 (6 per arm, flat vector)"
log "  Warmup Steps:         ${WARMUP_STEPS}"
log "  Freeze Vision:        ${FREEZE_VISION}"
log "  Gradient Checkpoint:  ${GRADIENT_CHECKPOINTING}"
log "  Output Dir:           ${OUTPUT_DIR}"
log ""
log "Training started at $(date)"
log "Log file: ${LOG_FILE}"
log ""

# Build the training command
if [ -n "${RESUME_FROM}" ]; then
    CMD="python -W ignore::UserWarning -W ignore::FutureWarning -W ignore::DeprecationWarning -m lerobot.scripts.lerobot_train \
        --dataset.repo_id=${DATASET_NAME} \
        --dataset.root=${DATASET_PATH} \
        --dataset.video_backend=pyav \
        --batch_size=${BATCH_SIZE} \
        --steps=${MAX_STEPS} \
        --save_freq=${SAVE_STEPS} \
        --log_freq=${LOG_FREQ} \
        --num_workers=${NUM_WORKERS} \
        --seed=${SEED} \
        --output_dir=${OUTPUT_DIR} \
        --job_name=smolvla_bimanual \
        --wandb.enable=false \
        ${RESUME_FLAG}"
else
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
        --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
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
        --job_name=smolvla_bimanual \
        --wandb.enable=false"
fi

# Camera name mapping for bimanual (3 cameras)
# Maps: head->camera1, left_wrist->camera2, right_wrist->camera3
# SmolVLA expects camera1, camera2, camera3 naming
CMD="${CMD} --rename_map={\"observation.images.head\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\",\"observation.images.right_wrist\":\"observation.images.camera3\"}"

# =============================================================================
# Run Training
# =============================================================================

log "Running command:"
log "${CMD}"
log ""
log "=============================================="

${CMD} 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
if [ ${EXIT_CODE} -eq 0 ]; then
    log ""
    log "=============================================="
    log "BIMANUAL Training completed successfully!"
    log "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
    log ""
    log "NOTE: This was experimental SmolVLA bimanual training."
    log "      If coordination is poor, consider trying xVLA instead."
    log "=============================================="
else
    log ""
    log "=============================================="
    log "Training failed with exit code: ${EXIT_CODE}"
    log "Check log file: ${LOG_FILE}"
    log "=============================================="
    exit ${EXIT_CODE}
fi
