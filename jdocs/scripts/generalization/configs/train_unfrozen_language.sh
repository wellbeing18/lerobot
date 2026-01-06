#!/bin/bash
# ============================================================================
# Experiment 2: Train SmolVLA with Unfrozen Language Encoder
# ============================================================================
#
# Purpose:
#   Test whether unfreezing language layers improves task discrimination.
#   Compare generalization on held-out task combinations vs frozen baseline.
#
# Hypothesis:
#   With frozen language (`train_expert_only=true`), task-specific embeddings
#   can't be learned. Unfreezing should enable better target discrimination
#   (e.g., "plate" vs "bin" becoming more separable in embedding space).
#
# Key Changes from Baseline:
#   - TRAIN_EXPERT_ONLY=false (unfreeze VLM language layers)
#   - FREEZE_VISION_ENCODER=false (keep vision trainable for consistency)
#   - Lower learning rate to prevent catastrophic forgetting
#
# Risk:
#   Catastrophic forgetting of VLM's general language understanding.
#   Mitigated by lower LR and monitoring loss curves.
#
# Usage:
#   ./train_unfrozen_language.sh
#   ./train_unfrozen_language.sh --max-steps 10000  # Quick test run
#
# Expected Outputs:
#   - Model checkpoint in outputs/smolvla_unfrozen_language_*
#   - Training logs for comparison with baseline
#   - Evaluation metrics on held-out tasks
#
# ============================================================================

set -e

# Parse arguments
MAX_STEPS=${1:-20000}

# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Input
DATASET_PATH="${PROJECT_ROOT}/datasets_bimanuel/multitasks"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/smolvla_unfrozen_language_${TIMESTAMP}"

# Baseline checkpoint for comparison (optional)
BASELINE_CHECKPOINT="${PROJECT_ROOT}/outputs/smolvla_bimanual_20260103_200201/checkpoints/020000/pretrained_model"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
POLICY_NAME="smolvla"
VLM_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Action/chunk settings (same as baseline)
CHUNK_SIZE=50
N_ACTION_STEPS=50
NUM_STEPS=10  # Denoising steps

# ============================================================================
# KEY EXPERIMENT SETTINGS
# ============================================================================
# EXPERIMENT: Unfreeze language encoder
FREEZE_VISION_ENCODER=false
TRAIN_EXPERT_ONLY=false  # <-- KEY CHANGE: Unfreeze language layers

# Camera mappings (same as bimanual training)
CAMERA_MAPPING="head:camera1,left_wrist:camera2,right_wrist:camera3"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE=16  # Reduced from 32 (more parameters trainable now)
MAX_STEPS=${MAX_STEPS}
LR="5e-5"  # Lower LR to prevent catastrophic forgetting (was 1e-4)
WARMUP_STEPS=2000  # Longer warmup for stability
WEIGHT_DECAY="1e-6"
GRAD_CLIP_NORM=5.0  # Tighter clipping

# Scheduler
SCHEDULER="cosine"
DECAY_STEPS=${MAX_STEPS}
DECAY_LR="1e-6"

# Memory optimization (more params = more VRAM)
GRADIENT_CHECKPOINTING=true

# ============================================================================
# VALIDATION
# ============================================================================
echo "============================================================"
echo "EXPERIMENT 2: Train with Unfrozen Language Encoder"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Dataset:              ${DATASET_PATH}"
echo "  Output:               ${OUTPUT_DIR}"
echo "  VLM Model:            ${VLM_MODEL}"
echo ""
echo "Key Settings (DIFFERENT from baseline):"
echo "  train_expert_only:    ${TRAIN_EXPERT_ONLY} (baseline: true)"
echo "  freeze_vision:        ${FREEZE_VISION_ENCODER}"
echo "  learning_rate:        ${LR} (baseline: 1e-4)"
echo "  batch_size:           ${BATCH_SIZE} (baseline: 32)"
echo "  warmup_steps:         ${WARMUP_STEPS} (baseline: 1000)"
echo ""
echo "Expected trainable parameters:"
echo "  - Vision encoder:     ~86M params"
echo "  - Language model:     ~260M params (NEW)"
echo "  - Action expert:      ~100M params"
echo "  - TOTAL:              ~446M params (vs ~186M baseline)"
echo ""
echo "============================================================"

# Check dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Save experiment config
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment": "unfrozen_language",
    "hypothesis": "Unfreezing language encoder enables task-specific embeddings",
    "baseline_checkpoint": "${BASELINE_CHECKPOINT}",
    "key_changes": {
        "train_expert_only": false,
        "learning_rate": "${LR}",
        "batch_size": ${BATCH_SIZE},
        "warmup_steps": ${WARMUP_STEPS}
    },
    "timestamp": "${TIMESTAMP}"
}
EOF

# ============================================================================
# TRAINING COMMAND
# ============================================================================
echo ""
echo "Starting training..."
echo ""

cd "${PROJECT_ROOT}"

python -m lerobot.scripts.train \
    --policy.type="${POLICY_NAME}" \
    --dataset.repo_id="${DATASET_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    \
    --policy.vlm_model_name="${VLM_MODEL}" \
    --policy.freeze_vision_encoder=${FREEZE_VISION_ENCODER} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    \
    --policy.chunk_size=${CHUNK_SIZE} \
    --policy.n_action_steps=${N_ACTION_STEPS} \
    --policy.num_steps=${NUM_STEPS} \
    \
    --batch_size=${BATCH_SIZE} \
    --steps=${MAX_STEPS} \
    --lr=${LR} \
    --warmup_steps=${WARMUP_STEPS} \
    --weight_decay=${WEIGHT_DECAY} \
    --grad_clip_norm=${GRAD_CLIP_NORM} \
    \
    --lr_scheduler="${SCHEDULER}" \
    --lr_scheduler.decay_steps=${DECAY_STEPS} \
    --lr_scheduler.final_lr=${DECAY_LR} \
    \
    --save_freq=5000 \
    --eval_freq=2500 \
    --log_freq=100 \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# ============================================================================
# POST-TRAINING
# ============================================================================
echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
echo ""
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo ""
echo "Next Steps:"
echo "  1. Compare training loss curves with baseline"
echo "  2. Run generalization evaluation:"
echo "     python evaluate_generalization.py \\"
echo "         --checkpoint ${OUTPUT_DIR}/checkpoints/*/pretrained_model \\"
echo "         --baseline ${BASELINE_CHECKPOINT}"
echo ""
echo "  3. Check for catastrophic forgetting:"
echo "     - Compare in-distribution task performance"
echo "     - Monitor if general language understanding degraded"
echo ""
