#!/bin/bash
# ===========================================================================
# GR00T 1.6 Training Script for Bimanual SO-101
# ===========================================================================
#
# This script finetunes GR00T N1.6-3B on bimanual SO-101 robot data.
#
# IMPORTANT: GR00T 1.6 uses SELECTIVE PARAMETER FREEZING, not LoRA.
# Default configuration:
#   - tune_llm=False, tune_visual=False: Freeze VLM backbone (~2.8B params)
#   - tune_projector=True, tune_diffusion_model=True: Train action processing (~214M params)
# This fits comfortably in 24GB VRAM without LoRA.
#
# Bimanual Configuration:
#   - State/Action: 12 DOF (6 per arm: 5 joints + 1 gripper)
#   - Cameras: 3 (head, left_wrist, right_wrist)
#   - Task format: "Use [left/right] arm to [action] [object] [target]"
#
# Memory Requirements:
#   - RTX 4090/5090 (24GB): batch_size=8 (~20-22GB peak)
#   - A100 40GB: batch_size=16-32
#   - If OOM: reduce GLOBAL_BATCH_SIZE to 4
#
# Usage:
#   # Basic training
#   bash train_groot16_bimanual.sh
#
#   # MVP run (1000 steps, ~1 hour)
#   MAX_STEPS=1000 bash train_groot16_bimanual.sh
#
#   # Custom dataset path
#   DATASET_PATH=/path/to/groot/dataset bash train_groot16_bimanual.sh
#
# ===========================================================================

set -e

# ===========================================================================
# KEY HYPERPARAMETERS
# ===========================================================================

# --- Dataset Configuration ---
# Must be in GR00T format (use convert_lerobot_to_groot.py first)
DATASET_PATH="${DATASET_PATH:-/path/to/datasets/multitasks_groot}"
MODALITY_CONFIG="${MODALITY_CONFIG:-jdocs/scripts/cloud/groot16/so101_bimanual_config.py}"

# --- Model Configuration ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"

# --- Training Hyperparameters ---
MAX_STEPS="${MAX_STEPS:-10000}"           # MVP: 1000, Full: 10000
LEARNING_RATE="${LEARNING_RATE:-1e-4}"    # NVIDIA recommended
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"   # 8 for 24GB VRAM
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Parameter Freezing (GR00T 1.6 default) ---
# Uncomment to override:
# TUNE_LLM="--tune_llm"             # Default: False (frozen)
# TUNE_VISUAL="--tune_visual"       # Default: False (frozen)
# TUNE_PROJECTOR="--tune_projector" # Default: True (trainable)
# TUNE_DIFFUSION="--tune_diffusion_model"  # Default: True (trainable)

# --- Checkpointing ---
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

# --- Output Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${OUTPUT_BASE:-outputs/groot16_bimanual}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}_${TIMESTAMP}}"

# --- Data Augmentation ---
# Color jitter helps with lighting variations
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"

# --- Resources ---
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

# --- Logging ---
USE_WANDB="${USE_WANDB:-false}"

# ===========================================================================
# END OF KEY HYPERPARAMETERS
# ===========================================================================

echo "=================================================================="
echo "GR00T 1.6 Bimanual Training Configuration"
echo "=================================================================="
echo "Model:          $BASE_MODEL"
echo "Dataset:        $DATASET_PATH"
echo "Output:         $OUTPUT_DIR"
echo "Max steps:      $MAX_STEPS"
echo "Batch size:     $GLOBAL_BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Warmup ratio:   $WARMUP_RATIO"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Save steps:     $SAVE_STEPS"
echo "Num GPUs:       $NUM_GPUS"
echo "Color jitter:   $COLOR_JITTER"
echo "=================================================================="

# Validate dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo ""
    echo "To convert LeRobot dataset to GR00T format:"
    echo "  python jdocs/scripts/cloud/groot16/convert_lerobot_to_groot.py \\"
    echo "      --input ./datasets_bimanuel/multitasks \\"
    echo "      --output ./datasets/multitasks_groot"
    exit 1
fi

# Validate modality config exists
if [ ! -f "$MODALITY_CONFIG" ]; then
    echo "ERROR: Modality config not found at $MODALITY_CONFIG"
    echo "Using default bimanual config..."
    # Could create default here or fail
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Logging Setup ---
LOG_FILE="$OUTPUT_DIR/training.log"
echo "Logging to: $LOG_FILE"

# W&B flag
WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

# Export for multi-GPU
export NUM_GPUS=$NUM_GPUS

# Save training configuration
{
    echo "=================================================================="
    echo "GR00T 1.6 Bimanual Training"
    echo "Started: $(date)"
    echo "=================================================================="
    echo "Model:          $BASE_MODEL"
    echo "Dataset:        $DATASET_PATH"
    echo "Modality:       $MODALITY_CONFIG"
    echo "Output:         $OUTPUT_DIR"
    echo "Max steps:      $MAX_STEPS"
    echo "Batch size:     $GLOBAL_BATCH_SIZE"
    echo "Learning rate:  $LEARNING_RATE"
    echo "Warmup ratio:   $WARMUP_RATIO"
    echo "Weight decay:   $WEIGHT_DECAY"
    echo "Save steps:     $SAVE_STEPS"
    echo "Num GPUs:       $NUM_GPUS"
    echo "Color jitter:   $COLOR_JITTER"
    echo "=================================================================="
    echo ""
} > "$LOG_FILE"

echo "Starting training..."

# Launch training
# For single GPU: use plain python
# For multi-GPU: use torchrun --nproc_per_node=$NUM_GPUS --master_port=29500
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "$BASE_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --modality_config_path "$MODALITY_CONFIG" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --num_gpus $NUM_GPUS \
    --output_dir "$OUTPUT_DIR" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --learning_rate $LEARNING_RATE \
    $WANDB_FLAG \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --color_jitter_params $COLOR_JITTER \
    --dataloader_num_workers $DATALOADER_WORKERS \
    2>&1 | tee -a "$LOG_FILE"

# Log completion
{
    echo ""
    echo "=================================================================="
    echo "Training Complete: $(date)"
    echo "=================================================================="
    echo "Checkpoint saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference test:"
    echo "     python jdocs/scripts/cloud/groot16/infer_groot16_bimanual.py \\"
    echo "         --checkpoint $OUTPUT_DIR/checkpoint-$MAX_STEPS"
    echo ""
    echo "  2. Test generalization (held-out tasks):"
    echo "     --task \"Use left arm to pick up the ice cream and place it on the plate\""
    echo "     --task \"Use right arm to pick up the tissue and place it on the plate\""
    echo "=================================================================="
} | tee -a "$LOG_FILE"
