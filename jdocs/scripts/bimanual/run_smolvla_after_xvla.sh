#!/bin/bash
# ===========================================================================
# Schedule SmolVLA Full Fine-Tuning After XVLA Training Completes
# ===========================================================================
#
# This script waits for the current XVLA training to finish, then starts
# SmolVLA full fine-tuning (with unfrozen language encoder).
#
# How it works:
#   - Uses "kill -0 $PID" to check if process is still running
#   - kill -0 sends signal 0 (null signal) which doesn't kill anything
#   - It just checks if the process exists and you have permission to signal it
#   - Returns 0 (success) if process exists, non-zero if not
#   - "2>/dev/null" suppresses error messages when process doesn't exist
#
# Usage:
#   # Run in background
#   nohup bash jdocs/scripts/bimanual/run_smolvla_after_xvla.sh > logs/scheduler.log 2>&1 &
#
#   # Monitor
#   tail -f logs/scheduler.log
#
# ===========================================================================

set -e

# XVLA training PID (found via: ps aux | grep train_xvla)
XVLA_PID=3632979

# Expected completion time: 3:00 AM tomorrow
# We'll sleep until then before starting the check loop
TARGET_HOUR=3
TARGET_MINUTE=0

echo "==========================================================="
echo "SmolVLA Full Fine-Tuning Scheduler"
echo "==========================================================="
echo "Started at: $(date)"
echo "XVLA training PID: $XVLA_PID"
echo "==========================================================="

# Calculate seconds until 3:00 AM tomorrow
calculate_sleep_seconds() {
    local now=$(date +%s)
    local target=$(date -d "tomorrow ${TARGET_HOUR}:${TARGET_MINUTE}:00" +%s)
    echo $((target - now))
}

SLEEP_SECONDS=$(calculate_sleep_seconds)
SLEEP_HOURS=$((SLEEP_SECONDS / 3600))
SLEEP_MINS=$(((SLEEP_SECONDS % 3600) / 60))

echo ""
echo "Sleeping until 3:00 AM tomorrow (~${SLEEP_HOURS}h ${SLEEP_MINS}m)..."
echo "Will wake up at: $(date -d "tomorrow ${TARGET_HOUR}:${TARGET_MINUTE}:00")"
echo "==========================================================="

sleep $SLEEP_SECONDS

echo ""
echo "[$(date)] Woke up! Now checking if XVLA training is still running..."

# Check if PID is still running after sleep
if ! kill -0 $XVLA_PID 2>/dev/null; then
    echo "[$(date)] XVLA training already finished! Proceeding to SmolVLA training..."
else
    echo "[$(date)] XVLA still running, entering check loop (every 10 minutes)..."

    # Wait for XVLA training to complete
    # Check every 10 minutes (600 seconds)
    while kill -0 $XVLA_PID 2>/dev/null; do
        echo "[$(date)] XVLA still running (PID $XVLA_PID)... checking again in 10 minutes"
        sleep 600
    done
fi

echo ""
echo "==========================================================="
echo "[$(date)] XVLA training completed!"
echo "Starting SmolVLA full fine-tuning in 10 seconds..."
echo "==========================================================="
sleep 10

# Start SmolVLA full fine-tuning
cd /home/jrobot/project/lerobot

echo "[$(date)] Starting SmolVLA full fine-tuning..."
echo "Parameters:"
echo "  TRAIN_EXPERT_ONLY=false (unfreeze language encoder)"
echo "  FREEZE_VISION=false"
echo "  BATCH_SIZE=4"
echo "  MAX_STEPS=200000"
echo "  LEARNING_RATE=5e-5"

nohup env \
    TRAIN_EXPERT_ONLY=false \
    FREEZE_VISION=false \
    BATCH_SIZE=4 \
    MAX_STEPS=200000 \
    LEARNING_RATE=5e-5 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > outputs/smolvla_full_finetune_200k.log 2>&1 &

SMOLVLA_PID=$!
echo ""
echo "==========================================================="
echo "[$(date)] SmolVLA training started!"
echo "  PID: $SMOLVLA_PID"
echo "  Log: outputs/smolvla_full_finetune_200k.log"
echo "==========================================================="
echo ""
echo "Monitor with: tail -f outputs/smolvla_full_finetune_200k.log"
