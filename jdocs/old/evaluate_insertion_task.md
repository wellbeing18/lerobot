# LeRobot ALOHA Insertion Task Evaluation Guide

This document provides step-by-step instructions for running the ALOHA Insertion task evaluation.

## Steps

### Step 1: Activate Environment

```bash
conda activate lerobot
```

Navigate to project root:
```bash
cd /home/jrobot/project/lerobot
```

### Step 2: Migrate the Insertion Model

```bash
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_insertion_human \
  --output-dir src/outputs/migrated/act_aloha_insertion_human
```

### Step 3: Run Evaluation

```bash
MUJOCO_GL=egl python -m lerobot.scripts.lerobot_eval --policy.path=/home/jrobot/project/lerobot/src/outputs/migrated/act_aloha_insertion_human --env.type=aloha --env.task=AlohaInsertion-v0 --eval.n_episodes=10 --eval.batch_size=2 --output_dir=outputs/eval/act_aloha_insertion 

# --policy.device=cpu
```

**Important**: Run the evaluation command as a single line with no line breaks.

## Key Problems Encountered

1. **Wrong environment name**: Used `AlohaInsertionHuman-v0` instead of `AlohaInsertion-v0`

2. **Multi-line command parsing errors**: Bash split the command incorrectly when using backslashes with spaces after them

3. **RTX 5090 CUDA incompatibility**: GPU has CUDA capability sm_120, but PyTorch 2.7.1+cu126 only supports up to sm_90

## Key Solutions

1. **Correct environment name**: Use `--env.task=AlohaInsertion-v0`

2. **Single-line command**: Run the entire evaluation command as one continuous line with no backslashes or line breaks

3. **CPU fallback**: Add `--policy.device=cpu` to run on CPU instead of incompatible GPU

## Summary

- Pretrained model: `lerobot/act_aloha_sim_insertion_human`
- Migrated path: `/home/jrobot/project/lerobot/src/outputs/migrated/act_aloha_insertion_human`
- Environment: `AlohaInsertion-v0`
- Device: CPU (due to GPU compatibility)
- Run from: `/home/jrobot/project/lerobot`
