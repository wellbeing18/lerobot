  nohup lerobot-train \
    --policy.type=smolvla \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --dataset.repo_id=pick_and_place \
    --dataset.root=/home/jrobot/project/lerobot/datasets/pick_and_place \
    --training.batch_size=8 \
    --training.steps=30000 \
    --training.save_freq=5000 \
    --training.log_freq=100 \
    --training.num_workers=4 \
    --output_dir=outputs/smolvla_unfrozen_vision_$(date +%Y%m%d_%H%M%S) \
    > outputs/smolvla_unfrozen_training.log 2>&1 &

nohup env FREEZE_VISION=false bash jdocs/scripts/train_smolvla_pickplace.sh > /dev/null 2>&1 &

tail -f jdocs/logs/train_smolvla_pickplace_20251225_163013.log

    python jdocs/scripts/infer_smolvla_so101.py \
    --checkpoint outputs/smolvla_pickplace_20251225_163013/checkpoints/010000/pretrained_model \
    --task "pick up the block and place it on the plate"

# bimanual
## smolvla
### train 

nohup env FREEZE_VISION=false bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh > outputs/smolvla_bimanual_training_multitasks.log 2>&1 &
tail -f /home/jrobot/project/lerobot/outputs/smolvla_bimanual_training_multitasks.log

#### multitasks training

nohup env FREEZE_VISION=false MAX_STEPS=40000 BATCH_SIZE=48 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > outputs/smolvla_bimanual_training_multitasks.log 2>&1 &

tail -f /home/jrobot/project/lerobot/outputs/smolvla_bimanual_training_multitasks.log

### inference

#### single task
- working right arm:
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20251231_175051/checkpoints/020000/pretrained_model \
      --right \
      --duration 20

  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20251231_175051/checkpoints/020000/pretrained_model \
      --left \
      --duration 20

- Run inference for LEFT arm task (default)
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20251231_175051/checkpoints/020000/pretrained_model \
      --left \
      --duration 60

  Or explicitly:
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20251231_175051/checkpoints/020000/pretrained_model \
      --task "Left arm pick up the tissue packet and place it on the plate" \
      --duration 60

  Dry run (no robot commands):
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20251231_175051/checkpoints/020000/pretrained_model \
      --left --dry-run

#### multi-tasks
  Final checkpoint (40K steps):
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model

  With specific task (use --task-key):
  # Left arm picks orange -> plate
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task-key left_orange_plate

  # Right arm picks banana -> plate  
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task-key right_corn_plate 

  # Left arm picks ketchup -> bin
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task-key left_yogurt_bin 

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the ice cream and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the used tissue and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the tissue package and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the tissue package and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the yogurt bottle and place it in the bin"



  Available task keys:
  | Plate Tasks        | Bin Tasks          |
  |--------------------|--------------------|
  | left_orange_plate  | left_icecream_bin  |
  | right_orange_plate | right_icecream_bin |
  | left_bread_plate   | left_ketchup_bin   |
  | right_bread_plate  | right_ketchup_bin  |
  | left_corn_plate    | left_yogurt_bin    |
  | right_corn_plate   | right_yogurt_bin   |
  | left_banana_plate  | left_tissue_bin    |
  | right_banana_plate | right_tissue_bin   |

#### issues:
- task name: garbage -> used tissue
- overfit to bin or plate

## xvla


### train
<!-- 60k, batch 16: 12.5 epochs, 80k/16: 17 epochs, 120k/12: 18.8 -->

  nohup env FREEZE_VISION=false MAX_STEPS=120000 BATCH_SIZE=12 \
    bash jdocs/scripts/bimanual/train_xvla_bimanual.sh \
    > outputs/xvla_bimanual_training_multitasks.log 2>&1 &

tail -f outputs/xvla_bimanual_training_multitasks.log

 Current Situation

  | Aspect                 | Status                       |
  |------------------------|------------------------------|
  | Training domain_id     | 0 (default for all 16 tasks) |
  | Soft prompt slots used | 1 out of 30                  |
  | Task differentiation   | Language conditioning only   |

  Impact on Current Training

  The model will still work, but:
  - All 16 tasks share the same soft prompt (slot 0)
  - XVLA's multi-domain capability is not leveraged
  - Task differentiation relies purely on language strings (like SmolVLA)

  Impact on Generalization Testing (from your proposal)

  Looking at Section 4.1 of your proposal:
  "Task-Type Soft Prompts | Medium-High | Medium | +10% | +5% | Maybe (Phase 3)"

  With current setup:
  - ❌ Cannot test soft prompt adaptation for different task types
  - ❌ Cannot leverage domain-specific learned features
  - ✅ Can still test language-based generalization

  Options

  | Option                              | Pros                             | Cons                        | Effort |
  |-------------------------------------|----------------------------------|-----------------------------|--------|
  | Keep domain_id=0                    | Simple, matches current training | No soft prompt benefits     | None   |
  | Re-train with task-category domains | Test soft prompts for plate/bin  | Requires data preprocessing | Medium |
  | Re-train with arm-based domains     | Test left/right adaptation       | Limited differentiation     | Low    |

  Recommendation

  For current training (already running): Keep domain_id=0. The model will rely on language conditioning.

  For future generalization testing, consider re-training with:

  # Option A: Task category based domains
  DOMAIN_MAPPING = {
      "plate_tasks": 1,  # orange/bread/corn/banana -> plate
      "bin_tasks": 2,    # icecream/ketchup/yogurt/tissue -> bin
  }

  # Option B: More granular - object type
  DOMAIN_MAPPING = {
      "orange": 1, "bread": 2, "corn": 3, "banana": 4,
      "icecream": 5, "ketchup": 6, "yogurt": 7, "tissue": 8,
  }

  # Option C: Arm-based
  DOMAIN_MAPPING = {
      "left_arm": 1,
      "right_arm": 2,
  }

  To implement (for future training):

  1. Add domain_id column to dataset during data collection:
  # In data collection script
  if "plate" in task:
      domain_id = 1
  elif "bin" in task:
      domain_id = 2

  2. Or override in training script:
  # train_xvla_bimanual.sh
  CMD="${CMD} --policy.domain_feature_key=task_category"

  Bottom Line: Current training will produce a working model with language-only task conditioning. For proper soft prompt testing per your Phase 3 proposal, you'll need to re-train with meaningful domain_id assignments after this training completes.

### inference
  python jdocs/scripts/bimanual/infer_xvla_bimanual.py \
      -c outputs/xvla_bimanual_TIMESTAMP/checkpoints/020000/pretrained_model \
      --left --duration 60


## pi0.5


## groot



1. recalibrate lefet arm

2. compare with backup

3. if there is difference then will this effect the trained model