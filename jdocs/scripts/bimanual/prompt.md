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
      --task-key left_ketchup_bin

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

## xvla


### train

nohup env FREEZE_VISION=false bash jdocs/scripts/bimanual/train_xvla_bimanual.sh > outputs/xvla_bimanual_training.log 2>&1 &

### inference
  python jdocs/scripts/bimanual/infer_xvla_bimanual.py \
      -c outputs/xvla_bimanual_TIMESTAMP/checkpoints/020000/pretrained_model \
      --left --duration 60


## pi0.5


## groot



1. recalibrate lefet arm

2. compare with backup

3. if there is difference then will this effect the trained model