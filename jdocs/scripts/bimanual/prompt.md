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

nohup env FREEZE_VISION=false bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh > outputs/smolvla_bimanual_training.log 2>&1 &
tail -f /home/jrobot/project/lerobot/outputs/smolvla_bimanual_training.log


### inference

 # Run inference for LEFT arm task (default)
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20241231_XXXXXX/checkpoints/020000/pretrained_model \
      --left \
      --duration 60

  Or explicitly:
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20241231_XXXXXX/checkpoints/020000/pretrained_model \
      --task "Left arm pick up the tissue packet and place it on the plate" \
      --duration 60

  Dry run (no robot commands):
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      -c outputs/smolvla_bimanual_20241231_XXXXXX/checkpoints/020000/pretrained_model \
      --left --dry-run

## xvla


## pi0.5


## groot