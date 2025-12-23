
MAX_STEPS=50000 SAVE_STEPS=10000 bash jdocs/scripts/train_act_pickplace.sh

MAX_STEPS=50000 BATCH_SIZE=32 bash jdocs/scripts/train_act_pickplace.sh

MAX_STEPS=30000 BATCH_SIZE=32 bash jdocs/scripts/train_act_pickplace.sh

python jdocs/scripts/infer_act_so101.py -c outputs/act_pickplace_*/checkpoints/050000/pretrained_model --duration 30

BATCH_SIZE=32 MAX_STEPS=30000 bash jdocs/scripts/train_smolvla_pickplace.sh

camera mapping
task name
log to file
warning suppress
- Current step / Total steps
- Iterations per second
- ETA