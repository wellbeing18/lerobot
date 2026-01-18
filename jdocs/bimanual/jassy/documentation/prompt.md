 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the bread and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the tissue packet and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the ketchup bottle and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the ketchup bottle and place it in the bin"






 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260107_072120/checkpoints/050000/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"


 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260107_072120/checkpoints/050000/pretrained_model \
      --task "Use right arm to pick up the ketchup bottle and place it in the bin"



python jdocs/scripts/bimanual/infer_xvla_bimanual.py \
      -c outputs/xvla_bimanual_20260105_234319/checkpoints/last/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"

Xvla error:

(lerobot) jrobot@jrobot-hp:~/project/lerobot$ python jdocs/scripts/bimanual/infer_xvla_bimanual.py \
      -c outputs/xvla_bimanual_20260105_234319/checkpoints/last/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"
/home/jrobot/anaconda3/envs/lerobot/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
2026-01-10 16:56:32,101 - INFO - Logging to: /home/jrobot/project/lerobot/jdocs/logs/inference_xvla_bimanual_20260110_165632.log
2026-01-10 16:56:32,101 - INFO - ======================================================================
2026-01-10 16:56:32,101 - INFO - xVLA BIMANUAL Robot Inference for SO-101
2026-01-10 16:56:32,101 - INFO - ======================================================================
2026-01-10 16:56:32,101 - INFO - Checkpoint:      outputs/xvla_bimanual_20260105_234319/checkpoints/last/pretrained_model
2026-01-10 16:56:32,101 - INFO - Task:            Use right arm to pick up the bread and place it on the plate
2026-01-10 16:56:32,101 - INFO - Domain ID:       0
2026-01-10 16:56:32,101 - INFO - Action Dim:      12 (6 per arm)
2026-01-10 16:56:32,101 - INFO - Duration:        60.0s
2026-01-10 16:56:32,101 - INFO - Device:          cuda
2026-01-10 16:56:32,101 - INFO - Dry run:         False
2026-01-10 16:56:32,101 - INFO - Dataset:         /home/jrobot/project/lerobot/datasets_bimanuel/multitasks
2026-01-10 16:56:32,101 - INFO - Log file:        /home/jrobot/project/lerobot/jdocs/logs/inference_xvla_bimanual_20260110_165632.log
2026-01-10 16:56:32,101 - INFO - ======================================================================
2026-01-10 16:56:32,104 - INFO - Loaded hardware config from: /home/jrobot/project/lerobot/jdocs/configs/hardware/xlerobot_bimanual.yaml
2026-01-10 16:56:32,104 - INFO - 
Loading dataset metadata...
2026-01-10 16:56:33,028 - INFO -   Dataset: /home/jrobot/project/lerobot/datasets_bimanuel/multitasks
2026-01-10 16:56:33,028 - INFO -   Episodes: 320
2026-01-10 16:56:33,028 - INFO - 
Loading xVLA policy...
Florence2ForConditionalGeneration has generative capabilities, as `prepare_inputs_for_generation` is explicitly defined. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
2026-01-10 16:56:41,887 - INFO -   Policy loaded on cuda
INFO:__main__:  Policy loaded on cuda
2026-01-10 16:56:41,888 - INFO -   Chunk size: 32
INFO:__main__:  Chunk size: 32
2026-01-10 16:56:41,888 - INFO -   Action mode: so101_bimanual
INFO:__main__:  Action mode: so101_bimanual
2026-01-10 16:56:41,888 - INFO -   Num denoising steps: 10
INFO:__main__:  Num denoising steps: 10
2026-01-10 16:56:42,203 - INFO -   Preprocessor and postprocessor created
INFO:__main__:  Preprocessor and postprocessor created
2026-01-10 16:56:42,203 - INFO -   Domain ID: 0 (used for soft prompt selection)
INFO:__main__:  Domain ID: 0 (used for soft prompt selection)
2026-01-10 16:56:42,204 - INFO - 
Initializing bimanual hardware...
INFO:__main__:
Initializing bimanual hardware...
2026-01-10 16:56:42,361 - INFO -   head camera initialized: 800x600
INFO:__main__:  head camera initialized: 800x600
2026-01-10 16:56:42,361 - INFO -   head camera crop enabled: 800x600 -> 640x480, y_offset=60
INFO:__main__:  head camera crop enabled: 800x600 -> 640x480, y_offset=60
2026-01-10 16:56:42,457 - INFO -   left_wrist camera initialized: 640x480
INFO:__main__:  left_wrist camera initialized: 640x480
2026-01-10 16:56:42,552 - INFO -   right_wrist camera initialized: 640x480
INFO:__main__:  right_wrist camera initialized: 640x480
2026-01-10 16:56:42,710 - INFO -   Bimanual robot connected:
INFO:__main__:  Bimanual robot connected:
2026-01-10 16:56:42,710 - INFO -     Left arm: /dev/ttyACM3
INFO:__main__:    Left arm: /dev/ttyACM3
2026-01-10 16:56:42,710 - INFO -     Right arm: /dev/ttyACM2
INFO:__main__:    Right arm: /dev/ttyACM2
2026-01-10 16:56:42,710 - INFO - 
Starting BIMANUAL inference loop (max 60.0s)...
INFO:__main__:
Starting BIMANUAL inference loop (max 60.0s)...
2026-01-10 16:56:42,710 - INFO - Task: Use right arm to pick up the bread and place it on the plate
INFO:__main__:Task: Use right arm to pick up the bread and place it on the plate
2026-01-10 16:56:42,710 - INFO - Domain ID: 0
INFO:__main__:Domain ID: 0
2026-01-10 16:56:42,710 - INFO - Action dim: 12 (6 per arm)
INFO:__main__:Action dim: 12 (6 per arm)
2026-01-10 16:56:42,710 - INFO - Action interval: 33.0ms (30.3Hz)
INFO:__main__:Action interval: 33.0ms (30.3Hz)
2026-01-10 16:56:42,710 - INFO - Press Ctrl+C to stop

INFO:__main__:Press Ctrl+C to stop

2026-01-10 16:56:44,803 - ERROR - 
Error: Got unsupported ScalarType BFloat16
ERROR:__main__:
Error: Got unsupported ScalarType BFloat16
Traceback (most recent call last):
  File "/home/jrobot/project/lerobot/jdocs/scripts/bimanual/infer_xvla_bimanual.py", line 1070, in main
    run_inference_loop(
  File "/home/jrobot/project/lerobot/jdocs/scripts/bimanual/infer_xvla_bimanual.py", line 717, in run_inference_loop
    raw_policy_action = raw_action.squeeze(0).cpu().numpy()
TypeError: Got unsupported ScalarType BFloat16
2026-01-10 16:56:44,803 - INFO - 
Cleaning up...
INFO:__main__:
Cleaning up...
2026-01-10 16:56:45,118 - INFO - Done
INFO:__main__:Done