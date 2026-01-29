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

## todos
1) generalizatin
- docs/design/vla_generalization_proposal.md
  - unfreeze llm for smolvla
  - unfreeze llm for xvla: and or multi domain ids
  - try pi0.5 & groot 1.6 full finetuning?
  - cloud training 

now you need to 1) assess whether we can support full finetuning smolvla(unfreeze everything) on my laptop rtx 5090 24GB vram 2) do research and analysis how to use pi0.5 lerobot or openai version(you can reference /home/jrobot/project/refs/openpi/LEROBOT_VS_OPENPI_INVESTIGATION.md), groot 1.5 or 1.6(i prefer 1.6 as it is newly, and I had successfully single arm trained groot1.6 as under /home/jrobot/project/Isaac-GR00T/custom/scripts/ver1_6). for you research, you should analyze whether we should do full finetuning, or partially unfreeze key components like vision encoder, language model, action model, as our goal is to make finetuning vla more general as expected in /home/jrobot/project/robotAgent/docs/design/vla_generalization_proposal.md(we encountered finetuned smolvla model on multi-tasks dataset which works well for tasks inside the dataset, but if we try task name with unseen objects in trained tasks, or reorganize task with different task name like seen object but to different target place than in trained task, vla ignores the task name and followed the trained trajectory). so we want to know whether we use more powerful vlas like pi0.5 groot 1.6 can solve the issue, or we need to do full finetuning(as current finetuned smolvla trained with language module frozen) 3) if we need to train pi0.5 and/or groot1.6, we could need to use cloud to do training, so you need to help prepare training script for pi0.5 and/or groot 1.6 for the bimanual dataset: datasets_bimanuel/multitasks. and put them under jdocs/scripts/cloud with guide.md

1) demo system
- demo and debug and improve
- website
- movable


## smolvla
### train 

nohup env FREEZE_VISION=false bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh > outputs/smolvla_bimanual_training_multitasks.log 2>&1 &
tail -f /home/jrobot/project/lerobot/outputs/smolvla_bimanual_training_multitasks.log

#### multitasks training

nohup env FREEZE_VISION=false MAX_STEPS=40000 BATCH_SIZE=48 \
    bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
    > outputs/smolvla_bimanual_training_multitasks.log 2>&1 &

tail -f /home/jrobot/project/lerobot/outputs/smolvla_bimanual_training_multitasks.log


#### full finetuning
  nohup env \
      TRAIN_EXPERT_ONLY=false \
      FREEZE_VISION=false \
      BATCH_SIZE=4 \
      MAX_STEPS=200000 \
      LEARNING_RATE=5e-5 \
      bash jdocs/scripts/bimanual/train_smolvla_bimanual.sh \
      > logs/smolvla_full_finetune_200k.log 2>&1 &

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

##### successful one(for demo)
 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the bread and place it in the bin"


 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model  \
      --task "Use right arm to pick up the bread and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the bread and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the banana and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the yogurt bottle and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the yogurt bottle and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the ketchup bottle and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the corn and place it on the plate"

we are encounter some issues when we use/test smolvla model using cmd like:  ```python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/success_smolvla_bimanual_20260103_multitask_visiononly/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the ketchup bottle and place it in the bin"```: 

case 1: log jdocs/logs/inference_smolvla_bimanual_20260118_115817.log
symptom: when left arm put the yogurt bottle to the bin, it went back home, but later it reached out trying to pick up something where the bottle used to be but now there is nothing. it is hallucination
note: there is another object banana on the right side of the table close to right arm

case 2: jdocs/logs/inference_smolvla_bimanual_20260118_120130.log
observation: after left arm put the yogurt bottle to the bin, it went back home, and stay still ongoing
note: not like case 1, there is no other object on the table

case 3: jdocs/logs/inference_smolvla_bimanual_20260118_120221.log
observation: after left arm put the yogurt bottle to the bin, it went back home, and stay still ongoing
note: there is no other object on the table, even there is a banana on the plate at the right side.

so from these 3 cases, the empirical observation gave me the impression that: when there are other object(s) on the table, current smolvla model kinds of hallucinating to pick up thing at the place where object used to be(even the object left on table is not the object in the command, and not in the area it is trying to pickup).

this is just one issue of many other different symptoms we experienced in different scenarios. there are several suspected areas: 1) the bias, the defects, the distribution of dataset we collected to do finetuning for the smolvla model: datasets_bimanuel/multitasks. 2) potential model's own defects like vision dominated the language(ie, from above case 1, give me the impression like vision dominates, even the command says clearly yogurt bottle, it is impacted by banana on the table, and generate hallucinated action to pick up where yogurt bottle used to be), or other issues we didn't figure out yet. 

but what I want now is as experts in vla model fields, you need to step back to research and think how to scientifically and systematically identify the root cause, then propose solution to solve the issue, more importantly during the process, we can generate process and tools which can facilitate us to spot and improve other issues. my suggestion is to use scientific facts/experiments based investigation and research methods: 1) first collect traces of those cases which become the key data for investigation, ie, logs and images/videos 2) use first principle to dissect smolvla model to find the root cause based on scientific explanation, for example, to check/visualize cross-attention on the picture, impact of vision and language(the model we finetuned unfroze only vision encoder and action part, not the language, we tried another version with unfrozon all, but the resulting model performed really bad), the generated action diffusion distribution coverage distribution, or other methodology vla research fields use to inspect and investigate vla model's internal reasoning and decision and generation logic 3) also check the dataset we collected for finetuning, create tools to analyze the data distributions or other features, which can help explain or resonate with the step 2)'s identified issues, which can further help /direct us to (re)collect missing data with goal otherwise we are facing curse of dimension issue of data collection and black box issue which we don't know what data are missing, what data we need more, what data we need to fix(for example, if the analysis shows there are data scarcity in certain use case, which caused the vla model doesn't have enough demo distribution which explain the hallucination) or collect more variety data(if above analysis shows the lack of data variety for certain situation caused the unexpected behavior), you get the idea, we need to use scientific mindset to use tools/experiements, collected data, visualized data, facts etc to spot root cause which explain the issue we experienced. then based on that to propose solutions, if it is data issue, we do dataset fix, if it is model issue, we fix model issue. but all needs to based on logically or facts based proof.

the research process can not be achieved in one shot, so you need to step back, design and plan a systematically process like: what data or logs or evidence to look into, what tools we need to build to generate more insightful evidence or probe deeper/embedded issues like what happened inside the smolvla model to produce hallucination for a situation, tools to scan and analyze dataset and model's internal we have then link the findings to the symptoms we spotted in the analysis. then from there we propose solutions which supported by evidences or facts.

this process is like academic or scientific research, which by iteself can later be turned into a research paper, like starting from the issue/symptom we facing, tools we built to help investigate, the evidences found to explain the issue(with some level of visualization is always a plus), then solutions proposed to solve the issue. then proved result to solve the issue. so follow this mindset or process to conduct your research plan. write your research plan into doc jdocs/bimanual/bin first, before start to implement and conduct the research. I can also help to run some experiements, collect data or evidence to iteratively promote the research process.

comments: 1) "Vision Encoder (SigLIP) - FROZEN“ is incorrect, we unfroze vision encoder during our finetuning 2) how to implement "4.2 Language Ablation Experiments", current smolvla cmd is based on finetuned task names, how to add this completion phrase during task execution?


##### others
  Final checkpoint (40K steps):
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model

  With specific task (use --task-key):
  <!-- Left arm picks orange -> plate -->
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task-key left_orange_plate

  <!-- Right arm picks banana -> plate   -->
  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task-key right_corn_plate 

  <!-- Left arm picks ketchup -> bin -->
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
      --task "Use RIGHT arm to pick up orange and place it on plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up yogurt bottle and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the tissue package and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use left arm to pick up the tissue package and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
      --task "Use right arm to pick up the yogurt bottle and place it in the bin"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model  --task "Use right arm to pick up corn and place it on plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model  --task "Use right arm to pick up banana and place it on plate"

  python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model  --task "Use right arm to pick up bread and place it on plate"

<!-- new 464 checkpoints -->
    python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use right arm to pick up corn and place it on plate"

        python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use right arm to pick up banana and place it on plate"

        python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use right arm to pick up bread and place it on plate"

        python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use right arm to pick up orange and place it on plate"

        python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use left arm to pick up used tissue and place it in the bin"

        python jdocs/scripts/bimanual/infer_smolvla_bimanual.py --checkpoint outputs/smolvla_bimanual_464_20260129/50000/pretrained_model  --task "Use left arm to pick up yogurt bottle and place it in the bin"

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

### full finetuning

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260107_072120/checkpoints/050000/pretrained_model \
      --task "Use right arm to pick up the ice cream and place it on the plate"

 python jdocs/scripts/bimanual/infer_smolvla_bimanual.py \
      --checkpoint outputs/smolvla_bimanual_20260107_072120/checkpoints/020000/pretrained_model \
      --task "Use right arm to pick up the bread and place it on the plate"

## xvla


### train
<!-- 60k, batch 16: 12.5 epochs, 80k/16: 17 epochs, 120k/12: 18.8 -->

  nohup env FREEZE_VISION=false MAX_STEPS=120000 BATCH_SIZE=12 \
    bash jdocs/scripts/bimanual/train_xvla_bimanual.sh \
    > outputs/xvla_bimanual_training_multitasks.log 2>&1 &

tail -f outputs/xvla_bimanual_training_multitasks.log

### inference

  python jdocs/scripts/bimanual/infer_xvla_bimanual.py \
      -c outputs/xvla_bimanual_20260105_234319/checkpoints/last/pretrained_model \
      --task "Use left arm to pick up the bread and place it on the plate"

### issues

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




## pi0.5


## groot



1. recalibrate lefet arm

2. compare with backup

3. if there is difference then will this effect the trained model