# SmolVLA Hallucination Investigation: Design Document

## Document Info
- **Created**: 2026-01-18
- **Updated**: 2026-01-19
- **Status**: Ready for Use
- **Related Files**:
  - `jdocs/scripts/bimanual/infer_smolvla_bimanual.py`
  - `jdocs/scripts/investigation/tools/` (all investigation tools)
  - `logs/investigation/` (output: cases, reports)
  - `src/lerobot/policies/smolvla/modeling_smolvla.py`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [SmolVLA Architecture Context](#2-smolvla-architecture-context-corrected)
3. [Language Ablation Experiments Design](#3-language-ablation-experiments-design)
4. [Tools Overview](#4-tools-overview)
5. [**Step-by-Step Usage Guide**](#5-step-by-step-usage-guide) ← START HERE
   - [Phase 1: Collect Evidence](#phase-1-collect-evidence-traces)
   - [Phase 2: Analyze Dataset](#phase-2-analyze-training-dataset)
   - [Phase 3: Model Introspection](#phase-3-model-introspection)
   - [Phase 4: Run Ablations](#phase-4-run-ablation-experiments)
   - [Phase 5: Aggregate & Report](#phase-5-aggregate-evidence--generate-report)
6. [Key Questions to Answer](#6-key-questions-to-answer)
7. [Success Criteria](#7-success-criteria)
8. [References](#8-references)

---

## 1. Problem Statement

The SmolVLA model exhibits "hallucination" behavior during bimanual robot inference where, after completing a pick-and-place task, the robot arm reaches back toward the location where an object **used to be** but no longer exists.

### Observed Cases

| Case | Log File | Behavior | Scene Context |
|------|----------|----------|---------------|
| 1 | `inference_smolvla_bimanual_20260118_115817.log` | After placing yogurt in bin, arm returned home (~step 180), then reached out again (~step 240+) | Banana present on table |
| 2 | `inference_smolvla_bimanual_20260118_120130.log` | Arm stayed still after task completion | No other objects |
| 3 | `inference_smolvla_bimanual_20260118_120221.log` | Arm stayed still after task completion | Banana on plate (far from action area) |

### Primary Hypothesis

**Vision-triggered hallucination**: When other objects are visually present near the workspace, the model's cross-attention mechanism attends to these visual features, triggering action patterns learned from training data—even when the task is complete.

---

## 2. SmolVLA Architecture Context (CORRECTED)

### Key Architectural Points

```
┌─────────────────────────────────────────┐
│  Vision Encoder (SigLIP) - UNFROZEN*    │  ← *Unfrozen during our finetuning
│  Language Embeddings - FROZEN           │
└──────────────┬──────────────────────────┘
               │ KV Cache (computed once per chunk)
               ▼
┌─────────────────────────────────────────┐
│  Cross-Attention Layer                  │
│  Action Expert (trainable) queries      │
│  VLM embeddings as K,V                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Flow-Matching Diffusion (10 steps)     │
│  x_t → denoise → action chunk (50)      │
└─────────────────────────────────────────┘
```

**CORRECTION**: The vision encoder (SigLIP) was **UNFROZEN** during our finetuning, not frozen. This is important because:
1. The vision encoder has been adapted to our specific scene/object distribution
2. Visual attention patterns may reflect our training data biases
3. Solutions involving vision encoder modification are viable

### KV Cache Behavior (Critical for Language Ablation)

From `modeling_smolvla.py:791-804`:
```python
# KV cache computed ONCE per action chunk
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images, img_masks, lang_tokens, lang_masks, state=state
)
# ...
_, past_key_values = self.vlm_with_expert.forward(
    # ...
    fill_kv_cache=True,  # Computed once, reused for all 10 denoising steps
)
```

**Implication**: The KV cache containing vision+language embeddings is computed **once** at the start of each 50-step action chunk. To change the task description mid-inference, we must:
1. Trigger a new chunk generation (invalidate current action queue)
2. Or wait until the next natural chunk boundary

---

## 3. Language Ablation Experiments Design

### 3.1 The Challenge

Current SmolVLA inference uses finetuned task names like:
- `"Use left arm to pick up the yogurt bottle and place it in the bin"`

The task string is passed once per inference step and embedded into the KV cache at chunk generation time.

### 3.2 Approach: Dynamic Task Modification

**Option A: Append Completion Phrase**
```python
# After completion detection (e.g., step > 180)
task = "Use left arm to pick up the yogurt bottle and place it in the bin. Task complete, stay still."
```

**Option B: Replace Task Entirely**
```python
task = "Hold current position. Do not move."
```

### 3.3 Implementation (DONE)

The `--dynamic-task` mode has been added to `infer_smolvla_bimanual.py`:

```bash
python infer_smolvla_bimanual.py \
    --task-key left_yogurt_bin \
    --dynamic-task \
    --completion-phrase "Task complete. Hold position." \
    --completion-step 180
```

### 3.4 Completion Detection: Research Findings

**IMPORTANT**: VLA models like SmolVLA do **NOT have built-in completion detection**.

Per research ([SeqVLA paper](https://roboticsproceedings.org/rss20/p112.pdf)), automatic heuristics
(gripper state, action variance) are **unreliable** and can cause false positives that ruin
normal execution. SeqVLA solves this by adding a learned "completion detection head" trained
jointly with action generation.

**For our research**, we only support **manual step count**:
- Observe when tasks typically complete during pilot runs
- Set `--completion-step` accordingly (e.g., 180 for yogurt-to-bin task)

| Strategy | Status | Notes |
|----------|--------|-------|
| `step_count` | ✅ Supported | Manual specification required |
| `gripper_close` | ❌ Removed | Unreliable, false positives |
| `action_variance` | ❌ Removed | Unreliable, false positives |
| Learned detector | 🔬 Future | Would require training completion head (SeqVLA approach) |

---

## 4. Tools Overview

All tools are located in `jdocs/scripts/investigation/tools/`.
Output goes to `logs/investigation/` (cases, reports).

| Tool | Purpose | Key Output |
|------|---------|------------|
| `trace_inference.py` | Capture detailed inference traces | `trace.jsonl`, images |
| `analyze_denoising.py` | Visualize denoising process | Trajectory plots, velocity analysis |
| `visualize_attention.py` | Attention heatmaps and analysis | Spatial attention maps |
| `analyze_dataset.py` | Training data distribution analysis | `report.md`, distribution plots |
| `run_ablation.py` | Systematic ablation experiments | Experiment results |
| `aggregate_evidence.py` | Synthesize all findings | Final investigation report |

---

## 5. Step-by-Step Usage Guide

### Prerequisites

```bash
# Navigate to tools directory
cd jdocs/scripts/investigation/tools

# Default checkpoint is used from infer_smolvla_bimanual.py:
# outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model
# Override with --checkpoint if needed

# Dataset path (for analyze_dataset.py)
DATASET="datasets_bimanuel/multitasks"

# Output directories (auto-created by tools)
# logs/investigation/cases/    - trace data
# logs/investigation/reports/  - analysis reports
```

---

### Phase 1: Collect Evidence (Traces)

**Goal**: Capture detailed traces from inference runs with and without hallucination.

**Defaults**:
- Images captured automatically (every 50 steps, ~1.7s at 30Hz)
- Output auto-generated to `logs/investigation/cases/<type>/case_<timestamp>/`
- Checkpoint uses default from main inference script

#### Step 1.1: Capture Hallucination Case (with distractor)

Set up the scene with a distractor object (e.g., banana on table), then run:

```bash
python trace_inference.py \
    --task-key left_yogurt_bin \
    --case-type hallucination \
    --duration 90 \
    --notes "Banana present on table near workspace"
```

#### Step 1.2: Capture Normal Case (no distractor)

Clear the scene of all objects except the task object, then run:

```bash
python trace_inference.py \
    --task-key left_yogurt_bin \
    --case-type normal \
    --duration 90 \
    --notes "No distractor objects"
```

#### Step 1.3: Capture with Dynamic Task (language ablation)

```bash
python trace_inference.py \
    --task-key left_yogurt_bin \
    --case-type ablation \
    --dynamic-task \
    --completion-phrase "Task complete. Hold position." \
    --completion-step 180 \
    --duration 90 \
    --notes "Testing completion phrase intervention"
```

#### Using custom task string

```bash
python trace_inference.py \
    --task "Use left arm to pick up the orange and place it on the plate" \
    --case-type hallucination \
    --notes "Custom task string test"
```

#### Output Structure
```
logs/investigation/cases/hallucination/case_20260119_143052/
├── metadata.json      # Run configuration + notes
├── trace.jsonl        # Per-step trace data
├── trace.log          # Execution log
└── images/            # Captured frames (every 50 steps)
    ├── step_0000_head.jpg
    ├── step_0050_head.jpg
    └── ...
```

Use `--case-name custom_name` to override auto-generated timestamp.

---

### Phase 2: Analyze Training Dataset

**Goal**: Identify distribution issues (imbalanced tasks, under-represented idle frames).

#### Step 2.1: Run Full Dataset Analysis

```bash
python analyze_dataset.py \
    --dataset-path /home/jrobot/project/lerobot/$DATASET \
    --output-dir ../../../../logs/investigation/reports/dataset_analysis \
    --analyze-phases \
    --analyze-post-completion \
    --max-episodes 100
```

#### Step 2.2: Review Output

```bash
# View the generated report
cat ../../../../logs/investigation/reports/dataset_analysis/report.md

# Key things to look for:
# 1. Task distribution - are all tasks equally represented?
# 2. Phase distribution - what % are "idle" frames?
# 3. Warnings about under-represented categories
```

#### Key Metrics to Check

| Metric | Healthy | Warning |
|--------|---------|---------|
| Episodes per task | 15-25 | < 10 |
| Idle frame ratio | > 10% | < 5% |
| Task balance ratio | < 1.5x | > 2x |

---

### Phase 3: Model Introspection

**Goal**: Understand what the model "sees" and where attention goes.

#### Step 3.1: Analyze Denoising Process

If you have a hallucination case with captured images:

```bash
# First, run inference and capture denoising data
# (Note: This requires modifying the inference to save denoising traces)

# Then visualize existing trace
python analyze_denoising.py \
    --trace-file ../../../../logs/investigation/cases/hallucination/case_001_banana_present/denoising_trace.json \
    --output-dir ../../../../logs/investigation/reports/denoising_analysis
```

#### Step 3.2: Compare Denoising Between Cases

```bash
python analyze_denoising.py \
    --compare \
    --case1 ../../../../logs/investigation/cases/hallucination/case_001_banana_present \
    --case2 ../../../../logs/investigation/cases/normal/case_001_no_distractor \
    --output-dir ../../../../logs/investigation/reports/denoising_comparison
```

#### Step 3.3: Attention Visualization

```bash
python visualize_attention.py \
    --checkpoint $CHECKPOINT \
    --case-dir ../../../../logs/investigation/cases/hallucination/case_001_banana_present \
    --output-dir ../../../../logs/investigation/reports/attention_analysis
```

#### What to Look For

1. **Denoising Trajectory**: Does the action "emerge" differently in hallucination cases?
2. **Velocity Field**: Are there anomalous velocity spikes at specific steps?
3. **Attention Maps**: Does attention focus on distractor objects?
4. **Attention Entropy**: Is attention more diffuse in hallucination cases?

---

### Phase 4: Run Ablation Experiments

**Goal**: Test hypotheses through controlled experiments.

#### Step 4.1: List Available Experiments

```bash
python run_ablation.py --list-experiments
```

Available experiments:
- `baseline_no_modification` - Control
- `language_completion_hold` - "Hold position" phrase
- `language_completion_stay_still` - "Stay still" phrase
- `language_completion_early` - Trigger at step 150
- `language_completion_late` - Trigger at step 220
- `language_completion_gripper` - Gripper-based trigger
- `language_completion_variance` - Variance-based trigger

#### Step 4.2: Run Single Experiment

```bash
python run_ablation.py \
    --experiment language_completion_hold \
    --checkpoint $CHECKPOINT \
    --task-key left_yogurt_bin \
    --output-dir ../../../../logs/investigation/reports/ablation_single \
    --dry-run  # Remove for real hardware
```

#### Step 4.3: Run Full Ablation Suite

```bash
python run_ablation.py \
    --config ablation_config.yaml \
    --checkpoint $CHECKPOINT \
    --task-key left_yogurt_bin \
    --output-dir ../../../../logs/investigation/reports/ablation_suite \
    --dry-run  # Remove for real hardware
```

#### Step 4.4: Review Ablation Results

```bash
# View suite report
cat ../../../../logs/investigation/reports/ablation_suite/suite_report.md

# Check which experiments reduced hallucination
grep -A2 "effect_size" ../../../../logs/investigation/reports/ablation_suite/suite_results.json
```

---

### Phase 5: Aggregate Evidence & Generate Report

**Goal**: Synthesize all findings into actionable insights.

#### Step 5.1: Run Evidence Aggregation

```bash
python aggregate_evidence.py \
    --investigation-dir ../../../../logs/investigation/cases \
    --ablation-results ../../../../logs/investigation/reports/ablation_suite \
    --dataset-analysis ../../../../logs/investigation/reports/dataset_analysis \
    --output-dir ../../../../logs/investigation/reports/final
```

#### Step 5.2: Review Final Report

```bash
# View the comprehensive investigation report
cat ../../../../logs/investigation/reports/final/investigation_report.md
```

#### Output Contains

1. **Evidence Correlation Matrix** - Which factors correlate with hallucination
2. **Ablation Comparison** - Which interventions were effective
3. **Root Cause Analysis** - Most likely cause with causal chain
4. **Recommended Solutions** - Prioritized list of fixes

---

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INVESTIGATION WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: COLLECT EVIDENCE
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │   Hallucination  │    │     Normal       │    │    Ablation      │
    │   Cases (3+)     │    │    Cases (2+)    │    │   Cases (5+)     │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
             │                       │                       │
             └───────────────────────┴───────────────────────┘
                                     │
                                     ▼
                     logs/investigation/cases/
                     (trace.jsonl, images/)

PHASE 2: DATASET ANALYSIS
    ┌──────────────────┐
    │ analyze_dataset  │ ──────► logs/investigation/reports/dataset_analysis/
    │    .py           │         ├── report.md
    └──────────────────┘         ├── task_distribution.png
                                 └── phase_distribution.png

PHASE 3: MODEL INTROSPECTION
    ┌──────────────────┐    ┌──────────────────┐
    │ analyze_denoising│    │visualize_attention│
    │      .py         │    │       .py         │
    └────────┬─────────┘    └────────┬──────────┘
             │                       │
             ▼                       ▼
    logs/investigation/     logs/investigation/
    reports/denoising/      reports/attention/
    ├── trajectory.png      ├── spatial_heatmap.png
    └── velocity.png        └── attention_summary.png

PHASE 4: ABLATION EXPERIMENTS
    ┌──────────────────┐
    │  run_ablation    │ ──────► logs/investigation/reports/ablation_suite/
    │      .py         │         ├── suite_results.json
    └──────────────────┘         └── suite_report.md

PHASE 5: SYNTHESIS
    ┌──────────────────┐
    │aggregate_evidence│ ◄────── ALL ABOVE OUTPUTS
    │       .py        │
    └────────┬─────────┘
             │
             ▼
    logs/investigation/reports/final/
    ├── investigation_report.md  ← MAIN OUTPUT
    ├── evidence_matrix.png
    └── evidence_data.json
```

---

### Quick Reference: Common Commands

```bash
cd jdocs/scripts/investigation/tools

# 1. Collect traces (images captured by default, output auto-generated)
python trace_inference.py -k left_yogurt_bin --case-type hallucination --notes "with banana"
python trace_inference.py -k left_yogurt_bin --case-type normal --notes "no distractor"

# 2. Collect with language ablation
python trace_inference.py -k left_yogurt_bin --case-type ablation \
    --dynamic-task --completion-step 180 --notes "completion phrase test"

# 3. Using custom task string
python trace_inference.py -t "Use left arm to pick up the orange" --case-type hallucination

# 4. View collected cases
ls -la ../../../../logs/investigation/cases/

# 5. Analyze dataset (specify dataset path)
python analyze_dataset.py -d datasets_bimanuel/multitasks --analyze-phases

# 6. View results
cat ../../../../logs/investigation/reports/*/report.md
```

### Parameter Reference

#### Task Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task-key` / `-k` | `left_yogurt_bin` | Predefined task key. Options: `left_orange_plate`, `right_yogurt_bin`, etc. |
| `--task` / `-t` | None | Custom task string (e.g., "Use left arm to pick up the orange"). Overrides `-k`. |

#### Output Organization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--case-type` | `hallucination` | **Folder category** for organizing traces. Options: `hallucination`, `normal`, `ablation`. Output goes to `logs/investigation/cases/<case-type>/`. |
| `--case-name` | `YYYYMMDD_HHMMSS` | Custom folder name. Default is auto-generated timestamp. |
| `--notes` | `""` | Free-text notes saved to `metadata.json`. Describe scene (e.g., "Banana on table near gripper"). |

#### Image Capture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--capture-interval` | `50` | Save camera image every N inference steps. At 30Hz, 50 steps ≈ 1.7 seconds. |
| `--no-capture-images` | `False` | Disable image capture. **Images are captured by default.** |

#### Language Ablation (Dynamic Task)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dynamic-task` | `False` | Enable language ablation mode. Injects completion phrase mid-inference. |
| `--completion-step` | `180` | Step number to trigger phrase injection. **Must observe real task completion to set correctly.** |
| `--completion-phrase` | `"Task complete. Hold position."` | Phrase injected at completion step. Forces KV cache recompute. |

#### Run Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--duration` | `60` | Max inference time in seconds. Script stops after this time. |
| `--dry-run` | `False` | Run without sending actions to robot. For testing trace collection. |
| `--checkpoint` / `-c` | (from script) | Model checkpoint path. Uses default from `infer_smolvla_bimanual.py`. |

---

## 6. Key Questions to Answer

### From Trace Analysis
- At what step does hallucination behavior start?
- Is there a consistent pattern (always after step 180)?
- Does hallucination correlate with distractor presence?

### From Dataset Analysis
- Are post-completion idle frames under-represented?
- Does the model see multi-object scenes during training?
- What do training trajectories do after task completion?

### From Model Introspection
- Where does attention go in hallucination cases?
- At which denoising step does the hallucination action emerge?
- Is attention entropy higher in hallucination cases?

### From Ablation Experiments
- Does adding completion phrase reduce hallucination?
- Which phrase is most effective?
- At what step should completion be triggered?

---

## 7. Success Criteria

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| **Diagnostic** | Can reproduce hallucination | 100% reproducibility |
| **Understanding** | Root cause identified | High confidence (>70%) |
| **Solution** | Intervention effectiveness | >80% reduction in hallucination |
| **Validation** | Task performance maintained | No degradation |

---

## 8. References

### Source Code
- SmolVLA policy: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Cross-attention: `src/lerobot/policies/smolvla/smolvlm_with_expert.py:309-422`
- Debug tracker: `src/lerobot/policies/rtc/debug_tracker.py`

### Investigation Tools
- All tools: `jdocs/scripts/investigation/tools/`
- Config: `jdocs/scripts/investigation/tools/ablation_config.yaml`
- Output: `logs/investigation/` (cases and reports)

### Scripts
- Inference: `jdocs/scripts/bimanual/infer_smolvla_bimanual.py`
- Existing attention viz: `jdocs/scripts/generalization/experiments/visualize_attention.py`

### Research Papers
- [SeqVLA: Completion-Aware VLA](https://roboticsproceedings.org/rss20/p112.pdf) - Learned completion detection for long-horizon tasks
- [Neural Task Success Classifiers](https://arxiv.org/abs/2107.00722) - Learning task completion from few demonstrations
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844) - SmolVLA architecture and training
