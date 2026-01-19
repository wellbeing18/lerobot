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
Output goes to `logs/` (cases and analysis results).

### Tool Status

| Tool | Purpose | Status | Key Output |
|------|---------|--------|------------|
| `trace_inference.py` | Capture detailed inference traces | **DONE** | `trace.jsonl`, images |
| `visualize_attention.py` | Vision encoder self-attention | **DONE** (insufficient) | Spatial attention maps |
| `cross_attention_capture.py` | **Action expert → VLM cross-attention** | **NEXT** | Denoising-step attention |
| `distractor_attention.py` | Quantify distractor attention | **NEXT** | Attention ratio metrics |
| `counterfactual_masking.py` | Object removal experiments | **NEXT** | Causal analysis |
| `analyze_denoising.py` | Visualize denoising process | Planned | Trajectory plots |
| `analyze_dataset.py` | Training data distribution | Planned | Distribution reports |

### Critical Finding: Why Self-Attention is Insufficient

**Problem**: Our initial `visualize_attention.py` captured **vision encoder self-attention** (how image patches attend to each other). The heatmaps looked similar between hallucination and normal cases because:

1. Vision encoder self-attention shows internal image processing
2. But the **action expert uses cross-attention** to query VLM embeddings
3. The real question is: **What does the action expert attend to when generating actions?**

**Solution**: Capture **cross-attention** at `smolvlm_with_expert.py:575` where the action expert queries the VLM prefix (image + language + state tokens).

---

## 4.1 Cross-Attention Analysis (NEW - CRITICAL)

### Why Cross-Attention Matters

```
Vision Encoder Self-Attention (INSUFFICIENT):
  Image Patches ←→ Image Patches (internal feature extraction)

Action Expert Cross-Attention (WHAT WE NEED):
  Action Tokens → [Image Patches + Language Tokens + State]
                   ↑
                   This shows what drives action generation
```

### Token Layout in VLM Prefix

The action expert attends to ~778 tokens in the VLM prefix:

```
Index Range    Token Type           Count    Notes
─────────────────────────────────────────────────────
[0-1]          Image special        2        <image_start>, global
[2-730]        Image patches        729      27×27 grid from SigLIP
[731]          Image end            1        <image_end>
[732-779]      Language tokens      ~48      Task description
[780]          State token          1        Robot joint state
```

### Cross-Attention Capture Hook

**File**: `jdocs/scripts/investigation/tools/cross_attention_capture.py`

**Hook location**: `smolvlm_with_expert.py:575` (after softmax in `eager_attention_forward`)

```python
# Captured attention shape: [batch, num_heads, 50_action_tokens, 778_prefix_tokens]
probs = nn.functional.softmax(masked_att_weights, dim=-1)
# ↑ Hook here to capture probs

# Map to spatial: extract attention to image patches (indices 2:731)
attn_to_image = probs[:, :, :, 2:731]  # [batch, heads, 50, 729]
spatial_attention = attn_to_image.mean(dim=2).reshape(-1, 27, 27)
```

### Temporal Evolution (10 Denoising Steps)

**Key insight**: Track attention across all 10 denoising steps to see WHEN hallucination emerges.

```
Step 0 (t=1.0): Initial noise → first action direction
Step 1-3:       Primary trajectory emerges
Step 4-6:       Trajectory refinement
Step 7-9:       Fine details + potential distractor influence
Step 9 (t=0.1): Final action output
```

**Hypothesis**: If distractor attention ratio increases in steps 7-9, hallucination emerges in late denoising.

### Distractor Attention Metrics

**File**: `jdocs/scripts/investigation/tools/distractor_attention.py`

```python
def compute_distractor_attention_ratio(spatial_attention, distractor_bbox):
    """
    Args:
        spatial_attention: [27, 27] attention map over image patches
        distractor_bbox: (x1, y1, x2, y2) in pixel coordinates

    Returns:
        ratio: 0.0-1.0 (fraction of attention to distractor region)
    """
    distractor_mask = create_patch_mask(distractor_bbox, grid=(27, 27))
    return (spatial_attention * distractor_mask).sum() / spatial_attention.sum()
```

**Success criteria**:
- Hallucination case: distractor attention ratio **increases** in steps 7-9
- Normal case: distractor attention ratio **stable/decreasing**

### Counterfactual Masking

**File**: `jdocs/scripts/investigation/tools/counterfactual_masking.py`

**Purpose**: Prove causality by removing distractor and re-running inference.

```python
def run_counterfactual_analysis(policy, observation, distractor_bbox):
    # Run with original image
    original_actions, original_attention = run_with_capture(observation)

    # Mask distractor (fill with image mean)
    masked_image = apply_object_mask(observation["image"], distractor_bbox, fill="mean")

    # Run with masked image
    masked_actions, masked_attention = run_with_capture(masked_observation)

    # Compare
    return {
        "action_delta": original_actions - masked_actions,  # Should be large if distractor caused hallucination
        "attention_delta": original_attention - masked_attention,
    }
```

**Expected result**: If masking banana eliminates hallucination → confirms visual distractor is root cause.

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

### Phase 3: Model Introspection (UPDATED)

**Goal**: Understand what the model "sees" and where attention goes.

#### Step 3.1: Cross-Attention Analysis (NEW - PRIORITY)

**This is the most important diagnostic tool.** It captures what the action expert attends to during action generation.

```bash
# Analyze hallucination case
python cross_attention_capture.py \
    --case-dir ../../../../logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --output-dir ../../../../logs/yogurt_banana_leftarm/cross_attention_analysis/case1

# Analyze normal case for comparison
python cross_attention_capture.py \
    --case-dir ../../../../logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --output-dir ../../../../logs/yogurt_banana_leftarm/cross_attention_analysis/case3
```

**Output**:
- `temporal_evolution.png`: 10-panel showing attention at each denoising step
- `distractor_ratio_plot.png`: Line graph of distractor attention over time
- `cross_attention_data.json`: Raw attention data for further analysis

#### Step 3.2: Distractor Attention Quantification

```bash
# Quantify attention to distractor region (requires bounding box)
python distractor_attention.py \
    --cross-attention-dir ../../../../logs/yogurt_banana_leftarm/cross_attention_analysis/case1 \
    --distractor-bbox 100,200,180,280 \
    --output-dir ../../../../logs/yogurt_banana_leftarm/distractor_analysis
```

**What to look for**:
- Distractor attention ratio should be < 0.15 for normal behavior
- Ratio > 0.20 indicates significant distractor attention
- Increasing ratio across denoising steps = hallucination emerging

#### Step 3.3: Counterfactual Masking (Causal Analysis)

```bash
# Prove causality by masking distractor and re-running inference
python counterfactual_masking.py \
    --case-dir ../../../../logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --distractor-bbox 100,200,180,280 \
    --output-dir ../../../../logs/yogurt_banana_leftarm/counterfactual
```

**Expected output**:
- If masking banana eliminates hallucination → confirms visual distractor is root cause
- Large `action_delta` in X,Y coordinates indicates distractor was driving motion

#### Step 3.4: Vision Encoder Self-Attention (SUPPLEMENTARY)

This captures internal image processing but is less useful for root cause analysis.

```bash
python visualize_attention.py \
    --checkpoint $CHECKPOINT \
    --case-dir ../../../../logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --output-dir ../../../../logs/yogurt_banana_leftarm/attention_analysis/case1
```

#### Step 3.5: Compare Cases

```bash
# Side-by-side comparison of cross-attention
python cross_attention_capture.py \
    --compare \
    --case1 ../../../../logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --case2 ../../../../logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --output-dir ../../../../logs/yogurt_banana_leftarm/cross_attention_comparison
```

#### What to Look For

| Metric | Normal Case | Hallucination Case |
|--------|-------------|-------------------|
| Distractor attention ratio | < 0.10 | > 0.20 |
| Attention trend (steps 7-9) | Stable/decreasing | Increasing |
| Attention entropy | Higher (diffuse) | Lower (focused on distractor) |
| Counterfactual action delta | N/A | Large in XY direction |

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

## 8. Current Hypotheses and Verification Status (2026-01-19)

### Critical Observation

**The hallucinating arm goes to where the bottle USED TO BE, NOT to the banana (distractor).**

This is the key insight for understanding the root cause.

### Hypothesis Status Summary

| # | Hypothesis | Status | Evidence |
|---|------------|--------|----------|
| H1 | Visual distractor (banana) triggers hallucination via cross-attention | **REJECTED** | Cross-attention shows LESS distractor attention in hallucination case; counterfactual masking shows no effect |
| H2 | Model "replays" previous pick action (goes to bottle's original location) | **NEEDS VERIFICATION** | Behavioral observation supports this |
| H3 | Training data lacks clear "stay still" patterns after task completion | **NEEDS VERIFICATION** | Dataset analysis required |
| H4 | Diffusion/flow-matching favors smooth trajectories over abrupt stops | **NEEDS VERIFICATION** | Denoising trajectory analysis required |
| H5 | KV cache retains "stale" visual information from early task phase | **NEEDS VERIFICATION** | KV cache analysis required |

### Detailed Hypothesis Analysis

#### H1: Visual Distractor Triggers Hallucination (REJECTED)

**Original claim**: The banana is visually detected, and the model's attention to it triggers pick actions.

**Evidence AGAINST**:
1. Cross-attention analysis: Hallucination case has 0.93% distractor attention vs normal case 1.00% - hallucination has LESS attention
2. Counterfactual masking: Removing banana changes action norm by only 0.12 (vs 3.5 total) - negligible effect
3. Behavioral observation: Arm goes to bottle's original location, NOT to banana location

**Status**: Rejected. Visual distractor is not the direct cause.

---

#### H2: Action Replay / Trajectory Memory (NEEDS VERIFICATION)

**Claim**: The model "remembers" the pick-up trajectory and replays it after task completion.

**Mechanism**: The VLM prefix (KV cache) may encode trajectory-related information that persists, causing the model to regenerate similar actions.

**Evidence FOR**:
- Arm goes to where bottle USED TO BE (original pick location)
- Gripper opens during hallucination (consistent with "pick" action)
- Pattern resembles initial pick-up motion

**Verification needed**:
1. Compare hallucination trajectory to initial pick-up trajectory
2. Analyze action similarity metrics across task phases
3. Check if KV cache encodes trajectory patterns

---

#### H3: Training Data Bias - Missing "Stay Still" Patterns (NEEDS VERIFICATION)

**Claim**: Training data doesn't have enough examples of arms staying still after task completion.

**Mechanism**: Without clear "idle" examples, the model defaults to exploratory/reaching behavior.

**Evidence FOR**:
- Hallucination occurs only after task completion
- The model continues generating actions instead of stopping

**Verification needed**:
1. Analyze training dataset: What % of frames are post-completion "idle"?
2. Check action distribution in final phase of episodes
3. Compare to successful no-hallucination case timing

---

#### H4: Diffusion/Flow-Matching Trajectory Smoothness (NEEDS VERIFICATION)

**Claim**: The diffusion process prefers smooth trajectory continuation over abrupt stops.

**Mechanism**: Flow-matching learns velocity fields; stopping requires predicting near-zero velocity, which may be under-represented.

**Evidence FOR**:
- Flow-matching is trained on continuous trajectories
- Stopping is a discontinuity that may be harder to model

**Verification needed**:
1. Analyze denoising velocity fields during normal vs hallucination
2. Check if velocity magnitude drops to zero in normal case
3. Compare action chunk boundaries

---

#### H5: KV Cache Stale Information (NEEDS VERIFICATION)

**Claim**: The KV cache, computed at chunk start, retains "stale" visual features from when bottle was present.

**Mechanism**: The VLM encodes scene state at chunk boundary; if bottle was still in hand, this may persist.

**Evidence FOR**:
- KV cache is computed once per 50-step chunk
- Visual features may encode object positions from chunk start

**Verification needed**:
1. Map KV cache updates to task phase
2. Check what scene state is encoded at each chunk boundary
3. Analyze if hallucination correlates with specific chunk boundaries

---

### Phase 2 Research Plan

#### Step 1: Understand Normal VLA Flow

Before explaining hallucination, understand how normal operation works:

1. **Task execution flow**: How do attention + diffusion work together during pick-place?
2. **Task completion detection**: How does the model know to stop?
3. **Normal action patterns**: What does "idle" look like in normal case?

**Tools needed**: Enhanced trace analysis, action trajectory comparison

#### Step 2: Trajectory Comparison (Verify H2)

Compare hallucination trajectory to initial pick-up:

```bash
# TODO: Create trajectory_comparison.py
python trajectory_comparison.py \
    --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --compare-phases "initial_pick" "hallucination" \
    --output-dir logs/yogurt_banana_leftarm/trajectory_analysis
```

**Expected output**: Similarity score, trajectory overlay, joint-by-joint comparison

#### Step 3: Dataset Analysis (Verify H3)

Analyze training data for post-completion patterns:

```bash
# TODO: Run dataset analysis
python analyze_dataset.py \
    --dataset-path datasets_bimanuel/multitasks \
    --focus post_completion \
    --output-dir logs/investigation/dataset_analysis
```

**Key questions**:
- What % of frames are in "idle/post-completion" phase?
- What are typical action values after task completion?
- Are there clear "stay still" patterns?

#### Step 4: Denoising Dynamics Analysis (Verify H4)

Analyze flow-matching behavior:

```bash
# TODO: Create denoising_analysis.py
python denoising_analysis.py \
    --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --compare-to logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --output-dir logs/yogurt_banana_leftarm/denoising_analysis
```

**Expected output**: Velocity field comparison, noise-to-action trajectory, convergence analysis

---

## 9. References

### Source Code - Key Locations for Cross-Attention Analysis

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Cross-attention forward | `smolvlm_with_expert.py` | 309-422 | Where cross-attention happens |
| Softmax computation | `smolvlm_with_expert.py` | 575 | **HOOK POINT** for attention capture |
| KV cache creation | `modeling_smolvla.py` | 797-804 | Prefix embeddings cached here |
| Denoising loop | `modeling_smolvla.py` | 809-841 | 10-step action generation |
| Debug tracker | `debug_tracker.py` | - | Existing debug infrastructure |

### Investigation Tools

| Tool | Location | Purpose |
|------|----------|---------|
| Trace collection | `jdocs/scripts/investigation/tools/trace_inference.py` | Capture inference traces |
| Self-attention viz | `jdocs/scripts/investigation/tools/visualize_attention.py` | Vision encoder attention |
| **Cross-attention** | `jdocs/scripts/investigation/tools/cross_attention_capture.py` | **Action expert → VLM** |
| **Distractor analysis** | `jdocs/scripts/investigation/tools/distractor_attention.py` | Quantify distractor attention |
| **Counterfactual** | `jdocs/scripts/investigation/tools/counterfactual_masking.py` | Object masking experiments |

### Collected Cases

| Case | Location | Condition |
|------|----------|-----------|
| Hallucination | `logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table` | Banana on table |
| Normal | `logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj` | No distractor |
| Analysis report | `logs/yogurt_banana_leftarm/hallucination_analysis_report.md` | Findings summary |

### Research Papers - VLA Hallucination Analysis

**Architecture & Training**:
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844) - SmolVLA architecture and training
- [SeqVLA: Completion-Aware VLA](https://roboticsproceedings.org/rss20/p112.pdf) - Learned completion detection

**Attention Analysis Methods**:
- [GMAR: Gradient-Driven Multi-Head Attention Rollout](https://arxiv.org/html/2504.19414v1) - Enhanced attention visualization
- [AttnLRP: Attention-Aware Layer-wise Relevance Propagation](https://arxiv.org/html/2402.05602v2) - Faithful attention attribution
- [Devils in Middle Layers of VLMs](https://openaccess.thecvf.com/content/CVPR2025/) - Middle-layer hallucination analysis

**VLA Robustness**:
- [LIBERO-Plus: VLA Robustness Analysis](https://arxiv.org/html/2510.13626.pdf) - Systematic perturbation studies
- [Mechanistic Interpretability for VLAs](https://vla-mech-interp.github.io/) - Activation steering for VLAs

**VLM Hallucination**:
- [VADE: Visual Attention Guided Hallucination Detection](https://aclanthology.org/2025.findings-acl.773.pdf)
- [VIB-Probe: Hallucination-Sensitive Head Detection](https://arxiv.org/html/2601.05547v1)

---

## 9. Phase 2: Mechanistic Understanding Investigation

### 9.1 Critical Re-evaluation of Previous Conclusions

**Previous conclusions that need revision:**

| Previous Finding | Problem | Revised Understanding |
|------------------|---------|----------------------|
| H3: "Training data lacks idle examples" | **Doesn't explain why 2 normal cases DO stay still** | Model CAN output idle actions; something specific triggers hallucination |
| H1 Rejection: "Visual distractor not the cause" | Cross-attention to banana is low, but distractor still correlates with hallucination | Distractor may affect processing differently than direct attention |

**The fundamental question remains unanswered:**
- In ALL three cases, the task completes successfully (bottle placed in bin)
- In 2 cases, arm stays still (correct behavior)
- In 1 case, arm reaches back (hallucination)
- **What is mechanistically different?**

### 9.2 VLA Mechanism: Complete Flow Diagram

Based on codebase analysis (`modeling_smolvla.py`, `smolvlm_with_expert.py`):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SmolVLA COMPLETE INFERENCE FLOW                          │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT STAGE (Per Chunk):
═══════════════════════════════════════════════════════════════════════════════════

    Raw Images              Task Text                Robot State
    (B×3×480×640)          "pick up bottle"         (B×12 joints)
         │                      │                        │
         ▼                      ▼                        ▼
    ┌─────────┐           ┌──────────┐            ┌──────────┐
    │ SigLIP  │           │Tokenizer │            │  Linear  │
    │ Vision  │           │ (48 tok) │            │  Project │
    │ Encoder │           │          │            │  (→768d) │
    └────┬────┘           └────┬─────┘            └────┬─────┘
         │                     │                       │
         ▼                     ▼                       ▼
    729 patches            48 tokens               1 token
    (B, 729, 768)         (B, 48, 768)           (B, 1, 768)
         │                     │                       │
         └──────────┬──────────┴───────────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │      PREFIX ASSEMBLY         │
         │  [img_special + patches +   │
         │   language + state]         │
         │  Total: ~778 tokens         │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │        KV CACHE             │  ← COMPUTED ONCE PER 50-STEP CHUNK
         │  key_states, value_states   │  ← FROZEN FOR ALL 10 DENOISING STEPS
         │  Contains scene+task context│
         └──────────────┬──────────────┘
                        │
                        │  (Reused for all 10 denoising steps)
                        │
════════════════════════╧═══════════════════════════════════════════════════════════

DENOISING LOOP (10 iterations per chunk):
═══════════════════════════════════════════════════════════════════════════════════

    x_0 = N(0,1)  ← Start with random noise (B, 50, 32)
         │
    ┌────┴────────────────────────────────────────────────────────────────────────┐
    │  FOR step = 0 to 9:                                                         │
    │                                                                              │
    │    time = 1.0 - step * 0.1   (1.0 → 0.1)                                   │
    │         │                                                                    │
    │         ▼                                                                    │
    │    ┌─────────────────────────────────┐                                      │
    │    │  SUFFIX EMBEDDING               │                                      │
    │    │  action_emb = project(x_t)     │                                      │
    │    │  time_emb = sinusoidal(time)   │                                      │
    │    │  suffix = MLP(concat(action,   │                                      │
    │    │                     time))     │                                      │
    │    └──────────────┬──────────────────┘                                      │
    │                   │                                                          │
    │                   ▼                                                          │
    │    ┌─────────────────────────────────────────────────────────────┐          │
    │    │              ACTION EXPERT CROSS-ATTENTION                   │          │
    │    │                                                              │          │
    │    │  Q = action_expert.q_proj(suffix)   ← Action tokens query   │          │
    │    │  K = kv_cache.key_states            ← From prefix (frozen)  │          │
    │    │  V = kv_cache.value_states          ← From prefix (frozen)  │          │
    │    │                                                              │          │
    │    │  attention = softmax(Q·K^T / √d)                            │          │
    │    │  output = attention · V                                      │          │
    │    │                                                              │          │
    │    │  ┌──────────────────────────────────────────────────────┐   │          │
    │    │  │  THIS IS THE DECISION POINT:                         │   │          │
    │    │  │  - Which image patches get high attention?           │   │          │
    │    │  │  - Which language tokens influence actions?          │   │          │
    │    │  │  - Does the model "see" task is complete?            │   │          │
    │    │  └──────────────────────────────────────────────────────┘   │          │
    │    └──────────────┬──────────────────────────────────────────────┘          │
    │                   │                                                          │
    │                   ▼                                                          │
    │    ┌─────────────────────────────────┐                                      │
    │    │  VELOCITY PREDICTION            │                                      │
    │    │  v_t = action_out_proj(output) │                                      │
    │    │  (B, 50, 32)                    │                                      │
    │    └──────────────┬──────────────────┘                                      │
    │                   │                                                          │
    │                   ▼                                                          │
    │    ┌─────────────────────────────────┐                                      │
    │    │  EULER UPDATE                   │                                      │
    │    │  x_{t+1} = x_t + dt * v_t       │                                      │
    │    │  (dt = -0.1)                    │                                      │
    │    └──────────────┬──────────────────┘                                      │
    │                   │                                                          │
    │                   ▼                                                          │
    │    x_t = x_{t+1}  (updated for next iteration)                              │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    x_final (B, 50, 32) ← Final action trajectory for next 50 steps
```

### 9.3 Key Insight: Where "Stay Still" vs "Move" is Decided

**The critical decision happens in CROSS-ATTENTION:**

For the model to output "stay still" (near-zero actions):
1. **Vision must encode "task complete"** - No target object in expected location
2. **Language context must not trigger movement** - "pick up" should not dominate
3. **Velocity field must predict near-zero** - v_t ≈ 0 at all denoising steps

**Questions to answer:**
- At step 200+ (post-task completion), what does KV cache contain?
- What image patches receive high cross-attention?
- How does the presence of banana change these patterns?

### 9.4 Revised Hypotheses

| ID | Hypothesis | Mechanism | Test Method |
|----|------------|-----------|-------------|
| **H4** | **KV Cache State Encoding** | KV cache encodes robot's earlier position (with bottle), causing "replay" | Compare KV cache contents at step 200 between cases |
| **H5** | **Cross-Attention to Scene Context** | Banana presence changes global scene representation (not direct attention to banana) | PCA/t-SNE of prefix embeddings |
| **H6** | **Denoising Trajectory Divergence** | Same inputs produce different trajectories due to velocity field instability | Compare denoising with same noise seed |
| **H7** | **Language Token Persistence** | "pick up" tokens retain high attention weight after task completion | Per-token attention analysis |
| **H8** | **Chunk Boundary Corruption** | Error introduced at chunk transition (step 150, 200, 250) amplifies | Analyze chunk encoder output differences |

### 9.5 Experiments to Run

#### Experiment 1: Normal Behavior Characterization

**Goal**: Document exactly how the model produces "stay still" in normal cases

```bash
python characterize_normal_behavior.py \
    --case-dir logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --output-dir logs/yogurt_banana_leftarm/normal_mechanism_analysis \
    --capture-kv-cache \
    --capture-denoising
```

**Metrics to collect**:
- Step at which action delta drops below 3.0 (transition to idle)
- KV cache statistics at transition point
- Denoising velocity magnitude over time
- Cross-attention distribution (image vs language vs state)

#### Experiment 2: Divergence Point Analysis

**Goal**: Find exactly WHERE in the pipeline hallucination diverges

```bash
python divergence_analysis.py \
    --hallucination-case logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --normal-case logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --compare-steps 150,200,210,250,300 \
    --output-dir logs/yogurt_banana_leftarm/divergence_analysis
```

**Comparison metrics**:
1. Vision features (SigLIP output): Cosine similarity
2. Prefix embeddings: L2 distance per token position
3. KV cache: Key/value state distances
4. Action output: Joint-by-joint comparison

#### Experiment 3: KV Cache Ablation

**Goal**: Test if forcing KV cache reset eliminates hallucination

```bash
python kv_cache_ablation.py \
    --scene hallucination \
    --reset-at-step 200 \
    --compare-with-baseline \
    --output-dir logs/yogurt_banana_leftarm/kv_ablation
```

**Success criterion**: If reset eliminates hallucination, KV cache staleness is confirmed as cause

#### Experiment 4: Denoising Trajectory Comparison

**Goal**: Find which denoising step introduces hallucination

```bash
python compare_denoising_trajectories.py \
    --hallucination-case logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --normal-case logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
    --inference-step 250 \
    --use-same-noise-seed \
    --output-dir logs/yogurt_banana_leftarm/denoising_comparison
```

**Expected visualization**:
```
Denoising Trajectory Comparison (Step 250)
     ▲
 15  │                              ╱ Hallucination
     │                           ╱
 10  │                        ╱
     │              ╭────────╯
  5  │         ╭────╯
     │      ╭──╯
  0  │──────╯  Normal (stays near zero)
     └─────┬──┬──┬──┬──┬──┬──┬──┬──┬──►
           0  1  2  3  4  5  6  7  8  9
                    Denoising Step
```

#### Experiment 5: Language Token Attention

**Goal**: Check if "pick up" tokens retain attention after task completion

```bash
python analyze_language_attention.py \
    --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
    --steps 100,200,250,300 \
    --output-dir logs/yogurt_banana_leftarm/language_attention
```

### 9.6 Tools to Build

| Priority | Tool | Purpose | Key Functions |
|----------|------|---------|---------------|
| HIGH | `characterize_normal_behavior.py` | Document normal "stay still" mechanism | `extract_transition_point()`, `analyze_idle_generation()` |
| HIGH | `divergence_analysis.py` | Compare hallucination vs normal at each pipeline stage | `compare_vision_features()`, `compare_kv_cache()`, `compare_actions()` |
| HIGH | `kv_cache_analysis.py` | Extract and analyze KV cache contents | `extract_kv_cache()`, `compute_token_distances()`, `visualize_kv_evolution()` |
| MEDIUM | `kv_cache_ablation.py` | Test KV cache reset intervention | `reset_kv_cache_at_step()`, `compare_with_baseline()` |
| MEDIUM | `compare_denoising_trajectories.py` | Compare denoising with controlled noise | `set_noise_seed()`, `compare_velocity_fields()`, `plot_trajectory_divergence()` |
| MEDIUM | `analyze_language_attention.py` | Per-token language attention over time | `extract_language_attention()`, `plot_token_heatmap()` |

### 9.7 Success Criteria

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| **Identify divergence stage** | Which probe shows largest difference | Clear identification with >3σ separation |
| **Quantify mechanism** | Metrics at identified stage | Reproducible across 5+ runs |
| **Causal verification** | Ablation eliminates hallucination | >80% reduction in hallucination rate |
| **Explain normal behavior** | Document why normal cases stay still | Complete mechanism diagram |

### 9.8 Visualization Dashboard

Create side-by-side comparison at key steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 HALLUCINATION vs NORMAL - Step 250                          │
├─────────────────────────────────┬───────────────────────────────────────────┤
│        HALLUCINATION            │              NORMAL                        │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  [Head Image]                   │  [Head Image]                             │
│   Arm reaching toward table     │   Arm at rest position                    │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  [Cross-Attention Heatmap]      │  [Cross-Attention Heatmap]                │
│   Image: 95%, Lang: 5%          │   Image: 92%, Lang: 8%                    │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  [Denoising Velocity Plot]      │  [Denoising Velocity Plot]                │
│   High magnitude, directed      │   Low magnitude, stationary               │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  Action Delta: 12.3             │  Action Delta: 2.1                        │
│  Gripper: OPENING (30.9)        │  Gripper: CLOSED (7.2)                    │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### 9.9 Investigation Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 2a | 2-3 days | Build probes, run Experiments 1-2 (normal characterization, divergence) |
| Phase 2b | 2-3 days | Run Experiments 3-4 (KV cache ablation, denoising comparison) |
| Phase 2c | 1-2 days | Run Experiment 5 (language attention), aggregate findings |
| Phase 2d | 1 day | Write mechanistic understanding report |

### 9.10 Key Code Locations for Probing

| Component | File | Lines | What to Capture |
|-----------|------|-------|-----------------|
| Vision encoding | `modeling_smolvla.py` | 403-443 | Image preprocessing, patch creation |
| Prefix embedding | `modeling_smolvla.py` | 598-690 | Token assembly, attention masks |
| KV cache creation | `modeling_smolvla.py` | 797-804 | `past_key_values` structure |
| Cross-attention | `smolvlm_with_expert.py` | 539-584 | `attn_weights` after softmax |
| Denoising loop | `modeling_smolvla.py` | 809-841 | `x_t`, `v_t` at each step |
| Action output | `modeling_smolvla.py` | 848-858 | Final action projection |

---

## 10. Phase 2a Findings: Trajectory Shape Analysis

### 10.1 Experimental Setup

Three cases collected with controlled banana position:

| Case ID | Banana Location | Hallucination? | Trace Path |
|---------|-----------------|----------------|------------|
| halluc_table | On table (near workspace) | **YES** | `case_20260119_131914_ha_bana_table` |
| normal_plate | On plate (away from workspace) | No | `case_20260119_132946_no_ha_plate` |
| normal_none | Not present | No | `case_20260119_133142_no_ha_no_other_obj` |

### 10.2 Key Discovery: action_raw vs action_final

**Critical insight**: The divergence is in `action_raw` (normalized model output), NOT in denormalization.

```
action_raw  → Normalized model output (around 0 = training mean)
action_final → Denormalized for robot execution
```

| Case | action_raw J1 (step 200) | action_final J1 (step 200) | Robot Behavior |
|------|--------------------------|----------------------------|----------------|
| Normal | -0.74 | -100.25 | Stays at -100.40 (still) |
| Halluc | -0.71 to +1.03 (ramp) | -99.00 to +6.12 (ramp) | Moves from -101 to +1 |

### 10.3 Trajectory Shape Comparison (3 Cases)

**Chunk 4 (Steps 200-249) - Critical Divergence Point:**

| Case | Start J1 | End J1 | Delta | Shape | Interpretation |
|------|----------|--------|-------|-------|----------------|
| halluc_table | -0.71 | +1.03 | **+1.74** | **RAMP UP ↑** | "Move forward for 50 steps" |
| normal_plate | -0.25 | -0.73 | -0.49 | RAMP DOWN ↓ | "Settle to still" |
| normal_none | -0.74 | -0.73 | +0.01 | **FLAT** | "Stay still for 50 steps" |

**Full Chunk Timeline:**

```
Chunk 3 (150-199): Return to home - ALL cases show RAMP DOWN (expected)
Chunk 4 (200-249): DIVERGENCE POINT
  - halluc_table: RAMP UP ↑ (+1.74) ← HALLUCINATION STARTS
  - normal_plate: RAMP DOWN ↓ (-0.49) ← Still settling
  - normal_none:  FLAT (+0.01) ← Perfect stay-still

Chunk 5 (250-299): Sustained behavior
  - halluc_table: RAMP UP ↑ (+0.75) ← Hallucination continues
  - normal_plate: FLAT (-0.01) ← Settled to stay-still
  - normal_none:  FLAT (+0.00) ← Continues stay-still

Chunk 6 (300-349): Late phase
  - halluc_table: RAMP DOWN ↓ (-0.35) ← Starting to return
  - normal_plate: FLAT (-0.01)
  - normal_none:  FLAT (+0.00)
```

### 10.4 Key Insights

1. **Banana position matters, not just presence**:
   - Banana on table → Hallucination (RAMP UP trajectory)
   - Banana on plate → No hallucination (self-corrects to FLAT)
   - No banana → No hallucination (FLAT from start)

2. **Normal-plate case self-corrects**:
   - Starts with perturbation in chunk 4 (-0.25, not -0.74)
   - Settles to FLAT by chunk 5
   - Model CAN handle distractor if not in workspace

3. **Hallucination is a sustained RAMP, not random noise**:
   - Consistent upward trend: -0.71 → +1.03 → +1.76
   - Model predicts "gradually move forward" trajectory
   - This is a coherent (but wrong) plan, not noise

4. **Divergence emerges WITHIN chunk, not at boundary**:
   - At chunk 4 start (step 200), difference is small (0.079)
   - Grows throughout chunk to 4.45 by step 249
   - The 50-step trajectory SHAPE is different

### 10.5 Remaining Questions

| Question | Why It Matters | How to Answer |
|----------|---------------|---------------|
| What visual features trigger RAMP vs FLAT? | Identifies root cause | Compare cross-attention heatmaps |
| At which denoising step does RAMP emerge? | Locates mechanism | Capture x_t at each of 10 steps |
| Why does banana-on-table prevent self-correction? | Explains sustained hallucination | Compare KV cache contents |
| Is the "ramp" trajectory learned from training? | Training vs inference issue | Analyze training trajectories |

### 10.6 Tools Built

| Tool | Purpose | Status |
|------|---------|--------|
| `characterize_normal_behavior.py` | Document normal "stay still" mechanism | ✅ Complete |
| `divergence_analysis.py` | Compare cases, find divergence point | ✅ Complete |
| `denoising_trajectory_capture.py` | Track x_t, v_t during 10-step denoising | 🔲 To build |
| `cross_attention_comparison.py` | Compare attention across 3 cases | 🔲 To build |

### 10.7 Next Steps

1. **Build denoising trajectory capture tool**:
   - Hook into `denoise_step` method
   - Capture x_t and velocity v_t at each of 10 steps
   - Compare trajectory evolution between cases

2. **Analyze cross-attention at chunk 4 start**:
   - What image patches get attention in each case?
   - Does banana-on-table get different attention than banana-on-plate?

3. **Test hypothesis: Workspace region attention**:
   - Banana on table is IN the workspace region
   - Banana on plate is OUTSIDE workspace region
   - Model may attend to workspace and see "object to interact with"
