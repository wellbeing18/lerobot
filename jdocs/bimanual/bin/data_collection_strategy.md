# Data Collection Strategy to Improve SmolVLA Performance

**Date**: 2026-01-25
**Status**: IMPLEMENTATION READY (Research Validated)
**Based on**: Investigation reports in `jdocs/investigation_reports/`
**Validated against**: External VLA research (see Sources section)

---

## Executive Summary

This plan addresses three SmolVLA performance issues through targeted data collection:
1. **Grasping emptiness after task completion** (hallucination)
2. **Object identification confusion** (ketchup vs yogurt, bread vs used tissue)
3. **Grasp accuracy degradation with multiple objects**

**Key Finding**: The training data has **0% IDLE frames with visible distractors**. This plan builds on that research and extends solutions to Problems 2 and 3.

---

## Hypothesis Evaluation

**Hypothesis**: "Pause a couple of seconds during data recording, then finish the episode"

**Verdict**: **Partially correct**, but needs nuance:

| Aspect | Assessment |
|--------|------------|
| Pausing after task completion | Correct - essential for Problem 1 |
| Duration (couple seconds) | Adjust to 1-2 seconds (30-60 frames) |
| Where to pause | End of episode; mid-task pauses are OK if consistent |

**Note**: Action chunking architectures (like ACT and SmolVLA's flow matching) can handle mid-task pauses if they're consistent. The key insight is about **coverage** - the training data needs examples of IDLE behavior with distractors visible, regardless of when pauses occur.

### Critical Insight

The key issue is **WHAT IS VISIBLE during the wait**:

```
CURRENT (problem):
  Task complete → Object on plate → Wait → End
  During wait: workspace is EMPTY (target moved to destination)
  Result: Model learns IDLE = empty workspace

NEEDED (fix):
  Task complete → Object on plate → Wait WITH DISTRACTOR VISIBLE → End
  During wait: Other object(s) still on TABLE/workspace
  Result: Model learns IDLE = can happen even with objects visible
```

**The fix is NOT just "wait longer" - it's ensuring distractor objects remain visible during the wait period.**

---

## Problem 1: Grasping at Emptiness After Task Completion

### Root Cause (Confirmed in Investigation Reports)

```
Training Data Gap:
  IDLE + Object visible on workspace = 0 frames (0%)

Model's Learned Behavior:
  P(MOVEMENT | object_visible) = 1.0  (always move when object visible)
  P(IDLE | object_visible + task_complete) = undefined  (never learned)
```

### Solution: Post-Completion IDLE with Visible Distractors

#### Episode Type 1A: Single Distractor IDLE (30 episodes)

| Parameter | Value |
|-----------|-------|
| Total frames | 250-350 per episode |
| Post-completion IDLE | 30-60 frames (1-2 seconds) |
| Distractor visibility | 100% during IDLE |

**Recording Protocol**:
1. Place target AND distractor on workspace
2. Execute pick-and-place of target (smooth, normal execution)
3. After placing target on plate/bin, return arm to neutral
4. **Stay completely still for 30-60 frames** with distractor visible
5. End episode

**Task Description**:
```
"Use [left/right] arm to pick up the [target] and place it on the plate"
```
(Mention only target - distractor is ignored)

**Distractor Placement Variations** (vary across episodes):
- Left of where target was
- Right of where target was
- Behind where target was
- Opposite side of table

#### Episode Type 1B: Multi-Distractor IDLE (15 episodes)

Same as 1A but with 2-3 distractor objects on table. Demonstrates ignoring multiple objects.

#### Episode Type 1C: Sequential Task with Persistent Distractor (10 episodes)

- Two objects on table (A and B)
- Task: "Pick up A and place on plate"
- Complete A, B remains, stay IDLE
- Shows model: completing one task != pick up all objects

### Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| IDLE with distractor frames | 0 | ~2,000-3,000 |
| Post-completion hallucination | ~95% | Significantly reduced |

---

## Problem 2: Object Identification Hallucination

### Root Cause: Visual-Language Grounding Failure

The model confuses visually similar objects because it lacks **contrastive training data** showing both confusable objects together.

### Identified Confusable Pairs

| Object A | Object B | Similarity | Priority |
|----------|----------|------------|----------|
| Ketchup | Yogurt | Both bottles, similar shape | Critical |
| Bread | Used tissue | Both soft, light-colored | Critical |
| Corn | Banana | Both yellow | High |
| Orange | Ice cream | Both round-ish | High |

### Solution: Contrastive Object Pair Training

#### Episode Type 2A: Confusable Pairs (40 episodes)

For each confusable pair, collect episodes with BOTH objects present:

```
Scene: Ketchup + Yogurt on table
Episode X: Task = "Pick up the ketchup bottle and place on plate"
Episode Y: Task = "Pick up the yogurt bottle and place in bin"
```

**Critical Requirements**:
- Both objects clearly visible in all cameras simultaneously
- Vary positions (prevent position-based shortcuts)
- Use exact object names in task descriptions

**Minimum per pair**: 8-10 episodes (4-5 picking each)

#### Episode Type 2B: Attribute-Based Discrimination (20 episodes)

| Scenario | Task Example |
|----------|--------------|
| Color discrimination | "Pick up the RED bottle" (ketchup vs yogurt) |
| Shape within color | "Pick up the ROUND yellow object" (orange vs corn) |
| Texture-based | "Pick up the bread" (bread vs used tissue) |

#### Episode Type 2C: Explicit Negative Examples (10 episodes)

Show the model clearly ignoring the wrong object:
- Camera captures approach to correct object
- Wrong object visible but untouched
- Clean grasp of correct object only

### Expected Impact (Estimates - validate empirically)

| Metric | Expected Improvement |
|--------|---------------------|
| Object confusion rate | Reduced (estimate: 40-60%) |
| Cross-object discrimination | Improved |

---

## Problem 3: Grasp Accuracy with Multiple Objects

### Root Cause: Spatial Attention Overwhelmed

With single objects: smooth, confident grasps
With multiple objects: jerky, stops above object, grasps air

The model's attention is distributed across multiple salient objects, degrading precision.

### Solution: Progressive Multi-Object Training

#### Episode Type 3A: Distractor Count Progression (30 episodes)

| Level | Setup | Episodes |
|-------|-------|----------|
| 1 | Target + 1 distractor (close) | 10 |
| 2 | Target + 2 distractors | 10 |
| 3 | Target + 3 distractors (cluttered) | 10 |

**Recording Requirements**:
- Smooth, confident approach (no hesitation)
- Clean grasp execution
- All objects visible throughout

#### Episode Type 3B: Close-Proximity Grasping (20 episodes)

Train precise discrimination when objects are very close:

| Separation | Episodes |
|------------|----------|
| 8 cm | 5 |
| 5 cm | 10 |
| 3 cm | 5 |

#### Episode Type 3C: Variable Object Sizes (15 episodes)

| Scenario | Purpose |
|----------|---------|
| Small target + large distractors | Don't be distracted by size |
| Large target + small distractors | Maintain precision |
| Mixed sizes | General robustness |

#### Episode Type 3D: Camera Coverage Optimization (10 episodes)

Ensure all cameras contribute useful information:
- Head camera: clear top-down view of ALL objects
- Wrist cameras: target visible during approach

### Expected Impact (Estimates - validate empirically)

| Metric | Expected Improvement |
|--------|---------------------|
| Multi-object grasp success | Improved (estimate: +50-70%) |
| Movement smoothness | Improved |
| Spatial precision | Enhanced |

---

## Complete Data Collection Plan

### Summary Table

| Problem | Episode Types | Episodes | Est. Frames |
|---------|---------------|----------|-------------|
| P1: Post-completion hallucination | 1A, 1B, 1C | 55 | ~15,000 |
| P2: Object confusion | 2A, 2B, 2C | 70 | ~18,000 |
| P3: Multi-object accuracy | 3A, 3B, 3C, 3D | 75 | ~20,000 |
| **Total** | | **200** | **~53,000** |

### Dataset Growth

| Metric | Current | After Collection |
|--------|---------|------------------|
| Episodes | 320 | 520 (+63%) |
| Frames | 76,597 | ~130,000 |
| IDLE with distractor | 0% | ~5-8% |
| Multi-object scenes | Low | ~40% of new data |

---

## Prioritized Collection Order

### Phase 1: Critical (First Priority) - Problem 1 Fix

| Type | Episodes | Rationale |
|------|----------|-----------|
| 1A (IDLE with distractor) | 30 | Direct fix for root cause |
| 1B (Multi-distractor IDLE) | 15 | Extension of fix |
| **Phase 1 Total** | **45** | |

**Validation**: After Phase 1, test on original hallucination scenario. Expect significant reduction in post-completion grasping.

### Phase 2: High Priority - Object Discrimination

| Type | Episodes |
|------|----------|
| 2A (Contrastive pairs) | 40 |
| 3A (Progressive distractors) | 30 |
| **Phase 2 Total** | **70** |

### Phase 3: Enhancement

| Type | Episodes |
|------|----------|
| 2B, 2C, 3B, 3C, 3D, 1C | 85 |

---

## Recording Guidelines

### General Protocol

1. **Pre-Recording**:
   - Position objects deliberately (not randomly)
   - Verify all cameras have clear views
   - Check lighting consistency

2. **During Recording**:
   - Execute movements smoothly and confidently
   - Avoid hesitation or mid-movement corrections
   - Maintain consistent speed

3. **Post-Completion IDLE** (for Problem 1 episodes):
   - Return arm to neutral position
   - Hold completely still for 30-60 frames (start with 30, increase if needed)
   - Ensure distractor objects remain visible
   - Do NOT move toward distractors

### Task Description Format

```
GOOD: "Use left arm to pick up the ketchup bottle and place it on the plate"
BAD:  "Use left arm to pick up the red bottle on the left and put it there"
```

- Use exact object names
- Specify arm (left/right)
- Clear destination (plate/bin)
- No position-based descriptions

---

## Verification Strategy

### After Each Phase

**Phase 1 Verification**:
- Test: Complete task with banana visible on workspace
- Expected: Robot stays IDLE after task, no grasping at empty space

**Phase 2 Verification**:
- Test: Ketchup + yogurt on table, task "pick up ketchup"
- Expected: Correct object picked, no confusion

**Phase 3 Verification**:
- Test: Target + 2 distractors, normal pick-and-place
- Expected: Smooth trajectory, successful grasp

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `jdocs/investigation_reports/FINAL_SUMMARY.md` | Root cause documentation |
| `jdocs/investigation_reports/dataset_defect_findings.md` | Quantified gaps |
| `datasets_bimanuel/multitasks/meta/info.json` | Dataset format reference |
| `datasets_bimanuel/multitasks/meta/tasks.parquet` | Task description format |

---

## Sources

### From Codebase Investigation Reports
- Root cause: 0% IDLE with distractor coverage
- Head camera 167% causal effect, right wrist 11%
- KV cache encodes distractor but model doesn't use it

### From External Research (Validated 2026-01-25)

**Distractor Handling & Hallucination:**
- [Causal Confusion in Robot Learning](https://arxiv.org/html/2507.22380v1): Confirms spurious correlation mechanism; Causal-ACT improves OOD from 0.23→0.88
- [NICE Scene Surgery](https://arxiv.org/pdf/2511.22777): Advocates enriching training with diverse distractors
- VLA Survey: "Stage hallucination" and task completion detection issues documented

**Object Discrimination:**
- [Sigma-Agent Contrastive IL](https://arxiv.org/html/2406.09738v1): +5.9% improvement via contrastive learning
- [ObjectVLA](https://arxiv.org/html/2502.19250v1): 64% success on 100+ novel objects via fine-grained discrimination

**Multi-Object Scenes:**
- [RT-1](https://robotics-transformer1.github.io/): 36% better distractor generalization, tested with 9+ distractors
- [GraspClutter6D](https://arxiv.org/html/2504.06866v1): 14.1 objects/scene benchmark for cluttered grasping

**Action Chunking & Pauses:**
- [ACT/ALOHA](https://tonyzhaozh.github.io/aloha/): "Action chunking mitigates non-Markovian issues from pauses"
- [π0 Flow Matching](https://arxiv.org/html/2410.24164v1): Smooth trajectory generation

**Data Collection:**
- [SmolVLA HuggingFace](https://huggingface.co/docs/lerobot/smolvla): Recommends 50+ episodes, 10 per variation
- [Is Diversity All You Need?](https://arxiv.org/html/2507.06219v1): Task diversity outweighs per-task quantity

---

## Appendix A: Existing Task Descriptions (for reference)

Current tasks in `datasets_bimanuel/multitasks`:

| Task Index | Task Description |
|------------|------------------|
| 0 | Use left arm to pick up the orange and place it on the plate |
| 1 | Use right arm to pick up the orange and place it on the plate |
| 2 | Use left arm to pick up the bread and place it on the plate |
| 3 | Use right arm to pick up the bread and place it on the plate |
| 4 | Use left arm to pick up the corn and place it on the plate |
| 5 | Use right arm to pick up the corn and place it on the plate |
| 6 | Use left arm to pick up the banana and place it on the plate |
| 7 | Use right arm to pick up the banana and place it on the plate |
| 8 | Use left arm to pick up the ice cream and place it in the bin |
| 9 | Use right arm to pick up the ice cream and place it in the bin |
| 10 | Use left arm to pick up the ketchup bottle and place it in the bin |
| 11 | Use right arm to pick up the ketchup bottle and place it in the bin |
| 12 | Use left arm to pick up the yogurt bottle and place it in the bin |
| 13 | Use right arm to pick up the yogurt bottle and place it in the bin |
| 14 | Use left arm to pick up the used tissue and place it in the bin |
| 15 | Use right arm to pick up the used tissue and place it in the bin |

---

## Appendix B: Proposed New Task Descriptions

These maintain the exact format of existing tasks:

### Problem 1 Episodes (IDLE with distractor)
Same task descriptions as existing - the change is in SCENE SETUP (distractor present) and BEHAVIOR (extended IDLE at end).

### Problem 2 Episodes (Contrastive pairs)
Same task descriptions - both objects present in scene but only one mentioned in task.

### Problem 3 Episodes (Multi-object accuracy)
Same task descriptions - multiple distractors in scene.

**Key Point**: The task descriptions don't change. What changes is:
1. Scene composition (distractors present)
2. Post-completion behavior (hold IDLE longer)
3. Object positioning (vary to prevent shortcuts)

---

## Appendix C: Dataset Technical Specs

| Spec | Value |
|------|-------|
| Robot | bi_so101_follower |
| Cameras | head, left_wrist, right_wrist |
| Resolution | 640x480 |
| FPS | 30 |
| Action dims | 12 (6 per arm) |
| Video codec | av1 |
| Data format | Parquet + MP4 |

---

## Appendix D: Recording Commands

```bash
# Standard recording
lerobot-record \
  --robot.type bi_so101_follower \
  --dataset.repo_id datasets_bimanuel/multitasks_v2 \
  --fps 30

# With specific episode count
lerobot-record \
  --robot.type bi_so101_follower \
  --dataset.repo_id datasets_bimanuel/multitasks_v2 \
  --num_episodes 45 \
  --fps 30
```

Refer to `lerobot-record --help` for full options.
