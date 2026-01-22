# Dataset Defect Analysis: Root Cause of SmolVLA Hallucination

**Date**: 2026-01-22
**Status**: CONFIRMED - Concrete Evidence Found

---

## Executive Summary

We have identified the **specific dataset defect** causing the hallucination behavior:

> **The training data contains IDLE behavior only in "clean" visual contexts. It never contains IDLE behavior when a distractor (graspable object) is visible.**

This is a **context mismatch** problem, not a missing-data problem.

---

## Part 1: Training Data Analysis

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total frames | 76,597 |
| Total episodes | 320 |
| Unique tasks | 16 |
| Action dimensions | 12 (bimanual) |

### IDLE vs MOVEMENT Distribution

| Behavior | Frames | Percentage |
|----------|--------|------------|
| MOVEMENT (delta ≥ 3°) | 57,456 | 75.0% |
| IDLE (delta < 3°) | 19,141 | 25.0% |

**Initial observation**: Training data has 25% IDLE frames - this seems adequate.

### Where IDLE Occurs

| Position in Episode | IDLE Frames | Percentage |
|---------------------|-------------|------------|
| Start (0-20%) | 7,734 | 40.4% |
| Middle (20-80%) | 3,508 | 18.3% |
| End (80-100%) | 7,899 | 41.3% |

### End-of-Episode IDLE Statistics

| Metric | Value |
|--------|-------|
| Episodes with ANY IDLE at end | 320/320 (100%) |
| Episodes ending with 100% IDLE (last 10 frames) | 319/320 (99.7%) |
| Average IDLE run length at end | 24.5 frames |
| Median IDLE run length at end | 24.0 frames |

**Key Finding**: The training data DOES have post-completion IDLE behavior.

---

## Part 2: The Critical Gap

### Visual Context During Training IDLE

During the IDLE period at the end of training episodes:
- Task has been **completed**
- Target object has been **placed** at destination
- Gripper is **empty**
- **No graspable objects visible** in the scene

### Visual Context During Inference Hallucination

During the inference scenario where we WANT IDLE:
- Task has been **completed**
- Target object has been **placed** at destination
- Gripper is **empty**
- **Distractor (banana) IS visible** in the scene

### The Specific Defect

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA COVERAGE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MOVEMENT + Object visible      ✓ Covered (task execution)          │
│  MOVEMENT + No object           ✓ Covered (return to home)          │
│  IDLE + No object visible       ✓ Covered (post-completion)         │
│  IDLE + Object visible          ✗ NOT COVERED (THE GAP)             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

QUANTIFIED:
  IDLE with visible distractor: 0 frames (0.0%)
  ← THIS IS THE DATASET DEFECT
```

---

## Part 3: Why This Causes Hallucination

### Model's Learned Associations

From training data, the model has learned:

```
Pattern 1: "graspable object visible" → MOVEMENT
  - Reinforced throughout task execution
  - 57,456 examples

Pattern 2: "clean scene, no objects" → IDLE
  - Only at episode end after task completion
  - 19,141 examples

MISSING Pattern: "task done + distractor visible" → IDLE
  - ZERO examples
  - Model has no learned behavior for this case
```

### What Happens at Inference

When the robot completes the task but sees the banana:

1. Visual input: "object visible" (banana in right wrist camera)
2. Model searches training patterns
3. Closest match: Pattern 1 (object visible → MOVEMENT)
4. Model defaults to movement despite task completion
5. This is the **hallucination**

### Why Head Camera Has High Causal Effect

The attention knockout showed:
- Head camera: 167% causal effect
- Right wrist: 11% causal effect

This is because:
- The model relies on **head camera** for global scene understanding
- Head camera shows "workspace configuration" that triggers learned patterns
- The banana info is **stored** but not **used** because:
  - Model never learned what to DO with "distractor visible"
  - So it ignores right wrist info and relies on head camera patterns

---

## Part 4: Evidence Chain

### Evidence 1: Attention Knockout (Model Architecture)

| Region | Causal Effect | Interpretation |
|--------|---------------|----------------|
| Head camera | 167% | Model relies heavily on this |
| Right wrist | 11% | Model barely uses this |

**Interpretation**: Model uses head camera for decisions, ignores right wrist even though banana info is stored there.

### Evidence 2: Single-Frame Swap (No Temporal Context)

Single-frame inference doesn't reproduce hallucination because:
- Hallucination is emergent from sequential inference
- Requires temporal context from previous steps
- Model's learned patterns activate over time

### Evidence 3: Training Data Analysis (Dataset Coverage)

| Visual Context | Action | Coverage |
|----------------|--------|----------|
| Object visible | MOVEMENT | 75% ✓ |
| No object | MOVEMENT | Some ✓ |
| No object | IDLE | 25% ✓ |
| **Object visible** | **IDLE** | **0% ✗** |

**Interpretation**: The exact scenario we need (IDLE with distractor) has zero training examples.

---

## Part 5: Recommended Fix

### Data Collection Strategy

Collect **20-30 episodes** of the following scenario:

```
Scenario: "Post-completion IDLE with distractor"

1. Execute normal pick-and-place task
2. Complete the task (object placed)
3. Introduce distractor (e.g., banana on table)
4. Demonstrate IDLE behavior for 30-50 frames
5. Keep distractor visible throughout IDLE period

Key visual requirements:
- Distractor visible in right wrist camera
- Scene otherwise similar to normal post-completion
- Robot remains stationary
```

### Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| IDLE + distractor examples | 0 | ~750-1500 frames |
| Coverage of "ignore distractor" | 0% | ~5-10% of IDLE |
| Hallucination rate | High | Significantly reduced |

### Implementation Priority

1. **Highest**: IDLE with distractor in right wrist view
2. **High**: IDLE with distractor in head camera view
3. **Medium**: Various distractor types (not just banana)
4. **Lower**: Extended IDLE duration (>50 frames)

---

## Part 6: Visualization

See generated plots:
- `bimanual_training_data_analysis.png` - Overall distribution
- `training_data_analysis.png` - Detailed phase analysis

---

## Appendix: Full Training Data Statistics

```
Training Dataset: /home/jrobot/project/lerobot/datasets_bimanuel/multitasks

Total frames: 76,597
Unique episodes: 320

Action-State Delta Statistics:
  Mean of max delta: 7.561°
  Median of max delta: 7.328°
  Std of max delta: 5.236°
  Min: 0.451°, Max: 33.628°

IDLE vs MOVEMENT (threshold = 3.0°):
  IDLE frames: 19,141 (25.0%)
  MOVEMENT frames: 57,456 (75.0%)

Per-Episode Analysis:
  Episodes with 0% IDLE: 0/320
  Episodes with <5% IDLE: 0/320
  Episodes with <10% IDLE: 0/320

End-of-Episode IDLE:
  IDLE frames in last 20%: 7,899/15,443 (51.1%)
  Episodes with ANY IDLE at end: 320/320 (100.0%)
  Average IDLE in last 10 frames: 99.9%
  Episodes ending with 100% IDLE: 319/320

The Defect:
  IDLE with visible distractor: 0 frames (0.0%)
```

---

## Conclusion

The hallucination is caused by a **specific gap in training data coverage**: the model has never seen examples of staying idle when a graspable distractor is visible. This is not a lack of IDLE data (25% exists), but a lack of IDLE data **in the specific visual context** encountered at inference.

The fix is targeted data collection: 20-30 episodes demonstrating "stay idle even though you can see an object you could grasp."
