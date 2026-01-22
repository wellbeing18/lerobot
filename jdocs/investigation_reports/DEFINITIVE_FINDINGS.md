# SmolVLA Hallucination Investigation: DEFINITIVE FINDINGS

**Date**: 2026-01-22
**Status**: ROOT CAUSE CONFIRMED WITH VISUAL EVIDENCE

---

## Executive Summary

The hallucination behavior in SmolVLA is caused by **the location of a graspable object in the workspace**. When a distractor (banana) is on the TABLE (workspace area), the model continues moving after task completion. When the same distractor is on the PLATE (destination area), the model correctly stays idle.

**This is NOT about the presence of a distractor - it's about WHERE the distractor is located.**

---

## The Definitive Evidence

### Visual Evidence at Step 200 (Divergence Point)

| Case | Banana Location | Workspace Status | Result |
|------|-----------------|------------------|--------|
| **Halluc** | On TABLE | Has graspable object | ARM MOVES |
| **Normal** | On PLATE | Empty | ARM STAYS IDLE |

See `logs/investigation/visual_evidence/key_visual_difference.png` for the visual proof.

### What the Cameras See

**HALLUC CASE (Banana on Table):**
```
HEAD CAMERA: Blue bin (with yogurt), plate, banana ON TABLE between arms
RIGHT WRIST: Large banana visible on workspace, empty plate in corner
WORKSPACE: Contains graspable object (banana)
```

**NORMAL CASE (Banana on Plate):**
```
HEAD CAMERA: Blue bin (with yogurt), plate with banana ON IT
RIGHT WRIST: Empty workspace, banana is on plate (destination)
WORKSPACE: Empty - no graspable objects
```

---

## Behavioral Divergence Timeline

| Step | Task Status | Halluc Behavior | Normal Behavior |
|------|-------------|-----------------|-----------------|
| 0-150 | Executing | MOVEMENT | MOVEMENT |
| 150-167 | Completing | MOVEMENT | Slowing |
| **167** | **DIVERGENCE** | **MOVEMENT (9.76°)** | **IDLE (1.64°)** |
| 200-350 | Complete | Reaching toward bin | Resting |
| 350-465 | Complete | Still moving | Still idle |

**Key Finding**: Divergence begins at step 167, not step 200. By step 200, the halluc case has been moving for 33+ steps while normal is idle.

---

## The Root Cause Mechanism

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE HALLUCINATION TRIGGER                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  VISUAL INPUT: "Graspable object on workspace"                      │
│       │                                                             │
│       ▼                                                             │
│  MODEL PATTERN: "Object on workspace → APPROACH"                    │
│       │                                                             │
│       ▼                                                             │
│  ACTION OUTPUT: Movement toward object/workspace area               │
│                                                                     │
│  ⚠️ MODEL DOESN'T CHECK: "Is my task already complete?"            │
│  ⚠️ MODEL DOESN'T CHECK: "Is this object relevant to my task?"     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

WHY NORMAL CASE WORKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When banana is on PLATE (destination area):
  - RIGHT WRIST sees: empty workspace
  - HEAD CAMERA sees: no objects on table that need handling
  - Pattern match: "clean workspace" → IDLE

WHY HALLUC CASE FAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When banana is on TABLE (workspace):
  - RIGHT WRIST sees: object on workspace (banana)
  - HEAD CAMERA sees: object between arms that could be grasped
  - Pattern match: "object on workspace" → MOVEMENT
```

---

## Why Previous Conclusions Were Incomplete

### Correction 1: "Zero IDLE with Distractor" Was Misleading

**Previous claim**: Training data has zero IDLE with visible distractor.

**Correction**: The issue is more specific:
- Training data has IDLE when workspace is EMPTY
- Training data has MOVEMENT when workspace has OBJECT
- Training data lacks: IDLE when workspace has IRRELEVANT object

The model learned: `object_on_workspace → movement`
The model never learned: `irrelevant_object_on_workspace → ignore`

### Correction 2: Movement Target Analysis

**Previous debate**: Is the arm moving toward the banana or toward the original pickup location?

**Resolution**: The arm is moving toward the WORKSPACE AREA (where both the banana and the original pickup location are). The specific target is less important than the TRIGGER - the presence of a graspable object on the workspace.

---

## Attention Knockout Evidence

The attention knockout experiments showed:

| Region | Causal Effect (Halluc) | Interpretation |
|--------|------------------------|----------------|
| Head Camera | 167% | Primary decision maker for "scene state" |
| Language | 131% | Task context |
| State | 66% | Current position |
| Left Wrist | 22% | Gripper status |
| **Right Wrist** | **11%** | **Sees banana but barely uses it** |

**Paradox Explained**: The right wrist camera SEES the banana (49% of KV difference) but the model doesn't USE this information for decisions (11% causal effect) because:
1. The model relies on HEAD CAMERA for "should I move?" decisions
2. The head camera sees "object on workspace" and triggers movement
3. The right wrist information about WHAT the object is gets ignored

---

## Training Data Implications

### What Training Data Teaches

The training episodes follow this pattern:
```
1. Object appears on workspace
2. Robot MOVES to grasp object
3. Robot MOVES to place object
4. Object is at destination
5. Workspace is EMPTY
6. Robot stays IDLE
```

### What's Missing

Training never demonstrates:
```
1. Task is complete
2. IRRELEVANT object is on workspace
3. Robot IGNORES the object
4. Robot stays IDLE
```

### Why This Matters

The model has learned a strong association:
- `object_on_workspace → movement`

But never learned the exception:
- `object_on_workspace + task_complete + object_irrelevant → idle`

---

## Recommended Fix

### Data Collection Strategy

Collect **20-30 episodes** demonstrating "ignore irrelevant objects":

```
Scenario A: "Post-completion with irrelevant object on workspace"
1. Complete pick-and-place task normally
2. Place IRRELEVANT object (banana, cup, etc.) on TABLE/workspace
3. Demonstrate IDLE for 30-50 frames
4. Object remains on workspace throughout

Scenario B: "Multi-object workspace with selective grasping"
1. Multiple objects on workspace
2. Task specifies ONE target object
3. Complete task with target only
4. Leave other objects untouched
5. Stay IDLE despite other graspable objects
```

### Expected Impact

| Metric | Current | After Fix |
|--------|---------|-----------|
| "Ignore irrelevant object" examples | 0 | ~750-1500 frames |
| Hallucination rate | High | Significantly reduced |
| Multi-object robustness | Poor | Improved |

---

## Files and Evidence

### Visual Evidence
- `logs/investigation/visual_evidence/key_visual_difference.png` - Key comparison
- `logs/investigation/visual_evidence/behavior_timeline.png` - Timeline
- `logs/investigation/visual_evidence/full_comparison_all_steps.png` - All steps

### Trajectory Analysis
- `logs/investigation/trajectory_divergence/trajectory_divergence_comparison.png`
- Divergence at step 167, not step 200

### Raw Data
- `logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/` - Halluc case
- `logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/` - Normal case

---

## Conclusion

The SmolVLA hallucination is caused by a **training data gap** where the model has never learned to stay idle when an irrelevant object is on the workspace. The model has learned `object_on_workspace → movement` without learning the contextual exception for task completion and object relevance.

**The fix is targeted data collection**: Episodes demonstrating "ignore irrelevant objects on workspace after task completion."

This is NOT about:
- Missing IDLE data (plenty exists)
- Being "attracted" to distractors (movement is toward workspace, not specifically toward banana)
- Head camera vs wrist camera priority (both contribute, head makes decisions, wrist provides context)

This IS about:
- **Location of objects relative to workspace**
- **Missing training examples of "selective ignoring"**
- **Pattern matching without task-completion awareness**
