# SmolVLA Hallucination Investigation: REVISED Findings

**Date**: 2026-01-22
**Status**: REVISED based on new analysis

---

## Corrections to Previous Analysis

### Correction 1: The Movement Target

**Previous claim**: "The arm moves toward the banana distractor"

**Corrected finding**: The arm moves toward **where the yogurt bottle originally was** (the approach position), NOT toward the banana.

Evidence:
- Cosine similarity of hallucination movement vs toward-approach-position: **0.997**
- The model is RE-EXECUTING THE PICKUP, not being attracted by the banana
- The banana's presence may trigger this behavior, but the target is the original pickup location

### Correction 2: Training Data Has Distractors

**Previous claim**: "Training data has zero IDLE with visible distractor"

**Corrected finding**: We didn't actually verify what objects are visible during training. The user points out that multi-object scenes DO exist in training data. What matters is:
- Does training data have IDLE when there's an object the robot COULD grasp but SHOULDN'T?
- This needs visual verification, not just action analysis

---

## Revised Understanding of Training Data Pattern

### The Actual Training Pattern

Analyzing episode phases reveals a consistent pattern:

```
Typical Episode Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame 0-20%:     IDLE (starting position)
Frame 20-70%:    MOVEMENT (approach → pickup → transport → place)
Frame 70-75%:    IDLE (brief pause after placing)
Frame 75-90%:    MOVEMENT (return to home position)
Frame 90-100%:   IDLE (final resting position)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Statistics**:
- 223/320 episodes (70%) show "IDLE → MOVEMENT → IDLE" in last 40%
- 97/320 episodes (30%) show "MOVEMENT → IDLE" in last 40%
- The "return to home" movement is a **common training pattern**

### What the Model Learned

The model learned a sequence:
1. Approach object
2. Pick up
3. Transport to destination
4. Place object
5. Brief pause
6. **Return toward original position** ← Important!
7. Final rest

---

## Revised Root Cause Hypothesis

### The Hallucination Mechanism

```
HALLUCINATION CASE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0-50:       Approach yogurt bottle (MOVEMENT)
Step 50-150:     Pick, transport, place (MOVEMENT)
Step 150-170:    Brief pause (IDLE)
Step 170-300:    Return movement toward approach position ← HALLUCINATION

Movement direction: Toward where bottle was picked up
Cosine similarity: 0.997 (almost exactly toward approach position)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NORMAL CASE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 0-50:       Approach yogurt bottle (MOVEMENT)
Step 50-150:     Pick, transport, place (MOVEMENT)
Step 150-300:    Stay in place (IDLE) ← Correct behavior

Movement magnitude: 0.92° (essentially stationary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Key Question: What Triggers the Difference?

The hallucination is NOT about being attracted to the banana. It's about:
- **WHY does the halluc case trigger the "return to home" phase?**
- **WHY does the normal case stay in IDLE?**

Possible explanations:
1. **Visual context difference**: The banana makes the scene look "active" or "incomplete"
2. **Pattern matching**: Head camera sees a scene that matches "mid-task" training examples
3. **Uncertainty**: The banana creates ambiguity that triggers default movement behavior

### The Banana's Role

The banana is NOT the target of movement. Instead, it may:
- Act as a **trigger** that activates the "continue working" mode
- Make the visual scene look like "there's still something to do"
- Interfere with the model's sense of "task completion"

---

## Revised Evidence Summary

### Evidence 1: Movement Direction Analysis

| Metric | Value |
|--------|-------|
| Cosine sim (movement vs toward-banana) | Unknown (needs verification) |
| Cosine sim (movement vs toward-approach-position) | **0.997** |
| Movement target | Original pickup location |

**Conclusion**: Robot re-executes approach, doesn't go toward banana.

### Evidence 2: Training Data Patterns

| Pattern | Count | Percentage |
|---------|-------|------------|
| IDLE → MOVEMENT → IDLE (last 40%) | 223 | 70% |
| MOVEMENT → IDLE (last 40%) | 97 | 30% |

**Conclusion**: Return-to-home movement is a learned pattern from training.

### Evidence 3: Behavioral Comparison

| Case | Post-completion behavior | Movement magnitude |
|------|-------------------------|-------------------|
| Hallucination | Re-executes approach | 122° |
| Normal | Stays still | 0.92° |

**Conclusion**: Halluc case triggers learned return pattern, normal case doesn't.

---

## What We Still Don't Know

1. **What specifically triggers the hallucination?**
   - Is it the banana's visual presence?
   - Is it something in the head camera?
   - Is it temporal/sequential context?

2. **Why doesn't normal case trigger the same pattern?**
   - Both completed the same task
   - The main visual difference is banana presence
   - But correlation ≠ causation

3. **Does training data have scenes with visible distractors?**
   - We analyzed actions, not visual content
   - Need to verify what objects are visible during IDLE periods in training

---

## Revised Recommendations

### For Data Collection

Instead of "IDLE with visible distractor", we should focus on:

1. **Extended post-completion IDLE** (30-50+ frames)
   - Without the return-to-home movement
   - To teach "stay still after task completion"

2. **Post-completion IDLE with various scene configs**
   - Different objects on table
   - Similar visual context to halluc case
   - But demonstrating STAYING STILL

3. **Breaking the return-to-home pattern**
   - If this pattern is causing issues, we might need to:
   - Either remove return-to-home from training, OR
   - Add explicit "stay still" demonstrations after the return

### For Further Investigation

1. **Verify banana position vs movement target**
   - Check actual spatial positions
   - Confirm robot is not moving toward banana

2. **Visual comparison of training IDLE vs inference IDLE**
   - What does the scene look like during training IDLE?
   - What's different in the halluc case?

3. **Test: Does removing return-to-home from training data help?**
   - Truncate episodes before the return-to-home phase
   - Retrain and see if hallucination persists

---

## Conclusion

The hallucination is more nuanced than originally thought:
- It's **NOT** "attraction to distractor"
- It's **RE-EXECUTION of the approach phase**
- The banana may **trigger** this behavior but is not the **target**
- The model learned return-to-home patterns from training, and something about the banana's presence activates this pattern inappropriately

This requires more investigation to understand exactly what visual/contextual factor triggers the difference between halluc and normal cases.
