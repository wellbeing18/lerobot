# Data Collection Checklist

**Quick Reference for Recording Sessions**

See `data_collection_strategy.md` for full details.

---

## Phase 1: IDLE with Distractor (45 episodes)

### Type 1A: Single Distractor (30 episodes)

**Setup**:
- [ ] Target object on table
- [ ] ONE distractor object visible (different from target)
- [ ] All cameras have clear view

**Recording**:
- [ ] Execute smooth pick-and-place of TARGET only
- [ ] Place target at destination
- [ ] Return arm to neutral
- [ ] **HOLD STILL 30-60 frames** (~1-2 sec)
- [ ] Distractor remains visible and untouched
- [ ] End episode

**Task description format**:
```
"Use [left/right] arm to pick up the [TARGET_NAME] and place it on the plate"
```

**Distractor positions to cover** (vary across episodes):
- [ ] Left of target starting position (8 episodes)
- [ ] Right of target starting position (8 episodes)
- [ ] Behind target starting position (7 episodes)
- [ ] Opposite side of table (7 episodes)

---

### Type 1B: Multi-Distractor (15 episodes)

**Setup**:
- [ ] Target object on table
- [ ] 2-3 distractor objects visible
- [ ] All cameras have clear view of all objects

**Recording**:
- Same as 1A but with multiple distractors

**Variation schedule**:
- [ ] 2 distractors (8 episodes)
- [ ] 3 distractors (7 episodes)

---

## Phase 2: Object Discrimination (70 episodes)

### Type 2A: Confusable Pairs (40 episodes)

**Pairs to cover** (10 episodes each, 5 picking each object):

#### Ketchup + Yogurt (10 episodes)
- [ ] 5 episodes: "Use [left/right] arm to pick up the ketchup bottle and place it in the bin"
- [ ] 5 episodes: "Use [left/right] arm to pick up the yogurt bottle and place it in the bin"
- Vary positions each episode

#### Bread + Used Tissue (10 episodes)
- [ ] 5 episodes: "Use [left/right] arm to pick up the bread and place it on the plate"
- [ ] 5 episodes: "Use [left/right] arm to pick up the used tissue and place it in the bin"
- Vary positions each episode

#### Corn + Banana (10 episodes)
- [ ] 5 episodes: "Use [left/right] arm to pick up the corn and place it on the plate"
- [ ] 5 episodes: "Use [left/right] arm to pick up the banana and place it on the plate"
- Vary positions each episode

#### Orange + Ice cream (10 episodes)
- [ ] 5 episodes: "Use [left/right] arm to pick up the orange and place it on the plate"
- [ ] 5 episodes: "Use [left/right] arm to pick up the ice cream and place it in the bin"
- Vary positions each episode

**Critical**: Both objects must be visible in all cameras throughout

---

### Type 3A: Progressive Distractors (30 episodes)

**Target + 1 distractor, close proximity** (10 episodes)
- [ ] Separation: 5-8 cm
- [ ] Smooth, confident approach
- [ ] Clean grasp of target

**Target + 2 distractors** (10 episodes)
- [ ] Various arrangements around target
- [ ] All visible in cameras

**Target + 3 distractors, cluttered** (10 episodes)
- [ ] Realistic cluttered scene
- [ ] Target not always in center

---

## Phase 3: Enhancement (85 episodes)

### Type 2B: Attribute-Based (20 episodes)
- [ ] Color discrimination tasks
- [ ] Shape discrimination tasks
- [ ] Texture discrimination tasks

### Type 2C: Explicit Negatives (10 episodes)
- [ ] Clear approach to correct object
- [ ] Wrong object highly visible
- [ ] No hesitation or deviation

### Type 3B: Close-Proximity (20 episodes)
- [ ] 8 cm separation (5 episodes)
- [ ] 5 cm separation (10 episodes)
- [ ] 3 cm separation (5 episodes)

### Type 3C: Variable Sizes (15 episodes)
- [ ] Small target + large distractors
- [ ] Large target + small distractors
- [ ] Mixed sizes

### Type 3D: Camera Coverage (10 episodes)
- [ ] Ensure head camera has clear top-down view
- [ ] Wrist cameras capture approach clearly

### Type 1C: Sequential with Persistent Distractor (10 episodes)
- [ ] Two objects A and B on table
- [ ] Task mentions only A
- [ ] Complete A, B remains
- [ ] IDLE with B visible

---

## Pre-Recording Checklist

Before each session:
- [ ] Lighting consistent
- [ ] Cameras calibrated and focused
- [ ] All objects clean and clearly visible
- [ ] Robot calibrated
- [ ] Recording system ready

---

## Task Description Template

```
"Use [left/right] arm to pick up the [EXACT_OBJECT_NAME] and place it [on the plate / in the bin]"
```

**DO**:
- Use exact object names (ketchup, yogurt, bread, etc.)
- Specify which arm
- Clear destination

**DON'T**:
- Use colors or positions ("the red one", "the left object")
- Use vague terms ("put it there", "that thing")
- Mention distractors in task description

---

## Episode Quality Checklist

Before marking episode complete:
- [ ] Smooth, confident movements (no jerking)
- [ ] No mid-movement corrections
- [ ] Clear grasp execution
- [ ] For IDLE episodes: held still for full 40-60 frames
- [ ] All objects visible throughout
- [ ] Task description accurate and specific

---

## Progress Tracker

| Phase | Type | Target | Completed |
|-------|------|--------|-----------|
| 1 | 1A - Single Distractor IDLE | 30 | ___ |
| 1 | 1B - Multi-Distractor IDLE | 15 | ___ |
| 2 | 2A - Confusable Pairs | 40 | ___ |
| 2 | 3A - Progressive Distractors | 30 | ___ |
| 3 | 2B - Attribute-Based | 20 | ___ |
| 3 | 2C - Explicit Negatives | 10 | ___ |
| 3 | 3B - Close-Proximity | 20 | ___ |
| 3 | 3C - Variable Sizes | 15 | ___ |
| 3 | 3D - Camera Coverage | 10 | ___ |
| 3 | 1C - Sequential | 10 | ___ |
| **Total** | | **200** | ___ |
