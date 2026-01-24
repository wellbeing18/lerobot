# Flow Matching Explanation: P(trajectory | velocity_field)

## Overview

This document explains how the velocity field `v(x_t, t, K)` drives trajectory generation in SmolVLA's flow matching architecture. Understanding this is crucial because **the velocity field magnitude determines whether the robot moves or stays idle**.

---

## 1. The Flow Matching Process

SmolVLA uses flow matching to generate action trajectories. The process transforms random noise into meaningful actions through iterative denoising.

### The Update Equation

```
x_{t+dt} = x_t + dt · v(x_t, t, K)

Where:
  x_t     = current noisy trajectory at time t
  v       = velocity field (direction + magnitude)
  K       = KV cache (visual + language + state conditioning)
  dt      = time step size (= -1/T for T denoising steps)
  x_{t+dt} = updated trajectory
```

### ASCII Diagram: The Denoising Process

```
                        FLOW MATCHING: NOISE → ACTION

    t=1.0           t=0.8           t=0.6           t=0.4           t=0.2           t=0.0
    (noise)                                                                         (action)

      ●───────────────●───────────────●───────────────●───────────────●───────────────●
      │               │               │               │               │               │
      │      v₁       │      v₂       │      v₃       │      v₄       │      v₅       │
      │    ──────▶    │    ──────▶    │    ──────▶    │    ──────▶    │    ──────▶    │
      │               │               │               │               │               │
    x_0 = noise    x_1 = x_0+v₁    x_2 = x_1+v₂    x_3 = x_2+v₃    x_4 = x_3+v₄    x_T = action


    EACH STEP:
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │  1. Current state x_t is embedded as "action tokens"                            │
    │  2. Action tokens attend to KV cache K via cross-attention                      │
    │  3. Cross-attention output is projected to velocity v                           │
    │  4. Trajectory is updated: x_{t+dt} = x_t + dt · v                             │
    │  5. Repeat for T=10 steps                                                       │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. How v is Computed: Cross-Attention Mechanism

The velocity field v is the output of cross-attention between action tokens and the KV cache.

### The Cross-Attention Formula

```
v(x_t, t, K) = ActionProjection( CrossAttention(Q_action, K, V) )

Where:
  Q_action = Query from action tokens (noisy trajectory + timestep embedding)
  K, V     = Key and Value from KV cache (visual + language + state)
```

### ASCII Diagram: Cross-Attention Computes v

```
                    CROSS-ATTENTION MECHANISM

    ┌─────────────────────┐              ┌─────────────────────┐
    │   ACTION TOKENS     │              │     KV CACHE        │
    │   ─────────────     │              │     ────────        │
    │                     │              │                     │
    │   x_t (noisy traj)  │              │   K_head   (α=HIGH) │ ◀── 167% effect
    │   t (timestep)      │              │   K_left   (α=MED)  │
    │                     │              │   K_right  (α=LOW)  │ ◀── 11% effect
    │         │           │              │   K_lang   (α=HIGH) │
    │         │           │              │   K_state  (α=MED)  │
    │         ▼           │              │         │           │
    │   ┌─────────┐       │              │         │           │
    │   │ Q_action│       │              │         ▼           │
    │   └────┬────┘       │              │   ┌───────────┐     │
    │        │            │              │   │  K  │  V  │     │
    └────────┼────────────┘              │   └─────┴─────┘     │
             │                           └─────────┼───────────┘
             │                                     │
             │         CROSS-ATTENTION             │
             └───────────────┬─────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Attention(Q,K,V)│
                    │                 │
                    │  = softmax(QK^T/√d) · V
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Action Projection│
                    │   (Linear)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  v(x_t, t, K)   │
                    │  ────────────   │
                    │  direction +    │
                    │  magnitude      │
                    └─────────────────┘


    KEY INSIGHT:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  The attention weights α_r determine how much each KV region affects v:    │
    │                                                                             │
    │  v ≈ Σ_r α_r · (information from K_r)                                      │
    │                                                                             │
    │  Since α_head >> α_right:                                                   │
    │    → Head camera DOMINATES v computation                                    │
    │    → Right wrist (which sees banana) has little effect on v                │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. How v Magnitude Determines Behavior

The **magnitude** of v determines whether the robot moves or stays idle.

### ASCII Diagram: v Magnitude → Behavior

```
                    v MAGNITUDE DETERMINES BEHAVIOR

    Action Space (2D projection for visualization)

            ▲ Action Dim 2
            │
            │         HALLUC CASE:
            │         |v| is HIGH
        1.0 │         ──────────▶ (large arrows)
            │        ╱
            │       ╱   Trajectory moves
            │      ╱    AWAY from origin
            │     ╱
        0.5 │    ●──────────────────▶ ■ Final: MOVEMENT
            │   ╱ Start (noise)
            │  ╱
            │ ╱
    ────────┼─────────────────────────────────────▶ Action Dim 1
            │       IDLE
            │      REGION
        -0.5│    ┌───────┐
            │    │       │   NORMAL CASE:
            │    │   ●───┼──▶ ■ Final: IDLE
            │    │       │   |v| is LOW (small arrows)
            │    └───────┘   Trajectory stays near origin
            │


    QUANTITATIVE THRESHOLDS:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   |v| < 3°  →  IDLE behavior (robot stays still)               │
    │   |v| > 3°  →  MOVEMENT behavior (robot moves)                 │
    │                                                                 │
    │   HALLUC CASE:  mean |v| = 7.45° post-completion  → MOVEMENT   │
    │   NORMAL CASE:  mean |v| = 2.41° post-completion  → IDLE       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 4. The 10 Denoising Steps

SmolVLA uses T=10 denoising steps. At each step, v is recomputed based on the current trajectory estimate.

### ASCII Diagram: Step-by-Step Denoising

```
    HALLUC vs NORMAL: 10 DENOISING STEPS

    Step 1 (t=1.0)     Step 5 (t=0.6)     Step 10 (t=0.1)
    ───────────────     ───────────────     ───────────────

         IDLE              IDLE               IDLE
        region            region             region
       ┌─────┐           ┌─────┐           ┌─────┐
       │     │           │     │           │     │
       │  G  │           │  G  │           │  G  │ ← Normal STAYS
       │ ●R  │           │     │    R      │     │     in IDLE
       └─────┘           └─────┘  ●        └─────┘
                                                    R
                                                   ●  ← Halluc EXITS
                                                        IDLE region

    Legend:
      ● = Current position
      R = Halluc (red) trajectory
      G = Normal (green) trajectory


    WHY TRAJECTORIES DIVERGE:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  Both start from SAME noise x_0                                            │
    │                                                                             │
    │  HALLUC: K_halluc → v_halluc has HIGH magnitude                            │
    │          Each step: x_{t+1} = x_t + large v                                │
    │          → Trajectory accumulates movement                                  │
    │          → Final x_T is FAR from origin (MOVEMENT)                         │
    │                                                                             │
    │  NORMAL: K_normal → v_normal has LOW magnitude                             │
    │          Each step: x_{t+1} = x_t + small v (toward origin)                │
    │          → Trajectory converges to origin                                   │
    │          → Final x_T is NEAR origin (IDLE)                                 │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Complete Causal Chain: K → v → trajectory

### ASCII Diagram: The Full Pipeline

```
    COMPLETE CAUSAL CHAIN: P(trajectory | v | K)

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   VISUAL     │     │   KV CACHE   │     │   VELOCITY   │     │  TRAJECTORY  │
    │   INPUT      │────▶│     (K)      │────▶│   FIELD (v)  │────▶│     (τ)      │
    │              │     │              │     │              │     │              │
    │ 3 cameras    │     │ K_head: 23%  │     │ direction +  │     │ action chunk │
    │ + language   │     │ K_wrist: 43% │     │ magnitude    │     │ (50 steps)   │
    │ + state      │     │ (of diff)    │     │              │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                │                    │
                                │                    │
                                ▼                    ▼
                         ┌─────────────┐      ┌─────────────┐
                         │Cross-Attn   │      │ Integration │
                         │Weights:     │      │             │
                         │α_head=HIGH  │      │ x_T = x_0 + │
                         │α_wrist=LOW  │      │ ∫v dt       │
                         └─────────────┘      └─────────────┘


    ═══════════════════════════════════════════════════════════════════════════════

    HALLUC CASE FLOW:
    ─────────────────

    [Banana on table] → [K encodes "obj on workspace"] → [α_head queries K_head]
                                                                    │
                                                                    ▼
    [Head: "Object detected"] → [v = HIGH magnitude] → [∫v dt = large]
                                                                    │
                                                                    ▼
                                                        [τ = MOVEMENT] ← HALLUCINATION!

    ═══════════════════════════════════════════════════════════════════════════════

    NORMAL CASE FLOW:
    ─────────────────

    [Banana on plate] → [K encodes "empty workspace"] → [α_head queries K_head]
                                                                    │
                                                                    ▼
    [Head: "Workspace empty"] → [v = LOW magnitude] → [∫v dt = small]
                                                                    │
                                                                    ▼
                                                        [τ = IDLE] ← CORRECT!

    ═══════════════════════════════════════════════════════════════════════════════
```

---

## 6. Why Training Bias Affects v

The velocity field v is learned from training data. The training distribution creates a bias in v:

### ASCII Diagram: Training Bias → v Bias

```
    HOW TRAINING DATA SHAPES v

    TRAINING DATA:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  Visual Context          Action Label        Learned v Mapping              │
    │  ──────────────          ────────────        ─────────────────              │
    │                                                                             │
    │  Object on workspace  →  MOVEMENT        →  v = HIGH magnitude              │
    │       (100% of such frames)                                                 │
    │                                                                             │
    │  Empty workspace      →  IDLE            →  v = LOW magnitude               │
    │       (100% of such frames)                                                 │
    │                                                                             │
    │  Object on workspace  →  IDLE            →  ???  (NEVER SEEN!)              │
    │       (0% - MISSING!)                                                       │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘


    RESULT:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  The model learns a CONDITIONAL velocity field:                             │
    │                                                                             │
    │  v(x_t, t, K) = {                                                          │
    │      HIGH magnitude,  if K encodes "object on workspace"                   │
    │      LOW magnitude,   if K encodes "empty workspace"                       │
    │  }                                                                          │
    │                                                                             │
    │  At inference with banana on table:                                         │
    │    → K_head encodes "object on workspace"                                   │
    │    → Model applies learned mapping: v = HIGH magnitude                      │
    │    → Trajectory integrates to MOVEMENT                                      │
    │    → HALLUCINATION                                                          │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. How to Read the Visualizations

### Figure: p_trajectory_given_velocity_field.png

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TOP-LEFT: Flow Matching Process Diagram                                        │
│  ───────────────────────────────────────                                        │
│                                                                                 │
│    Shows the denoising chain: x_0 → x_1 → ... → x_T                            │
│    Arrows labeled 'v' show velocity at each step                               │
│    Update equation: x_{t+dt} = x_t + dt · v                                    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TOP-MIDDLE: Cross-Attention Computes v                                         │
│  ──────────────────────────────────────                                         │
│                                                                                 │
│    Shows: KV Cache + Action Tokens → Cross-Attention → v                       │
│    Key insight: α_head = HIGH, α_wrist = LOW                                   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TOP-RIGHT: Velocity Field Comparison                                           │
│  ────────────────────────────────────                                           │
│                                                                                 │
│    Arrow field showing v direction/magnitude                                    │
│    RED arrows (Halluc): Point toward movement (upper-right)                    │
│    GREEN arrows (Normal): Point toward origin (stay still)                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  BOTTOM-LEFT: Trajectory Integration                                            │
│  ───────────────────────────────────                                            │
│                                                                                 │
│    Shows actual trajectory paths in action space                               │
│    Blue circle = IDLE region                                                    │
│    RED path: Exits IDLE region → MOVEMENT                                      │
│    GREEN path: Stays in IDLE region → CORRECT                                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  BOTTOM-MIDDLE: Real Data |v| Over Time                                         │
│  ──────────────────────────────────────                                         │
│                                                                                 │
│    X-axis: Inference step (200-300, post-completion)                           │
│    Y-axis: |v| = action delta in degrees                                       │
│    Horizontal line: IDLE threshold (3°)                                        │
│    RED line above threshold → HALLUC has high |v|                              │
│    GREEN line below threshold → NORMAL has low |v|                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Figure: denoising_steps_visualization.png

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  10 PANELS: One for each denoising step (t=1.0 to t=0.1)                       │
│  ─────────────────────────────────────────────────────────                      │
│                                                                                 │
│    Each panel shows:                                                            │
│    • Blue circle = IDLE region (where trajectory should end for idle)          │
│    • RED path = Halluc trajectory (accumulates, exits IDLE)                    │
│    • GREEN path = Normal trajectory (converges, stays in IDLE)                 │
│    • Arrows = velocity v at current step                                        │
│                                                                                 │
│    WHAT TO OBSERVE:                                                             │
│    • Step 1: Both start from same noise (near each other)                      │
│    • Steps 2-5: Trajectories begin to diverge                                  │
│    • Steps 6-10: Clear separation - RED exits IDLE, GREEN stays               │
│                                                                                 │
│    KEY INSIGHT:                                                                 │
│    The v arrows show WHY they diverge:                                         │
│    • Halluc v points AWAY from origin (movement)                               │
│    • Normal v points TOWARD origin (stay still)                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | Halluc Case | Normal Case |
|-----------|-------------|-------------|
| K (KV Cache) | Encodes "object on workspace" | Encodes "empty workspace" |
| Cross-Attn | α_head queries "object present" | α_head queries "workspace empty" |
| v magnitude | HIGH (mean 7.45°) | LOW (mean 2.41°) |
| ∫v dt | Large displacement | Small displacement |
| Final τ | MOVEMENT (outside IDLE) | IDLE (inside IDLE region) |

**Root Cause**: Training data created a mapping where `v(K_with_object) = HIGH`, but never taught `v(K_with_irrelevant_object) = LOW`.

---

## Appendix: Visualization Scripts and Output Files

### Scripts for Generating Flow Matching Diagrams

| Script | Output | Description |
|--------|--------|-------------|
| `jdocs/scripts/investigation/tools/flow_matching_visualization.py` | `p_trajectory_given_velocity_field.png`, `denoising_steps_visualization.png` | Flow matching mechanism visualizations |
| `jdocs/scripts/investigation/tools/trajectory_divergence_analysis.py` | `trajectory_divergence_comparison.png` | Trajectory divergence over time |

### Output Files

| File | Description |
|------|-------------|
| `logs/investigation/first_principles/p_trajectory_given_velocity_field.png` | 6-panel flow matching mechanism |
| `logs/investigation/first_principles/denoising_steps_visualization.png` | 10-step denoising process |
| `logs/investigation/trajectory_divergence/trajectory_divergence_comparison.png` | Halluc vs Normal trajectory comparison |

### Running the Scripts

```bash
# Activate conda environment
eval "$(conda shell.bash hook)" && conda activate lerobot

# Generate flow matching visualizations
python jdocs/scripts/investigation/tools/flow_matching_visualization.py

# Generate trajectory divergence analysis
python jdocs/scripts/investigation/tools/trajectory_divergence_analysis.py
```

### Related Documentation

- `jdocs/investigation_reports/first_principles_explanation.md` - Training distribution bias
- `jdocs/investigation_reports/mechanism_explanation.md` - Information-action gap
- `jdocs/investigation_reports/visualization_guide.md` - How to read the diagrams
