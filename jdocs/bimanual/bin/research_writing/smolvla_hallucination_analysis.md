# SmolVLA Hallucination Analysis: Architecture and Introspection

## 1. SmolVLA Model Architecture

### 1.1 High-Level Architecture Overview

SmolVLA is a Vision-Language-Action (VLA) model that combines a Vision-Language Model (VLM) backbone with a flow matching action expert. The architecture processes visual observations, language instructions, and robot state to generate action trajectories.

```
                            SMOLVLA ARCHITECTURE OVERVIEW
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                              │
    │  │  CAMERAS    │  │  LANGUAGE   │  │   STATE     │                              │
    │  │  (3 views)  │  │  INSTRUCTION│  │  (joints)   │                              │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                              │
    │         │                │                │                                      │
    │         ▼                ▼                ▼                                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                              │
    │  │   SigLIP    │  │  Tokenizer  │  │  state_proj │                              │
    │  │   Encoder   │  │  + Embed    │  │   (Linear)  │                              │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                              │
    │         │                │                │                                      │
    │         └────────────────┼────────────────┘                                      │
    │                          │                                                       │
    │                          ▼                                                       │
    │              ┌───────────────────────┐                                           │
    │              │   PREFIX EMBEDDING    │                                           │
    │              │   (241 tokens total)  │                                           │
    │              └───────────┬───────────┘                                           │
    │                          │                                                       │
    │                          ▼                                                       │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │                  VLM WITH EXPERT (SmolVLM + Expert Gemma)               │   │
    │   │  ┌─────────────────────────────────────────────────────────────────┐    │   │
    │   │  │                       TRANSFORMER LAYERS                        │    │   │
    │   │  │                                                                 │    │   │
    │   │  │   Layer 1 ─▶ Layer 2 ─▶ ... ─▶ Layer 16                        │    │   │
    │   │  │      │                                │                         │    │   │
    │   │  │      ▼                                ▼                         │    │   │
    │   │  │   ┌─────────────────────────────────────────┐                  │    │   │
    │   │  │   │              KV CACHE                   │                  │    │   │
    │   │  │   │   (Keys and Values for all layers)      │                  │    │   │
    │   │  │   └─────────────────────────────────────────┘                  │    │   │
    │   │  │                                                                 │    │   │
    │   │  └─────────────────────────────────────────────────────────────────┘    │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                          │                                                       │
    │                          ▼                                                       │
    │              ┌───────────────────────┐                                           │
    │              │   FLOW MATCHING       │                                           │
    │              │   ACTION GENERATION   │                                           │
    │              │   (10 denoising steps)│                                           │
    │              └───────────┬───────────┘                                           │
    │                          │                                                       │
    │                          ▼                                                       │
    │              ┌───────────────────────┐                                           │
    │              │   ACTION TRAJECTORY   │                                           │
    │              │   (50 steps × 12 dim) │                                           │
    │              └───────────────────────┘                                           │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Detailed Component Architecture

#### 1.2.1 Prefix Embedding Structure

The prefix embedding combines visual, language, and state information into a 241-token sequence:

```
                        PREFIX EMBEDDING LAYOUT (241 tokens)
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │  TOKEN POSITION:  0        64       128      192      240   241                 │
    │                   │        │        │        │        │     │                   │
    │                   ▼        ▼        ▼        ▼        ▼     ▼                   │
    │   ┌──────────────┬────────────────┬────────────────┬───────┬──────┐            │
    │   │  HEAD CAMERA │  LEFT WRIST    │  RIGHT WRIST   │ LANG  │STATE │            │
    │   │   64 tokens  │  64 tokens     │  64 tokens     │ 48 tok│1 tok │            │
    │   │   (8×8 grid) │  (8×8 grid)    │  (8×8 grid)    │       │      │            │
    │   └──────────────┴────────────────┴────────────────┴───────┴──────┘            │
    │                                                                                  │
    │   VISUAL TOKENS (192 total = 3 cameras × 64 patches each):                      │
    │   ┌────────────────────────────────────────────────────────────────────────┐    │
    │   │  Each camera image (512×512) → SigLIP encoder → 64 patch embeddings   │    │
    │   │  Each patch: 64×64 pixels → 768-dim embedding                          │    │
    │   │                                                                        │    │
    │   │  HEAD CAMERA: Bird's-eye view of workspace                            │    │
    │   │  LEFT WRIST:  Left arm gripper view                                    │    │
    │   │  RIGHT WRIST: Right arm gripper view                                   │    │
    │   └────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                  │
    │   LANGUAGE TOKENS (48):                                                          │
    │   ┌────────────────────────────────────────────────────────────────────────┐    │
    │   │  "Use left arm to pick up the yogurt bottle and place it on the plate" │    │
    │   │                                                                        │    │
    │   │  Tokenized → 48 tokens → Embedding lookup → 768-dim per token         │    │
    │   └────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                  │
    │   STATE TOKEN (1):                                                               │
    │   ┌────────────────────────────────────────────────────────────────────────┐    │
    │   │  12-dim joint positions → Linear projection → 768-dim                  │    │
    │   └────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

#### 1.2.2 KV Cache and Cross-Attention

The KV cache stores pre-computed keys and values from the prefix processing. During action generation, action tokens attend to this cache via cross-attention:

```
                        KV CACHE AND CROSS-ATTENTION MECHANISM
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   STEP 1: FILL KV CACHE (done once per inference)                               │
    │   ─────────────────────────────────────────────────                              │
    │                                                                                  │
    │   Prefix (241 tokens)                                                            │
    │         │                                                                        │
    │         ▼                                                                        │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │                    TRANSFORMER FORWARD PASS                              │   │
    │   │                                                                          │   │
    │   │   For each layer l = 1 to 16:                                           │   │
    │   │     K_l = W_k × prefix_embeddings  (shape: 241 × 768)                   │   │
    │   │     V_l = W_v × prefix_embeddings  (shape: 241 × 768)                   │   │
    │   │     Store K_l, V_l in cache                                              │   │
    │   │                                                                          │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                    │                                                             │
    │                    ▼                                                             │
    │            ┌───────────────┐                                                     │
    │            │   KV CACHE    │  past_key_values[layer][key_or_value]              │
    │            │  (FROZEN)     │  Shape: (16 layers × 2 × batch × 241 × 768)        │
    │            └───────────────┘                                                     │
    │                                                                                  │
    │                                                                                  │
    │   STEP 2: CROSS-ATTENTION (repeated 10 times for denoising)                     │
    │   ──────────────────────────────────────────────────────────                     │
    │                                                                                  │
    │   Action Tokens (50 per chunk)                                                   │
    │         │                                                                        │
    │         ▼                                                                        │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │                    CROSS-ATTENTION COMPUTATION                           │   │
    │   │                                                                          │   │
    │   │   Q = W_q × action_embeddings      (shape: 50 × 768)                    │   │
    │   │   K = KV_cache[keys]               (shape: 241 × 768)                   │   │
    │   │   V = KV_cache[values]             (shape: 241 × 768)                   │   │
    │   │                                                                          │   │
    │   │   Attention = softmax(Q × K^T / √768) × V                               │   │
    │   │                                                                          │   │
    │   │   Output shape: 50 × 768                                                 │   │
    │   │                                                                          │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                    │                                                             │
    │                    ▼                                                             │
    │            ┌───────────────┐                                                     │
    │            │ action_out_   │  Linear(768 → 12)                                   │
    │            │    proj       │                                                     │
    │            └───────────────┘                                                     │
    │                    │                                                             │
    │                    ▼                                                             │
    │            ┌───────────────┐                                                     │
    │            │ VELOCITY v_t  │  Shape: 50 × 12 (action dimension)                 │
    │            └───────────────┘                                                     │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

#### 1.2.3 Flow Matching Action Generation

  ANALOGY:                                                                                                              
  ─────────                                                                                                             
    x_0 = blank canvas (random paint splatters)                                                                         
    KV cache = the reference image you want to paint                                                                    
    v_t = brush strokes guided by the reference                                                                         
    x_10 = final painting that matches the reference                                                                    
                                                                                                                        
  So the denoising process is:                                                                                          
  1. Start with random noise (x_0)                                                                                      
  2. Ask "given the current visual scene, task, and arm state (all in KV cache), what velocity should push this noise   
  toward a valid action?"                                                                                               
  3. Apply that velocity to refine the trajectory                                                                       
  4. Repeat 10 times until noise becomes a clean action trajectory  

Flow matching generates action trajectories by iteratively denoising from random noise:

```
                        FLOW MATCHING: 10-STEP DENOISING PROCESS
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   MATHEMATICAL FORMULATION:                                                      │
    │   ───────────────────────                                                        │
    │                                                                                  │
    │   x_{t+dt} = x_t + dt × v(x_t, t, K)                                            │
    │                                                                                  │
    │   Where:                                                                         │
    │     x_t   = noisy trajectory at time t                                          │
    │     v     = velocity field (learned, depends on KV cache K)                     │
    │     dt    = time step = -1/10 = -0.1                                            │
    │     K     = KV cache from prefix processing                                      │
    │                                                                                  │
    │                                                                                  │
    │   10 DENOISING STEPS:                                                            │
    │   ────────────────────                                                           │
    │                                                                                  │
    │   t=1.0      t=0.9      t=0.8            t=0.1      t=0.0                       │
    │   (noise)                                           (action)                     │
    │                                                                                  │
    │    x_0 ─────▶ x_1 ─────▶ x_2 ─────▶ ... ─────▶ x_9 ─────▶ x_10                 │
    │         +v₀       +v₁       +v₂            +v₈       +v₉                        │
    │                                                                                  │
    │                                                                                  │
    │   EACH STEP IN DETAIL:                                                           │
    │   ────────────────────                                                           │
    │                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │  Step i (time = 1.0 - i×0.1):                                           │   │
    │   │                                                                          │   │
    │   │  1. Embed noisy trajectory:                                              │   │
    │   │     action_emb = action_in_proj(x_t)    # Linear(12 → 768)              │   │
    │   │                                                                          │   │
    │   │  2. Embed timestep:                                                      │   │
    │   │     time_emb = sinusoidal_encoding(t)   # Shape: 768                    │   │
    │   │                                                                          │   │
    │   │  3. Combine action + time:                                               │   │
    │   │     action_time = MLP([action_emb, time_emb])                           │   │
    │   │                                                                          │   │
    │   │  4. Cross-attend to KV cache:                                            │   │
    │   │     output = CrossAttention(action_time, K, V)                          │   │
    │   │                                                                          │   │
    │   │  5. Project to velocity:                                                 │   │
    │   │     v_t = action_out_proj(output)       # Linear(768 → 12)              │   │
    │   │                                                                          │   │
    │   │  6. Update trajectory:                                                   │   │
    │   │     x_{t+1} = x_t + (-0.1) × v_t                                        │   │
    │   │                                                                          │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                  │
    │                                                                                  │
    │   FINAL OUTPUT:                                                                  │
    │   ─────────────                                                                  │
    │                                                                                  │
    │   x_10 = Action trajectory                                                       │
    │        = 50 timesteps × 12 joint dimensions                                      │
    │        = 600 action values total                                                 │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Introspection for Hallucination Behavior

### 2.1 The Hallucination Problem

After completing a task (e.g., "pick up the yogurt bottle and place it on the plate"), the model sometimes continues to generate movement actions toward empty space, as if grasping an invisible object. This occurs specifically when a **distractor object** (e.g., a banana) is visible in the workspace.

### 2.2 Key Insight: Conditional Flow Distribution

The core of understanding hallucination lies in the conditional flow distribution:

```
                    P(trajectory | KV_cache) = P(τ | K)
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   The model generates trajectories by sampling from:                             │
    │                                                                                  │
    │   τ ~ FlowMatching(noise, v(·, ·, K))                                           │
    │                                                                                  │
    │   Where v is the learned velocity field conditioned on KV cache K.              │
    │                                                                                  │
    │                                                                                  │
    │   CRITICAL OBSERVATION:                                                          │
    │   ─────────────────────                                                          │
    │                                                                                  │
    │   • K encodes visual patterns from cameras                                       │
    │   • The velocity field v depends on K                                            │
    │   • Different K → Different v → Different trajectory τ                          │
    │                                                                                  │
    │                                                                                  │
    │   ┌────────────────────────────────────────────────────────────────────────┐    │
    │   │                                                                        │    │
    │   │  HALLUC CASE:          K_halluc          NORMAL CASE:        K_normal │    │
    │   │  (banana on table)     encodes           (banana on plate)   encodes  │    │
    │   │                        "object on                            "empty   │    │
    │   │                        workspace"                            workspace"│    │
    │   │                             │                                    │     │    │
    │   │                             ▼                                    ▼     │    │
    │   │                        ┌─────────┐                          ┌─────────┐│    │
    │   │                        │ v = HIGH│                          │ v = LOW ││    │
    │   │                        │ magnitude│                          │magnitude││    │
    │   │                        └────┬────┘                          └────┬────┘│    │
    │   │                             │                                    │     │    │
    │   │                             ▼                                    ▼     │    │
    │   │                        ┌─────────┐                          ┌─────────┐│    │
    │   │                        │MOVEMENT │                          │  IDLE   ││    │
    │   │                        │(WRONG!) │                          │(CORRECT)││    │
    │   │                        └─────────┘                          └─────────┘│    │
    │   │                                                                        │    │
    │   └────────────────────────────────────────────────────────────────────────┘    │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Reading the p_traj_given_kv_cache.png Visualization

The visualization `logs/investigation/causal_distribution/p_traj_given_kv_cache.png` shows two key panels that reveal the hallucination mechanism:

```
                    p_traj_given_kv_cache.png - TOP 2 PANELS
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   PANEL 1: 2D CONDITIONAL DISTRIBUTION P(action | KV_cache)                     │
    │   ───────────────────────────────────────────────────────                        │
    │                                                                                  │
    │   Action                                                                         │
    │   Delta    │                        ╔══════════════════╗                        │
    │   (degrees)│                        ║  HIGH DENSITY    ║                        │
    │            │                        ║  (red contours)  ║                        │
    │       14   │                        ║       ★          ║  ★ = HALLUC mean      │
    │            │                        ║   Halluc case    ║      (should be LOW   │
    │       10   │                        ║   samples here   ║       but is HIGH!)   │
    │            │                        ╚════════════════╤═╝                        │
    │        6   │    MOVEMENT REGION                      │                          │
    │            │    (action > 3°)                        │                          │
    │   ─────────┼─────────────────────────────────────────┼──────────────────────    │
    │        3   │═══════════════════════════════════════════  IDLE THRESHOLD        │
    │            │                                                                     │
    │        2   │  △      ╔══════════════════╗                △ = NORMAL mean       │
    │            │         ║  HIGH DENSITY    ║                    (correctly LOW)   │
    │        0   │  STAY   ║ Normal case      ║  STILL                               │
    │            └─────────╚══════════════════╝──────────────────────────────────▶   │
    │            0.0                0.5                1.0                             │
    │                        KV Cache Proxy                                            │
    │                   (0 = no distractor, 1 = distractor visible)                   │
    │                                                                                  │
    │                                                                                  │
    │   HOW TO READ:                                                                   │
    │   ────────────                                                                   │
    │   • X-axis: KV cache conditioning (proxy for distractor visibility)             │
    │   • Y-axis: Action delta (movement magnitude in degrees)                        │
    │   • Contour colors: Probability density (darker = higher probability)           │
    │   • Horizontal line at y=3: IDLE threshold                                      │
    │   • ★ (star): Hallucination case mean (should be in STAY STILL but is not!)   │
    │   • △ (triangle): Normal case mean (correctly in STAY STILL region)            │
    │                                                                                  │
    │   KEY INSIGHT:                                                                   │
    │   The contours show P(action | KV) - when KV encodes "distractor visible"       │
    │   (right side), the action distribution shifts UP into the MOVEMENT region!     │
    │                                                                                  │
    ├──────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                  │
    │   PANEL 2: MARGINAL HISTOGRAMS                                                   │
    │   ──────────────────────────────                                                 │
    │                                                                                  │
    │   Density                                                                        │
    │      │                                                                           │
    │      │     ┌────┐                                                                │
    │      │     │    │      Halluc case                                               │
    │      │     │    │      (red)                                                     │
    │      │  ┌──┤    │                                                                │
    │      │  │  │    │      Distribution shifted                                      │
    │      │  │  │    │      RIGHT (high velocity)                                     │
    │      │  │  └────┘                                                                │
    │      │  │                                                                        │
    │   ───┼──┼────────────────────────────────────────────────────────────▶          │
    │      │  │    │                                                                   │
    │      │  │    │ IDLE threshold (3°)                                              │
    │      │  │    ▼                                                                   │
    │      │  │                                                                        │
    │      │  └───────┐                                                                │
    │      │          │  Normal case                                                   │
    │      │          │  (green)                                                       │
    │      │          │  Distribution shifted                                          │
    │      │          │  LEFT (low velocity)                                          │
    │      │          │                                                                │
    │      └──────────┴──────────────────────────────────────────────────▶            │
    │                 0        3        6        9       12       15                   │
    │                           Action Delta (degrees)                                 │
    │                                                                                  │
    │   HOW TO READ:                                                                   │
    │   ────────────                                                                   │
    │   • Shows P(action) for each case separately as histograms                      │
    │   • Red histogram: Halluc case - centered around 7-10° (MOVEMENT)               │
    │   • Green histogram: Normal case - centered around 1-3° (IDLE)                  │
    │   • Vertical line at x=3: IDLE/MOVEMENT boundary                                │
    │   • Almost no overlap - distributions are completely separated!                 │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 The Training Distribution Bias

The root cause of hallucination is a bias in the training data distribution:

```
                    TRAINING DATA DISTRIBUTION BIAS
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   WHAT THE TRAINING DATA SHOWS:                                                  │
    │   ─────────────────────────────                                                  │
    │                                                                                  │
    │   Visual Context           Action Label         Training %                       │
    │   ───────────────          ────────────         ──────────                       │
    │   Object on workspace  →   MOVEMENT         →   100% of such frames             │
    │   Empty workspace      →   IDLE             →   100% of such frames             │
    │   Object on workspace  →   IDLE             →   0% (NEVER SEEN!)                │
    │                                                      ↑                           │
    │                                                      │                           │
    │                                             THE MISSING PATTERN!                 │
    │                                                                                  │
    │                                                                                  │
    │   WHAT THE MODEL LEARNS:                                                         │
    │   ──────────────────────                                                         │
    │                                                                                  │
    │   P(movement | object_on_workspace) = 1.0   (always move when object visible)  │
    │   P(idle | empty_workspace) = 1.0           (always idle when empty)            │
    │   P(idle | object_on_workspace) = ???       (UNDEFINED - never trained!)        │
    │                                                                                  │
    │                                                                                  │
    │   VISUALIZATION OF THE GAP:                                                      │
    │   ─────────────────────────                                                      │
    │                                                                                  │
    │   Action    │                                                                    │
    │   Velocity  │    ×××××××                                                         │
    │   (degrees) │   ×××××××××          TRAINING: Object on workspace                │
    │        10   │  ××××××××××××        (all frames show MOVEMENT)                   │
    │             │ ××××××××××××××                                                     │
    │         6   │×××××××××××××××                                                     │
    │             │                                                                    │
    │    ─────────┼─────────────────────────────────────────────────  IDLE threshold  │
    │         3   │                                                                    │
    │             │                       ┌────────────────────────┐                  │
    │         1   │  ○○○○○○○○○○○○○        │  MISSING IN TRAINING!  │                  │
    │             │ ○○○○○○○○○○○○○○○       │  Object visible + IDLE │                  │
    │         0   │○○○○○○○○○○○○○○○○○      │  = 0 examples          │                  │
    │             └───────────────────────┴────────────────────────┴───────────▶      │
    │             0.0              0.5               1.0                               │
    │                       Visual Context                                             │
    │                  (0 = empty, 1 = object visible)                                │
    │                                                                                  │
    │   Legend:                                                                        │
    │     × = Training frames with object visible → ALWAYS movement                   │
    │     ○ = Training frames with empty workspace → ALWAYS idle                      │
    │     Empty region in bottom-right = THE DATASET DEFECT                           │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 The Cross-Attention Mechanism and Information-Action Gap

The cross-attention mechanism reveals why the model hallucinated despite encoding distractor information:

```
                    CROSS-ATTENTION: INFORMATION-ACTION GAP
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   KV CACHE DIFFERENCE (What is encoded):                                        │
    │   ──────────────────────────────────────                                         │
    │                                                                                  │
    │   % of Total                                                                     │
    │   KV Diff    │                                                                   │
    │              │                                                                   │
    │         50%  │              ┌───────┐                                            │
    │              │              │       │  42.9%  RIGHT WRIST                        │
    │         40%  │              │       │  (banana is here!)                         │
    │              │              │       │                                            │
    │         30%  │     ┌───────┐│       │                                            │
    │              │     │       ││       │  23.2%  HEAD CAMERA                        │
    │         20%  │     │       ││       │                                            │
    │              │     │       ││       │                                            │
    │         10%  │┌───┐│       ││       │┌───┐┌───┐                                  │
    │              ││   ││       ││       ││   ││   │                                  │
    │          0%  └┴───┴┴───────┴┴───────┴┴───┴┴───┴─────────────▶                   │
    │              Head  Left    Right   Lang  State                                   │
    │              Cam   Wrist   Wrist                                                 │
    │                                                                                  │
    │                                                                                  │
    │   CROSS-ATTENTION WEIGHTS (What is used):                                       │
    │   ───────────────────────────────────────                                        │
    │                                                                                  │
    │   Causal                                                                         │
    │   Effect     │                                                                   │
    │              │                                                                   │
    │        150%  │┌───────┐                                                          │
    │              ││       │  167%  HEAD CAMERA                                       │
    │        100%  ││       │  (dominant driver!)                                      │
    │              ││       │                                                          │
    │         50%  ││       │                   ┌───┐                                  │
    │              ││       │              ┌───┐│   │                                  │
    │          0%  └┴───────┴───────┬──────┴───┴┴───┴─────────────▶                   │
    │              Head  Left    Right   Lang  State                                   │
    │              Cam   Wrist   Wrist                                                 │
    │                       ↑                                                          │
    │                    11% ONLY!                                                     │
    │                    (banana info                                                  │
    │                     ignored!)                                                    │
    │                                                                                  │
    │                                                                                  │
    │   THE GAP:                                                                       │
    │   ─────────                                                                      │
    │                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                         │   │
    │   │  RIGHT WRIST: 42.9% of KV difference (banana encoded)                  │   │
    │   │               BUT only 11% causal effect on action                     │   │
    │   │               → INFORMATION ENCODED BUT NOT USED                       │   │
    │   │                                                                         │   │
    │   │  HEAD CAMERA: 23.2% of KV difference                                   │   │
    │   │               BUT 167% causal effect on action                          │   │
    │   │               → DOMINATES THE ACTION DECISION                          │   │
    │   │                                                                         │   │
    │   │  The model ENCODES the banana in KV cache but IGNORES it because      │   │
    │   │  cross-attention weights favor the head camera, which shows an         │   │
    │   │  "occupied workspace" pattern that training associated with MOVEMENT.  │   │
    │   │                                                                         │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.6 Complete Causal Chain: Visual → KV Cache → Velocity → Trajectory

```
                    COMPLETE CAUSAL CHAIN OF HALLUCINATION
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                  │
    │   STEP 1: VISUAL INPUT                                                           │
    │   ────────────────────                                                           │
    │                                                                                  │
    │   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐                │
    │   │  HEAD CAMERA   │    │  LEFT WRIST    │    │  RIGHT WRIST   │                │
    │   │                │    │                │    │                │                │
    │   │  ┌──────────┐  │    │  ┌──────────┐  │    │  ┌──────────┐  │                │
    │   │  │Workspace │  │    │  │ Gripper  │  │    │  │ BANANA   │  │                │
    │   │  │ overview │  │    │  │  view    │  │    │  │ VISIBLE! │  │                │
    │   │  └──────────┘  │    │  └──────────┘  │    │  └──────────┘  │                │
    │   └────────────────┘    └────────────────┘    └────────────────┘                │
    │            │                    │                    │                          │
    │            ▼                    ▼                    ▼                          │
    │                                                                                  │
    │   STEP 2: KV CACHE ENCODING                                                      │
    │   ─────────────────────────                                                      │
    │                                                                                  │
    │   ┌──────────────────────────────────────────────────────────────────────────┐  │
    │   │                           KV CACHE                                        │  │
    │   │                                                                           │  │
    │   │   K_head (23.2%)  K_left (20.5%)  K_right (42.9%)  K_lang  K_state      │  │
    │   │   ┌──────────┐   ┌──────────┐    ┌──────────┐    ┌────┐  ┌────┐        │  │
    │   │   │ workspace│   │ gripper  │    │  BANANA  │    │task│  │12D │        │  │
    │   │   │ "objects"│   │ position │    │ features │    │desc│  │pos │        │  │
    │   │   └──────────┘   └──────────┘    └──────────┘    └────┘  └────┘        │  │
    │   │                                       ↑                                   │  │
    │   │                                       │                                   │  │
    │   │                              Banana encoded here                          │  │
    │   │                              but attention weight                         │  │
    │   │                              is ONLY 11%!                                 │  │
    │   └──────────────────────────────────────────────────────────────────────────┘  │
    │            │                                                                     │
    │            ▼                                                                     │
    │                                                                                  │
    │   STEP 3: CROSS-ATTENTION                                                        │
    │   ───────────────────────                                                        │
    │                                                                                  │
    │   Action tokens query KV cache:                                                  │
    │                                                                                  │
    │   Q_action × K^T → Attention weights:                                           │
    │     α_head  = 0.45 (167% effect)  ←──── DOMINANT                               │
    │     α_left  = 0.25                                                               │
    │     α_right = 0.15 (11% effect)   ←──── IGNORED despite banana info            │
    │     α_lang  = 0.10                                                               │
    │     α_state = 0.05                                                               │
    │                                                                                  │
    │            │                                                                     │
    │            ▼                                                                     │
    │                                                                                  │
    │   STEP 4: VELOCITY FIELD COMPUTATION                                             │
    │   ──────────────────────────────────                                             │
    │                                                                                  │
    │   v = Σ_r α_r × V_r                                                             │
    │     ≈ 0.45 × V_head + 0.25 × V_left + 0.15 × V_right + ...                     │
    │                                                                                  │
    │   Since V_head encodes "object visible on workspace":                           │
    │   → Training distribution says: object visible → MOVEMENT                       │
    │   → v has HIGH MAGNITUDE (7.45° average post-completion)                        │
    │                                                                                  │
    │            │                                                                     │
    │            ▼                                                                     │
    │                                                                                  │
    │   STEP 5: TRAJECTORY GENERATION                                                  │
    │   ─────────────────────────────                                                  │
    │                                                                                  │
    │   Over 10 denoising steps:                                                       │
    │   x_T = x_0 + Σ_t (dt × v_t)                                                    │
    │                                                                                  │
    │   With high |v|: trajectory accumulates MOVEMENT                                │
    │   Final x_T exits IDLE region → HALLUCINATION                                   │
    │                                                                                  │
    │   ┌─────────────────────────────────────────────────────────────────────────┐   │
    │   │                                                                         │   │
    │   │   Action Space:                                                         │   │
    │   │                                                                         │   │
    │   │       ▲                                                                 │   │
    │   │       │                    ★ HALLUC final                              │   │
    │   │       │                   ╱  (MOVEMENT region)                          │   │
    │   │       │                 ╱                                               │   │
    │   │       │               ╱                                                 │   │
    │   │       │    ┌───────┐╱                                                   │   │
    │   │       │    │IDLE   │                                                    │   │
    │   │       │    │region │ ← NORMAL final (stays here)                       │   │
    │   │       │    │  △    │                                                    │   │
    │   │       │    └───────┘                                                    │   │
    │   │       └─────────────────────────────────────────────▶                   │   │
    │   │                                                                         │   │
    │   └─────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                  │
    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Concepts Summary

### 3.1 Flow Matching

Flow matching learns a velocity field `v(x_t, t, K)` that transforms noise into actions:

| Component | Description |
|-----------|-------------|
| x_t | Noisy trajectory at time t |
| v | Velocity field (direction + magnitude) |
| K | KV cache (conditioning from visual + language + state) |
| Update | x_{t+dt} = x_t + dt × v |

The **magnitude of v** determines behavior:
- **|v| < 3°** → IDLE (robot stays still)
- **|v| > 3°** → MOVEMENT (robot moves)

### 3.2 KV Cache Conditioning

The KV cache stores pre-computed keys and values from the prefix:

| Region | Tokens | Content |
|--------|--------|---------|
| Head camera | 0-63 | Top-down workspace view |
| Left wrist | 64-127 | Left arm perspective |
| Right wrist | 128-191 | Right arm perspective (banana here!) |
| Language | 192-239 | Task description |
| State | 240 | Joint positions |

### 3.3 Cross-Attention Weights

Cross-attention determines how much each KV region affects action generation:

| Region | KV Difference | Causal Effect | Gap |
|--------|---------------|---------------|-----|
| Head camera | 23.2% | 167% | Over-attended |
| Right wrist | 42.9% | 11% | Under-attended |

### 3.4 Training Distribution Bias

The root cause is a dataset defect:

| Visual Context | Action in Training | Probability |
|----------------|-------------------|-------------|
| Object on workspace | MOVEMENT | 100% |
| Empty workspace | IDLE | 100% |
| Object on workspace | IDLE | **0%** (MISSING!) |

---

## 4. References

### Related Documentation

| Document | Description |
|----------|-------------|
| `jdocs/investigation_reports/first_principles_explanation.md` | Mathematical explanation of training bias |
| `jdocs/investigation_reports/flow_matching_explanation.md` | P(trajectory \| velocity_field) explanation |
| `jdocs/investigation_reports/visualization_guide.md` | How to read all analysis diagrams |
| `jdocs/investigation_reports/mechanism_explanation.md` | Information-action gap details |

### Key Visualizations

| File | Description |
|------|-------------|
| `logs/investigation/causal_distribution/p_traj_given_kv_cache.png` | P(trajectory \| KV_cache) analysis |
| `logs/investigation/first_principles/context_velocity_scatter_clean.png` | Training distribution bias |
| `logs/investigation/first_principles/p_trajectory_given_velocity_field.png` | Flow matching mechanism |

### Generation Scripts

| Script | Output |
|--------|--------|
| `jdocs/scripts/investigation/tools/generate_p_traj_kv.py` | p_traj_given_kv_cache.png |
| `jdocs/scripts/investigation/tools/context_velocity_scatter.py` | context_velocity_scatter_clean.png |
| `jdocs/scripts/investigation/tools/flow_matching_visualization.py` | p_trajectory_given_velocity_field.png |
| `jdocs/scripts/investigation/tools/comprehensive_mechanism_visualization.py` | Complete mechanism diagrams |

---

## 5. Conclusion

The SmolVLA hallucination behavior is not a model architecture bug, but a **dataset distribution problem**. The model correctly learns P(trajectory | KV_cache) from training data, but the training data never showed examples of "stay idle while distractor object is visible." When a distractor (banana) appears in the workspace post-completion, the model applies its learned rule: object visible → movement, leading to the hallucination of grasping at empty space.

The fix requires adding training data with the missing pattern: **object visible on workspace + IDLE behavior**.
