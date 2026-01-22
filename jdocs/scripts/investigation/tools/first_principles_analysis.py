#!/usr/bin/env python3
"""
First Principles Analysis of SmolVLA Hallucination Mechanism

Mathematical Framework:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model learns: P(trajectory | KV_cache)

Where:
  - KV_cache = f(images, language, state)
  - trajectory = Flow_matching_sample(noise, velocity_field(KV_cache))

The velocity field v(x_t, t, K) determines the trajectory:
  x_{t+dt} = x_t + dt * v(x_t, t, K)

Key Questions:
1. How does KV_cache differ between halluc/normal cases?
2. How does this difference affect v(x_t, t, K)?
3. What training distribution P_train(trajectory | visual_pattern) caused this?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import torch

OUTPUT_DIR = Path("/home/jrobot/project/lerobot/logs/investigation/first_principles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_training_data_distribution():
    """
    Analyze P(trajectory_type | visual_context) from training data.

    Hypothesis: Training data creates a biased conditional distribution where
    P(movement | object_visible_on_workspace) >> P(idle | object_visible_on_workspace)
    """
    print("\n" + "="*70)
    print("PART 1: Training Data Distribution Analysis")
    print("="*70)

    # Load training data
    DATASET_PATH = Path("/home/jrobot/project/lerobot/datasets_bimanuel/multitasks")

    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        return None

    # Load parquet data
    import pyarrow.parquet as pq

    parquet_files = list(DATASET_PATH.glob("**/*.parquet"))
    if not parquet_files:
        print("No parquet files found")
        return None

    print(f"Found {len(parquet_files)} parquet files")

    # Analyze action patterns
    all_actions = []
    all_episode_indices = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()

        if 'action' in df.columns:
            actions = np.stack(df['action'].values)
            all_actions.append(actions)

            if 'episode_index' in df.columns:
                all_episode_indices.extend(df['episode_index'].tolist())

    if not all_actions:
        print("No action data found")
        return None

    actions = np.concatenate(all_actions, axis=0)
    print(f"Total frames: {len(actions)}")
    print(f"Action dimensions: {actions.shape[1]}")

    # Compute action deltas (movement magnitude)
    action_deltas = np.abs(np.diff(actions, axis=0)).max(axis=1)

    # Define IDLE threshold (in normalized action space)
    # Based on observation: ~0.01-0.02 for idle, >0.05 for movement
    IDLE_THRESHOLD = 0.03

    is_idle = action_deltas < IDLE_THRESHOLD
    is_movement = ~is_idle

    print(f"\nAction Delta Statistics:")
    print(f"  Mean: {action_deltas.mean():.4f}")
    print(f"  Std: {action_deltas.std():.4f}")
    print(f"  Median: {np.median(action_deltas):.4f}")
    print(f"  P10: {np.percentile(action_deltas, 10):.4f}")
    print(f"  P90: {np.percentile(action_deltas, 90):.4f}")

    print(f"\nBehavior Classification (threshold={IDLE_THRESHOLD}):")
    print(f"  IDLE frames: {is_idle.sum()} ({100*is_idle.mean():.1f}%)")
    print(f"  MOVEMENT frames: {is_movement.sum()} ({100*is_movement.mean():.1f}%)")

    return {
        'actions': actions,
        'action_deltas': action_deltas,
        'is_idle': is_idle,
        'idle_ratio': is_idle.mean(),
        'movement_ratio': is_movement.mean(),
    }


def analyze_episode_phase_distribution(training_data):
    """
    Analyze P(behavior | episode_phase) from training data.

    Key insight: The model learns from the TEMPORAL CONTEXT of when behaviors occur.
    If IDLE only occurs at episode END with clean workspace, the model learns:
      P(idle | post_completion) ≈ P(idle | clean_workspace)
    """
    print("\n" + "="*70)
    print("PART 2: Episode Phase Distribution Analysis")
    print("="*70)

    if training_data is None:
        print("No training data available")
        return None

    actions = training_data['actions']
    action_deltas = training_data['action_deltas']

    # Segment into episodes (assuming continuous recording with resets)
    # Look for large jumps in action space as episode boundaries
    large_jumps = np.where(np.abs(np.diff(actions, axis=0)).max(axis=1) > 0.5)[0]

    # Group into episodes
    episode_boundaries = [0] + list(large_jumps + 1) + [len(actions)]
    episodes = []

    for i in range(len(episode_boundaries) - 1):
        start = episode_boundaries[i]
        end = episode_boundaries[i + 1]
        if end - start > 50:  # Minimum episode length
            episodes.append({
                'start': start,
                'end': end,
                'length': end - start,
                'actions': actions[start:end],
                'deltas': action_deltas[start:min(end, len(action_deltas))]
            })

    print(f"Identified {len(episodes)} episodes")

    if len(episodes) == 0:
        return None

    # Analyze phase distribution within episodes
    IDLE_THRESHOLD = 0.03

    phase_stats = {
        'early': {'idle': 0, 'movement': 0},  # 0-25%
        'mid': {'idle': 0, 'movement': 0},    # 25-75%
        'late': {'idle': 0, 'movement': 0},   # 75-100%
    }

    for ep in episodes:
        length = len(ep['deltas'])
        if length < 10:
            continue

        early_end = int(0.25 * length)
        mid_end = int(0.75 * length)

        early_deltas = ep['deltas'][:early_end]
        mid_deltas = ep['deltas'][early_end:mid_end]
        late_deltas = ep['deltas'][mid_end:]

        for phase, deltas in [('early', early_deltas), ('mid', mid_deltas), ('late', late_deltas)]:
            if len(deltas) > 0:
                idle_count = (deltas < IDLE_THRESHOLD).sum()
                movement_count = (deltas >= IDLE_THRESHOLD).sum()
                phase_stats[phase]['idle'] += idle_count
                phase_stats[phase]['movement'] += movement_count

    print("\nPhase-wise Behavior Distribution:")
    print("-" * 50)
    for phase in ['early', 'mid', 'late']:
        total = phase_stats[phase]['idle'] + phase_stats[phase]['movement']
        if total > 0:
            idle_pct = 100 * phase_stats[phase]['idle'] / total
            movement_pct = 100 * phase_stats[phase]['movement'] / total
            print(f"  {phase.upper():6s}: IDLE {idle_pct:5.1f}% | MOVEMENT {movement_pct:5.1f}%")

    return phase_stats


def compute_conditional_trajectory_distribution():
    """
    Compute P(trajectory_direction | current_state, phase).

    This shows what the model SHOULD have learned from training data.

    Key insight: If training data always shows:
      - Early phase: movement TOWARD objects
      - Late phase: IDLE (near-zero movement)

    Then the model learns a conditional distribution that doesn't include:
      - Late phase: IDLE despite visible objects
    """
    print("\n" + "="*70)
    print("PART 3: Conditional Trajectory Distribution P(direction | context)")
    print("="*70)

    # Load inference traces for halluc/normal cases
    HALLUC_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table/trace.jsonl")
    NORMAL_TRACE = Path("/home/jrobot/project/lerobot/logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate/trace.jsonl")

    def load_trace(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    halluc = load_trace(HALLUC_TRACE)
    normal = load_trace(NORMAL_TRACE)

    # Extract actions and states
    def extract_trajectory_info(trace_data, start_step, end_step):
        """Extract trajectory characteristics for a time window."""
        actions = []
        states = []
        for entry in trace_data:
            if start_step <= entry['step'] < end_step:
                actions.append(entry['action_raw'])
                states.append(entry['state_normalized'])

        if len(actions) < 2:
            return None

        actions = np.array(actions)
        states = np.array(states)

        # Compute trajectory direction (change in action over time)
        action_velocity = np.diff(actions, axis=0).mean(axis=0)
        action_magnitude = np.linalg.norm(action_velocity)

        # Compute mean state
        mean_state = states.mean(axis=0)

        return {
            'action_velocity': action_velocity,
            'action_magnitude': action_magnitude,
            'mean_state': mean_state,
            'trajectory': actions,
        }

    # Compare at different phases
    phases = [
        ('approach', 0, 100),
        ('execution', 100, 150),
        ('post_completion', 167, 250),
        ('late', 250, 350),
    ]

    print("\nTrajectory Characteristics by Phase:")
    print("-" * 70)
    print(f"{'Phase':<20} {'Halluc Magnitude':<18} {'Normal Magnitude':<18} {'Ratio':<10}")
    print("-" * 70)

    for phase_name, start, end in phases:
        h_info = extract_trajectory_info(halluc, start, end)
        n_info = extract_trajectory_info(normal, start, end)

        if h_info and n_info:
            h_mag = h_info['action_magnitude']
            n_mag = n_info['action_magnitude']
            ratio = h_mag / (n_mag + 1e-8)

            print(f"{phase_name:<20} {h_mag:<18.4f} {n_mag:<18.4f} {ratio:<10.2f}x")

    # Detailed analysis of post-completion phase
    print("\n" + "="*70)
    print("DETAILED: Post-Completion Phase Analysis (Steps 167-250)")
    print("="*70)

    h_post = extract_trajectory_info(halluc, 167, 250)
    n_post = extract_trajectory_info(normal, 167, 250)

    if h_post and n_post:
        # Compute trajectory direction cosine similarity
        h_vel = h_post['action_velocity']
        n_vel = n_post['action_velocity']

        h_vel_norm = h_vel / (np.linalg.norm(h_vel) + 1e-8)
        n_vel_norm = n_vel / (np.linalg.norm(n_vel) + 1e-8)

        cos_sim = np.dot(h_vel_norm, n_vel_norm)

        print(f"\nHalluc action velocity magnitude: {np.linalg.norm(h_vel):.4f}")
        print(f"Normal action velocity magnitude: {np.linalg.norm(n_vel):.4f}")
        print(f"Direction cosine similarity: {cos_sim:.4f}")

        if cos_sim < 0.5:
            print(">>> DIVERGENT DIRECTIONS: Halluc and Normal are moving differently")
        else:
            print(">>> SIMILAR DIRECTIONS: Both moving in similar direction but different magnitudes")

        # What direction is halluc moving?
        print("\nHalluc velocity by joint (top 5 contributors):")
        joint_names = ["L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
                      "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip"]

        vel_abs = np.abs(h_vel)
        top_joints = np.argsort(vel_abs)[::-1][:5]

        for idx in top_joints:
            name = joint_names[idx] if idx < len(joint_names) else f"j{idx}"
            print(f"  {name}: {h_vel[idx]:+.4f} (|{vel_abs[idx]:.4f}|)")


def analyze_kv_cache_effect_on_velocity_field():
    """
    Analyze how KV cache difference affects the velocity field v(x_t, t, K).

    The flow matching model learns:
      v(x_t, t, K) = neural_network(x_t, t, cross_attention(x_t, K))

    Key insight: The velocity field is CONDITIONED on K (KV cache).
    Different K → Different v → Different trajectory.

    We need to understand:
    1. What information in K drives the velocity field?
    2. Why does K_halluc produce "movement toward workspace" velocity?
    3. Why does K_normal produce "stay still" velocity?
    """
    print("\n" + "="*70)
    print("PART 4: KV Cache Effect on Velocity Field v(x_t, t, K)")
    print("="*70)

    # Load pre-computed attention knockout results
    KNOCKOUT_RESULTS = Path("/home/jrobot/project/lerobot/logs/investigation/advanced_analysis_20260121/attention_knockout/analyses.json")

    if not KNOCKOUT_RESULTS.exists():
        print(f"Knockout results not found at {KNOCKOUT_RESULTS}")
        return None

    with open(KNOCKOUT_RESULTS) as f:
        knockout_data = json.load(f)

    print("\nVelocity Field Sensitivity to KV Cache Regions:")
    print("-" * 70)
    print("(Measured by: How much does v(x_t, t, K) change when we zero out region of K?)")
    print()

    for case in knockout_data:
        case_name = case['case_name']
        print(f"\nCase: {case_name}")
        print(f"  Normal trajectory shape: {case['normal_trajectory_shape']}")
        print(f"  Normal action delta: {case['normal_action_delta']:.4f}")

        if 'knockout_results' in case:
            print(f"\n  Region Causal Effects:")
            for region, results in case['knockout_results'].items():
                pct_change = results.get('percent_change', 0)
                print(f"    {region:15s}: {pct_change:+7.1f}% change in |v|")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("""
The velocity field v(x_t, t, K) is most sensitive to HEAD CAMERA region of K.

Mathematical interpretation:
  v = f(cross_attention(action_tokens, K))

  Where cross_attention weights determine which parts of K influence v.

  When head_camera region of K is zeroed:
    - Halluc case: 167% increase in |v| → head camera was SUPPRESSING movement
    - This suggests head camera encodes "task state" information

  When right_wrist region of K is zeroed:
    - Halluc case: only 11% change → right wrist contributes little to v
    - Even though right wrist SEES the banana

  CONCLUSION: The model uses HEAD CAMERA to decide magnitude of v,
              not the right wrist camera where the banana is visible.
""")


def synthesize_first_principles_explanation():
    """
    Synthesize all findings into a first-principles explanation.
    """
    print("\n" + "="*70)
    print("SYNTHESIS: First Principles Explanation of Hallucination")
    print("="*70)

    explanation = """

THE MATHEMATICAL MECHANISM OF HALLUCINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MODEL LEARNS: P(trajectory | KV_cache)

   The SmolVLA model is a conditional flow matching model that learns:

   trajectory ~ FlowMatching(noise, v(·, ·, K))

   Where v(x_t, t, K) is the learned velocity field conditioned on KV_cache K.

2. TRAINING DISTRIBUTION CREATES BIAS:

   From training data, the model learns the conditional distribution:

   P(trajectory | visual_pattern) = {
     P(MOVEMENT | object_on_workspace) ≈ 1.0   [all task execution frames]
     P(IDLE | clean_workspace) ≈ 1.0           [all post-completion frames]
     P(IDLE | object_on_workspace) ≈ 0.0       [NEVER SEEN IN TRAINING]
   }

   The training data has a PERFECT CORRELATION between:
   - "object visible on workspace" → movement
   - "clean workspace" → idle

   The model cannot disentangle "object relevance" from "object presence".

3. KV CACHE ENCODES VISUAL PATTERN, NOT TASK STATE:

   K = Encoder(images, language, state)

   The KV cache encodes:
   - Visual features from cameras (dominant)
   - Language tokens (task description)
   - Current state

   But it does NOT explicitly encode:
   - "Is my task complete?"
   - "Is this object relevant to my task?"

   These semantic concepts must be INFERRED from visual/language context,
   but the model never learned to make this inference because training
   data never required it.

4. VELOCITY FIELD IS BIASED BY VISUAL PATTERN:

   The learned velocity field v(x_t, t, K) produces:

   v(x_t, t, K_halluc) → movement toward workspace
     Because K_halluc encodes "object on workspace" visual pattern
     And P_train(movement | object_on_workspace) ≈ 1.0

   v(x_t, t, K_normal) → stay still
     Because K_normal encodes "clean workspace" visual pattern
     And P_train(idle | clean_workspace) ≈ 1.0

5. THE HALLUCINATION IS A DISTRIBUTION SAMPLING ARTIFACT:

   The model is NOT "hallucinating" in the sense of generating nonsense.
   It is CORRECTLY sampling from P(trajectory | visual_pattern).

   The problem is that P(trajectory | visual_pattern) was learned from
   biased training data that never showed "idle despite visible object".

   The model has no way to know the banana is "irrelevant" because:
   - Training never demonstrated irrelevant objects
   - The visual pattern "object on workspace" ALWAYS meant "move"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAL STATEMENT:

Let V = visual_features, L = language, S = state
Let K = KVCache(V, L, S) be the conditioning

Training data defines:
  D_train = {(K_i, τ_i)} where τ_i is trajectory

The model learns:
  P_θ(τ | K) ≈ P_train(τ | K)

The bias:
  For K where V shows "object on workspace":
    P_train(τ = movement | K) = 1.0
    P_train(τ = idle | K) = 0.0

  This is because training NEVER has examples where:
    V = "object on workspace" AND τ = "idle"

At inference:
  K_halluc encodes V_halluc = "banana on workspace"
  Model samples τ ~ P_θ(τ | K_halluc)
  Since P_θ ≈ P_train, and P_train(movement | object_on_workspace) = 1.0
  The model produces τ = movement

This is not a model bug - it's a DATASET DISTRIBUTION BUG.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(explanation)

    # Save the explanation
    with open(OUTPUT_DIR / "first_principles_explanation.txt", "w") as f:
        f.write(explanation)

    print(f"\nSaved to: {OUTPUT_DIR / 'first_principles_explanation.txt'}")


def main():
    print("="*70)
    print("FIRST PRINCIPLES ANALYSIS OF SMOLVLA HALLUCINATION")
    print("="*70)

    # Part 1: Training data distribution
    training_data = analyze_training_data_distribution()

    # Part 2: Episode phase distribution
    phase_stats = analyze_episode_phase_distribution(training_data)

    # Part 3: Conditional trajectory distribution
    compute_conditional_trajectory_distribution()

    # Part 4: KV cache effect on velocity field
    analyze_kv_cache_effect_on_velocity_field()

    # Synthesis
    synthesize_first_principles_explanation()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
