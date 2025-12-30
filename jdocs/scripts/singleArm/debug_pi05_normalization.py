#!/usr/bin/env python3
"""
Debug script to verify Pi0.5 state normalization and discretization.

This script simulates the normalization and discretization process
to identify if there's an issue with how states are being encoded.
"""

import json
import numpy as np

# Load dataset stats
with open('datasets/pick_and_place/meta/stats.json') as f:
    stats = json.load(f)

# Get state stats
state_stats = stats['observation.state']
state_mean = np.array(state_stats['mean'])
state_std = np.array(state_stats['std'])
state_q01 = np.array(state_stats['q01'])
state_q99 = np.array(state_stats['q99'])

# Simulated robot state at inference time (from trace)
robot_state = np.array([0.97, -98.23, 100.0, 60.26, -8.39, 0.0])

print("=" * 60)
print("Pi0.5 State Normalization Debug")
print("=" * 60)

print("\n=== Robot State at Inference ===")
print(f"  Raw state: {[round(v, 2) for v in robot_state]}")

# Joint names for reference
joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
for i, name in enumerate(joint_names):
    print(f"  {name}: {robot_state[i]:.2f}°")

print("\n=== Dataset Statistics ===")
print(f"  Mean:  {[round(v, 2) for v in state_mean]}")
print(f"  Std:   {[round(v, 2) for v in state_std]}")
print(f"  Q01:   {[round(v, 2) for v in state_q01]}")
print(f"  Q99:   {[round(v, 2) for v in state_q99]}")

# Method 1: QUANTILES normalization (Pi0.5 default)
print("\n=== QUANTILES Normalization ===")
# Formula: 2.0 * (x - q01) / (q99 - q01) - 1.0
q_range = state_q99 - state_q01
q_normalized = 2.0 * (robot_state - state_q01) / q_range - 1.0
print(f"  Normalized (QUANTILES): {[round(v, 4) for v in q_normalized]}")
print(f"  Min: {q_normalized.min():.4f}, Max: {q_normalized.max():.4f}")

for i, name in enumerate(joint_names):
    if q_normalized[i] < -1.0 or q_normalized[i] > 1.0:
        print(f"  ⚠️  {name}: {q_normalized[i]:.4f} is OUTSIDE [-1, 1]!")

# Method 2: MEAN_STD normalization
print("\n=== MEAN_STD Normalization ===")
# Formula: (x - mean) / std
ms_normalized = (robot_state - state_mean) / state_std
print(f"  Normalized (MEAN_STD): {[round(v, 4) for v in ms_normalized]}")
print(f"  Min: {ms_normalized.min():.4f}, Max: {ms_normalized.max():.4f}")

for i, name in enumerate(joint_names):
    if ms_normalized[i] < -1.0 or ms_normalized[i] > 1.0:
        print(f"  ⚠️  {name}: {ms_normalized[i]:.4f} is OUTSIDE [-1, 1]!")

# Discretization (256 bins)
print("\n=== Discretization (256 bins) ===")
print("Discretization assumes values in [-1, 1] range.")

bins = np.linspace(-1, 1, 257)[:-1]

for norm_name, normalized in [("QUANTILES", q_normalized), ("MEAN_STD", ms_normalized)]:
    discretized = np.digitize(normalized, bins) - 1
    # Clamp to valid range
    discretized = np.clip(discretized, 0, 255)

    print(f"\n  {norm_name} discretized bins: {discretized.tolist()}")
    for i, name in enumerate(joint_names):
        if discretized[i] == 0:
            print(f"    ⚠️  {name}: bin 0 (value was < -1, CLIPPED!)")
        elif discretized[i] == 255:
            print(f"    ⚠️  {name}: bin 255 (value was > 1, CLIPPED!)")

# Check what the bins represent
print("\n=== Bin Value Mapping ===")
print("Bins map normalized values back to original scale:")

# For QUANTILES, bin 0 = q01, bin 255 = q99
print("\n  QUANTILES: bin 0 → q01, bin 127 → median, bin 255 → q99")
for i, name in enumerate(joint_names):
    bin_val = q_normalized[i]
    # Unnormalize: (bin_val + 1.0) * q_range / 2.0 + q01
    unnorm = (bin_val + 1.0) * q_range[i] / 2.0 + state_q01[i]
    print(f"    {name}: normalized {bin_val:.4f} → original {unnorm:.2f}° (actual: {robot_state[i]:.2f}°)")

print("\n=== Analysis ===")
# Check if the issue is shoulder_lift
sl_idx = 1  # shoulder_lift index
print(f"\nShoulder Lift Analysis:")
print(f"  Robot position: {robot_state[sl_idx]:.2f}°")
print(f"  Dataset q01 (minimum): {state_q01[sl_idx]:.2f}°")
print(f"  Dataset q99 (maximum): {state_q99[sl_idx]:.2f}°")
print(f"  Dataset mean: {state_mean[sl_idx]:.2f}°")

# The robot at -98° is very close to q01 of -99.29°
# With QUANTILES, this normalizes to close to -1.0
print(f"\n  QUANTILES normalized value: {q_normalized[sl_idx]:.4f}")
print(f"  MEAN_STD normalized value: {ms_normalized[sl_idx]:.4f}")

# For MEAN_STD, check if it's clipped
if ms_normalized[sl_idx] < -1.0:
    print(f"\n  ⚠️  With MEAN_STD, shoulder_lift = {ms_normalized[sl_idx]:.4f}")
    print(f"      This is OUTSIDE [-1, 1] and will be CLIPPED to bin 0!")
    print(f"      The model sees the same input as if the joint was at the minimum position.")

print("\n=== Elbow Analysis ===")
el_idx = 2  # elbow index
print(f"  Robot position: {robot_state[el_idx]:.2f}°")
print(f"  Dataset q01: {state_q01[el_idx]:.2f}°")
print(f"  Dataset q99: {state_q99[el_idx]:.2f}°")
print(f"  QUANTILES normalized: {q_normalized[el_idx]:.4f}")
print(f"  MEAN_STD normalized: {ms_normalized[el_idx]:.4f}")

if q_normalized[el_idx] > 1.0:
    print(f"\n  ⚠️  With QUANTILES, elbow = {q_normalized[el_idx]:.4f}")
    print(f"      This is OUTSIDE [-1, 1] and will be CLIPPED to bin 255!")
