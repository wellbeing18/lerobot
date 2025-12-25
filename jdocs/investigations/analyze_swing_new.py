import json
import numpy as np
import sys
import os

def analyze_swing(trace_path):
    print(f"Analyzing trace for swing behavior: {trace_path}")
    trace_file = os.path.join(trace_path, "trace.jsonl")
    if not os.path.exists(trace_file):
        print(f"Error: {trace_file} does not exist.")
        return

    steps = []
    joint_data = {'shoulder_lift': [], 'elbow_flex': []}
    action_data = {'shoulder_lift': [], 'elbow_flex': []}

    with open(trace_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            steps.append(data['step'])
            states = data.get('joint_states', [])
            if states and len(states) == 6:
                joint_data['shoulder_lift'].append(states[1])
                joint_data['elbow_flex'].append(states[2])
            else:
                joint_data['shoulder_lift'].append(None)
                joint_data['elbow_flex'].append(None)

            actions = data.get('action_executed', [])
            if actions and len(actions) == 6:
                action_data['shoulder_lift'].append(actions[1])
                action_data['elbow_flex'].append(actions[2])
            else:
                action_data['shoulder_lift'].append(None)
                action_data['elbow_flex'].append(None)

    for joint_name in ['shoulder_lift', 'elbow_flex']:
        print(f"\n--- {joint_name} ---")
        actions = action_data[joint_name]
        for i in range(1, len(actions)):
            if actions[i] is not None and actions[i-1] is not None:
                delta = actions[i] - actions[i-1]
                if abs(delta) > 10.0:
                    print(f"Step {steps[i]}: Large Action Jump! {actions[i-1]:.2f} -> {actions[i]:.2f} (Delta: {delta:.2f})")

if __name__ == "__main__":
    analyze_swing(sys.argv[1])


