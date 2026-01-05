#!/usr/bin/env python3
"""
Interactive port identification script.
Run this to identify which physical arm is on which USB port.
"""
import sys
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
import time

def wiggle_port(port):
    """Wiggle elbow on a port to identify which arm it controls."""
    try:
        motors = {name: Motor(id=i+1, model='sts3215', norm_mode=MotorNormMode.DEGREES)
                  for i, name in enumerate(['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'])}
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        pos = bus.read('Present_Position', 'elbow_flex', normalize=False)
        if pos is None:
            print(f"  Could not read position")
            bus.disconnect()
            return False

        pos = int(pos)
        bus.write('Torque_Enable', 'elbow_flex', 1, normalize=False)

        # Wiggle
        for p in [pos+150, pos-150, pos+150, pos]:
            bus.write('Goal_Position', 'elbow_flex', p, normalize=False)
            time.sleep(0.4)

        bus.write('Torque_Enable', 'elbow_flex', 0, normalize=False)
        bus.disconnect()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("="*60)
    print("PORT IDENTIFICATION TEST")
    print("="*60)
    print()
    print("Watch carefully which arm moves for each port.")
    print()
    print("Options:")
    print("  lf = Left Follower")
    print("  rf = Right Follower")
    print("  ll = Left Leader")
    print("  rl = Right Leader")
    print("  n  = None/didn't move")
    print()
    input("Press Enter when ready to start...")
    print()

    results = {}

    for port in ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3']:
        print(f"Testing {port}...")
        print(f"  Wiggling now - WATCH which arm moves!")
        success = wiggle_port(port)

        if success:
            response = input(f"  Which arm moved? (lf/rf/ll/rl/n): ").strip().lower()
            results[port] = response
            print()
        else:
            results[port] = 'error'
            print(f"  Skipped due to error\n")

    print("="*60)
    print("RESULTS - Copy this to Claude!")
    print("="*60)
    mapping = {
        'lf': 'Left Follower',
        'rf': 'Right Follower',
        'll': 'Left Leader',
        'rl': 'Right Leader',
        'n': 'None/Unknown',
        'error': 'Error'
    }
    for port, code in results.items():
        print(f"  {port} = {mapping.get(code, code)}")

if __name__ == "__main__":
    main()
