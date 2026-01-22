import sys
from pathlib import Path
sys.path.insert(0, "src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    # Point to the local directory
    ds = LeRobotDataset(repo_id="multitasks", root="datasets_bimanuel/multitasks")
    print(f"Loaded dataset with {len(ds)} frames")
    print(f"Sample action: {ds[0]['action']}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
