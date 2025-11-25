# LeRobot Complete Pipeline Explained

A comprehensive guide to understanding the LeRobot robotics framework for someone new to robotics.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Data Collection & Datasets](#data-collection--datasets)
3. [Dataset Structure & Storage](#dataset-structure--storage)
4. [Configuration System](#configuration-system)
5. [Preprocessors & Normalization](#preprocessors--normalization)
6. [Policy Implementations](#policy-implementations)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation & Deployment](#evaluation--deployment)
9. [Robot Support & Control](#robot-support--control)
10. [Model Inference on Robots](#model-inference-on-robots)

---

## Architecture Overview

LeRobot is built around a modular architecture with clear separation of concerns:

```
LeRobot Framework
├── Data Pipeline (Datasets)
│   ├── Collection from robots/simulations
│   ├── Storage in standardized format
│   └── Loading with preprocessing
├── Configuration System (Hydra)
│   ├── Default configs
│   ├── Policy configs
│   ├── Environment configs
│   └── Robot configs
├── Policy Layer
│   ├── ACT (Action Chunking Transformer)
│   ├── Diffusion Policy
│   ├── TDMPC
│   └── VQ-BeT
├── Training Engine
│   ├── Offline training
│   ├── Online training (RL)
│   └── Evaluation during training
├── Evaluation System
│   ├── Batch rollouts
│   ├── Metric computation
│   └── Video recording
└── Robot Control
    ├── Real robot interface
    ├── Simulation environments
    └── Inference on hardware
```

---

## Data Collection & Datasets

### How Data Collection Works

In LeRobot, data is collected through **teleoperation** or **manual control**:

1. **Teleoperation Setup**: A human operator controls a "leader" robot while a "follower" robot mirrors the movements
2. **Recording**: During this teleoperation:
   - Camera frames are captured from multiple angles
   - Robot joint states (positions, velocities) are recorded
   - Actions (target joint positions) are recorded
   - Everything is synchronized to a fixed FPS (e.g., 10 Hz, 30 Hz)
3. **Episodes**: Each complete task execution is recorded as an "episode"
   - Contains sequential observations and actions
   - Marked with timestamps for synchronization
4. **Storage**: Data is stored locally first, then converted to standardized format for the Hub

### Data Control Scripts

From `lerobot/scripts/control_robot.py`:
- `calibrate`: Calibrates robot joints to ensure consistency across units
- `teleoperate`: Manual control without recording (highest frequency ~200 Hz)
- `record`: Records teleoperated episodes with controlled FPS
- `replay`: Replays recorded episodes (useful for verification)

Example recording command:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id user/koch_pick_place \
    --num-episodes 50 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

---

## Dataset Structure & Storage

### The LeRobotDataset Format

LeRobot uses a unified dataset format accessible via the Hugging Face Hub. Load a dataset simply:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/pusht")  # Or any dataset ID
```

### Internal Structure

A LeRobotDataset contains:

```
dataset attributes:
├── hf_dataset: Hugging Face dataset (Arrow/parquet backend)
│   ├── observation.images.cam_high (VideoFrame)
│   │   └── {'path': path to mp4, 'timestamp': float}
│   ├── observation.state (float32 array): joint positions
│   ├── observation.other_modalities
│   ├── action (float32 array): target positions
│   ├── episode_index (int64): which episode this frame belongs to
│   ├── frame_index (int64): index within episode (starts at 0)
│   ├── timestamp (float32): time in the episode
│   └── next.done (bool): is this the last frame?
│
├── episode_data_index: Start/end frame indices for each episode
│   ├── from: (num_episodes,) - first frame index
│   └── to: (num_episodes,) - last frame index
│
├── stats: Statistics for normalization
│   ├── observation.images.cam_high
│   │   ├── mean: (c, 1, 1) tensor
│   │   ├── std: (c, 1, 1) tensor
│   │   ├── min: (c, 1, 1) tensor
│   │   └── max: (c, 1, 1) tensor
│   └── ... (for all features)
│
├── info: Metadata
│   ├── codebase_version: version when created
│   ├── fps: frame rate (e.g., 10, 30)
│   ├── video: bool (whether videos or PNGs)
│   └── encoding: ffmpeg options used
│
├── videos_dir: Location of mp4 videos or png files
└── camera_keys: List of image observation keys
```

### File Format Details

- **hf_dataset**: Parquet format (via Hugging Face datasets library)
- **videos**: MP4 files (compressed) or PNG files (for specific frames)
- **episode_data_index**: Safetensors format
- **stats**: Safetensors format
- **info**: JSON format

### Accessing Data

PyTorch-style indexing:
```python
item = dataset[idx]
# Returns dict with all features at that index
```

### Key Feature: Delta Timestamps

Load multiple frames at different times relative to the target frame:

```python
delta_timestamps = {
    # Load 4 images: 1s ago, 0.5s ago, 0.2s ago, current
    "observation.image": [-1.0, -0.5, -0.2, 0.0],
    # Load 8 past state vectors
    "observation.state": [-1.5, -1.0, -0.5, -0.2, -0.1, -0.02, -0.01, 0.0],
    # Load 64 future actions (0.0 to 6.3 seconds)
    "action": [i/10 for i in range(64)],  # assuming 10 FPS
}

dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
item = dataset[0]
# item['observation.image'].shape = (4, 3, 96, 96)  # 4 images
# item['observation.state'].shape = (8, state_dim)
# item['action'].shape = (64, action_dim)
```

This is critical for policies that need to:
- See history of observations (transformer-based policies)
- Predict sequences of actions (diffusion, ACT policies)

---

## Configuration System

### Hydra Configuration Framework

LeRobot uses **Hydra** for configuration management. Hydra allows:
- YAML-based config files
- Command-line overrides
- Composition of configs
- Automatic experiment logging

### Configuration Hierarchy

```
configs/
├── default.yaml (root config, selects defaults)
├── env/
│   ├── pusht.yaml
│   ├── aloha.yaml
│   ├── xarm.yaml
│   └── koch_real.yaml
├── policy/
│   ├── act.yaml
│   ├── diffusion.yaml
│   ├── tdmpc.yaml
│   └── vqbet.yaml
└── robot/
    ├── koch.yaml
    └── koch_bimanual.yaml
```

### Default Configuration Structure

From `default.yaml`:

```yaml
resume: false
device: cuda
seed: 1000
dataset_repo_id: lerobot/pusht
video_backend: pyav

training:
  offline_steps: ???  # Must be specified
  batch_size: ???
  num_workers: 4
  eval_freq: ???
  save_freq: ???
  save_checkpoint: true
  log_freq: 200
  
  # Online training (RL)
  online_steps: ???
  online_rollout_n_episodes: 1
  online_rollout_batch_size: 1
  online_steps_between_rollouts: null
  online_sampling_ratio: 0.5
  
  # Data augmentation
  image_transforms:
    enable: false
    max_num_transforms: 3
    brightness:
      weight: 1
      min_max: [0.8, 1.2]
    contrast, saturation, hue, sharpness: ...

eval:
  n_episodes: 1
  batch_size: 1
  use_async_envs: true

wandb:
  enable: false
  project: lerobot
```

### Policy Configuration Example (ACT)

From `policy/act.yaml`:

```yaml
policy:
  name: act
  
  # Input/output structure
  n_obs_steps: 1
  chunk_size: 100  # Actions predicted per forward pass
  n_action_steps: 100
  
  input_shapes:
    observation.images.top: [3, 480, 640]
    observation.state: [${env.state_dim}]
  output_shapes:
    action: [${env.action_dim}]
  
  # Normalization strategy
  input_normalization_modes:
    observation.images.top: mean_std
    observation.state: mean_std
  output_normalization_modes:
    action: mean_std
  
  # Architecture
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  use_vae: true
  latent_dim: 32
  
  # Training
  dropout: 0.1
  kl_weight: 10.0
```

### Using Configurations

Command-line overrides:
```bash
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=lerobot/aloha_sim_insertion_human \
    training.offline_steps=100000 \
    training.batch_size=8
```

In Python:
```python
from hydra import compose, initialize

initialize(config_path="configs")
cfg = compose(config_name="default")
# cfg is an OmegaConf DictConfig object
```

---

## Preprocessors & Normalization

### Why Normalization Matters

Neural networks train better when inputs and outputs are normalized:
- **Stability**: Prevents exploding/vanishing gradients
- **Convergence**: Faster, more stable training
- **Transfer**: Enables using pretrained models

### The Normalization System

#### `Normalize` Module (Pre-training)

Used during training to normalize inputs before the model sees them:

```python
from lerobot.common.policies.normalize import Normalize

normalizer = Normalize(
    shapes={
        "observation.image": [3, 96, 96],
        "observation.state": [6],
    },
    modes={
        "observation.image": "mean_std",  # (x - mean) / std
        "observation.state": "min_max",   # (x - min) / (max - min) * 2 - 1
    },
    stats=dataset.stats  # Load from dataset statistics
)

# During forward pass
batch = normalizer(batch)  # Normalizes in-place
output = policy(batch)
```

#### `Unnormalize` Module (Post-prediction)

Used after model inference to convert predictions back to action space:

```python
from lerobot.common.policies.normalize import Unnormalize

unnormalizer = Unnormalize(
    shapes={"action": [7]},
    modes={"action": "mean_std"},
    stats=dataset.stats
)

# Model outputs normalized actions
normalized_actions = policy.select_action(observation)
# Convert back to robot action space
real_actions = unnormalizer({"action": normalized_actions})["action"]
env.step(real_actions)
```

### Normalization Modes

1. **mean_std**: Standard normalization
   - Formula: `(x - mean) / (std + 1e-8)`
   - Good for: Generally stable features
   - Used for: Images, states with good scale

2. **min_max**: Min-max scaling to [-1, 1]
   - Formula: `2 * (x - min) / (max - min) - 1`
   - Good for: Actions with known bounds
   - Used for: Joint angles, gripper positions

### Computing Statistics

The framework automatically computes statistics across the entire dataset:

```python
# In compute_stats.py
stats = {
    "observation.image": {
        "mean": mean_tensor,
        "std": std_tensor,
        "min": min_tensor,
        "max": max_tensor,
    },
    "observation.state": {...},
    "action": {...},
}
# Stored in dataset for reproducibility
```

### Transforms (Data Augmentation)

Applied AFTER normalization during training:

```yaml
image_transforms:
  enable: true
  max_num_transforms: 3
  brightness:
    weight: 1
    min_max: [0.8, 1.2]
  contrast:
    weight: 1
    min_max: [0.8, 1.2]
  saturation:
    weight: 1
    min_max: [0.5, 1.5]
```

Uses `RandomSubsetApply` to sample and apply transforms:
- Selects `max_num_transforms` from available transforms
- Applies them in order (or random order)
- Helps prevent overfitting

---

## Policy Implementations

### Policy Protocol

All policies follow the `Policy` protocol (structural typing, no inheritance required):

```python
class Policy(Protocol):
    name: str
    
    def reset(self):
        """Called when environment resets. Clears internal state."""
        
    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Training forward pass.
        
        Returns: dict with 'loss' key (Tensor) and other metrics
        """
        
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        """Inference: returns single action for environment step.
        
        Handles caching/buffering for temporal models.
        """
```

### Supported Policies

#### 1. ACT (Action Chunking Transformer)

**Core Idea**: Predict a "chunk" of multiple future actions at once.

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

cfg = ACTConfig(
    n_obs_steps=1,
    chunk_size=100,  # Predict 100 actions per forward pass
    n_action_steps=100,
    vision_backbone="resnet18",
    dim_model=512,
    use_vae=True,
)
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
```

**Architecture**:
- Vision encoder (ResNet18) for images
- Transformer encoder (processes observations)
- Transformer decoder (generates action predictions)
- VAE on top of decoder output (provides latent space)

**Forward Pass**:
```
Input image/state → ResNet18 backbone → Positional encoding
  → Transformer encoder (4 layers) 
  → Transformer decoder (1 layer)
  → VAE bottleneck
  → Linear output → Predicted action chunk (100 actions)
```

**Action Selection**:
- Predicts all 100 actions at once
- Uses temporal ensembling or queuing to select single actions
- Efficient but requires buffering predictions

#### 2. Diffusion Policy

**Core Idea**: Generate action sequences by iterative denoising.

```python
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig

cfg = DiffusionConfig(
    n_obs_steps=2,
    horizon=16,
    n_action_steps=8,
    vision_backbone="resnet18",
    down_dims=[512, 1024, 2048],
    num_train_timesteps=100,  # Number of noise levels
)
policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
```

**Architecture**:
- Vision encoder (ResNet18 with spatial softmax)
- History aggregation (concatenate past observations)
- U-Net denoiser (removes noise iteratively)
- Noise scheduler (DDPM, defines noise schedule)

**Training** (Forward):
- Add noise to ground truth action: `a_t = sqrt(alpha) * a_0 + sqrt(1-alpha) * eps`
- Predict noise at each level
- Loss: MSE between predicted and actual noise

**Inference** (Reverse):
- Start with random noise
- Iteratively denoise: predict noise, subtract from action
- Repeat for 10-100 steps (configurable)
- Returns final denoised action sequence

#### 3. TDMPC (Temporal Difference Model Predictive Control)

**Core Idea**: Learn a world model + MPC-style action planning.

- Learns to predict future observations
- Uses learned model for planning
- Combines model learning with RL

#### 4. VQ-BeT (Vector Quantized Behavior Embeddings Transformer)

**Core Idea**: Discretize actions into tokens, use transformer.

- Quantizes actions into discrete vocabulary
- Transformer operates on action tokens
- Faster training than continuous predictions

---

## Training Pipeline

### Offline Training (Imitation Learning)

Training on a fixed dataset of demonstrations:

```python
# From lerobot/scripts/train.py

# 1. Load dataset
dataset = make_dataset(cfg)  # Uses delta_timestamps from config

# 2. Create policy
policy = make_policy(cfg, dataset_stats=dataset.stats)

# 3. Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.training.batch_size,
    num_workers=cfg.training.num_workers,
    shuffle=True,
    pin_memory=True,
)

# 4. Training loop
optimizer = make_optimizer_and_scheduler(cfg, policy)
for step in range(cfg.training.offline_steps):
    batch = next(dataloader)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    output_dict = policy.forward(batch)
    loss = output_dict["loss"]
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    optimizer.step()
    optimizer.zero_grad()
    
    # Periodic evaluation
    if step % cfg.training.eval_freq == 0:
        eval_policy(eval_env, policy, n_episodes=50)
        
    # Checkpointing
    if step % cfg.training.save_freq == 0:
        checkpoint = {
            'step': step,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoint_{step}.pt")
```

### Online Training (Reinforcement Learning)

After offline training, optionally fine-tune with environment interaction:

```python
# Setup
online_env = make_env(cfg)
online_buffer = OnlineBuffer(
    data_spec=policy.config.input_shapes,
    buffer_capacity=cfg.training.online_buffer_capacity,
)

# Loop structure
for online_step in range(cfg.training.online_steps):
    # 1. Collect rollouts with current policy
    with torch.no_grad():
        eval_info = eval_policy(
            online_env,
            policy,
            n_episodes=cfg.training.online_rollout_n_episodes,
            return_episode_data=True,
        )
    
    # 2. Add to buffer
    online_buffer.add_data(eval_info["episodes"])
    
    # 3. Update sampler weights (mix offline + online data)
    sampler_weights = compute_sampler_weights(
        offline_dataset,
        online_dataset=online_buffer,
        online_sampling_ratio=cfg.training.online_sampling_ratio,
    )
    
    # 4. Training steps on mixed data
    for train_step in range(cfg.training.online_steps_between_rollouts):
        batch = next(combined_dataloader)  # Sample from offline+online
        loss = policy.forward(batch)["loss"]
        loss.backward()
        optimizer.step()
```

### Key Training Concepts

1. **Batch Composition**:
   - Each batch contains multiple (obs, action) pairs
   - Thanks to delta_timestamps, each item has history and future
   - Example: batch_size=64 → 64 items per batch

2. **Gradient Clipping**:
   - Prevents exploding gradients
   - `torch.nn.utils.clip_grad_norm_` limits max gradient magnitude

3. **Episode-Aware Sampling**:
   - Samples respect episode boundaries
   - Doesn't mix data across episodes
   - Prevents unrealistic transitions

4. **Learning Rate Scheduling**:
   - Warmup: gradually increase LR (stabilizes early training)
   - Decay: cosine or exponential decay (fine-tunes later)
   - Policy-specific: ACT has lower backbone LR

---

## Evaluation & Deployment

### Evaluation Process

```python
# From lerobot/scripts/eval.py

def eval_policy(env, policy, n_episodes=50):
    """
    Runs policy in environment and collects metrics.
    """
    policy.eval()
    
    all_rewards = []
    all_successes = []
    
    for episode in range(n_episodes):
        policy.reset()  # Clear internal state
        obs, info = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            # Preprocess observation (convert numpy to torch, normalize)
            obs = preprocess_observation(obs)
            obs = {k: v.to(device) for k, v in obs.items()}
            
            # Get action from policy
            with torch.inference_mode():
                action = policy.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            episode_reward += reward
            done = terminated or truncated
        
        all_rewards.append(episode_reward)
        all_successes.append(info.get("is_success", False))
    
    return {
        "avg_reward": np.mean(all_rewards),
        "success_rate": np.mean(all_successes),
    }
```

### Metrics Collected

- **sum_reward**: Total reward per episode
- **success**: Boolean, whether task succeeded
- **eval_s**: Time taken
- Videos of rollouts (optional)

### Checkpoint Management

Checkpoints are saved during training:

```
outputs/train/2024-11-12/12-34-56_pusht_diffusion/
├── checkpoints/
│   ├── 000250/
│   │   ├── pretrained_model/
│   │   │   ├── config.json          # Policy config
│   │   │   ├── config.yaml          # Full hydra config
│   │   │   ├── model.safetensors    # Model weights
│   │   │   └── README.md            # Model card
│   │   └── training_state.pth       # Optimizer/scheduler state
│   ├── 000500/
│   └── last/ (symlink to last checkpoint)
├── eval/
│   └── videos_step_000250/
├── logs/
│   └── wandb/
└── .hydra/
    └── config.yaml
```

### Inference/Deployment

Converting trained model to deployment:

1. **Load model**:
   ```python
   policy = DiffusionPolicy.from_pretrained("path/to/checkpoint")
   policy.eval()
   ```

2. **On real robot**:
   ```python
   while True:
       # Read observations from sensors
       image = camera.read()
       state = arm.read_joint_positions()
       
       # Prepare for inference
       obs = {
           "observation.image": torch.from_numpy(image).unsqueeze(0),
           "observation.state": torch.from_numpy(state).unsqueeze(0),
       }
       
       # Predict action
       with torch.inference_mode():
           action = policy.select_action(obs)
       
       # Execute on robot
       arm.set_target_positions(action.cpu().numpy()[0])
   ```

3. **Optimization options**:
   - Reduce diffusion steps (10 instead of 100) for speed
   - Use ONNX/TorchScript for C++ deployment
   - Quantize model weights

---

## Robot Support & Control

### Robot Device Architecture

```
robot_devices/
├── robots/
│   ├── koch.py              # Koch arm implementation
│   └── factory.py           # Robot factory
├── motors/
│   ├── dynamixel.py         # Dynamixel servo interface
│   └── motors_bus.py        # Motor communication
├── cameras/
│   ├── opencv.py            # USB camera interface
│   └── utils.py
└── utils.py
```

### Supported Robots

1. **Koch Arm** (Open-source, affordable)
   - Multiple Dynamixel servos (XL430, etc.)
   - 6 DOF + gripper
   - Real-world support

2. **Simulation Environments**:
   - ALOHA (via gym-aloha)
   - PushT (via gym-pusht)
   - xArm (via gym-xarm)

### Dynamixel Motor Control

```python
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

# Initialize
motor_bus = DynamixelMotorsBus(
    port="/dev/ttyUSB0",
    motor_ids=[1, 2, 3, 4, 5, 6, 7],  # 6 joints + gripper
)

# Set operating modes
motor_bus.write("Operating_Mode", OperatingMode.EXTENDED_POSITION.value)

# Control motors
motor_bus.write("Goal_Position", [0, 0, 0, 0, 0, 0, 0])  # Set target positions

# Read state
positions = motor_bus.read("Present_Position")
velocities = motor_bus.read("Present_Velocity")
```

### Calibration Process

Before use, robots must be calibrated:

```python
from lerobot.common.robot_devices.robots.koch import run_arm_calibration

# 1. Move robot to "zero position" manually
# 2. Run calibration
run_arm_calibration(arm_motor_bus, arm_name="left", arm_type="follower")

# This computes:
# - Homing offset (0-position alignment)
# - Drive mode (invert rotation if needed)
# Stores in cache for future use
```

### Multi-Arm Configuration

For bimanual systems (leader-follower):

```yaml
robot:
  name: koch_bimanual
  leader:
    right:
      port: /dev/ttyUSB0
      motor_ids: [1, 2, 3, 4, 5, 6, 100]
  follower:
    right:
      port: /dev/ttyUSB1
      motor_ids: [1, 2, 3, 4, 5, 6, 100]
```

---

## Model Inference on Robots

### End-to-End Inference Loop

```python
import torch
from lerobot.common.robot_devices.robots.koch import KochRobot
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# 1. Initialize robot
robot = KochRobot(config_path="config_koch.yaml")

# 2. Load pretrained policy
policy = DiffusionPolicy.from_pretrained("outputs/train/koch_policy/checkpoints/last/pretrained_model")
policy.eval()

# 3. Run control loop
fps = 30
dt = 1.0 / fps

policy.reset()

try:
    while True:
        # Read observations
        start_time = time.time()
        
        # Get images from cameras
        images = {}
        for cam_name in policy.config.image_keys:
            images[cam_name] = robot.cameras[cam_name].read()
        
        # Get joint states
        state = robot.arm.read_state()  # positions, velocities
        
        # Format for policy
        observation = {
            "observation.image.cam_high": torch.from_numpy(images["cam_high"]).unsqueeze(0),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
        }
        
        # Inference
        with torch.inference_mode():
            action = policy.select_action(observation)  # Shape: (1, action_dim)
        
        # Execute on robot
        target_positions = action[0].cpu().numpy()
        robot.arm.set_target_positions(target_positions)
        
        # Maintain frequency
        elapsed = time.time() - start_time
        time.sleep(max(0, dt - elapsed))
        
except KeyboardInterrupt:
    robot.arm.disable_torque()
    print("Stopped")
```

### Key Considerations

1. **Frequency Control**: Real-time systems need consistent timing
   - Use `time.sleep()` or thread schedulers
   - Monitor actual vs desired FPS

2. **Safety**:
   - Always have manual override (e.g., kill switch)
   - Set velocity/torque limits
   - Disable on errors

3. **Latency**:
   - Minimize data transfer time
   - Use GPU for inference (100-500ms vs 1-2s on CPU)
   - Cache calibration data

4. **Synchronization**:
   - Match training FPS in deployment
   - Account for camera exposure time
   - Sync multiple cameras if used

### Recording Evaluation Data

During evaluation/deployment, record data to create datasets:

```python
dataset_recorder = DatasetRecorder(
    root_dir="data/my_collected_data",
    fps=30,
)

for episode in range(num_episodes):
    dataset_recorder.start_episode()
    
    while not done:
        frame = camera.read()
        state = arm.read_state()
        action = policy.select_action(obs)
        
        dataset_recorder.record_frame(
            image=frame,
            state=state,
            action=action,
        )
    
    dataset_recorder.end_episode()

# Convert and upload
upload_dataset_to_hub(dataset_recorder, "user/my_dataset")
```

---

## Complete Training Workflow Example

### Step 1: Prepare Data

```bash
# Record demonstration data
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id user/koch_demo \
    --num-episodes 50
```

### Step 2: Upload to Hub

```bash
# Convert to LeRobot format and upload
python lerobot/scripts/push_dataset_to_hub.py \
    --raw-dir data/koch_demo_raw \
    --out-dir data \
    --repo-id user/koch_demo \
    --raw-format aloha_hdf5  # or other format
```

### Step 3: Train Model

```bash
# Train with default ACT config
python lerobot/scripts/train.py \
    policy=act \
    env=koch_real \
    dataset_repo_id=user/koch_demo \
    training.offline_steps=100000 \
    training.batch_size=8 \
    seed=1000
```

### Step 4: Evaluate

```bash
# Evaluate checkpoint
python lerobot/scripts/eval.py \
    -p outputs/train/2024-11-12/.../checkpoints/100000/pretrained_model \
    eval.n_episodes=50
```

### Step 5: Deploy on Robot

```python
# Use inference loop above
python deploy_model.py --model-path outputs/train/.../pretrained_model
```

---

## Key Design Patterns

### 1. Factory Pattern

Create objects from configs without knowing concrete types:

```python
# Policy factory
policy_cls, config_cls = get_policy_and_config_classes("diffusion")
policy = policy_cls(config)

# Environment factory
env = make_env(cfg)  # Creates appropriate gym environment

# Dataset factory
dataset = make_dataset(cfg)  # Creates appropriate dataset type
```

### 2. Protocol-Based Typing

Instead of inheritance, policies follow a protocol:
- Enables flexibility
- Type-safe without coupling
- Easy to add new policies

### 3. Configuration Composition

Configs built from parts:
```yaml
defaults:
  - env: pusht
  - policy: diffusion
```

Can override anything:
```bash
training.batch_size=32 policy.horizon=20
```

### 4. Separate Normalization

Input/output normalization decoupled from models:
- Models work with normalized data
- Normalization stored separately
- Easy to inspect/debug

### 5. Delta Timestamps

Generic mechanism for temporal data:
```python
delta_timestamps = {
    "observation": [-2, -1, 0],  # Past observations
    "action": [0, 1, ..., 16],   # Future actions
}
```

Works for any policy with minimal changes.

---

## Summary

**LeRobot Pipeline Flow**:

```
Raw Demonstrations
    ↓
Data Collection (teleoperation)
    ↓
Convert to LeRobotDataset Format
    ↓
Upload to Hugging Face Hub
    ↓
Load Dataset (with delta_timestamps)
    ↓
Compute Statistics (mean, std, min, max)
    ↓
Create Policy (with normalization buffers)
    ↓
Training Loop:
  - Sample from dataset
  - Normalize batch
  - Forward pass (compute loss)
  - Backward + optimize
  - Periodic evaluation
  - Checkpointing
    ↓
Fine-tune with RL (optional)
    ↓
Final Evaluation
    ↓
Deploy on Real Robot
    ↓
Inference Loop:
  - Read sensors
  - Normalize observations
  - Policy forward
  - Execute actions
```

---

## Further Reading

Key files in the codebase:
- `/home/jrobot/project/lerobot/lerobot/scripts/train.py` - Main training loop
- `/home/jrobot/project/lerobot/lerobot/scripts/eval.py` - Evaluation
- `/home/jrobot/project/lerobot/lerobot/common/datasets/lerobot_dataset.py` - Dataset loading
- `/home/jrobot/project/lerobot/lerobot/common/policies/factory.py` - Policy creation
- `/home/jrobot/project/lerobot/lerobot/common/policies/normalize.py` - Normalization
