# Hierarchical VLM+VLA Robot Control on XLeRobot: Iterative MVP Roadmap

Your xlerobot setup combined with LeRobot and an RTX 5090 provides an excellent foundation for implementing hierarchical robot control. The critical insight: **xlerobot is a complete $660 dual-arm mobile platform fully integrated with LeRobot**, not just a standalone arm. This means you can leverage extensive existing infrastructure to achieve working inference within days rather than weeks.

## Understanding your xlerobot hardware

**XLeRobot is a complete dual-arm mobile household robot** ($660) built entirely on the LeRobot framework. It consists of **2x SO100/SO101 6-DOF arms** (Feetech STS3215 servos), a Lekiwi mobile base with omni-wheels, an IKEA RÅSKOG cart for stability, and multiple cameras (2x wrist RGB + 1x head depth). Each arm provides **6 DOF + 1-DOF gripper** with 40cm reach and 600-1000g payload.

**Crucially for your workflow**: The SO100/SO101 arms have **native LeRobot integration** with official drivers (`so100.yaml`, `so101.yaml`). Commands like `lerobot-teleoperate`, `lerobot-record`, and `lerobot-train` work directly. No custom drivers needed. The platform is part of a thriving 10,000+ member LeRobot community with extensive documentation at xlerobot.readthedocs.io and active GitHub support.

**Control interfaces available**: Python via LeRobot library (primary), keyboard/gamepad control (Bluetooth), Meta Quest 3 VR teleoperation via XLeVR, and simulation control through ManiSkill. The hardware communicates via UART/USB-C with the STS3215 servos using 12-bit magnetic encoders.

## Critical path: MVP roadmap for fastest verification

### MVP Step 1: Verify xlerobot basic control (Start here today)

**Objective**: Confirm hardware functionality and establish baseline control in 2-4 hours

**Execute these commands**:
```bash
# Install LeRobot with xlerobot support
pip install lerobot[feetech,intelrealsense]

# Identify USB ports for your arms
ls /dev/ttyACM* /dev/ttyUSB*

# Calibrate follower arms (CRITICAL - do this first)
python -m lerobot.calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_left_arm

python -m lerobot.calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=xlerobot_right_arm

# Test basic keyboard control
python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_test
```

**Why start with hardware**: Verifying real hardware first reveals any mechanical issues, calibration needs, or workspace constraints before investing time in simulation. XLeRobot's native LeRobot support makes this trivial (unlike custom robot platforms).

**Expected outcome**: Arms respond to commands, move smoothly through workspace, grippers open/close reliably. If issues arise, check motor IDs via Feetech Software (Windows) or FT_SCServo_Debug_Qt (Ubuntu).

**Common issues**: USB permissions (add user to dialout group: `sudo usermod -a -G dialout $USER`), incorrect motor IDs (reconfigure with motor software), insufficient power supply (use provided Anker 288Wh battery or adequate power supply).

### MVP Step 2: Run pre-trained VLA inference (ACT model, fastest path)

**Objective**: Get ANY VLA model running inference on your hardware within 1-2 days

**Recommended starting model: ACT (Action Chunking Transformer)**

**Why ACT first**:
- Smallest model (80M parameters vs 3-7B for others)
- Fastest training (3 hours on RTX 5090 vs days for larger models)
- **Fully integrated with LeRobot** - zero custom code needed
- Data efficient (works with 50-100 demonstrations)
- Runs at 50Hz control frequency
- Proven on ALOHA platform (similar dual-arm setup)

**Implementation steps**:

```bash
# 1. Download existing ACT checkpoint trained on similar tasks
# (Start with pre-trained models to verify pipeline before collecting data)
from lerobot.common.policies.act import ACTPolicy

# Option A: Use pre-trained ALOHA model as baseline
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

# Option B: If you have collected data already
policy = ACTPolicy.from_pretrained("your_username/xlerobot_act_policy")

# 2. Set up observation collection from xlerobot
from lerobot.common.robot_devices.robots.so101 import SO101Robot
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

robot = SO101Robot(port="/dev/ttyACM0", config_path="so101.yaml")
camera = IntelRealSenseCamera(serial_number="your_camera_serial")

# 3. Run inference loop (50Hz)
while True:
    observation = {
        "observation.images.cam_high": camera.read(),
        "observation.state": robot.get_joint_positions(),
    }
    
    action = policy.select_action(observation)
    robot.send_action(action)
    time.sleep(0.02)  # 50Hz control
```

**Quick verification task**: Simple pick-and-place of a cube. Place a distinctive object (bright colored block) in consistent location, run inference, observe if policy attempts to reach toward it. **Success criteria**: Any coherent reaching motion toward target, even if not perfect grasping.

**If ACT doesn't work initially**: This is expected for new robot platforms. The pre-trained models are trained on different robots. This verifies your pipeline works; proceed to Step 3 to collect xlerobot-specific data.

**Alternative: SmolVLA for faster iteration**

If you want language conditioning from the start:
```bash
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

observation = {
    "observation.images.cam_high": image,
    "observation.state": joint_positions,
    "task": "pick up the red cube"  # Natural language input
}
action = policy.select_action(observation)
```

SmolVLA is lightweight (450M parameters), runs at 30Hz, and supports language instructions while being small enough for fast iteration on RTX 5090.

### MVP Step 3: Simple pick-and-place demonstration (real hardware with teleoperation)

**Objective**: Collect 50-100 high-quality demonstrations for fine-tuning in 1-2 days

**Critical decision: Teleoperation hardware selection**

Based on research findings, **recommended teleoperation approach for xlerobot**:

**Option 1: SO-ARM101 Leader arm ($350) - RECOMMENDED for LeRobot ecosystem**

**Why this is best for your use case**:
- **Zero custom code** - native LeRobot support
- Automatic calibration (`lerobot-calibrate`)
- Kinematically equivalent to follower (ensures feasible demonstrations)
- Direct recording to HuggingFace dataset format
- TTS audio feedback during data collection
- Proven in xlerobot community

**Purchase and setup**:
```bash
# 1. Buy SO-ARM101 leader kit from WowRobo (~$350)
# https://shop.wowrobo.com/products/so-arm101-leader

# 2. Assemble (1-2 hours with provided instructions)

# 3. Calibrate leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0

# 4. Start teleoperation with recording
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=xlerobot_left_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=xlerobot_leader
```

**Option 2: GELLO (~$300) - Best performance-to-cost ratio**

If you want maximum data quality and don't mind assembly:
- Proven superior in peer-reviewed studies (30-50% faster completion, higher success rates)
- Open-source designs available at github.com/wuphilipp/gello_mechanical
- 30-minute assembly after 3D printing
- Requires custom LeRobot plugin (~200 lines Python code)

**Bill of materials**: 6-7x Dynamixel XL430-W250 motors ($200-250), 3D printed parts ($20-40 filament), hardware ($20-30). See github.com/wuphilipp/gello_software for integration code.

**Option 3: SpaceMouse Compact ($150) - For initial testing only**

Good for workspace exploration and initial testing but **not recommended for production data collection**:
- Works via end-effector control (requires inverse kinematics)
- Learning curve of 5+ hours
- Less intuitive than leader-follower for manipulation
- Use `pyspacemouse` Python library for integration

**Avoid**: Game controllers for manipulation data collection (only 4-6 DOF control, too abstract), Meta Quest VR unless you need bimanual dexterous tasks (high setup complexity, 1-3 days initial setup).

**Data collection workflow**:

```bash
# With SO-ARM leader (recommended)
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyUSB0 \
  --dataset.repo_id=your_username/xlerobot_pickup_cube \
  --dataset.fps=50 \
  --num-episodes=100

# This creates LeRobot-format dataset directly on HuggingFace Hub
```

**Pick-and-place task specification**:
- **Object**: Bright red cube (10cm) for easy visual detection
- **Start position**: Fixed location on table, marked with tape
- **Goal**: Pick up cube, move to target zone 30cm away, release
- **Workspace**: 30x30cm area within arm reach (40cm max)
- **Lighting**: Consistent (important for vision models)
- **Camera angles**: Head camera (third-person) + wrist camera (egocentric)

**Quality over quantity**: 50 high-quality demonstrations (smooth motions, consistent success) beats 200 noisy demonstrations. Reset to exact start position each time. Take breaks every 10 demonstrations to maintain quality.

**Training on collected data**:

```bash
lerobot-train \
  --dataset.repo_id=your_username/xlerobot_pickup_cube \
  --policy.type=act \
  --training.num_epochs=3000 \
  --training.batch_size=16 \
  --training.lr=1e-4 \
  --output_dir=outputs/train/xlerobot_act_policy

# Training time on RTX 5090: ~2-3 hours for ACT
```

**Deployment and verification**:

```bash
lerobot-record \
  --policy.path=outputs/train/xlerobot_act_policy/checkpoint-3000 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --num-episodes=10 \
  --run-inference

# Success criteria: 70%+ task completion rate
```

### MVP Step 4: Integrate VLM for high-level planning (hierarchical architecture)

**Objective**: Enable natural language control with "pick up the red cube" instead of hard-coded tasks (2-3 days)

**Architecture pattern: Hi Robot style (Physical Intelligence, 2025)**

This is the state-of-the-art approach achieving 40% better instruction accuracy than GPT-4o while running efficiently on your RTX 5090:

```
User: "pick up the red object"
    ↓
High-Level VLM (PaliGemma-3B, 1Hz)
    → Generates: "grasp red cube at position (0.3, 0.2, 0.05)"
    ↓
Low-Level VLA (ACT or SmolVLA, 50Hz)
    → Executes: smooth approach, grasp, lift actions
    ↓
XLeRobot hardware execution
```

**Recommended VLM: PaliGemma-3B** (best balance for your RTX 5090)

**Why PaliGemma**:
- 3B parameters fits comfortably in RTX 5090 VRAM alongside VLA
- Runs at 1-5Hz (sufficient for planning updates)
- Used in Physical Intelligence's Hi Robot system
- Excellent spatial reasoning for manipulation
- Open weights, commercial use allowed

**Implementation strategy**:

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

class HierarchicalController:
    def __init__(self):
        # High-level VLM (runs on GPU 0)
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-448",
            torch_dtype=torch.bfloat16
        ).to("cuda:0")
        self.vlm_processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
        
        # Low-level VLA (your trained ACT policy)
        self.vla = ACTPolicy.from_pretrained(
            "your_username/xlerobot_act_policy"
        ).to("cuda:0")
        
        self.current_subgoal = None
        self.update_interval = 50  # Update VLM every 1 sec (50 steps at 50Hz)
        
    def run(self, robot, instruction="pick up the red cube"):
        step = 0
        
        while True:
            # Update high-level plan every 1 second
            if step % self.update_interval == 0:
                obs_image = robot.get_camera_image()
                
                # VLM generates semantic subgoal
                prompt = f"Robot task: {instruction}. What should the robot do next? Provide specific gripper position and action."
                inputs = self.vlm_processor(prompt, obs_image, return_tensors="pt").to("cuda:0")
                output = self.vlm.generate(**inputs, max_new_tokens=100)
                self.current_subgoal = self.vlm_processor.decode(output[0], skip_special_tokens=True)
                
                print(f"VLM subgoal: {self.current_subgoal}")
            
            # Execute low-level control at 50Hz
            observation = {
                "observation.images.cam_high": robot.get_camera_image(),
                "observation.state": robot.get_joint_positions(),
                "language_instruction": self.current_subgoal  # Pass VLM output to VLA
            }
            
            action = self.vla.select_action(observation)
            robot.send_action(action)
            
            step += 1
            time.sleep(0.02)  # 50Hz
```

**Action chunking pattern** (more efficient):

```python
def run_with_chunking(self, robot, instruction):
    action_buffer = []
    buffer_idx = 0
    
    while True:
        # Refill buffer when empty
        if len(action_buffer) == 0 or buffer_idx >= 50:
            # Get new subgoal from VLM
            subgoal = self.vlm_plan(robot.get_camera_image(), instruction)
            
            # Generate 50-step action chunk from VLA
            action_buffer = self.vla.generate_actions(
                observation=robot.get_observation(),
                goal=subgoal,
                horizon=50  # 1 second of actions
            )
            buffer_idx = 0
        
        # Execute from buffer
        robot.send_action(action_buffer[buffer_idx])
        buffer_idx += 1
        time.sleep(0.02)
```

**LoRA fine-tuning for robot-specific adaptation**:

Instead of full fine-tuning (expensive, causes forgetting), use **LoRA adapters** (5% trainable parameters):

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA for PaliGemma
lora_config = LoraConfig(
    r=16,  # Rank (higher = more capacity, typical 8-32)
    lora_alpha=32,  # Scaling factor (2x rank is typical)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply to frozen base model
base_vlm = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448")
vlm_with_lora = get_peft_model(base_vlm, lora_config)

# Only ~150M parameters trainable (vs 3B total)
# Train on robot-specific data: object positions, workspace constraints, etc.
```

**Alternative VLM options**:

- **Qwen-VL-2B**: Better spatial reasoning, slightly larger
- **InstructBLIP-7B**: Best for complex reasoning (requires more VRAM)
- **SmolVLM-500M**: Smallest option if you need to run both VLM+VLA on single GPU with other processes

### MVP Step 5: Collect diverse demonstration data for scaling

**Objective**: Scale beyond single task to multiple household manipulation tasks (1-2 weeks)

**Once basic pipeline works**, expand dataset systematically:

**Task suite for household robot** (priority order):
1. **Pick-and-place variations** (50 demos each):
   - Red cube → target zone
   - Blue cylinder → different target
   - Soft object (sponge) → container
   - Heavy object (can) → stable surface

2. **Bimanual coordination** (100 demos):
   - Open drawer with one arm, retrieve item with other
   - Hold container with one arm, place item with other
   - Bimanual carrying of large objects

3. **Dynamic interactions** (50 demos):
   - Opening fridge door (use mobile base)
   - Wiping table surface (requires force control)
   - Pouring from bottle (advanced dexterity)

**Data collection best practices**:
- **Consistent environment setup**: Mark robot base position, fix lighting, use consistent camera positions
- **Variation in objects**: Same task with different object colors/shapes improves generalization
- **Record failures**: Include failed attempts in dataset (helps policy learn boundaries)
- **Multiple camera angles**: Head camera (third-person), wrist cameras (egocentric) provide complementary views
- **Annotation**: Label key events (contact, grasp, release) for debugging

**Dataset format** (LeRobot automatically handles this):
```
xlerobot_manipulation_dataset/
├── episode_0000.hdf5  # Contains images, joint states, actions
├── episode_0001.hdf5
├── ...
├── metadata.json  # Task descriptions, camera info
└── stats.json  # Normalization statistics
```

**Training with multiple tasks**:

```bash
# Multi-task training with task conditioning
lerobot-train \
  --dataset.repo_id=your_username/xlerobot_multitask \
  --policy.type=act \
  --policy.use_language_conditioning=true \
  --training.num_epochs=5000 \
  --training.batch_size=32 \
  --output_dir=outputs/xlerobot_multitask_policy
```

## Simulation strategy: ManiSkill for sim2real transfer

**When to use simulation**: After Step 3 (real hardware verification) but before large-scale data collection in Step 5. Simulation enables rapid policy iteration and testing.

**ManiSkill setup for xlerobot** (15-20 minutes):

```bash
# 1. Install ManiSkill
conda create -n lerobot python=3.10
conda activate lerobot
pip install mani-skill

# 2. Download xlerobot URDF files
git clone https://github.com/Vector-Wangel/XLeRobot.git
cd XLeRobot

# 3. Manual integration (until official support)
# Copy xlerobot URDF to ManiSkill package directory
MANISKILL_PATH=$(python -c "import mani_skill; print(mani_skill.__path__[0])")
cp -r simulation/xlerobot_urdf $MANISKILL_PATH/assets/robots/

# 4. Test in ReplicaCAD environment
python -m mani_skill.examples.demo_ctrl_action_ee_keyboard \
  -e "ReplicaCAD_SceneManipulation-v1" \
  -r "xlerobot" \
  --render-mode="human" \
  --shader="default" \
  -c "pd_joint_delta_pos_dual_arm"
```

**Sim2real transfer workflow**:

```bash
# 1. Train policy in ManiSkill (GPU-parallelized, ~1 hour on RTX 5090)
python train_rl.py \
  --env="ReplicaCAD_PickCube-v1" \
  --robot="xlerobot" \
  --num-envs=4096 \
  --total-timesteps=10_000_000

# 2. Export trained policy
# (ManiSkill → LeRobot format conversion)

# 3. Deploy on real xlerobot hardware
lerobot-record \
  --policy.path=outputs/maniskill_policy \
  --robot.type=so101_follower \
  --run-inference \
  --num-episodes=20

# Success: 70%+ sim2real transfer for simple tasks
```

**Domain randomization for robust sim2real**:
- **Visual**: Randomize lighting, textures, camera pose (±5cm)
- **Dynamics**: Randomize friction (±20%), mass (±10%), motor torque (±15%)
- **Perception**: Add Gaussian noise to depth images (σ=2mm), RGB images (±10 brightness)

**Use lerobot-sim2real repository** for proven sim2real pipeline:
```bash
git clone https://github.com/StoneT2000/lerobot-sim2real.git
# Includes GPU-parallelized training, domain randomization, zero-shot transfer examples
```

**When NOT to use simulation**: For tasks requiring precise force control (wiping, pouring) or deformable objects (cloth, liquids). Real-world data is essential for these.

## Pre-trained VLA model comparison for xlerobot

Based on research, here's the decision matrix:

### For immediate deployment (this week):

**ACT** ⭐⭐⭐⭐⭐
- **Pros**: Smallest (80M params), fastest training (2-3 hrs), native LeRobot support, 50Hz control
- **Cons**: Not a foundation model, requires task-specific data
- **Use when**: You want simplest path to working system
- **Command**: `lerobot-train --policy.type=act`

**π0-FAST** ⭐⭐⭐⭐
- **Pros**: Best performance, 50Hz control, foundation model (10M+ training steps)
- **Cons**: JAX ecosystem (not PyTorch), requires `uv` package manager, more complex setup
- **Use when**: You need highest quality manipulation after initial verification
- **Setup**: Clone github.com/Physical-Intelligence/openpi, follow Libero example

### For language-conditioned control:

**SmolVLA** ⭐⭐⭐⭐⭐
- **Pros**: Lightweight (450M params), 30Hz control, native language support, LeRobot integration
- **Cons**: Newer (less tested), smaller training dataset
- **Use when**: You want language conditioning without massive compute
- **Command**: `policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`

**OpenVLA** ⭐⭐⭐
- **Pros**: Most mature VLA, 7B params, huge training dataset (970K trajectories)
- **Cons**: Slow inference (6-10Hz), requires fine-tuning for xlerobot, large VRAM footprint
- **Use when**: You need open-vocabulary object understanding
- **Consider**: OpenVLA-OFT variant (25-50x faster inference) released March 2025

### Action space adaptation for xlerobot

**Critical consideration**: VLA models expect specific action formats. For xlerobot (SO101 arms):

**Native action space**: 6 joint angles + 1 gripper per arm
- Joint 1-6: shoulder, elbow, wrist rotations (radians)
- Joint 7: gripper state (0=closed, 1=open)

**VLA expected format** (most models): End-effector delta pose
- (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)

**Conversion required**: Implement inverse kinematics (IK)

```python
def xlerobot_action_adapter(vla_action, current_joint_state):
    """Convert VLA end-effector deltas to xlerobot joint commands"""
    # VLA outputs: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
    
    # 1. Get current end-effector pose from forward kinematics
    current_ee_pose = forward_kinematics(current_joint_state)
    
    # 2. Apply delta to get target pose
    target_ee_pose = current_ee_pose + vla_action[:6]
    
    # 3. Solve IK for target joint angles
    target_joints = inverse_kinematics(target_ee_pose, current_joint_state)
    
    # 4. Return xlerobot-compatible action
    return {
        'joint_positions': target_joints,
        'gripper': vla_action[6]
    }
```

**IK solutions for SO101**:
- **PyBullet IK**: Fast, built-in solver (use xlerobot URDF)
- **MoveIt**: ROS-based, more robust but heavier
- **Custom analytic IK**: Fastest but requires deriving equations

**Normalization is critical**:

```python
# Collect statistics from demonstrations
joint_ranges = {
    'joint_1': (-180, 180),  # degrees
    'joint_2': (-90, 90),
    # ... for all 6 joints
}

# Normalize during training
normalized_action = (action - mean) / (std + 1e-6)

# Denormalize during inference
robot_action = (predicted_action * std) + mean
```

## Hierarchical VLM+VLA integration: Implementation patterns

### Asynchronous multi-rate control (recommended)

**Pattern: Fixed-rate VLM with event-driven updates**

```python
import threading
import queue

class AsynchronousHierarchicalController:
    def __init__(self):
        self.vlm = load_vlm("paligemma-3b")
        self.vla = load_vla("lerobot/act_xlerobot")
        
        # Shared state with thread-safe queue
        self.command_queue = queue.Queue(maxsize=1)
        self.current_command = "wait"
        
    def vlm_thread(self):
        """Runs at 1Hz, updates semantic commands"""
        while True:
            obs = self.get_observation()
            
            # VLM generates high-level command
            command = self.vlm.generate(
                image=obs['image'],
                instruction=obs['user_instruction'],
                history=obs['command_history']
            )
            
            # Update command queue (non-blocking)
            try:
                self.command_queue.put_nowait(command)
            except queue.Full:
                pass  # Skip if VLA hasn't consumed yet
            
            time.sleep(1.0)  # 1Hz
    
    def vla_thread(self):
        """Runs at 50Hz, executes low-level control"""
        while True:
            # Check for new commands (non-blocking)
            try:
                self.current_command = self.command_queue.get_nowait()
                print(f"New VLM command: {self.current_command}")
            except queue.Empty:
                pass  # Continue with current command
            
            # Generate action conditioned on current command
            obs = self.get_observation()
            action = self.vla.select_action(
                observation=obs,
                language_instruction=self.current_command
            )
            
            # Execute
            self.robot.send_action(action)
            time.sleep(0.02)  # 50Hz
    
    def run(self):
        # Start both threads
        vlm_thread = threading.Thread(target=self.vlm_thread, daemon=True)
        vla_thread = threading.Thread(target=self.vla_thread, daemon=True)
        
        vlm_thread.start()
        vla_thread.start()
        
        # Keep main thread alive
        vlm_thread.join()
```

### ROS2 integration for production deployment

**When to use ROS2**: If you need multi-robot coordination, advanced sensor fusion, or integration with existing ROS ecosystem.

**Complete ROS2 architecture**:

```python
# high_level_planner.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

class VLMPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner')
        
        # Publishers
        self.cmd_pub = self.create_publisher(String, '/semantic_commands', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )
        
        # Timer for periodic planning (1Hz)
        self.timer = self.create_timer(1.0, self.plan_callback)
        
        self.vlm = load_vlm("paligemma-3b")
        self.latest_image = None
        
    def image_callback(self, msg):
        self.latest_image = self.convert_ros_image(msg)
    
    def plan_callback(self):
        if self.latest_image is None:
            return
        
        command = self.vlm.generate_command(
            image=self.latest_image,
            instruction=self.get_current_task()
        )
        
        # Publish semantic command
        msg = String()
        msg.data = command
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'Published: {command}')

# low_level_controller.py
class VLAControllerNode(Node):
    def __init__(self):
        super().__init__('vla_controller')
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            String, '/semantic_commands', self.command_callback, 10
        )
        
        # Publishers (joint commands)
        self.joint_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )
        
        # Timer for control loop (50Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        
        self.vla = load_vla("lerobot/act_xlerobot")
        self.current_command = None
    
    def command_callback(self, msg):
        self.current_command = msg.data
        self.get_logger().info(f'Received command: {self.current_command}')
    
    def control_loop(self):
        if self.current_command is None:
            return
        
        obs = self.get_robot_observation()
        action = self.vla.select_action(obs, self.current_command)
        
        # Publish joint commands
        joint_msg = self.create_joint_state_msg(action)
        self.joint_pub.publish(joint_msg)

# launch.py
def main():
    rclpy.init()
    
    vlm_node = VLMPlannerNode()
    vla_node = VLAControllerNode()
    
    # Use MultiThreadedExecutor for concurrent execution
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(vlm_node)
    executor.add_node(vla_node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        vlm_node.destroy_node()
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Launch with ros2_control**:

```yaml
# xlerobot_controller.yaml
controller_manager:
  ros__parameters:
    update_rate: 50  # Match VLA control frequency
    
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    
    vla_controller:
      type: joint_trajectory_controller/JointTrajectoryController

vla_controller:
  ros__parameters:
    joints:
      - joint_1
      - joint_2
      - joint_3
      - joint_4
      - joint_5
      - joint_6
    
    command_interfaces:
      - position
    
    state_interfaces:
      - position
      - velocity
```

### LoRA adaptation strategy for preserving world knowledge

**Key principle**: Fine-tune only 5% of parameters to maintain generalization while adapting to robot specifics.

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA for VLM
lora_config = LoraConfig(
    r=16,  # Rank (controls adapter capacity)
    lora_alpha=32,  # Scaling factor (typically 2x rank)
    target_modules=["q_proj", "v_proj"],  # Apply to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply to frozen base model
base_vlm = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448")
vlm_with_lora = get_peft_model(base_vlm, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in vlm_with_lora.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in vlm_with_lora.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
# Output: Trainable: 150M / 3B (5.0%)

# Train on robot-specific data
trainer = VLMTrainer(
    model=vlm_with_lora,
    train_dataset=xlerobot_instruction_dataset,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
    )
)
trainer.train()

# Save only adapter weights (small checkpoint)
vlm_with_lora.save_pretrained("xlerobot_lora_adapter")  # ~300MB vs 6GB full model
```

**Multi-task adapter strategy**:

```python
# Train separate adapters for different task families
adapters = {
    "manipulation": train_lora_adapter(manipulation_data),
    "navigation": train_lora_adapter(navigation_data),
    "bimanual": train_lora_adapter(bimanual_data)
}

# Runtime adapter switching
def adaptive_controller(task_type):
    base_model = load_base_vlm()
    
    if "pick" in task_type or "place" in task_type:
        model = load_adapter(base_model, adapters["manipulation"])
    elif "move to" in task_type:
        model = load_adapter(base_model, adapters["navigation"])
    else:
        model = base_model  # Use base model for novel tasks
    
    return model
```

## Common pitfalls and troubleshooting for xlerobot

### Hardware-specific issues

**Motor calibration drift**:
- **Symptom**: Arms don't return to exact same position after power cycle
- **Solution**: Run `lerobot-calibrate` before each session, save calibration offsets
- **Prevention**: Use homing routine on startup

**USB communication errors**:
- **Symptom**: "Cannot open port /dev/ttyACM0" or intermittent disconnections
- **Solution**: Add user to dialout group (`sudo usermod -a -G dialout $USER`), check USB cable quality
- **Prevention**: Use powered USB hub, avoid long USB cables (\>2m)

**Insufficient torque/payload**:
- **Symptom**: Arms shake or fail to lift objects \>500g
- **Solution**: Reduce payload, upgrade to 12V motors (Pro Kit), optimize gripper design
- **Prevention**: Test with light objects first (100-200g)

### Action space normalization issues

**Symptom**: Policy outputs actions outside valid range, causes jerky motion or errors

**Root cause**: Training data normalization doesn't match inference normalization

**Solution**:
```python
# Compute and save statistics during data collection
def compute_action_stats(dataset):
    all_actions = np.concatenate([episode['actions'] for episode in dataset])
    stats = {
        'mean': np.mean(all_actions, axis=0),
        'std': np.std(all_actions, axis=0),
        'min': np.min(all_actions, axis=0),
        'max': np.max(all_actions, axis=0)
    }
    np.save('action_stats.npy', stats)
    return stats

# Use exact same stats for normalization and denormalization
stats = np.load('action_stats.npy', allow_pickle=True).item()

# During training
normalized_action = (action - stats['mean']) / (stats['std'] + 1e-8)

# During inference
robot_action = (model_output * stats['std']) + stats['mean']

# Safety clipping
robot_action = np.clip(robot_action, stats['min'], stats['max'])
```

### Coordinate frame mismatches

**Symptom**: Policy reaches in wrong direction (e.g., approaches from below instead of above)

**Root cause**: Camera frame vs robot base frame vs world frame confusion

**Solution**: Define clear frame conventions
```python
# Example: Camera is mounted 50cm above table, pointing down
# Robot base is on table surface

def camera_to_robot_frame(point_camera):
    """Transform point from camera frame to robot base frame"""
    # Camera frame: X right, Y down, Z forward
    # Robot frame: X forward, Y left, Z up
    
    T_camera_to_robot = np.array([
        [0, 0, 1, 0.0],    # Robot X = Camera Z
        [-1, 0, 0, 0.0],   # Robot Y = -Camera X
        [0, -1, 0, 0.5],   # Robot Z = -Camera Y + offset
        [0, 0, 0, 1]
    ])
    
    point_robot = T_camera_to_robot @ point_camera
    return point_robot[:3]
```

### VLA inference latency

**Symptom**: Policy runs slowly (\<10Hz), causing delayed reactions

**Solutions**:
1. **Use smaller model**: SmolVLA (450M) instead of OpenVLA (7B)
2. **Quantization**: INT8 quantization with bitsandbytes
   ```python
   from transformers import BitsAndBytesConfig
   
   quantization_config = BitsAndBytesConfig(
       load_in_8bit=True,
       llm_int8_threshold=6.0
   )
   
   model = AutoModelForVision2Seq.from_pretrained(
       "openvla/openvla-7b",
       quantization_config=quantization_config
   )
   # 2-3x speedup with minimal accuracy loss
   ```
3. **Action chunking**: Generate 50 actions per forward pass (amortize cost)
4. **Compile model**: Use `torch.compile()` for 20-30% speedup
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

### Sim2real transfer failure

**Symptom**: Policy works in simulation but fails on real hardware

**Root causes and solutions**:
1. **Visual domain gap**: 
   - Add Gaussian noise to sim images during training
   - Use domain randomization (lighting, textures)
   - Collect small real-world fine-tuning dataset (10-20 episodes)

2. **Dynamics mismatch**:
   - Tune simulation friction, damping to match real hardware
   - Use system identification to estimate real parameters
   - Record real trajectory, replay in sim, adjust params until match

3. **Sensor noise**:
   - Simulation sensors too perfect
   - Add realistic noise models based on real sensor specs
   - For RealSense D435: depth noise σ=2mm, RGB SNR~40dB

4. **Action delay**:
   - Real robot has 10-50ms actuation delay
   - Add action delay in simulation
   - Use action history as input to policy

## Timeline and success metrics

### Week-by-week breakdown

**Week 1: Hardware verification and basic control**
- **Day 1-2**: Verify xlerobot hardware, calibrate arms, test basic control
- **Day 3-4**: Set up LeRobot environment, run pre-trained ACT model (may not work yet)
- **Day 5**: Acquire teleoperation hardware (order SO-ARM leader or start GELLO build)
- **Success metric**: Arms respond to commands, move smoothly, cameras capture images

**Week 2: First working VLA inference**
- **Day 1-3**: Collect 50 pick-and-place demonstrations with teleoperation
- **Day 4-5**: Train ACT policy on demonstrations (2-3 hours training time)
- **Day 6-7**: Deploy policy, iterate on data quality if needed
- **Success metric**: 50%+ success rate on simple pick-and-place task

**Week 3: Hierarchical VLM integration**
- **Day 1-2**: Implement PaliGemma-3B high-level planner
- **Day 3-4**: Integrate with ACT low-level controller (asynchronous pattern)
- **Day 5-7**: Test natural language instructions, debug coordination
- **Success metric**: Robot responds correctly to 3+ different language instructions

**Week 4: Scaling and robustness**
- **Day 1-3**: Collect 100+ demonstrations across multiple tasks
- **Day 4-5**: Train multi-task policy with language conditioning
- **Day 6-7**: Set up ManiSkill simulation for rapid iteration
- **Success metric**: 70%+ success on 5 different tasks

**Weeks 5-8: Advanced capabilities** (optional)
- Implement LoRA fine-tuning for VLM
- Add mobile base navigation
- Bimanual coordination tasks
- Deploy ROS2 architecture for production

### Critical success metrics

**MVP Step 2 (VLA inference)**:
- ✅ Policy loads without errors
- ✅ Inference runs at \>20Hz
- ✅ Actions are within valid joint limits
- ✅ **ANY coherent motion toward target** (even if not successful grasp)

**MVP Step 3 (Pick-and-place)**:
- ✅ 70%+ success rate on fixed-position cube pickup
- ✅ Smooth trajectories (no jerky motion)
- ✅ Consistent grasp quality
- ✅ \<5% dataset contamination (failed demos removed)

**MVP Step 4 (Hierarchical control)**:
- ✅ VLM updates at 1-5Hz without crashing
- ✅ VLA executes at 50Hz consistently
- ✅ Language instructions correctly understood (80%+ accuracy)
- ✅ No race conditions or deadlocks in multi-threaded execution

**MVP Step 5 (Multi-task)**:
- ✅ Policy generalizes to 3+ different tasks
- ✅ 60%+ success on novel object instances (different colored cubes)
- ✅ Graceful failure handling (doesn't damage hardware)

## Key resources and repositories

**Essential repositories** (bookmark these):
- **LeRobot**: github.com/huggingface/lerobot - Core framework, all VLA models
- **XLeRobot**: github.com/Vector-Wangel/XLeRobot - Hardware docs, URDF files, examples
- **GELLO**: github.com/wuphilipp/gello_mechanical - Teleoperation hardware designs
- **Physical Intelligence OpenPI**: github.com/Physical-Intelligence/openpi - π0 models
- **OpenVLA**: github.com/openvla/openvla - VLA with largest training dataset
- **lerobot-sim2real**: github.com/StoneT2000/lerobot-sim2real - Simulation training

**Documentation**:
- XLeRobot docs: xlerobot.readthedocs.io
- LeRobot docs: huggingface.co/docs/lerobot
- SO101 setup: huggingface.co/docs/lerobot/so101
- ManiSkill: maniskill.readthedocs.io

**Pre-trained models** (HuggingFace Hub):
- lerobot/pi0 - π0 base model
- lerobot/smolvla_base - Lightweight VLA
- lerobot/act_aloha_sim_transfer_cube_human - ACT baseline
- google/paligemma-3b-mix-448 - VLM for high-level planning

**Community**:
- LeRobot Discord: 10,000+ members, active support
- XLeRobot GitHub Issues: Direct support from creator
- HuggingFace Forums: LeRobot section

**Papers** (read these for depth):
- Hi Robot (Physical Intelligence, 2025): arxiv.org/abs/2502.19417
- HiRT (CoRL 2024): arxiv.org/abs/2410.05273
- OpenVLA (2024): arxiv.org/abs/2406.09246
- ACT (CoRL 2022): Original action chunking paper

## Starting today: Your immediate action plan

**This afternoon** (2-3 hours):
1. Connect xlerobot arms to computer, identify USB ports
2. Install LeRobot: `pip install lerobot[feetech,intelrealsense]`
3. Run calibration on both arms
4. Test keyboard control to verify hardware works
5. Record 5 test episodes with keyboard control (learning the data format)

**This week** (before weekend):
1. Order SO-ARM101 leader kit ($350) from shop.wowrobo.com - **critical path item**
2. While waiting for delivery, study ACT architecture and LeRobot data format
3. Set up camera calibration (intrinsics, extrinsics)
4. Define your first task precisely (e.g., "pick red cube at (30cm, 0cm, 0cm), place at (30cm, 20cm, 0cm)")
5. Load and test ACT pre-trained model to verify inference pipeline

**Next week** (when teleoperation arrives):
1. Assemble and calibrate SO-ARM leader (2-3 hours)
2. Practice teleoperation, aim for smooth consistent motions
3. Collect 50 high-quality demonstrations over 2-3 days
4. Train overnight (3 hours on RTX 5090)
5. Deploy and evaluate - iterate on data collection if needed

**Following weeks**:
1. Add PaliGemma VLM for language conditioning
2. Implement asynchronous hierarchical control
3. Expand to multiple tasks
4. Set up ManiSkill for sim2real

**Remember**: The goal is to see **SOMETHING working quickly** - even imperfect grasping is a huge milestone. Perfect manipulation comes with iteration. Your RTX 5090 + LeRobot + XLeRobot stack is an excellent foundation. Start simple, verify each step, then scale systematically.

The most important decision you'll make this week: **Order teleoperation hardware TODAY**. This is your critical path blocker. SO-ARM101 leader ($350) is the lowest-friction option for your LeRobot setup. Everything else can proceed in parallel, but quality demonstration data requires good teleoperation.