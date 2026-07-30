# UR5e ROS 2 Package

ROS 2 integration for the UR5e arm: bringup, teleop data collection (LeRobot), replay, and COMET policy evaluation.

# Workspace

A single colcon workspace at the repo root: `ws/`. Activate it with `pixi shell` or from inside the
Docker container — see the repo README for both paths.

# Launch files

| Command | What it does |
|---------|--------------|
| `ros2 launch osx_ur5e connect_real_robot.launch.py` | Real UR5e via `ur_robot_driver` + this cell's calibration, filtered wrench on `/wrench/filtered`, FDCC loaded inactive |
| `ros2 launch osx_ur5e camera_bringup.launch.py` | The two RealSense cameras on `/front_camera/...` and `/wrist_camera/...` |
| `ros2 launch osx_ur5e gz_bringup.launch.py` | Gazebo (gz Harmonic) simulation with the same controllers, wrench topics and FDCC |
| `ros2 launch ur_description view_ur.launch.py ur_type:=ur5e` | Just view the arm model in RViz |

Gello teleoperation lives in a separate `osx_gello` package that is not part of this workspace.

# Data Collection
Command:
`ros2 run osx_ur5e data_collection dataset.task="Wipe the table" dataset.num_episodes=5`

## LeRobot pipeline

All three scripts use [Hydra](https://hydra.cc/). **Data collection** and **replay** load configs from `config/hydra/`. **Policy evaluation** loads a COMET eval config from `dependencies/comet/configs/` (must still define `dataset` and `controller` for `FDCCEnv`).

Prerequisites: robot in Remote mode, `connect_real_robot.launch.py` (+ `camera_bringup.launch.py`
for collection). Against `gz_bringup.launch.py` instead, add `+use_gazebo_sim=true` — it is a node
parameter read from the Hydra config, and it defaults to `false` (i.e. real robot).

### `data_collection.py`

Teleoperate with Gello and save a LeRobot dataset.

```bash
ros2 run osx_ur5e data_collection
ros2 run osx_ur5e data_collection --config-name=my_task dataset.num_episodes=10
```

| Key | Meaning |
|-----|---------|
| `Enter` | End episode early (save) |
| `r` | Discard episode, re-record |
| `q` | Stop and finalize dataset |

Writes to `{dataset.dir}/{repo_id}/`. Saves resolved Hydra config under `meta/hydra_config.yaml`.

### `replay_episode.py`

Replay one recorded episode on the real robot via `FDCCEnv` (open-loop).

```bash
ros2 run osx_ur5e replay_episode
ros2 run osx_ur5e replay_episode dataset.episode_idx=3
```

Uses `dataset.replay` (e.g. `raw_actions`) and the matching action group (`dataset.raw_actions`). Press `Enter` to start after reset.

### `evaluate_policy.py`

Run a trained COMET diffusion policy on the robot via `FDCCEnv`.

```bash
ros2 run osx_ur5e evaluate_policy
ros2 run osx_ur5e evaluate_policy eval.num_rollouts=5 eval.max_timesteps=500
```

Checkpoint path comes from the COMET config (`eval.base.load_ckpt`). Logs and plots go to Hydra’s output directory.

### Config layout (`config/hydra/`)

| File | Role |
|------|------|
| `<task>.yaml` | Task/dataset: `repo_id`, `dir`, `fps`, cameras, states/actions, replay keys |
| `controller/ur5e.yaml` | Robot: FDCC gains, safety limits, `init_qpos` |

Composed config shape (what the code reads):

- `cfg.dataset.*` — dataset path, features, replay settings  
- `cfg.controller.*` — arm control and `controller.safety_parameters`

Example default: `test_task.yaml` + `defaults: [controller: ur5e, _self_]`.

### New task YAML

1. Copy `config/hydra/test_task.yaml` → `config/hydra/my_task.yaml`.
2. Edit at least:
   - `dataset.repo_id` — list with one ID, e.g. `[my_task]`
   - `dataset.task` — language instruction stored in frames
   - `dataset.dir` / `dataset.root` — where data is stored
   - `dataset.num_episodes`, `episode_time_s`, `fps`
   - `dataset.cameras` — names must match the `camera_name` of a live camera, since
     `ImageRecorder` subscribes to `/<name>/color/image_raw` (see `camera_bringup.launch.py`)
3. Adjust `states` / `actions` only if you change what you log.
4. For replay, set `dataset.replay` and the matching block (e.g. `raw_actions`).
5. Tune `config/hydra/controller/ur5e.yaml` for a new cell (`init_qpos`, workspace/safety) or add `controller/my_robot.yaml` and set `defaults: [controller: my_robot, _self_]`.
6. Run with `--config-name=my_task`.

CLI overrides: `dataset.fps=30`, `controller.stiffness=1000`, etc.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Robot Setup](#robot-setup)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Useful Scripts](#useful-scripts)

## Prerequisites

Before using this package, ensure you have:
- A UR5e robot with Polyscope interface, running the External Control URCap
- ROS 2 Jazzy and this workspace built and sourced (`pixi shell`, or the Docker container)
- This cell's calibration in `config/ur5e_calibration.yaml` (extract it with `ur_calibration`)

## Robot Setup

### 1. Power On Sequence

**Step 1: Power on the Teach Pendant (Polyscope)**
- Press the "Power button" on the tablet

**Step 2: Power on the Robot Controller**
- Tap the red "power off" indicator in the bottom left corner
- Press "Start" twice to activate the robot's motors
- Close the popup window

**Step 3: Enable Remote Mode**
- Change from "Local" mode to "Remote" mode in the upper right section

### 2. Manual Robot Positioning

**Step 1: Switch to Local Mode**
- Change from "Remote" mode to "Local" mode in the upper right section

**Step 2: Stop Current Program**
- Press the `Pause` or `Stop` button in the bottom bar

**Step 3: Manual Movement**
- Hold the button behind the tablet while moving the robot arm to desired position

**Step 4: Return to Remote Mode**
- Change from "Local" mode back to "Remote" mode

**Step 5: Get Joint Positions**
```bash
ros2 run ur_control_examples joint_position_keyboard
```
- Press `P` to print the current joint configuration

## Environment Setup

See the repo README for the two supported setups: a native `pixi` install (`pixi shell`) or the
Docker container. ROS 2 needs no roscore.

## Usage

Each of these goes in its own terminal, with the workspace sourced.

**Connect to the UR5e:**
```bash
ros2 launch osx_ur5e connect_real_robot.launch.py robot_ip:=10.0.2.15
```
Add `use_mock_hardware:=true` to bring the whole stack up without a robot on the network.

**Connect to the cameras:**
```bash
ros2 launch osx_ur5e camera_bringup.launch.py
```
Color only by default; `enable_depth:=true` also streams depth and color-aligned depth.

**Simulate instead of using hardware:**
```bash
ros2 launch osx_ur5e gz_bringup.launch.py           # gui:=false for a headless server
```

**MoveIt**
```bash
ros2 launch osx_examples moveit_examples.launch.py
```

## Useful Scripts

### Testing and Control

**Keyboard Teleoperation for Robot:**
```bash
ros2 run ur_control_examples joint_position_keyboard
```
Press SPACE to see the key mappings

### Advanced Usage

For advanced robot control using the `arm` interface, see the example scripts:
```bash
ros2 run ur_control_examples controller_examples
ros2 run osx_examples ur_control_examples
```

## Troubleshooting

- **Connection Issues**: Ensure the robot is in "Remote" mode before attempting to connect
- **Joint Position Errors**: Verify the robot is not in a collision state

## Support

For issues or questions, please refer to the ROS package documentation or contact the development team.
