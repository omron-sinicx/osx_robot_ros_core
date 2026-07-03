# UR5e ROS Package

ROS integration for the UR5e arm: bringup, teleop data collection (LeRobot), replay, and COMET policy evaluation.

# Workspaces
- catkin_ws
- underlay_ws

# Important packages
- catkin_ws/osx_ur5e
    - connect to the robot `roslaunch osx_ur5e connect_real_robot.launch`
    - connect to cameras `roslaunch osx_ur5e camera_bringup.launch`
- catkin_ws/osx_gello
    - connect to the robot `roslaunch osx_gello dynamixel_controllers.launch use_rviz:=0 enable_gravity_compensation:=1`

# Data Collection
Command:
`rosrun osx_ur5e data_collection.py --config config/data_collection.yaml --task "Wipe the table" --num-episodes 5 --resume`

## LeRobot pipeline

All three scripts use [Hydra](https://hydra.cc/). **Data collection** and **replay** load configs from `config/hydra/`. **Policy evaluation** loads a COMET eval config from `dependencies/comet/configs/` (must still define `dataset` and `controller` for `FDCCEnv`).

Prerequisites: robot in Remote mode, `connect_real_robot.launch` (+ cameras for collection).

### `data_collection.py`

Teleoperate with Gello and save a LeRobot dataset.

```bash
rosrun osx_ur5e data_collection.py
rosrun osx_ur5e data_collection.py --config-name=my_task dataset.num_episodes=10
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
rosrun osx_ur5e replay_episode.py
rosrun osx_ur5e replay_episode.py dataset.episode_idx=3
```

Uses `dataset.replay` (e.g. `raw_actions`) and the matching action group (`dataset.raw_actions`). Press `Enter` to start after reset.

### `evaluate_policy.py`

Run a trained COMET diffusion policy on the robot via `FDCCEnv`.

```bash
rosrun osx_ur5e evaluate_policy.py
rosrun osx_ur5e evaluate_policy.py eval.num_rollouts=5 eval.max_timesteps=500
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
   - `dataset.cameras` — names must match live camera topics
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
- A UR5e robot with Polyscope interface
- ROS environment properly configured
- Docker container setup (if using the provided environment)

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
rosrun ur_control joint_position_keyboard.py
```
- Press `P` to print the current joint configuration

## Environment Setup

### Using the Provided Docker Environment

The easiest way to set up the environment is using the provided Terminator terminal setup:

1. **Launch the Docker Environment**
   ```bash
   cd ~/scu-hand-env
   ./LAUNCH-TERMINATOR-TERMINAL.sh
   ```

2. **Start ROS Core**
   - In the "roscore" terminal, press `r`

3. **Connect to UR5e Robot**
   - In the "bring-up robots" terminal, press `r`

4. **Connect to SCU HAND Dynamixel Motors**
   - In the "dynamixel_service" terminal, press `r`

## Usage

### Manual Connection (Alternative to the LAUNCH-TERMINATOR-TERMINAL Setup)

If you're not using the provided LAUNCH-TERMINATOR-TERMINAL environment, you can connect manually on different terminals:

**Connect to UR5e Robot:**
```bash
rosrun osx_ur5e roslaunch osx_ur5e connect_real_robot.launch
```

**Connect to camera:**
```bash
roslaunch osx_ur5e camera_bringup.launch
```

**Launch MoveIt**
```bash
roslaunch osx_moveit_ur5e osx_moveit_planning_execution.launch
```

## Useful Scripts

### Testing and Control

**Keyboard Teleoperation for Robot:**
```bash
rosrun ur_control joint_position_keyboard.py
```
Press SPACE to see the key mappings

### Advanced Usage

For advanced robot control using the `arm` interface, see the example script:
```bash
./underlay_ws/src/ur_python_utilities/ur_control/scripts/controller_examples.py
```

## Troubleshooting

- **Connection Issues**: Ensure the robot is in "Remote" mode before attempting to connect
- **Joint Position Errors**: Verify the robot is not in a collision state

## Support

For issues or questions, please refer to the ROS package documentation or contact the development team.
