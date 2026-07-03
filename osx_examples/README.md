# osx_examples

Interactive examples for motion planning (MoveIt 2) and direct arm control (`ur_control`).
These replace the ROS 1 scripts that depended on `osx_robot_control` / Gazebo Classic.

Dependencies live in [ur_python_utilities](https://github.com/omron-sinicx/ur_python_utilities)
(`ur_control`, `ur_gripper_gz`, `ur_gripper_gz_moveit_config`).

## Prerequisites

- ROS 2 Jazzy workspace sourced (`underlay_ws` then `catkin_ws`)
- `ros-jazzy-moveit-py` installed (Python MoveIt 2 API)
- A running gz-sim bringup with ros2_control controllers

---

## MoveIt 2 examples

`ros2 run osx_examples moveit_examples` — step-by-step motions via **MoveItPy**
(`ur_manipulator` planning group from `ur_gripper_gz_moveit_config`):

1. Joint-space goal
2. Cartesian pose goals in `base_link`
3. Relative motions (base frame, tool0 frame, world/base offset)
4. Named SRDF pose `home`

### Bringup (two terminals)

**Terminal 1 — gz-sim + ros2_control**

UR5e + Robotiq 2F-85 (default for `moveit_examples`):

```bash
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py
```

UR3e + Hand-E:

```bash
ros2 launch ur_gripper_gz ur_gz_control.launch.py
```

**Terminal 2 — MoveIt move_group (+ optional RViz)**

Match `ur_type` and `gripper` to the sim you launched:

```bash
ros2 launch ur_gripper_gz_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e gripper:=robotiq_2f85 launch_rviz:=true
```

Hand-E:

```bash
ros2 launch ur_gripper_gz_moveit_config ur_moveit.launch.py \
  ur_type:=ur3e gripper:=hande launch_rviz:=true
```

**Terminal 3 — run the example**

```bash
ros2 run osx_examples moveit_examples
# Hand-E bringup:
ros2 run osx_examples moveit_examples --ur-type ur3e --gripper hande
```

Press Enter at each prompt. Planning and execution go through MoveIt (not raw `ur_control`).

> **Note:** The old ROS 1 examples used `osx_moveit_config`, dual-arm namespaces (`a_bot`), and
> a `cutting_board` frame. The ROS 2 port targets the single-arm `ur_gripper_gz` cell with
> goals in `base_link`.

---

## Direct ur_control examples

`ros2 run osx_examples ur_control_examples` — **bypasses MoveIt** and commands the arm
directly through `ur_control.arm.Arm` (joint trajectory + IK Cartesian targets). Use with care.

### Bringup (one terminal)

Any gz bringup that exposes `scaled_joint_trajectory_controller` and `/joint_states`:

```bash
# Full cell (arm + gripper + FT + compliance controller):
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py

# Arm-only minimal sim:
ros2 launch osx_ur5e gz_bringup.launch.py
```

### Run

```bash
ros2 run osx_examples ur_control_examples
ros2 run osx_examples ur_control_examples --gripper auto
ros2 run osx_examples ur_control_examples --real-robot --no-use-gazebo-sim
```

Gripper `--gripper auto` reads `/active_gripper` published by the bringup launch.

---

## Migration notes (ROS 1 → 2)

| ROS 1 | ROS 2 |
|---|---|
| `rosrun osx_examples moveit_examples` | `ros2 run osx_examples moveit_examples` |
| `osx_robot_control.OSXCore` | `moveit_py.MoveItPy` + `ur_gripper_gz_moveit_config` |
| `moveit_commander` | `moveit_py` planning components |
| `rospy` | `rclpy` + background executor |
| `osx_gazebo` + `osx_moveit_config` | `ur_gripper_gz` + `ur_gripper_gz_moveit_config` |
| `TRAC_IK` | `IKSolverType.EAIK` (TRAC-IK unavailable on Jazzy) |

See also [ur_python_utilities README](https://github.com/omron-sinicx/ur_python_utilities/blob/jazzy/README.md)
for keyboard teleop and compliance-control examples:

```bash
ros2 run ur_control_examples joint_position_keyboard
ros2 run ur_control_examples controller_examples --move
```
