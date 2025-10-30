# UR5e ROS Package

This package provides ROS integration for controlling the UR5e robot arm.

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
