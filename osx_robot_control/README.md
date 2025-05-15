# Introduction

This package contains:

1) (in `src`) Python classes to control the complete robot system (robots, tools, vision)

# "Controller" classes

- `common.py` extends `core.py` and offers a class to control the entire robot system. This class (`Common`) owns other classes that control different subsystems:
    - The robot arms (via `robot_base.py`, `ur_robot.py`, `dual_arm.py`, `robotiq_gripper.py` and `ur_force_control.py`)
    - The vision system (via `vision_client.py`)

# Movement Sequences

We implement command sequences, in which the next motion is either planned while the previous motion is being executed, or the joint trajectories can be saved/loaded to/from a file, to be executed without requiring any planning time.

Also see [pilz_robot_programming](https://github.com/PilzDE/pilz_industrial_motion/tree/melodic-devel/pilz_robot_programming) for an alternative Python implementation, which is probably cleaner, and from which we took inspiration.

# Gym environments
## Checklist

### FZI Cartesian compliance controller
- [ ] check the force-torque topic used by the controller:
    - [ ] It is raw but the default payload of the gripper is already offset. 
        - In the **real robot**, if the pay load is properly specify in the installations option, then it will be fine. Make sure the cables are not too tight and pushing/pulling the cameras or the FT reading will be very different.
        - In **gazebo**, it should be computed and offset using the payload_estimation and gravity_compensation launch files on `osx_scene_description`. Make sure that the initial state of the sensor is correct, in the default position the reading should be around [0,0,-20], if it is too big and changing widely without any robot motion, re-launch the simulation (FIXME bug).

- [ ] 
