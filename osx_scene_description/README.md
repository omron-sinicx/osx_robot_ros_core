# Introduction

This package contains the description of our robot system and scenes.

# Payload Estimation

For contact-rich tasks, it is important to accurately estimate the contact force with the external objects. To that end, it is important to offset from the force sensor the forces due to gravity of the payload (gripper, tools, etc.). While for the UR robots the payload can be estimated and set from the teach pendant, it is not very accurate. To obtain a better estimation we use the ROS package `force_torque_tools`.
As the package is use very rarely, it needs to be manually installed from source for Noetic. 


1. Configure a number of poses for the robot, the more the better and non coplanar. Example file `catkin_ws/src/osx_scene_description/config/b_bot_ft_calib_config.yaml`
2. In the teach-pendant set the payload mass and COM as zeros on the `Installation > Payload` configuration option.
3. Execute the calibration sequence: `roslaunch osx_scene_description payload_estimation.launch robot:=b_bot`
4. Apply offset using the teach pendant by setting the estimated mass and COM on the `Installation > Payload` configuration option. Alternatively you could use the `gravity_compensation` node from the `force_torque_tools` just be careful of not mixing both of them. 
5. A simple way to test the payload estimation is to execute the same calibration sequence with the gravity compensation and observe the compensated force signal, it should be as close as possible to zero. Note: best results so far is +-2N, given the 3.5N precision/accuracy of the sensor, it is probably as good as it may be.

# Known issues

Increasing the size of the boxes too much causes them to be unstable in the Gazebo simulation.
