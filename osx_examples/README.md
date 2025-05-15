# Basic examples

See the following scripts with basic examples for motion control of the robots:

## MoveIt

`scripts/moveit_examples.py` for basic functions to command the robots using the motion planning framework MoveIt.

It can be run either using only the moveit demo or a combination of Gazebo+Moveit:

1) Only MoveIt demo:
    ```shell
    roslaunch osx_moveit_config demo.launch
    ```
2) Using Gazebo and MoveIt. It requires two terminals:
    ```shell
    roslaunch osx_gazebo osx_gazebo.launch
    ```
    ```shell
    roslaunch osx_moveit_config osx_moveit_planning_execution.launch sim:=true
    ```

Then execute the script and follow the instructions:
```shell
rosrun osx_examples moveit_examples
```

## UR Python Utilities
`scripts/ur_control_examples.py` for more advance functions that enables direct control of the robots without motion planning (Be very careful!)

Needs to use Gazebo
```shell
roslaunch osx_gazebo osx_gazebo.launch
```

Then execute the script and follow the instructions
```shell
rosrun osx_examples ur_control_examples
```
