#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2021, OMRON SINIC X
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of OMRON SINIC X nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Cristian C. Beltran-Hernandez

from osx_robot_control import helpers
import rospy

import numpy as np

from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import traj_utils, conversions
from ur_control.constants import ExecutionResult, GripperType
from osx_robot_control.utils import transform_pose
from osx_robot_control.helpers import get_direction_index, get_orthogonal_plane, get_random_valid_direction


class URForceController(CompliantController):
    """ 
    A controller for Universal Robots (UR) that implements force control and compliant motion.

    This class extends the CompliantController to provide specialized force control
    capabilities for UR robots, including linear pushing, circular/spiral trajectories,
    and insertion operations with force feedback.

    The controller supports both simulation (Gazebo) and real robot environments,
    with appropriate configuration for each.
    """

    def __init__(self, robot_name, listener, tcp_link='gripper_tip_link', **kwargs):
        """
        Initialize the UR Force Controller.

        Args:
            robot_name (str): Name of the robot (e.g., 'a_bot', 'b_bot')
            listener (tf.TransformListener): TF listener for coordinate transformations
            tcp_link (str, optional): Tool center point link name. Defaults to 'gripper_tip_link'
            **kwargs: Additional arguments passed to the parent CompliantController
        """
        self.listener = listener
        self.default_tcp_link = robot_name + '_' + tcp_link
        self.robot_name = robot_name

        use_gazebo_sim = rospy.get_param("use_gazebo_sim", False)
        ft_topic = 'wrench/ft_compensated' if use_gazebo_sim else 'wrench'
        use_real_robot = rospy.get_param("use_real_robot", False)
        gripper_type = GripperType.ROBOTIQ if use_real_robot else GripperType.GENERIC

        CompliantController.__init__(self, ft_topic=ft_topic, namespace=robot_name,
                                     joint_names_prefix=robot_name+'_',
                                     ee_link=tcp_link, gripper_type=gripper_type,
                                     **kwargs)

        self.max_force_torque = [30., 30., 30., 4., 4., 4.]
        self.p_gains = [0.05, 0.05, 0.05, 1.0, 1.0, 1.0]

    def force_control(self, target_force=None, target_positions=None,
                      force_position_selection_matrix=None,
                      timeout=10.0, stop_on_target_force=False,
                      termination_criteria=None, end_effector_link=None,
                      stop_at_wrench=None):
        """ 
        Execute force control with the robot's end effector.

        This method implements hybrid force/position control where some directions
        are force-controlled while others are position-controlled based on the
        selection matrix.

        Args:
            target_force (list[6], optional): Target force/torque for each direction [x,y,z,ax,ay,az].
                Defaults to [0,0,0,0,0,0].
            target_positions (array[array[7]] or array[7], optional): Target pose(s) for the end effector.
                Can be a single pose or a trajectory of multiple poses. Defaults to current end effector pose.
            force_position_selection_matrix (list[6], optional): Selection matrix defining which directions
                are force-controlled (0.0) or position-controlled (1.0). Defaults to None.
            timeout (float, optional): Maximum duration in seconds for the force control. Defaults to 10.0.
            stop_on_target_force (bool, optional): Whether to stop when target force is reached. Defaults to False.
            termination_criteria (callable, optional): Custom function that returns True when force control
                should terminate. Defaults to None.
            end_effector_link (str, optional): Custom end effector link to use instead of the default.
                Defaults to None.
            stop_at_wrench (list[6], optional): Wrench values at which to stop the force control.
                Defaults to [0,0,0,0,0,0].

        Returns:
            ExecutionResult: Result of the force control execution.

        Note:
            Use with caution as force control can be dangerous if not properly configured.
        """
        if end_effector_link and end_effector_link != self.default_tcp_link:
            # Init IK and FK solvers with new end effector link
            self.set_end_effector_link(end_effector_link)

        self.zero_ft_sensor()  # offset the force sensor

        self.set_control_mode("parallel")
        if force_position_selection_matrix is not None:
            self.update_selection_matrix(force_position_selection_matrix)
        self.update_pd_gains(p_gains=self.p_gains)

        target_positions = self.end_effector() if target_positions is None else np.array(target_positions)
        target_force = np.array([0., 0., 0., 0., 0., 0.]) if target_force is None else np.array(target_force)
        stop_at_wrench = np.array([0., 0., 0., 0., 0., 0.]) if stop_at_wrench is None else np.array(stop_at_wrench)

        result = self.execute_compliance_control(target_positions, target_force, max_force_torque=self.max_force_torque, duration=timeout,
                                                 stop_on_target_force=stop_on_target_force, termination_criteria=termination_criteria,
                                                 stop_at_wrench=stop_at_wrench)

        if end_effector_link and end_effector_link != self.default_tcp_link:
            # Init IK and FK solvers with default end effector link
            self.set_end_effector_link(end_effector_link)

        return result

    def execute_circular_trajectory(self, *args, **kwargs):
        """
        Execute a circular trajectory on a given plane with respect to the robot's end effector.

        This is a wrapper around execute_trajectory that sets the trajectory_type to "circular".

        Args:
            *args: Positional arguments passed to execute_trajectory
            **kwargs: Keyword arguments passed to execute_trajectory

        Returns:
            ExecutionResult: Result of the trajectory execution

        Note:
            Assumes the robot is already in its starting position.
        """
        kwargs.update({"trajectory_type": "circular"})
        return self.execute_trajectory(*args, **kwargs)

    def execute_spiral_trajectory(self, *args, **kwargs):
        """
        Execute a spiral trajectory on a given plane with respect to the robot's end effector.

        This is a wrapper around execute_trajectory that sets the trajectory_type to "spiral".

        Args:
            *args: Positional arguments passed to execute_trajectory
            **kwargs: Keyword arguments passed to execute_trajectory

        Returns:
            ExecutionResult: Result of the trajectory execution

        Note:
            Assumes the robot is already in its starting position.
        """
        kwargs.update({"trajectory_type": "spiral"})
        return self.execute_trajectory(*args, **kwargs)

    def execute_trajectory(self, plane, max_radius, radius_direction=None,
                           steps=100, revolutions=5,
                           wiggle_direction=None, wiggle_angle=0.0, wiggle_revolutions=0.0,
                           target_force=None, force_position_selection_matrix=None, timeout=10.,
                           termination_criteria=None, end_effector_link=None, trajectory_type="spiral"):
        """
        Execute a trajectory (circular or spiral) on a given plane with force control.

        Args:
            plane (str): The plane on which to execute the trajectory ('XY', 'YZ', or 'XZ')
            max_radius (float): Maximum radius of the trajectory
            radius_direction (str, optional): Direction for the radius vector. Defaults to None (random).
            steps (int, optional): Number of points in the trajectory. Defaults to 100.
            revolutions (int, optional): Number of complete revolutions. Defaults to 5.
            wiggle_direction (str, optional): Direction for wiggle motion. Defaults to None.
            wiggle_angle (float, optional): Angle of wiggle motion. Defaults to 0.0.
            wiggle_revolutions (float, optional): Number of wiggle revolutions. Defaults to 0.0.
            target_force (list[6], optional): Target force/torque for each direction. Defaults to None.
            force_position_selection_matrix (list[6], optional): Selection matrix for force/position control. Defaults to None.
            timeout (float, optional): Maximum duration in seconds. Defaults to 10.0.
            termination_criteria (callable, optional): Custom termination function. Defaults to None.
            end_effector_link (str, optional): Custom end effector link. Defaults to None.
            trajectory_type (str, optional): Type of trajectory ('circular' or 'spiral'). Defaults to "spiral".

        Returns:
            ExecutionResult: Result of the trajectory execution
        """
        from_center = True if trajectory_type == "spiral" else False
        eff = self.default_tcp_link if not end_effector_link else end_effector_link
        direction = radius_direction if radius_direction else helpers.get_random_valid_direction(plane)
        dummy_trajectory = traj_utils.compute_trajectory([0, 0, 0, 0, 0, 0, 1.],
                                                         plane, max_radius, direction, steps, revolutions,
                                                         from_center=from_center, trajectory_type=trajectory_type,
                                                         wiggle_direction=wiggle_direction, wiggle_angle=wiggle_angle,
                                                         wiggle_revolutions=wiggle_revolutions)
        # convert dummy_trajectory (initial pose frame id) to robot's base frame
        transform2target = conversions.from_pose(transform_pose(self.listener.buffer, self.base_link, eff).pose)

        if not transform2target:
            return False

        trajectory = []
        for p in dummy_trajectory:
            ps = conversions.to_pose_stamped(self.base_link, p)
            trajectory.append(conversions.from_pose_to_list(conversions.transform_pose(self.base_link, transform2target, ps).pose))

        sm = force_position_selection_matrix if force_position_selection_matrix else [1., 1., 1., 1., 1., 1.]  # no force control by default
        return self.force_control(target_force=target_force, target_positions=trajectory, force_position_selection_matrix=sm,
                                  timeout=timeout, termination_criteria=termination_criteria,
                                  end_effector_link=end_effector_link)

    def linear_push(self, force, direction, max_translation=None,
                    timeout=10.0,
                    force_position_selection_matrix=None,
                    end_effector_link=None):
        """
        Apply force control in one direction until contact with the specified force.

        This method pushes the end effector in a specified direction with a given force
        until contact is detected or a maximum translation is reached.

        Args:
            force (float): Desired force magnitude to apply
            direction (str): Direction for linear push, format: "+X", "-Y", "+Z", etc.
                The sign indicates the direction along the axis.
            max_translation (float, optional): Maximum translation distance before stopping.
                Defaults to None (no limit).
            timeout (float, optional): Maximum duration in seconds. Defaults to 10.0.
            force_position_selection_matrix (list[6], optional): Selection matrix for force/position control.
                Defaults to None (automatically determined based on target force).
            end_effector_link (str, optional): Custom end effector link. Defaults to None.

        Returns:
            bool: True if the push was successful, False otherwise
        """
        sign = 1. if '+' in direction else -1.

        target_force = np.array([0., 0., 0., 0., 0., 0.])
        target_force[get_direction_index(direction[1])] = (force + 3) * sign

        stop_at_wrench = np.array([0., 0., 0., 0., 0., 0.])
        stop_at_wrench[get_direction_index(direction[1])] = force * (sign*-1)

        if force_position_selection_matrix is None:
            force_position_selection_matrix = np.zeros(6)
            force_position_selection_matrix = np.array(target_force == 0.0) * 1.0  # define the selection matrix based on the target force

        initial_pose = self.end_effector(tip_link=end_effector_link)[get_direction_index(direction[1])]

        if max_translation is not None:
            def termination_criteria(current_pose): return abs(initial_pose - current_pose[get_direction_index(direction[1])]) >= max_translation
        else:
            termination_criteria = None

        self.set_solver_parameters(error_scale=1.0)
        pd_gains = [0.015, 0.015, 0.015, 1.0, 1.0, 1.0]
        self.update_pd_gains(pd_gains)
        result = self.force_control(target_force=target_force, force_position_selection_matrix=force_position_selection_matrix,
                                    timeout=timeout, stop_on_target_force=True, termination_criteria=termination_criteria,
                                    stop_at_wrench=stop_at_wrench, end_effector_link=end_effector_link)

        if result in (ExecutionResult.TERMINATION_CRITERIA, ExecutionResult.DONE, ExecutionResult.STOP_ON_TARGET_FORCE):
            rospy.loginfo("Completed linear_push: %s" % result)
            return True
        rospy.logerr("Fail to complete linear_push %s" % result)
        return False

    def do_insertion(self, target_pose_in_target_frame, insertion_direction, timeout,
                     radius=0.0, radius_direction=None, revolutions=3, force=1.0, goal_tolerance_if_lockup=0.0,
                     wiggle_direction=None, wiggle_angle=0.0, wiggle_revolutions=0.0,
                     force_position_selection_matrix=None):
        """
        Execute an insertion operation with force control and spiral search.

        This method attempts to insert an object by applying force in a specified direction
        while using a spiral search pattern to find the insertion point.

        Args:
            target_pose_in_target_frame (geometry_msgs/PoseStamped): Target pose in the target frame
            insertion_direction (str): Direction for insertion, format: "+X", "-Y", "+Z", etc.
                The sign indicates the direction along the axis.
            timeout (float): Maximum duration in seconds for the insertion attempt
            radius (float, optional): Maximum radius for the spiral search. Defaults to 0.0.
            radius_direction (str, optional): Direction for the radius vector. Defaults to None (random).
            revolutions (int, optional): Number of spiral revolutions. Defaults to 3.
            force (float, optional): Force to apply during insertion. Defaults to 1.0.
            goal_tolerance_if_lockup (float, optional): Tolerance to consider insertion successful
                if the robot gets stuck. Defaults to 0.0.
            wiggle_direction (str, optional): Direction for wiggle motion. Defaults to None.
            wiggle_angle (float, optional): Angle of wiggle motion. Defaults to 0.0.
            wiggle_revolutions (float, optional): Number of wiggle revolutions. Defaults to 0.0.
            force_position_selection_matrix (list[6], optional): Selection matrix for force/position control.
                Defaults to None (automatically determined based on target force).

        Returns:
            ExecutionResult: Result of the insertion operation

        Note:
            The insertion direction is currently limited to one axis in the robot's base frame.
            Future improvements may allow defining the insertion axis relative to the target object.
        """
        axis = get_direction_index(insertion_direction[1])
        plane = get_orthogonal_plane(insertion_direction[1])
        radius_direction = get_random_valid_direction(plane) if radius_direction is None else radius_direction

        offset = 1 if "Z" in insertion_direction or self.robot_name == "b_bot" else -1
        target_force = [0., 0., 0., 0., 0., 0.]
        sign = 1. if '+' in insertion_direction else -1.
        target_force[get_direction_index(insertion_direction[1])] = force * sign
        target_force *= offset

        if force_position_selection_matrix is None:
            force_position_selection_matrix = np.array(target_force == 0.0) * 0.8  # define the selection matrix based on the target force

        transform2target = conversions.from_pose(transform_pose(self.listener.buffer, target_pose_in_target_frame.header.frame_id, self.ns + "_base_link").pose)

        start_pose_robot_base = conversions.to_pose_stamped(self.ns + "_base_link", self.end_effector())
        start_pose_in_target_frame = conversions.transform_pose(target_pose_in_target_frame.header.frame_id, transform2target, start_pose_robot_base)
        start_pose_of = conversions.from_pose_to_list(start_pose_in_target_frame.pose)
        target_pose_of = conversions.from_pose_to_list(target_pose_in_target_frame.pose)
        more_than = start_pose_of[axis] < target_pose_of[axis]

        def termination_criteria(current_pose):
            current_pose_robot_base = conversions.to_pose_stamped(self.ns + "_base_link", current_pose)
            current_pose_in_target_frame = conversions.transform_pose(target_pose_in_target_frame.header.frame_id, transform2target, current_pose_robot_base)
            current_pose_of = conversions.from_pose_to_list(current_pose_in_target_frame.pose)
            # print("check cp,tp", current_pose_of[axis], target_pose_of[axis])
            if more_than:
                return current_pose_of[axis] >= target_pose_of[axis] or \
                    (current_pose_of[axis] >= target_pose_of[axis] - goal_tolerance_if_lockup)
            return current_pose_of[axis] <= target_pose_of[axis] or \
                (current_pose_of[axis] <= target_pose_of[axis] + goal_tolerance_if_lockup)

        result = self.execute_spiral_trajectory(plane, max_radius=radius, radius_direction=radius_direction, steps=100, revolutions=revolutions,
                                                wiggle_direction=wiggle_direction, wiggle_angle=wiggle_angle, wiggle_revolutions=wiggle_revolutions,
                                                target_force=target_force, force_position_selection_matrix=force_position_selection_matrix, timeout=timeout,
                                                termination_criteria=termination_criteria)

        if result in (ExecutionResult.TERMINATION_CRITERIA, ExecutionResult.DONE, ExecutionResult.STOP_ON_TARGET_FORCE):
            rospy.loginfo("Completed insertion with state: %s" % result)
        else:
            rospy.logerr("Fail to complete insertion with state %s" % result)
        return result
