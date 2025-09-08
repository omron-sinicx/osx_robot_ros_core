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
# Author: Cristian C. Beltran-Hernandez, Felix von Drigalski

from typing import List, Tuple, Union, Optional
import actionlib
import copy
import rosbag
import rospkg
import rospy
import tf2_ros

import numpy as np
import geometry_msgs.msg
import moveit_msgs.msg
import moveit_msgs.srv

from moveit_commander import MoveGroupCommander
from osx_robot_control import helpers
from osx_robot_control.utils import transform_pose
from std_msgs.msg import Bool
from ur_control import conversions, transformations


class RobotBase:
    """ 
    Base class for robot arm control via MoveIt.

    This class provides a comprehensive interface for controlling robot arms using the MoveIt framework.
    It includes methods for forward and inverse kinematics, motion planning, and execution of various
    types of movements (linear, joint-based, circular, etc.).

    Attributes:
        ns (str): Namespace for the robot group
        robot_group (MoveGroupCommander): MoveIt commander for the robot group
        listener (tf2_ros.TransformListener): TF2 transform listener
        buffer (tf2_ros.Buffer): TF2 buffer for transforms
        marker_counter (int): Counter for visualization markers
        sequence_move_group (actionlib.SimpleActionClient): Action client for sequence movements
        run_mode_ (bool): Flag indicating if the robot is in run mode
        pause_mode_ (bool): Flag indicating if the robot is in pause mode
        test_mode_ (bool): Flag indicating if the robot is in test mode
        moveit_ik_srv (rospy.ServiceProxy): Service proxy for inverse kinematics
        moveit_fk_srv (rospy.ServiceProxy): Service proxy for forward kinematics
    """

    def __init__(self, group_name: str, tf_listener: tf2_ros.TransformListener, ns: str = "", ns_is_group_name: bool = False):
        """
        Initialize the RobotBase class.

        Args:
            group_name (str): Name of the MoveIt group to control
            tf_listener (tf2_ros.TransformListener): TF2 transform listener
            ns (str): Namespace for the robot group
        """
        self.robot_group = MoveGroupCommander(group_name)
        self.ns = group_name if ns_is_group_name else ns

        self.listener: tf2_ros.TransformListener = tf_listener
        self.buffer: tf2_ros.Buffer = self.listener.buffer
        self.marker_counter = 0

        self.sequence_move_group = actionlib.SimpleActionClient("/sequence_move_group", moveit_msgs.msg.MoveGroupSequenceAction)

        self.run_mode_ = True     # The modes limit the maximum speed of motions. Used with the safety system @WRS2020
        self.pause_mode_ = False
        self.test_mode_ = False

        self.sub_run_mode_ = rospy.Subscriber("/run_mode", Bool, self.run_mode_callback)
        self.sub_pause_mode_ = rospy.Subscriber("/pause_mode", Bool, self.pause_mode_callback)
        self.sub_test_mode_ = rospy.Subscriber("/test_mode", Bool, self.test_mode_callback)

        rospy.wait_for_service('compute_ik')
        rospy.wait_for_service('compute_fk')

        self.moveit_ik_srv = rospy.ServiceProxy('/compute_ik', moveit_msgs.srv.GetPositionIK)
        self.moveit_fk_srv = rospy.ServiceProxy('/compute_fk', moveit_msgs.srv.GetPositionFK)

    def run_mode_callback(self, msg: Bool) -> None:
        """
        Callback for run mode changes.

        Args:
            msg (Bool): Message containing the new run mode state
        """
        self.run_mode_ = msg.data

    def pause_mode_callback(self, msg: Bool) -> None:
        """
        Callback for pause mode changes.

        Args:
            msg (Bool): Message containing the new pause mode state
        """
        self.pause_mode_ = msg.data

    def test_mode_callback(self, msg: Bool) -> None:
        """
        Callback for test mode changes.

        Args:
            msg (Bool): Message containing the new test mode state
        """
        self.test_mode_ = msg.data

    def compute_fk(self, robot_state: Optional[Union[List[float], Tuple[float, ...], np.ndarray, moveit_msgs.msg.RobotState]] = None,
                   tcp_link: Optional[str] = None,
                   frame_id: Optional[str] = None) -> Union[geometry_msgs.msg.PoseStamped, bool]:
        """
        Compute the Forward Kinematics for a move group using the MoveIt service.

        Args:
            robot_state: Joint state values or RobotState message. If passed as list, tuple, or numpy array,
                        assumes that the joint values are in the same order as defined for that group.
            tcp_link: Tool center point link name. If None, uses the default end effector link.
            frame_id: Frame to transform the result to. If None, returns in the original frame.

        Returns:
            geometry_msgs.msg.PoseStamped: The computed pose, or False if computation failed
        """
        if robot_state:
            if isinstance(robot_state, moveit_msgs.msg.RobotState):
                robot_state_ = robot_state
            elif isinstance(robot_state, (list, tuple, np.ndarray)):
                robot_state_ = moveit_msgs.msg.RobotState()
                robot_state_.joint_state.name = self.robot_group.get_active_joints()
                robot_state_.joint_state.position = list(robot_state)
            else:
                rospy.logerr("Unsupported type of robot_state %s" % type(robot_state))
                raise
        else:
            return self.compute_fk(robot_state=self.robot_group.get_current_joint_values())
        req = moveit_msgs.srv.GetPositionFKRequest()
        req.fk_link_names = [tcp_link if tcp_link else self.robot_group.get_end_effector_link()]
        req.robot_state = robot_state_
        res = self.moveit_fk_srv.call(req)
        if res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS:
            rospy.logwarn("compute FK failed with code: %s" % res.error_code.val)
            return False
        else:
            if frame_id:
                return transform_pose(self.buffer, frame_id, res.pose_stamped[0])
            return res.pose_stamped[0]

    def compute_ik(self, target_pose: geometry_msgs.msg.PoseStamped,
                   joints_seed: Optional[List[float]] = None,
                   timeout: float = 0.01,
                   end_effector_link: str = "",
                   retry: bool = False,
                   allow_collisions: bool = False) -> Optional[List[float]]:
        """
        Compute the Inverse Kinematics for a move group using the MoveIt service.

        Args:
            target_pose: Target pose to compute IK for
            joints_seed: Initial joint configuration for IK solver. If None, uses current joint values.
            timeout: Timeout for the IK solver in seconds. Higher values may improve success rate.
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            retry: If True, retries the IK computation for up to 10 seconds if it fails.
            allow_collisions: If True, allows collisions with other objects during IK computation.
                             Self-collisions are always considered.

        Returns:
            List[float]: Joint values solution, or None if computation failed
        """
        if isinstance(target_pose, geometry_msgs.msg.PoseStamped):
            ik_request = moveit_msgs.msg.PositionIKRequest()
            ik_request.avoid_collisions = not allow_collisions
            ik_request.timeout = rospy.Duration(timeout)
            ik_request.pose_stamped = target_pose
            ik_request.group_name = self.robot_group.get_name()
            ik_request.ik_link_name = end_effector_link
            ik_request.robot_state.joint_state.name = self.robot_group.get_active_joints()
            ik_request.robot_state.joint_state.position = joints_seed if joints_seed is not None else self.robot_group.get_current_joint_values()
        else:
            rospy.logerr("Unsupported type of target_pose %s" % type(target_pose))
            raise

        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request = ik_request
        res = self.moveit_ik_srv.call(req)

        if retry:
            start_time = rospy.get_time()
            while res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS \
                    and not rospy.is_shutdown() and (rospy.get_time() - start_time < 10):
                res = self.moveit_ik_srv.call(req)

        if res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS:
            rospy.logwarn("compute IK failed with code: %s" % res.error_code.val)
            return None

        solution = []
        for joint_name in self.robot_group.get_active_joints():
            solution.append(res.solution.joint_state.position[res.solution.joint_state.name.index(joint_name)])
        return solution

    def set_up_move_group(self, speed: float, acceleration: Optional[float], planner: str = "OMPL") -> Tuple[float, float]:
        """
        Set up the move group interface with planner, speed scaling, and acceleration scaling.

        Args:
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            planner: Planner to use ("OMPL", "LINEAR", "PTP", or "CIRC")

        Returns:
            Tuple[float, float]: The applied speed and acceleration scaling factors
        """
        assert not rospy.is_shutdown()
        (speed_, accel_) = self.limit_speed_and_acc(speed, acceleration)
        group = self.robot_group
        rospy.logdebug("Setting velocity scaling to " + str(speed_))
        rospy.logdebug("Setting acceleration scaling to " + str(accel_))
        group.set_max_velocity_scaling_factor(speed_)
        group.set_max_acceleration_scaling_factor(accel_)
        self.set_planner(planner)
        return speed_, accel_

    def set_planner(self, planner: str = "OMPL") -> None:
        """
        Set the planner for the move group.

        Args:
            planner: Planner to use ("OMPL", "LINEAR", "PTP", or "CIRC")

        Raises:
            ValueError: If an unsupported planner is specified
        """
        group = self.robot_group
        if planner == "OMPL":
            group.set_planning_pipeline_id("ompl")
            group.set_planner_id("RRTConnect")
            group.set_goal_joint_tolerance(1e-3)
        elif planner == "LINEAR":
            group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            group.set_planner_id("LIN")
        elif planner == "PTP":
            group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            group.set_planner_id("PTP")
        elif planner == "CIRC":
            group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            group.set_planner_id("CIRC")
        else:
            raise ValueError("Unsupported planner: %s" % planner)

    def limit_speed_and_acc(self, speed: float, acceleration: Optional[float]) -> Tuple[float, float]:
        """
        Limit speed and acceleration based on robot mode and safety constraints.

        Args:
            speed: Desired speed scaling factor
            acceleration: Desired acceleration scaling factor

        Returns:
            Tuple[float, float]: The limited speed and acceleration scaling factors
        """
        if self.pause_mode_ or self.test_mode_:
            if speed > self.reduced_mode_speed_limit:
                rospy.loginfo("Reducing speed from " + str(speed) + " to " + str(self.reduced_mode_speed_limit) + " because robot is in test or pause mode")
                speed = self.reduced_mode_speed_limit
        sp = copy.copy(speed)
        acc = copy.copy(acceleration)
        if sp > 1.0:
            sp = 1.0
        if acc is None:
            rospy.logdebug("Setting acceleration to " + str(sp) + " by default.")
            acc = sp/2.0
        else:
            if acc > sp:
                rospy.logdebug("Setting acceleration to " + str(sp) + " instead of " + str(acceleration) + " to avoid jerky motion.")
                acc = sp
        return (sp, acc)

    def check_goal_pose_reached(self, goal_pose: geometry_msgs.msg.PoseStamped) -> bool:
        """
        Check if the current pose matches the goal pose within a tolerance.

        Args:
            goal_pose: The target pose to check against

        Returns:
            bool: True if the goal pose has been reached, False otherwise
        """
        current_pose = self.robot_group.get_current_pose()
        if current_pose.header.frame_id != goal_pose.header.frame_id:
            gp = transform_pose(self.buffer, current_pose.header.frame_id, goal_pose)
        else:
            gp = goal_pose
        return helpers.all_close(gp.pose, current_pose.pose, 0.01)

    def joint_configuration_changes(self, start: List[float], end: List[float], tolerance: float = 0.1) -> bool:
        """
        Check if the sign of any joint angle changes during the motion.

        Args:
            start: Initial joint configuration
            end: Final joint configuration
            tolerance: Tolerance for small joint angles (in radians)

        Returns:
            bool: True if the sign of any joint angle changes and the joint angle is not near zero,
                 False otherwise
        """
        signs = np.sign(np.array(start)*np.array(end))

        if np.all(signs > 0):
            return False  # = all OK

        joint_changes_small = True
        for i in range(len(signs)):

            if signs[i] < 0:
                if abs(start[i] < tolerance) or abs(end[i] < tolerance):
                    rospy.logdebug("Joint changes sign, but the change is small. Ignoring.")
                    rospy.logdebug("start[i] = %d6, end[i] = %d6", (start[i], end[i]))
                    continue
                rospy.logerr("Joint angle " + str(i) + " would change sign!")
                print("start[i] = %d6, end[i] = %d6", (start[i], end[i]))
                joint_changes_small = False
        if joint_changes_small:
            return False  # = all OK
        else:
            return True  # Joints change

    def get_current_pose_stamped(self, end_effector_link: str = '') -> geometry_msgs.msg.PoseStamped:
        """
        Get the current pose of the end effector as a PoseStamped message.

        Args:
            end_effector_link: End effector link name. If empty, uses the default end effector link.

        Returns:
            geometry_msgs.msg.PoseStamped: The current pose of the end effector
        """
        return self.robot_group.get_current_pose(end_effector_link)

    def get_current_pose(self, end_effector_link: str = '') -> geometry_msgs.msg.Pose:
        """
        Get the current pose of the end effector as a Pose message.

        Args:
            end_effector_link: End effector link name. If empty, uses the default end effector link.

        Returns:
            geometry_msgs.msg.Pose: The current pose of the end effector
        """
        return self.robot_group.get_current_pose(end_effector_link).pose

    def get_named_pose_target(self, name: str) -> List[float]:
        """
        Get joint values for a named pose target.

        Args:
            name: Name of the pose target in the MoveIt configuration

        Returns:
            List[float]: Joint values for the named pose target
        """
        return helpers.ordered_joint_values_from_dict(self.robot_group.get_named_target_values(name), self.robot_group.get_active_joints())

    def save_plan(self, filename: str, plan: moveit_msgs.msg.RobotTrajectory) -> None:
        """
        Store a given plan to a file.

        Args:
            filename: Name of the file to save the plan to
            plan: The plan to save
        """
        rp = rospkg.RosPack()
        bagfile = rp.get_path("osx_robot_control") + "/config/saved_plans/" + filename
        with rosbag.Bag(bagfile, 'w') as bag:
            bag.write(topic="saved_plan", msg=plan)

    def load_saved_plan(self, filename: str) -> moveit_msgs.msg.RobotTrajectory:
        """
        Load a plan from a file.

        Args:
            filename: Name of the file to load the plan from

        Returns:
            moveit_msgs.msg.RobotTrajectory: The loaded plan
        """
        rp = rospkg.RosPack()
        bagfile = rp.get_path("osx_robot_control") + "/config/saved_plans/" + filename
        with rosbag.Bag(bagfile, 'r') as bag:
            for (topic, plan, ts) in bag.read_messages():
                return plan

    def execute_saved_plan(self, filename: str = "", plan: List[moveit_msgs.msg.RobotTrajectory] = [], wait: bool = True) -> bool:
        """
        Execute a saved plan from a file or a provided plan.

        Args:
            filename: Name of the file to load the plan from. If empty, uses the provided plan.
            plan: Plan to execute. If empty and filename is provided, loads the plan from the file.
            wait: Whether to wait for the plan to complete

        Returns:
            bool: True if the plan was executed successfully, False otherwise
        """
        if filename and not plan:
            plan = self.load_saved_plan(filename)
        return self.execute_plan(plan, wait)

    # ------ Robot motion functions

    def execute_plan(self, plan: moveit_msgs.msg.RobotTrajectory, wait: bool = True) -> bool:
        """
        Execute a planned trajectory.

        Args:
            plan: The plan to execute
            wait: Whether to wait for the plan to complete

        Returns:
            bool: True if the plan was executed successfully, False otherwise
        """
        result = self.robot_group.execute(plan, wait=wait)
        self.robot_group.clear_pose_targets()
        return result

    def set_pose_goal(self, pose_goal_stamped: geometry_msgs.msg.PoseStamped,
                      speed: float = 0.25,
                      acceleration: Optional[float] = None,
                      end_effector_link: str = "",
                      move_lin: bool = False,
                      wait: bool = True,
                      plan_only: bool = False,
                      initial_joints: Optional[List[float]] = None,
                      allow_joint_configuration_flip: bool = False,
                      move_ptp: bool = False,
                      retime: bool = False) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Move robot to a given PoseStamped goal.

        This method plans and optionally executes a motion to reach a target pose.
        It supports different planning strategies (OMPL, PTP, LINEAR) and can be configured
        to only plan without execution.

        Args:
            pose_goal_stamped: Target pose to move to
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            move_lin: If True, force use of Pilz linear planner
            wait: Whether to wait for the motion to complete
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            allow_joint_configuration_flip: If True, allow joint configuration to flip during motion
            move_ptp: If True, plan first using Pilz PTP planner, in case of failure, retry with OMPL
            retime: If True, retime plan using time_optimal_trajectory_generation

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        move_ptp = False if move_lin else move_ptp  # Override if move_lin is set (Linear takes priority since PTP is the default value)

        planner = "LINEAR" if move_lin else ("PTP" if move_ptp else "OMPL")
        speed_, accel_ = self.set_up_move_group(speed, acceleration, planner)

        group = self.robot_group
        group.clear_pose_targets()

        if not end_effector_link:
            end_effector_link = self.robot_group.get_end_effector_link()
        group.set_end_effector_link(end_effector_link)

        robots_in_simultaneous = rospy.get_param("/osx/simultaneous", False)
        if initial_joints is None:
            group.set_start_state_to_current_state()
        else:
            group.set_start_state(helpers.to_robot_state(group, initial_joints))

        group.set_pose_target(pose_goal_stamped)
        max_retries = 3
        for attempt in range(max_retries):
            success, plan, planning_time, error = group.plan()
            if success:
                break
            rospy.logwarn(f"Planning attempt {attempt + 1} failed, retrying...")

        rospy.loginfo(f"go_to_pose_goal planning time: {planning_time:.3f}")

        if success:
            if self.joint_configuration_changes(plan.joint_trajectory.points[0].positions,
                                                plan.joint_trajectory.points[-1].positions) \
                    and not allow_joint_configuration_flip:
                success = False
                rospy.logwarn("Joint configuration would have flipped.")

        if success:
            if planner != "LINEAR" and retime:
                # retime
                plan = self.robot_group.retime_trajectory(self.robot_group.get_current_state(), plan, algorithm="time_optimal_trajectory_generation",
                                                          velocity_scaling_factor=speed_, acceleration_scaling_factor=accel_)
            if plan_only:
                group.set_start_state_to_current_state()
                group.clear_pose_targets()
                return plan, planning_time
            else:
                success = self.execute_plan(plan, wait=wait)

        if not success:
            rospy.logerr("go_to_pose_goal failed times! Broke out, published failed pose. simultaneous=" + str(robots_in_simultaneous))
            helpers.publish_marker(pose_goal_stamped, "pose", self.ns + "_move_lin_failed_pose_" + str(self.marker_counter))
            self.marker_counter += 1
        else:
            helpers.publish_marker(pose_goal_stamped, "pose", self.ns + "_go_to_pose_goal_failed_pose_" + str(self.marker_counter), marker_topic="osx_success_markers")
            self.marker_counter += 1

        group.clear_pose_targets()

        return success

    def set_linear_eef_trajectory(self, trajectory: List[Tuple[geometry_msgs.msg.PoseStamped, float, Optional[float]]],
                                  speed: float = 1.0,
                                  acceleration: Optional[float] = None,
                                  end_effector_link: str = "",
                                  plan_only: bool = False,
                                  initial_joints: Optional[List[float]] = None,
                                  allow_joint_configuration_flip: bool = False,
                                  timeout: float = 10,
                                  retime: bool = False) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Compute and execute a linear trajectory through multiple waypoints using the Pilz Linear planner.

        Args:
            trajectory: List of tuples, each containing (pose, blend_radius, speed).
                       If speed is not provided in the tuple, the global speed parameter is used.
            speed: Default speed scaling factor (0.0 to 1.0) for waypoints without specified speed
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            allow_joint_configuration_flip: If True, allow joint configuration to flip during motion
            timeout: Timeout for planning in seconds
            retime: If True, retime plan using time_optimal_trajectory_generation

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        # TODO: Add allow_joint_configuration_flip
        speed_, accel_ = self.set_up_move_group(speed, acceleration, "LINEAR")

        if not end_effector_link:
            end_effector_link = self.robot_group.get_end_effector_link()

        group = self.robot_group

        group.set_end_effector_link(end_effector_link)
        # Do we need this transformation?
        if len(trajectory[0]) == 2:  # Speed per point was not defined
            waypoints = [(ps, blend_radius, speed) for ps, blend_radius in trajectory]
        elif len(trajectory[0]) == 3:
            waypoints = [(ps, blend_radius, speed) for ps, blend_radius, speed in trajectory]

        motion_plan_requests = []

        # Start from current pose
        if initial_joints:
            initial_pose = self.compute_fk(initial_joints, end_effector_link)
            group.set_pose_target(initial_pose)
        else:
            group.set_pose_target(group.get_current_pose(end_effector_link))
        msi = moveit_msgs.msg.MotionSequenceItem()
        msi.req = group.construct_motion_plan_request()
        msi.blend_radius = 0.0

        if initial_joints:
            msi.req.start_state = helpers.to_robot_state(self.robot_group, initial_joints)
        else:
            msi.req.start_state = helpers.to_robot_state(self.robot_group, self.robot_group.get_current_joint_values())

        motion_plan_requests.append(msi)

        for wp, blend_radius, spd in waypoints:
            self.set_up_move_group(spd, acceleration, planner="LINEAR")
            group.clear_pose_targets()
            group.set_pose_target(wp)
            msi = moveit_msgs.msg.MotionSequenceItem()
            msi.req = group.construct_motion_plan_request()
            msi.req.start_state = moveit_msgs.msg.RobotState()
            msi.blend_radius = blend_radius
            motion_plan_requests.append(msi)

        # Force last point to be 0.0 to avoid raising an error in the planner
        motion_plan_requests[-1].blend_radius = 0.0

        # Make MotionSequence
        goal = moveit_msgs.msg.MoveGroupSequenceGoal()
        goal.request = moveit_msgs.msg.MotionSequenceRequest()
        goal.request.items = motion_plan_requests
        # Plan only always for compatibility with simultaneous motions
        goal.planning_options.plan_only = True

        self.sequence_move_group.send_goal_and_wait(goal)
        response = self.sequence_move_group.get_result()

        group.clear_pose_targets()

        if response.response.error_code.val == 1:
            plan = response.response.planned_trajectories[0]  # support only one plan?
            planning_time = response.response.planning_time
            rospy.loginfo(f"move_lin_trajectory planning time: {planning_time:.3f}")
            if retime:
                # retime
                plan = self.robot_group.retime_trajectory(self.robot_group.get_current_state(), plan,
                                                          algorithm="time_optimal_trajectory_generation",
                                                          velocity_scaling_factor=speed_, acceleration_scaling_factor=accel_)
            if plan_only:
                return plan, planning_time
            else:
                return self.execute_plan(plan, wait=True)

        rospy.logerr("move_lin_trajectory failed times! Broke out, published failed pose.")
        helpers.publish_marker(waypoints[-1][0], "pose", self.ns + "_move_lin_failed_pose_" + str(self.marker_counter))
        self.marker_counter += 1

        rospy.logerr("Failed to plan linear trajectory. error code: %s" % response.response.error_code.val)
        return False

    def set_relative_motion_goal(self, relative_translation: List[float] = [0, 0, 0],
                                 relative_rotation: List[float] = [0, 0, 0],
                                 speed: float = 0.5,
                                 acceleration: Optional[float] = None,
                                 relative_to_robot_base: bool = False,
                                 relative_to_tcp: bool = False,
                                 wait: bool = True,
                                 end_effector_link: str = "",
                                 plan_only: bool = False,
                                 initial_joints: Optional[List[float]] = None,
                                 allow_joint_configuration_flip: bool = False,
                                 pose_only: bool = False,
                                 retime: bool = False) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], geometry_msgs.msg.PoseStamped, bool]:
        """
        Perform a linear motion relative to the current position of the robot.

        Args:
            relative_translation: Translation relative to current TCP position, expressed in world frame
            relative_rotation: Rotation relative to current TCP position, expressed in world frame (roll, pitch, yaw)
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            relative_to_robot_base: If True, uses the robot_base coordinates for the relative motion
            relative_to_tcp: If True, uses the robot's end effector link coordinates for the relative motion
            wait: Whether to wait for the motion to complete
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            allow_joint_configuration_flip: If True, allow joint configuration to flip during motion
            pose_only: If True, return only the computed target pose without planning or executing
            retime: If True, retime plan using time_optimal_trajectory_generation

        Returns:
            If pose_only is True, returns the computed target pose.
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        if not end_effector_link:
            end_effector_link = self.ns + "_gripper_tip_link"

        group = self.robot_group
        group.set_end_effector_link(end_effector_link)

        if initial_joints:
            w2b = self.buffer.lookup_transform("world", self.ns + "_base_link", rospy.Time(0), rospy.Duration(1.0))
            t_w2b = transformations.pose_to_transform(list(w2b[0]) + list(w2b[1]))  # transform robot's base to world frame
            b2tcp = self.compute_fk(initial_joints, tcp_link=end_effector_link, frame_id=self.ns + "_base_link")  # forward kinematics
            t_b2tcp = conversions.from_pose(b2tcp.pose)  # transform tcp to robot's base
            if relative_to_tcp:
                new_pose = conversions.to_pose_stamped(end_effector_link, [0, 0, 0, 0, 0, 0.])
            elif relative_to_robot_base:
                new_pose = self.compute_fk(initial_joints, tcp_link=end_effector_link, frame_id=self.ns + "_base_link")
            else:
                t_w2tcp = transformations.concatenate_matrices(t_w2b, t_b2tcp)
                new_pose = conversions.to_pose_stamped("world", transformations.pose_from_matrix(t_w2tcp))
        else:
            new_pose = group.get_current_pose()

            if relative_to_robot_base:
                new_pose = transform_pose(self.buffer, self.ns + "_base_link", new_pose)
            elif relative_to_tcp:
                new_pose = transform_pose(self.buffer, self.ns + "_gripper_tip_link", new_pose)

        new_position = conversions.from_point(new_pose.pose.position) + relative_translation
        new_pose.pose.position = conversions.to_point(new_position)
        new_pose.pose.orientation = helpers.rotateQuaternionByRPYInUnrotatedFrame(relative_rotation[0], relative_rotation[1],
                                                                                  relative_rotation[2], new_pose.pose.orientation)

        if initial_joints:
            newpose = conversions.from_pose_to_list(new_pose.pose)  # new relative transformation
            t_newpose = transformations.pose_to_transform(newpose)
            if relative_to_tcp:
                # manually compute the transform from TCP to world since we are doing offline planning
                t_w2tcp = transformations.concatenate_matrices(t_w2b, t_b2tcp, t_newpose)
                new_pose = conversions.to_pose_stamped("world", transformations.pose_from_matrix(t_w2tcp))
            if relative_to_robot_base:
                # manually compute the transform from base to world since we are doing offline planning
                t_w2tcp = transformations.concatenate_matrices(t_w2b, t_newpose)
                new_pose = conversions.to_pose_stamped("world", transformations.pose_from_matrix(t_w2tcp))

        if pose_only:
            return new_pose
        else:
            return self.set_pose_goal(new_pose, speed=speed, acceleration=acceleration,
                                      end_effector_link=end_effector_link,  wait=wait,
                                      move_lin=True, plan_only=plan_only, initial_joints=initial_joints,
                                      allow_joint_configuration_flip=allow_joint_configuration_flip,
                                      retime=retime)

    def go_to_named_pose(self, pose_name: str,
                         speed: float = 0.25,
                         acceleration: Optional[float] = None,
                         wait: bool = True,
                         plan_only: bool = False,
                         initial_joints: Optional[List[float]] = None,
                         move_ptp: bool = False,
                         retime: bool = False) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Move the robot to a named pose defined in the MoveIt configuration.

        Args:
            pose_name: Name of the pose in the MoveIt configuration (e.g., "home", "back")
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            wait: Whether to wait for the motion to complete
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            move_ptp: If True, use Pilz PTP planner instead of OMPL
            retime: If True, retime plan using time_optimal_trajectory_generation

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        speed_, accel_ = self.set_up_move_group(speed, acceleration, planner=("PTP" if move_ptp else "OMPL"))
        group = self.robot_group

        group.set_named_target(pose_name)

        if initial_joints:
            group.set_start_state(helpers.to_robot_state(group, initial_joints))
        else:
            group.set_start_state_to_current_state()

        success, plan, planning_time, error = group.plan()
        rospy.loginfo(f"go_to_named_pose planning time: {planning_time:.3f}")

        if success:
            if retime:
                plan = self.robot_group.retime_trajectory(self.robot_group.get_current_state(), plan, algorithm="time_optimal_trajectory_generation",
                                                          velocity_scaling_factor=speed_, acceleration_scaling_factor=accel_)
            group.clear_pose_targets()
            group.set_start_state_to_current_state()
            if plan_only:
                return plan, planning_time
            else:
                success = self.execute_plan(plan, wait=wait)

        return success

    def set_joint_position_goal(self, joint_pose_goal: List[float],
                                speed: float = 0.6,
                                acceleration: Optional[float] = None,
                                wait: bool = True,
                                plan_only: bool = False,
                                initial_joints: Optional[List[float]] = None,
                                move_ptp: bool = False,
                                retime: bool = False) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Move the robot to a specific joint configuration.

        Args:
            joint_pose_goal: Target joint values
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            wait: Whether to wait for the motion to complete
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            move_ptp: If True, use Pilz PTP planner instead of OMPL
            retime: If True, retime plan using time_optimal_trajectory_generation

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        speed_, accel_ = self.set_up_move_group(speed, acceleration, planner=("PTP" if move_ptp else "OMPL"))
        group = self.robot_group

        group.set_joint_value_target(joint_pose_goal)

        if initial_joints:
            group.set_start_state(helpers.to_robot_state(group, initial_joints))
        else:
            group.set_start_state_to_current_state()

        success, plan, planning_time, error = group.plan()

        rospy.loginfo(f"move_joints planning time: {planning_time:.3f}")

        if success:
            if retime:
                # retime
                plan = self.robot_group.retime_trajectory(self.robot_group.get_current_state(), plan, algorithm="time_optimal_trajectory_generation",
                                                          velocity_scaling_factor=speed_, acceleration_scaling_factor=accel_)
            group.set_start_state_to_current_state()
            if plan_only:
                return plan, planning_time
            else:
                return self.execute_plan(plan, wait=wait)

        return False

    def move_joints_trajectory(self, trajectory: List[Tuple[Union[str, List[float], Tuple[float, ...], geometry_msgs.msg.PoseStamped], float, float]],
                               speed: float = 1.0,
                               acceleration: Optional[float] = None,
                               plan_only: bool = False,
                               initial_joints: Optional[List[float]] = None,
                               end_effector_link: str = "",
                               planner: str = "OMPL") -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Compute and execute a joint trajectory through multiple waypoints.

        Args:
            trajectory: List of tuples, each containing (joint_values, blend_radius, speed).
                       joint_values can be a string (named pose), a list/tuple of joint values,
                       or a PoseStamped (will be converted to joint values).
            speed: Default speed scaling factor (0.0 to 1.0) for waypoints without specified speed
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            planner: Planner to use ("OMPL", "PTP")

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        speed_, accel_ = self.set_up_move_group(speed, acceleration, planner=planner)

        group = self.robot_group

        try:
            if not end_effector_link:
                end_effector_link = self.robot_group.get_end_effector_link()
            group.set_end_effector_link(end_effector_link)
        except:
            pass

        waypoints = []
        for point, blend_radius, speed in trajectory:
            if isinstance(point, str):
                joint_values = helpers.ordered_joint_values_from_dict(group.get_named_target_values(point), group.get_active_joints())
            elif isinstance(point, tuple) or isinstance(point, list) or isinstance(point, geometry_msgs.msg.PoseStamped):
                joint_values = point
            else:
                rospy.logerr("Joint trajectory with invalid point: type=%s" % type(point))
                return False
            waypoints.append((joint_values, blend_radius, speed))

        group.set_joint_value_target(initial_joints if initial_joints else group.get_current_joint_values())
        # Start from current pose
        msi = moveit_msgs.msg.MotionSequenceItem()
        msi.req = group.construct_motion_plan_request()
        msi.blend_radius = 0.0
        msi.req.start_state = helpers.to_robot_state(group, initial_joints if initial_joints else group.get_current_joint_values())

        motion_plan_requests = []
        motion_plan_requests.append(msi)

        for wp, blend_radius, spd in waypoints:
            self.set_up_move_group(spd, spd/2.0, planner=planner)
            group.clear_pose_targets()
            try:
                group.set_joint_value_target(wp)
            except Exception as e:
                rospy.logerr("Can set joint traj point: %s. Abort" % e)
                break
            msi = moveit_msgs.msg.MotionSequenceItem()
            msi.req = group.construct_motion_plan_request()
            msi.req.start_state = moveit_msgs.msg.RobotState()
            msi.blend_radius = blend_radius
            motion_plan_requests.append(msi)

        # Force last point to be 0.0 to avoid raising an error in the planner
        motion_plan_requests[-1].blend_radius = 0.0

        # Make MotionSequence
        goal = moveit_msgs.msg.MoveGroupSequenceGoal()
        goal.request = moveit_msgs.msg.MotionSequenceRequest()
        goal.request.items = motion_plan_requests
        # Plan only always for compatibility with simultaneous motions
        goal.planning_options.plan_only = True

        self.sequence_move_group.send_goal_and_wait(goal)
        response = self.sequence_move_group.get_result()

        group.clear_pose_targets()

        if response.response.error_code.val == 1:  # Success
            plan = response.response.planned_trajectories[0]  # support only one plan?
            # retime
            plan = self.robot_group.retime_trajectory(self.robot_group.get_current_state(), plan, algorithm="time_optimal_trajectory_generation",
                                                      velocity_scaling_factor=speed_, acceleration_scaling_factor=accel_)
            planning_time = response.response.planning_time
            if plan_only:
                return plan, planning_time
            else:
                return self.execute_plan(plan, wait=True)

        return False

    def move_circ(self, pose_goal_stamped: geometry_msgs.msg.PoseStamped,
                  constraint_point: List[float],
                  constraint_type: str = "center",
                  speed: float = 0.25,
                  acceleration: Optional[float] = None,
                  wait: bool = True,
                  end_effector_link: str = "",
                  plan_only: bool = False,
                  initial_joints: Optional[List[float]] = None) -> Union[Tuple[moveit_msgs.msg.RobotTrajectory, float], bool]:
        """
        Move the robot in a circular arc to a target pose.

        Args:
            pose_goal_stamped: Target pose to move to
            constraint_point: Point to constrain the circular motion (center or interim point)
            constraint_type: Type of constraint ("center" or "interim")
            speed: Speed scaling factor (0.0 to 1.0)
            acceleration: Acceleration scaling factor. If None, set to half of speed.
            wait: Whether to wait for the motion to complete
            end_effector_link: End effector link name. If empty, uses the default end effector link.
            plan_only: If True, return only the plan and planning time without executing
            initial_joints: Initial joint configuration for planning. If None, uses current joint values.

        Returns:
            If plan_only is True, returns a tuple of (plan, planning_time).
            Otherwise, returns True if the motion was successful, False otherwise.
        """
        if not self.set_up_move_group(speed, acceleration, "CIRC"):
            return False

        group = self.robot_group
        group.clear_pose_targets()

        if not end_effector_link:
            end_effector_link = self.ns + "_gripper_tip_link"
        group.set_end_effector_link(end_effector_link)

        if initial_joints:
            group.set_start_state(helpers.to_robot_state(group, initial_joints))

        # Do we need this transformation?
        # pose_goal_world = transform_pose(self.buffer, "world", pose_goal_stamped)
        group.set_pose_target(pose_goal_stamped)

        constraint = moveit_msgs.msg.Constraints()
        if constraint_type not in ("center", "interim"):
            rospy.logerr("Invalid parameter: %s" % constraint_type)
            return False
        constraint.name = constraint_type
        pc = moveit_msgs.msg.PositionConstraint()
        if constraint_type == "center":
            constraint_pose = conversions.from_pose_to_list(self.get_current_pose())[:3] - constraint_point
            constraint_pose = conversions.to_pose(constraint_pose.tolist()+[0, 0, 0])
        else:
            constraint_pose = conversions.to_pose(constraint_point+[0, 0, 0])  # Pose
        pc.constraint_region.primitive_poses = [constraint_pose]
        constraint.position_constraints = [pc]
        group.set_path_constraints(constraint)

        success, plan, planning_time, error = group.plan()

        if success:
            if plan_only:
                group.clear_pose_targets()
                group.set_start_state_to_current_state()
                return plan, planning_time
            else:
                self.execute_plan(plan, wait=wait)

        group.clear_pose_targets()
        return success
