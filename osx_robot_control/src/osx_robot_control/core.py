#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2023, OMRON SINIC X
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

import copy
import os
import timeit

import geometry_msgs.msg
from moveit_commander import PlanningSceneInterface
import moveit_msgs.msg
import osx_msgs.msg
import ur_msgs.msg
import rospkg
from osx_robot_control.markers_scene import MarkersScene
from osx_robot_control.robot_base import RobotBase
import rospy
import tf2_ros
import yaml
from osx_robot_control import helpers
from osx_robot_control.dual_arm import DualArm
from osx_robot_control.helpers import *
from osx_robot_control.ur_robot import URRobot
from ur_gazebo.gazebo_spawner import GazeboModels


class OSXCore(object):
    """
    This class contains the basic helper and convenience functions used in the routines.
    The basic functions include the initialization of the services and actions,
    and shorthand functions for the most common actions.
    """

    def __init__(self):
        start_time = timeit.default_timer()

        self.listener = tf2_ros.TransformListener(tf2_ros.Buffer())

        # Status variables and settings
        self.use_real_robot = rospy.get_param("use_real_robot", False)
        self.use_gazebo_sim = rospy.get_param("use_gazebo_sim", False)
        if self.use_gazebo_sim:
            self.gazebo_scene = GazeboModels('osx_gazebo')

        # Miscellaneous helpers
        self.planning_scene_interface: PlanningSceneInterface = PlanningSceneInterface(synchronous=True)
        self.markers_scene = MarkersScene(self.listener)

        # Action clients and movegroups
        self.a_bot = URRobot("a_bot", self.listener, self.markers_scene)
        self.a_bot.activate_ros_control_on_ur()
        self.b_bot = URRobot("b_bot", self.listener, self.markers_scene)
        self.b_bot.activate_ros_control_on_ur()
        self.ab_bot = DualArm("ab_bot", self.a_bot, self.b_bot, self.listener)
        # For compatibility let's wrap the robots
        self.active_robots: dict[str, RobotBase] = {'a_bot': self.a_bot, 'b_bot': self.b_bot, 'ab_bot': self.ab_bot}

        for robot in ["a_bot", "b_bot"]:
            self.active_robots[robot].get_status_from_param_server()

        rospy.logerr(f"Core loading time: {timeit.default_timer()-start_time:.2f} secs")

    @check_for_real_robot
    def activate_led(self, LED_name="b_bot", on=True):
        if self.use_gazebo_sim:
            return True
        request = ur_msgs.srv.SetIORequest()
        request.fun = ur_msgs.srv.SetIORequest.FUN_SET_DIGITAL_OUT
        request.pin = 4
        if on:
            request.state = ur_msgs.srv.SetIORequest.STATE_ON
        else:
            request.state = ur_msgs.srv.SetIORequest.STATE_OFF

        if LED_name == "b_bot":
            return self.b_bot.set_io.call(request)
        elif LED_name == "a_bot":
            return self.a_bot.set_io.call(request)
        else:
            rospy.logerr("Invalid LED name")

    def allow_collisions_with_robot_hand(self, link_name, robot_name, allow=True):
        """Allow collisions of a link with the robot hand"""
        hand_links = [
            robot_name + "_tip_link",
            robot_name + "_left_inner_finger_pad",
            robot_name + "_right_inner_finger_pad",
            robot_name + "_left_inner_finger",
            robot_name + "_right_inner_finger",
            robot_name + "_left_inner_knuckle",
            robot_name + "_right_inner_knuckle",
            robot_name + "_left_outer_finger",
            robot_name + "_right_outer_finger",
        ]
        if allow:
            self.planning_scene_interface.allow_collisions(hand_links, link_name)
        else:
            self.planning_scene_interface.disallow_collisions(hand_links, link_name)
        return

    def disable_scene_object_collisions(self):
        """ Disables collisions between all world objects (except tools) and everything else.
            Used because our meshes are so heavy that they impact performance too much.
        """
        object_names = self.planning_scene_interface.get_known_object_names()
        rospy.loginfo("Disabling collisions for all scene objects (except tools).")
        objects_without_tools = []
        for n in object_names:
            if not "tool" in n:
                objects_without_tools.append(n)
        print(objects_without_tools)
        self.planning_scene_interface.allow_collisions(objects_without_tools, "")

    def confirm_to_proceed(self, next_task_name):
        # Ignore during simultaneous motions
        rospy.loginfo("Press enter to proceed to: " + next_task_name)
        i = input()
        if i == "":
            if not rospy.is_shutdown():
                return True
        raise Exception("User caused exit!")

    def publish_robot_status(self):
        for robot in ["a_bot", "b_bot"]:
            self.active_robots[robot].publish_robot_status()

    def reset_scene_and_robots(self):
        """ Also see reset_assembly_visualization in common.py """
        self.a_bot.robot_status = osx_msgs.msg.RobotStatus()
        self.b_bot.robot_status = osx_msgs.msg.RobotStatus()
        self.planning_scene_interface.remove_attached_object()  # Detach objects
        rospy.sleep(0.5)  # Wait half a second for the detach to finish so that we can remove the object
        self.planning_scene_interface.remove_world_object()  # Clear all objects
        self.publish_robot_status()
        # self.reset_assembly_visualization()
        self.markers_scene.delete_all()

###########
# Sequences related methods
# A sequence is a collection of actions that can be:
# - Record and replay. Record sequences are stored in a file (./config/saved_plans).
# - Plan subsequent actions while the current one is being executed.
# - Sequences can be defined from a config file (./config/playback_sequences) or with code (see equip_unequip_realign_tool() for an example)
###########

    def read_playback_sequence(self, routine_filename, default_frame="world"):
        """
        default_frame: used for task-space-in-frame waypoints with undefined frame_id or where the value is specified as default, then the default_frame is used.
        """
        path = rospkg.RosPack().get_path("osx_assembly") + ("/config/playback_sequences/%s.yaml" % routine_filename)
        with open(path, 'r') as f:
            routine = yaml.safe_load(f)
        robot_name = routine["robot_name"]
        waypoints = routine["waypoints"]

        trajectory = []

        playback_trajectories = []

        for waypoint in waypoints:
            is_trajectory = waypoint.get('is_trajectory', False)

            if waypoint.get("frame_id", "default") == "default":
                waypoint.update({"frame_id": default_frame})

            if is_trajectory:
                pose = waypoint.get("pose", None)
                if waypoint["pose_type"] == "joint-space-goal-cartesian-lin-motion":
                    p = self.active_robots[robot_name].compute_fk(pose)  # Forward Kinematics
                elif waypoint["pose_type"] == "task-space-in-frame":
                    # Convert orientation to radians!
                    frame_id = waypoint.get("frame_id", "default")
                    p = conversions.to_pose_stamped(frame_id, np.concatenate([pose[:3], np.deg2rad(pose[3:])]))
                elif waypoint["pose_type"] == "master-slave":
                    p = waypoint  # forward all the info to construct the master-slave trajectory
                else:
                    raise ValueError("Unsupported trajectory for pose type: %s" % waypoint["pose_type"])
                blend = waypoint.get("blend", 0.0)
                trajectory_speed = waypoint.get("speed", 0.5)
                trajectory.append([p, blend, trajectory_speed])
            else:
                # Save any on-going trajectory
                if trajectory:
                    playback_trajectories.append(["trajectory", trajectory])
                    trajectory = []

                waypoint.update({"retime": True})
                playback_trajectories.append(["waypoint", waypoint])

        if trajectory:
            playback_trajectories.append(["trajectory", trajectory])

        return robot_name, playback_trajectories

    def execute_sequence(self, robot_name, sequence, sequence_name, end_effector_link="", plan_while_moving=True, save_on_success=False, use_saved_plans=False):
        if use_saved_plans:
            # TODO(cambel): check that the original plan's file has not been updated. if so, try to do the online planning
            bagfile = helpers.get_plan_full_path(sequence_name)
            if not os.path.exists(bagfile):
                rospy.logwarn("Attempted to execute saved sequence, but file not found: %s" % bagfile)
                rospy.logwarn("Try to execute sequence with online planning")
            else:
                rospy.logwarn("Executing saved sequence: " + bagfile)
                return self.execute_saved_sequence(sequence_name)

        robot = self.active_robots[robot_name]
        all_plans = []
        if not plan_while_moving:
            for i, point in enumerate(sequence):
                rospy.loginfo("Sequence point: %i - %s" % (i+1, point[0]))
                self.confirm_to_proceed("playback_sequence")
                if point[0] == "waypoint":
                    waypoint_params = point[1]
                    res = self.move_to_sequence_waypoint(robot_name, waypoint_params)
                elif point[0] == "trajectory":
                    trajectory = point[1]
                    res = robot.set_linear_eef_trajectory(trajectory)
                elif point[0] == "joint_trajectory":
                    trajectory = point[1]
                    res = robot.move_joints_trajectory(trajectory)
                if not res:
                    rospy.logerr("Fail to complete playback sequence: %s" % sequence_name)
                    return False
        else:
            rospy.loginfo("(plan_while_moving, %s) Sequence name: %s" % (robot_name, sequence_name))
            all_plans.append(robot_name)
            active_plan = None
            active_plan_start_time = rospy.Time(0)
            active_plan_duration = 0.0
            previous_plan = None
            backlog = []
            previous_plan_type = ""
            for i, point in enumerate(sequence):
                gripper_action = None

                rospy.logdebug("(plan_while_moving, %s) Sequence point: %i - %s" % (robot_name, i+1, point[0]))
                # self.confirm_to_proceed("playback_sequence")

                res = self.plan_waypoint(robot_name, point, previous_plan, end_effector_link=end_effector_link)  # res = plan, planning_time

                if not res:
                    rospy.logerr("%s: Fail to complete playback sequence: %s" % (robot_name, sequence_name))
                    return False

                if isinstance(res[0], dict):
                    gripper_action = copy.copy(res[0])
                    res = None, 0.0

                plan, _ = res

                # begin - For logging only
                if point[0] == "waypoint":
                    previous_plan_type = point[1]["pose_type"]
                else:
                    previous_plan_type = point[0]
                # end

                if save_on_success:
                    if not gripper_action:
                        if i == 0:  # Just save the target joint configuration
                            all_plans.append(helpers.get_trajectory_joint_goal(plan))
                        else:  # otherwise save the computed plan
                            all_plans.append(plan)
                    else:  # or the gripper action
                        all_plans.append(gripper_action)

                if gripper_action:
                    backlog.append((gripper_action, (i+1), previous_plan_type))
                    continue

                backlog.append((plan, (i+1), previous_plan_type))
                previous_plan = plan

                if active_plan:
                    execution_time = (rospy.Time.now() - active_plan_start_time).secs
                    remaining_time = execution_time - active_plan_duration
                    if remaining_time < 0.1:  # Not enough time for queue up another plan; wait for execution to complete
                        if not robot.robot_group.wait_for_motion_result():  # wait for motion max of 45 seconds
                            rospy.logerr("%s: MoveIt aborted the motion" % robot_name)
                        rospy.logdebug("%s: Waited for motion result" % robot_name)
                    else:
                        # Try planning another point
                        continue

                    if not self.check_plan_goal_reached(robot_name, active_plan):
                        rospy.logerr("%s: Failed to execute sequence plan: target pose not reached" % robot_name)
                        return False

                # execute next plan
                next_plan, index, plan_type = backlog.pop(0)
                rospy.logdebug(robot_name + ": Executing sequence plan: index, " + str(index) + " type " + str(plan_type))
                wait = True if i == len(sequence) - 1 else False
                self.execute_waypoint_plan(robot_name, next_plan, wait=wait)

                if isinstance(next_plan, dict):  # Gripper action
                    active_plan = None
                    active_plan_start_time = rospy.Time(0)
                    active_plan_duration = 0.0
                elif isinstance(next_plan, moveit_msgs.msg.RobotTrajectory):
                    active_plan = next_plan
                    active_plan_start_time = rospy.Time.now()
                    active_plan_duration = helpers.get_trajectory_duration(next_plan)

            # wait for last executed motion
            robot.robot_group.wait_for_motion_result()  # wait for motion max of 45 seconds
            # Finished preplanning the whole sequence: Execute remaining waypoints
            while backlog:
                if active_plan and not self.check_plan_goal_reached(robot_name, active_plan):
                    rospy.logerr("%s: Fail to execute plan: target pose not reach" % robot_name)
                    return False

                next_plan, index, plan_type = backlog.pop(0)
                rospy.logdebug(robot_name + ": Executing plan (backlog loop): index, " + str(index) + " type " + str(plan_type))
                self.execute_waypoint_plan(robot_name, next_plan, True)
                if isinstance(next_plan, (moveit_msgs.msg.RobotTrajectory)):
                    active_plan = next_plan

        if plan_while_moving and save_on_success:
            helpers.save_sequence_plans(name=sequence_name, plans=all_plans)

        return True

    def check_plan_goal_reached(self, robot_name, plan):
        current_joints = self.active_robots[robot_name].robot_group.get_current_joint_values()
        plan_goal = helpers.get_trajectory_joint_goal(plan, self.active_robots[robot_name].robot_group.get_active_joints())
        return helpers.all_close(plan_goal, current_joints, 0.01)

    def execute_waypoint_plan(self, robot_name, plan, wait):
        if isinstance(plan, dict):  # gripper action
            if not self.execute_gripper_action(robot_name, plan):
                return False
        elif isinstance(plan, moveit_msgs.msg.RobotTrajectory):
            if not self.active_robots[robot_name].execute_plan(plan, wait=wait):
                rospy.logerr("plan execution failed")
                return False
        return True

    def plan_waypoint(self, robot_name, point, previous_plan, end_effector_link=""):
        """
          Plan a point defined in a dict
          a point can be of type `waypoint` or `trajectory`
          a `waypoint` can be a robot motion (joints, cartesian, linear, relative...) or a gripper action
          a `trajectory` can be for a single robot or for a master-slave relationship
          previous_plan: defines the initial state for the new `point`
          end_effector_link: defines the end effector for the plan. FIXME: so far, it is only used for move_lin_trajectory 
        """
        initial_joints = self.active_robots[robot_name].robot_group.get_current_joint_values() if not previous_plan else helpers.get_trajectory_joint_goal(previous_plan,
                                                                                                                                                           self.active_robots[robot_name].robot_group.get_active_joints())
        if point[0] == "waypoint":
            rospy.loginfo("Sequence point type: %s > %s" % (point[1]["pose_type"], point[1].get("desc", '')))
            waypoint_params = point[1]
            gripper_action = waypoint_params["pose_type"] == 'gripper'
            if gripper_action:
                return waypoint_params["gripper"], 0.0
            else:
                if waypoint_params["pose_type"] == 'master-slave':
                    joint_list = self.active_robots[waypoint_params["master_name"]].robot_group.get_active_joints() + self.active_robots[waypoint_params["slave_name"]].robot_group.get_active_joints()
                    initial_joints_ = None if not previous_plan else helpers.get_trajectory_joint_goal(previous_plan, joint_list)
                else:
                    initial_joints_ = initial_joints
                return self.move_to_sequence_waypoint(robot_name, waypoint_params, plan_only=True, initial_joints=initial_joints_)
        elif point[0] == "trajectory":
            trajectory = point[1]
            if isinstance(trajectory[0][0], geometry_msgs.msg.PoseStamped):
                return self.active_robots[robot_name].set_linear_eef_trajectory(trajectory, plan_only=True, initial_joints=initial_joints, end_effector_link=end_effector_link)
            elif isinstance(trajectory[0][0], dict):
                joint_list = self.active_robots[trajectory[0][0]["master_name"]].robot_group.get_active_joints() + self.active_robots[trajectory[0][0]["slave_name"]].robot_group.get_active_joints()
                initial_joints_ = None if not previous_plan else helpers.get_trajectory_joint_goal(previous_plan, joint_list)
                return self.master_slave_trajectory(robot_name, trajectory, plan_only=True, initial_joints=initial_joints_)
            else:
                rospy.logerr("Trajectory: %s" % trajectory)
                raise ValueError("Invalid trajectory type %s" % type(trajectory[0][0]))
        elif point[0] == "joint_trajectory":
            eef = point[2] if len(point) > 2 else ""  # optional end effector
            # TODO(cambel): support master-slave trajectories
            return self.active_robots[robot_name].move_joints_trajectory(point[1], plan_only=True, initial_joints=initial_joints, end_effector_link=eef)
        else:
            raise ValueError("Invalid sequence type: %s" % point[0])

    def playback_sequence(self, routine_filename, default_frame="world", plan_while_moving=True, save_on_success=True, use_saved_plans=True):
        # TODO(felixvd): Remove this after the bearing procedure is fixed.
        if routine_filename == "bearing_orient_b_bot":
            rospy.logwarn("FIXME: Allow collision between b_bot_cam_cables_link and tray")
            self.planning_scene_interface.allow_collisions("b_bot_cam_cables_link", "tray")
            rospy.sleep(1.0)
        robot_name, playback_trajectories = self.read_playback_sequence(routine_filename, default_frame)

        return self.execute_sequence(robot_name, playback_trajectories, routine_filename, plan_while_moving=plan_while_moving, save_on_success=save_on_success, use_saved_plans=use_saved_plans)

    def execute_gripper_action(self, robot_name, gripper_params):
        gripper_action = gripper_params.get("action", None)
        opening_width = gripper_params.get("open_width", 0.140)
        force = gripper_params.get("force", 80.)
        velocity = gripper_params.get("velocity", 0.03)
        wait = gripper_params.get("wait", True)

        pre_operation_callback = gripper_params.get("pre_callback", None)
        post_operation_callback = gripper_params.get("post_callback", None)

        if pre_operation_callback:
            pre_operation_callback()

        success = False
        if robot_name == "ab_bot":
            if gripper_action == 'open':
                success = self.a_bot.gripper.open(opening_width=opening_width, velocity=velocity, wait=wait)
                success &= self.b_bot.gripper.open(opening_width=opening_width, velocity=velocity, wait=wait)
            elif gripper_action == 'close':
                success = self.a_bot.gripper.close(force=force, velocity=velocity, wait=wait)
                success &= self.b_bot.gripper.close(force=force, velocity=velocity, wait=wait)
            elif isinstance(gripper_action, float):
                success = self.a_bot.gripper.send_command(gripper_action, force=force, velocity=velocity, wait=wait)
                success &= self.b_bot.gripper.send_command(gripper_action, force=force, velocity=velocity, wait=wait)
            else:
                raise ValueError("Unsupported gripper action: %s of type %s" % (gripper_action, type(gripper_action)))
        else:
            robot = self.active_robots[robot_name]
            if gripper_action == 'open':
                success = robot.gripper.open(opening_width=opening_width, velocity=velocity, wait=wait)
            elif gripper_action == 'close':
                success = robot.gripper.close(force=force, velocity=velocity, wait=wait)
            elif gripper_action == 'close-open':
                robot.gripper.close(velocity=velocity, force=force, wait=wait)
                success = robot.gripper.open(opening_width=opening_width, velocity=velocity, wait=wait)
            elif isinstance(gripper_action, float):
                success = robot.gripper.send_command(gripper_action, force=force, velocity=velocity, wait=wait)
            else:
                raise ValueError("Unsupported gripper action: %s of type %s" % (gripper_action, type(gripper_action)))

        if post_operation_callback:  # Assume gripper always succeeds
            post_operation_callback()
        return success

    def move_to_sequence_waypoint(self, robot_name, params, plan_only=False, initial_joints=None):
        success = False
        robot = self.active_robots[robot_name]

        pose = params.get("pose", None)
        pose_type = params["pose_type"]
        speed = params.get("speed", 0.5)
        acceleration = params.get("acc", speed/2.)
        gripper_params = params.get("gripper", None)
        end_effector_link = params.get("end_effector_link", None)
        retime = params.get("retime", False)

        if pose_type == 'joint-space':
            success = robot.set_joint_position_goal(pose, speed=speed, acceleration=acceleration, plan_only=plan_only, initial_joints=initial_joints)
        elif pose_type == 'joint-space-goal-cartesian-lin-motion':
            p = robot.compute_fk(pose)  # Forward Kinematics
            success = robot.set_pose_goal(p, speed=speed, acceleration=acceleration, plan_only=plan_only, initial_joints=initial_joints, end_effector_link=end_effector_link, move_lin=True)
        elif pose_type == 'task-space-in-frame':
            frame_id = params.get("frame_id", "world")
            # Convert orientation to radians!
            if robot_name == "ab_bot":
                a_bot_pose = conversions.to_pose_stamped(frame_id, pose)
                b_bot_pose = conversions.to_pose_stamped(frame_id, params["pose2"])
                planner = params.get("planner", "LINEAR")
                success = self.ab_bot.go_to_goal_poses(a_bot_pose, b_bot_pose, plan_only=plan_only, initial_joints=initial_joints, planner=planner)
            else:
                p = conversions.to_pose_stamped(frame_id, np.concatenate([pose[:3], np.deg2rad(pose[3:])]))
                move_linear = params.get("move_linear", True)
                success = robot.set_pose_goal(p, speed=speed, acceleration=acceleration, move_lin=move_linear, plan_only=plan_only,
                                              initial_joints=initial_joints, end_effector_link=end_effector_link)
        elif pose_type == 'relative-tcp':
            success = robot.set_relative_motion_goal(relative_translation=pose[:3], relative_rotation=np.deg2rad(pose[3:]), speed=speed, acceleration=acceleration,
                                                     relative_to_tcp=True, plan_only=plan_only, initial_joints=initial_joints, end_effector_link=end_effector_link)
        elif pose_type == 'relative-world':
            success = robot.set_relative_motion_goal(relative_translation=pose[:3], relative_rotation=np.deg2rad(pose[3:]), speed=speed,
                                                     acceleration=acceleration, plan_only=plan_only, initial_joints=initial_joints)
        elif pose_type == 'relative-base':
            success = robot.set_relative_motion_goal(relative_translation=pose[:3], relative_rotation=np.deg2rad(pose[3:]), speed=speed,
                                                     acceleration=acceleration, relative_to_robot_base=True, plan_only=plan_only, initial_joints=initial_joints)
        elif pose_type == 'named-pose':
            success = robot.go_to_named_pose(pose, speed=speed, acceleration=acceleration, plan_only=plan_only, initial_joints=initial_joints)
        elif pose_type == 'master-slave':
            ps = conversions.to_pose_stamped(params["frame_id"], np.concatenate([pose[:3], np.deg2rad(pose[3:])]))
            success = robot.master_slave_control(params['master_name'], params['slave_name'], ps, params['slave_relation'], speed=speed, plan_only=plan_only, initial_joints=initial_joints)
        elif not plan_only and pose_type == 'gripper':
            success = self.execute_gripper_action(robot_name, gripper_params)
        else:
            raise ValueError("Invalid pose_type: %s" % pose_type)

        if plan_only and success and retime:
            rospy.loginfo("retiming waypoint")
            plan, planning_time = success
            plan = robot.robot_group.retime_trajectory(robot.robot_group.get_current_state(), plan, algorithm="time_optimal_trajectory_generation",
                                                       velocity_scaling_factor=speed, acceleration_scaling_factor=acceleration)
            return plan, planning_time

        return success

    def execute_saved_sequence(self, name):
        sequence = helpers.load_sequence_plans(name)
        robot_name = sequence[0]
        robot = self.active_robots[robot_name]

        robot.set_joint_position_goal(sequence[1])
        for seq in sequence[2:]:
            success = False
            if isinstance(seq, dict):
                success = self.execute_gripper_action(robot_name, seq)
            else:
                # TODO(cambel): validate that the plan is still valid before execution
                success = robot.execute_plan(seq, wait=True)

            if not success:
                rospy.logerr("Fail to execute saved plan from sequence. Abort")
                return False

        return True

    def master_slave_trajectory(self, robot_name, trajectory, plan_only=False, initial_joints=None):
        master_trajectory = []
        waypoints = [wp for wp, _, _ in trajectory]  # Ignoring blend and speed as the whole waypoints dict is forwarded in the first value
        for waypoint in waypoints:
            pose = waypoint["pose"]
            frame_id = waypoint["frame_id"]
            ps = conversions.to_pose_stamped(frame_id, np.concatenate([pose[:3], np.deg2rad(pose[3:])]))
            master_trajectory.append((ps, waypoint.get("blend", 0), waypoints[0].get("speed", 0.5)))  # poseStamped, blend, speed

        slave_initial_joints = initial_joints[6:] if initial_joints is not None else None
        master_initial_joints = initial_joints[:6] if initial_joints is not None else None

        master_plan, _ = self.active_robots[waypoints[0]["master_name"]].set_linear_eef_trajectory(
            master_trajectory, speed=waypoints[0].get("speed", 0.5), plan_only=True, initial_joints=master_initial_joints)

        master_slave_plan, planning_time = self.active_robots[robot_name].compute_master_slave_plan(waypoints[0]["master_name"],
                                                                                                    waypoints[0]["slave_name"],
                                                                                                    waypoints[0]["slave_relation"],
                                                                                                    slave_initial_joints,
                                                                                                    master_plan)

        if plan_only:
            return master_slave_plan, planning_time
        else:
            return self.active_robots[robot_name].execute_plan(master_slave_plan)
