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

"""
OSX Common Robot Control Module

This module provides common functionality for robot control operations,
including vision-assisted manipulation, object handling, and tool management.
It extends the OSXCore class with additional methods for common robot tasks.
"""

import copy
from math import radians, tau

import numpy as np
from osx_assembly_database.assembly_reader import AssemblyReader
from osx_robot_control.utils import transform_pose
import rospy
import rospkg
import yaml

import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from shape_msgs.msg import SolidPrimitive

from osx_robot_control.core import OSXCore
from osx_robot_control.helpers import get_direction_index, rotatePoseByRPY, rotateQuaternionByRPY, to_sequence_gripper, to_sequence_item
from osx_vision.vision_client import VisionClient

from ur_control import conversions
from ur_control.math_utils import quaternion_normalize


class OSXCommon(OSXCore):
    """
    Implementation of useful methods for manipulation with or without vision assistance.

    This class extends OSXCore with additional functionality for:
    - Vision-based object detection and pose estimation
    - Object manipulation (pick and place operations)
    - Tool management and collision object handling
    - Scene management for MoveIt
    """

    def __init__(self):
        """Initialize the OSXCommon class with vision client and other components."""
        self.pub_status_text = rospy.Publisher("/osx_text_to_image", String, queue_size=1)
        self.publish_status_text("Initializing ...")
        super(OSXCommon, self).__init__()

        # Vision components
        self.vision = VisionClient()
        self.use_dummy_vision = False

        self.objects_in_tray = dict()
        self.object_in_tray_dimensions = dict()

        # Assembly database
        self.assembly_database: AssemblyReader = None

        # Tool management
        self.tools = {}
        self.dummy_pose = [-0.06, -0.1, 0.0, 0.0, 0.0, tau/4]

        # Publish status text to image

    def publish_status_text(self, text):
        """ Publish a string to the status topic, which is then converted to an image and displayed in Rviz.
        """
        rospy.loginfo("Published status: " + text)
        self.pub_status_text.publish(String(text))

    # ===== VISION METHODS =====

    def define_local_tray_views(self, high_height=.385, low_height=.24,
                                robot_name="b_bot", include_rotated_views=False,
                                frame_id="tray_center", offsets=[.0, 0.0]):
        """
        Define the poses used to position the camera to look into the tray.

        Args:
            high_height (float): Height for the high view position
            low_height (float): Height for the low view position
            robot_name (str): Name of the robot ("a_bot" or "b_bot")
            include_rotated_views (bool): Whether to include rotated views
            frame_id (str): Reference frame for the poses
            offsets (list): [x_offset, y_offset] for corner views

        Returns:
            tuple: (tray_view_high, close_tray_views) where:
                - tray_view_high is a PoseStamped for the high view
                - close_tray_views is a list of PoseStamped for various close views

        Example:
            self.b_bot.go_to_pose_goal(self.tray_view_high,
                                      end_effector_link="a_bot_outside_camera_color_frame",
                                      speed=.1, acceleration=.04)
        """
        # Set offsets based on robot
        x_offset = offsets[0]
        y_offset = offsets[1]

        # Create base pose
        ps = geometry_msgs.msg.PoseStamped()
        ps.header.frame_id = frame_id
        if frame_id == "cutting_board_surface":
            orientation = [-0.630, 0.002, 0.777, 0.002]
        else:  # tray_center
            orientation = [0.003, 0.657, -0.013, 0.753]
        ps.pose.orientation = geometry_msgs.msg.Quaternion(*quaternion_normalize(orientation))
        ps.pose.position.y = -0.01 if robot_name == "b_bot" else 0
        ps.pose.position.x = 0 if robot_name == "b_bot" else 0.04
        ps.pose.position.z = high_height

        # Centered views (high up and close)
        tray_view_high = copy.deepcopy(ps)
        ps.pose.position.z = low_height
        tray_view_low = copy.deepcopy(ps)

        # Close views in corners
        ps.pose.position.x = x_offset
        ps.pose.position.y = y_offset
        tray_view_close_front_b = copy.deepcopy(ps)
        ps.pose.position.x = -x_offset
        ps.pose.position.y = y_offset
        tray_view_close_back_b = copy.deepcopy(ps)
        ps.pose.position.x = x_offset
        ps.pose.position.y = -y_offset
        tray_view_close_front_a = copy.deepcopy(ps)
        ps.pose.position.x = -x_offset
        ps.pose.position.y = -y_offset
        tray_view_close_back_a = copy.deepcopy(ps)

        # Combine all close views
        close_tray_views = [tray_view_low, tray_view_close_front_b, tray_view_close_back_b,
                            tray_view_close_front_a, tray_view_close_back_a]

        # Add rotated views if requested
        if include_rotated_views:
            rot_20 = [rotatePoseByRPY(radians(20), 0, 0, pose) for pose in close_tray_views]
            rot_n20 = [rotatePoseByRPY(radians(-20), 0, 0, pose) for pose in close_tray_views]
            rot_50 = [rotatePoseByRPY(radians(50), 0, 0, pose) for pose in close_tray_views]
            rot_90 = [rotatePoseByRPY(radians(90), 0, 0, pose) for pose in close_tray_views]
            close_tray_views += rot_20
            close_tray_views += rot_n20
            close_tray_views += rot_50
            close_tray_views += rot_90

        return tray_view_high, close_tray_views

    def get_3d_poses_from_vision_server(self, frame_id):
        """
        Get object poses as estimated by the vision server.
        """
        try:
            # Retry up to 5 times until any results are detected
            max_attempts = 2
            attempt = 0
            res = []

            while attempt < max_attempts and not res:
                attempt += 1
                rospy.loginfo(f"Vision detection attempt {attempt}/{max_attempts}")
                res = self.vision.read_from_vision_server(frame_id)
                if not res and attempt < max_attempts:
                    rospy.sleep(0.5)  # Short delay between attempts

            found_objects = []

            for pose3d in res:
                object_name = self.assembly_database.id_to_name(pose3d.class_id)
                found_objects.append(object_name)
                self.objects_in_tray[object_name] = {
                    "pose": pose3d.pose,
                    "width": pose3d.width,
                    "length": pose3d.length,
                    "confidence": pose3d.confidence,
                    "info": pose3d.info
                }

            return found_objects
        except Exception as e:
            rospy.logerr(f"Exception at get_3d_poses_from_vision_server: {e}")
            return False

    def look_and_get_object_pose(self, object_id, robot_name="b_bot", frame_id="tray_center", multiple_views=True, spawn_all_objects=False):
        """
        Look at the tray from above and get grasp points of items.

        This method positions the camera to view the tray and attempts to detect the specified object.
        It performs a light feasibility check before returning the object pose.

        Args:
            object_id (str): ID of the object to detect
            robot_name (str): Name of the robot to use for camera positioning
            frame_id (str): Reference frame for the poses
            multiple_views (bool): Whether to use multiple camera views for detection

        Returns:
            PoseStamped: Pose of the detected object or False if not found
        """
        # Activate camera and LED
        self.vision.activate_camera(robot_name + "_outside_camera")
        self.activate_led(robot_name)

        # Get object info from database
        object_id, object_name = self.assembly_database.get_object_info(object_id)

        # Clear previous detection for this object
        if object_name in self.objects_in_tray:
            del self.objects_in_tray[object_name]

        # Use dummy vision if in simulation mode
        if self.use_dummy_vision or (not self.use_real_robot and not self.use_gazebo_sim):
            rospy.logwarn("Using dummy vision! Setting object pose to tray center.")
            self.objects_in_tray[object_name] = {
                "pose": conversions.to_pose_stamped(frame_id, self.dummy_pose),
                "width": 0.0, "length": 0.0, "confidence": 0.0, "info": ""
            }
            return self.objects_in_tray[object_name]["pose"]

        # Define camera views
        tray_view_high, close_tray_views = self.define_local_tray_views(
            robot_name=robot_name,
            include_rotated_views=multiple_views,
            high_height=0.45,
            low_height=0.45,
            frame_id=frame_id,
            offsets=[0.02, 0.02]
        )

        # Adjust for cutting board if needed
        if 'cutting_board' in frame_id:
            tray_view_high.pose.position.x -= 0.05
            tray_view_high.pose.position.z -= 0.15
            tray_view_high.pose.position.y -= 0.02
        # Select which views to use
        if multiple_views:
            tray_views = [tray_view_high] + close_tray_views
        else:
            tray_views = [tray_view_high]

        # Try each view until object is found
        for view_pose in tray_views:
            assert not rospy.is_shutdown()

            # Move robot to view position
            self.active_robots[robot_name].go_to_pose_goal(view_pose,
                                                           end_effector_link=robot_name + "_outside_camera_color_frame",
                                                           speed=.5, acceleration=.3, wait=True, move_lin=True)
            rospy.sleep(0.5)  # wait for robot to stop moving for cleaner vision

            # Try detection multiple times
            found_objects = self.get_3d_poses_from_vision_server(frame_id)
            print(f"Found {len(found_objects)} objects: {found_objects}")
            if frame_id != "tray_center":
                for obj in found_objects:
                    self.objects_in_tray[obj]["pose"] = transform_pose(self.listener.buffer, frame_id, self.objects_in_tray[obj]["pose"])
                    if "slice" not in obj:
                        self.objects_in_tray[obj]["pose"].pose.position.x += -0.01
                        self.objects_in_tray[obj]["pose"].pose.position.y += -0.01
                    else:
                        self.objects_in_tray[obj]["pose"].pose.position.x += 0.01
                        self.objects_in_tray[obj]["pose"].pose.position.y += 0.01

            if spawn_all_objects:
                for obj in found_objects:
                    if 'cutting_board' in frame_id:
                        self.objects_in_tray[obj]["pose"].pose.position.z = 0.005
                    else:
                        self.objects_in_tray[obj]["pose"].pose.position.z = 0.001
                    self.spawn_object(obj, self.objects_in_tray[obj]["pose"])
            else:
                if object_name in found_objects:
                    if 'cutting_board' in frame_id:
                        self.objects_in_tray[object_name]["pose"].pose.position.z = 0.005
                    else:
                        self.objects_in_tray[object_name]["pose"].pose.position.z = 0.001
                    self.spawn_object(object_name, self.objects_in_tray[object_name]["pose"])

            # Check if object was found
            object_pose = copy.deepcopy(self.objects_in_tray.get(object_name, None))
            if object_pose:
                return object_pose['pose']

        rospy.logerr(f"Could not find item id {object_id} in tray!")
        return False

    def visualize_object(self, object_id, robot_name="a_bot"):
        """
        Visualize an object in the scene based on its detected pose and dimensions.

        Args:
            object_id (str): ID of the object to visualize
            robot_name (str): Name of the robot to use for detection

        Returns:
            bool: True if visualization was successful, False otherwise
        """
        obj_id = self.get_object_id(object_id)
        obj_name = self.assembly_database.id_to_name(obj_id)

        # Get object pose
        if not self.look_and_get_object_pose(object_id=obj_id, robot_name=robot_name,
                                             frame_id="tray_center", multiple_views=False):
            return False

        # Get object dimensions
        dims = self.object_in_tray_dimensions.get(obj_id)  # width, length
        if not dims:
            return False

        width, length = dims
        rospy.logdebug(f"Object dimensions - width: {width}, length: {length}")

        # Special handling for cucumber objects
        if obj_name == "cucumber":
            z_scale = length / 0.192 * 0.001
            xy_scale = width / 0.03 * 0.001
            rospy.logdebug(f"Scaling cucumber with factors: xy={xy_scale}, z={z_scale}")
            self.spawn_object(obj_name, self.objects_in_tray.get(obj_id), "tray_center",
                              scale=(xy_scale, xy_scale, z_scale))
        else:
            self.visionary_load_object(self.objects_in_tray)

        return True

    # ===== MOVEIT SCENE METHODS =====

    def spawn_object(self, object_name, object_pose, object_reference_frame="", scale=(0.001, 0.001, 0.001), alias=""):
        """
        Add an object to the MoveIt planning scene.

        Args:
            object_name (str): Name of the object to spawn
            object_pose: Pose of the object (PoseStamped, Pose, or list/numpy array)
            object_reference_frame (str): Reference frame for the pose if not provided in PoseStamped
            scale (tuple): Scale factors for the object (x, y, z)
            alias (str): Alternative name for the object in the planning scene
        """
        # Remove existing object with same name
        if object_name in self.planning_scene_interface.get_known_object_names():
            self.despawn_object(object_name)

        # Get collision object from database
        collision_object = self.assembly_database.get_collision_object(object_name, scale=scale)

        # Handle different pose types
        if isinstance(object_pose, geometry_msgs.msg._PoseStamped.PoseStamped):
            co_pose = object_pose.pose
            collision_object.header.frame_id = object_pose.header.frame_id
        elif isinstance(object_pose, geometry_msgs.msg._Pose.Pose):
            co_pose = object_pose
        elif isinstance(object_pose, list) or isinstance(object_pose, np.ndarray):
            co_pose = conversions.to_pose(object_pose)
        else:
            raise ValueError(f"Unsupported pose type: {type(object_pose)}")

        # Set reference frame if not provided in PoseStamped
        if not isinstance(object_pose, geometry_msgs.msg._PoseStamped.PoseStamped):
            if not object_reference_frame:
                raise ValueError("object_reference_frame is required when providing a pose type Pose or List")
            else:
                collision_object.header.frame_id = object_reference_frame

        # Set pose and ID
        collision_object.pose = co_pose
        if alias:
            collision_object.id = alias

        # Add to planning scene
        self.planning_scene_interface.add_object(collision_object)

    def despawn_object(self, object_name, collisions_only=False):
        """
        Remove an object from the MoveIt planning scene.

        Args:
            object_name (str): Name of the object to remove
            collisions_only (bool): If True, only remove collision objects, not visual markers
        """
        # Remove only if it exists
        if object_name not in self.planning_scene_interface.get_known_object_names():
            return

        # Remove from planning scene
        self.planning_scene_interface.remove_attached_object(name=object_name)
        rospy.sleep(0.5)  # Wait for detach to complete
        self.planning_scene_interface.remove_world_object(object_name)

        # Remove visual markers if needed
        if not collisions_only:
            self.markers_scene.detach_item(object_name)
            self.markers_scene.despawn_item(object_name)

    def define_tool_collision_objects(self):
        """
        Load and define collision objects for tools from configuration file.

        This method reads tool definitions from a YAML file and creates
        collision objects for each tool in the planning scene.
        """
        # Define primitive types
        PRIMITIVES = {"BOX": SolidPrimitive.BOX, "CYLINDER": SolidPrimitive.CYLINDER, "CONE": SolidPrimitive.CONE}

        # Load tool definitions from YAML
        path = rospkg.RosPack().get_path("osx_cooking") + "/config/tool_collision_objects.yaml"
        with open(path, 'r') as f:
            tools = yaml.safe_load(f)

        # Create collision objects for each tool
        for tool_key, tool in tools.items():
            tool_co = moveit_msgs.msg.CollisionObject()
            tool_co.header.frame_id = tool["frame_id"]
            tool_co.id = tool["id"]

            # Set tool pose
            toolpose = tool.get("pose", [0, 0, 0, 0, 0, 0])
            tool_co.pose = conversions.to_pose(conversions.to_float(toolpose))

            # Define collision geometry
            primitive_num = len(tool['primitives'])
            tool_co.primitives = [SolidPrimitive() for _ in range(primitive_num)]
            tool_co.primitive_poses = [geometry_msgs.msg.Pose() for _ in range(primitive_num)]

            # Set primitive properties
            for i, primitive in enumerate(tool["primitives"]):
                try:
                    tool_co.primitives[i].type = PRIMITIVES[(primitive['type'])]
                except KeyError as e:
                    rospy.logerr(f"Invalid Collision Object Primitive type: {primitive['type']}")
                    raise
                tool_co.primitives[i].dimensions = primitive['dimensions']
                tool_co.primitive_poses[i] = conversions.to_pose(conversions.to_float(primitive['pose']))

            # Set operation and subframes
            tool_co.operation = tool_co.ADD
            tool_co.subframe_poses = [conversions.to_pose(conversions.to_float(tool["subframe"]["pose"]))]
            tool_co.subframe_names = [tool["subframe"]["name"]]

            # Store tool
            self.tools[tool["id"]] = tool_co

    def spawn_tool(self, tool_name):
        """
        Add a tool to the MoveIt planning scene.

        Args:
            tool_name (str): Name of the tool to spawn

        Returns:
            bool: True if tool was spawned successfully, False otherwise
        """
        # Remove existing tool with same name
        if tool_name in self.planning_scene_interface.get_known_object_names():
            self.despawn_tool(tool_name)

        # Add tool if it exists
        if tool_name in self.tools:
            rospy.loginfo(f"Spawn: {tool_name}")
            self.planning_scene_interface.add_object(self.tools[tool_name])
            return True
        else:
            rospy.logerr(f"Cannot spawn tool: {tool_name} because it has not been loaded")
            return False

    def despawn_tool(self, tool_name):
        """
        Remove a tool from the MoveIt planning scene.

        Args:
            tool_name (str): Name of the tool to remove

        Returns:
            bool: True if tool was removed successfully, False otherwise
        """
        if tool_name in self.tools:
            rospy.loginfo(f"Despawn: {tool_name}")
            self.planning_scene_interface.remove_attached_object(name=self.tools[tool_name].id)
            self.planning_scene_interface.remove_world_object(self.tools[tool_name].id)
            return True
        else:
            rospy.logerr(f"Cannot despawn tool: {tool_name} because it has not been loaded")
            return False

    # ===== MANIPULATION METHODS =====

    def get_transformed_grasp_pose(self, object_name, grasp_name, target_frame="tray_center", alias=""):
        """
        Get an object's grasp pose in the target frame.

        Args:
            object_name (str): Name of the object
            grasp_name (str): Name of the grasp pose
            target_frame (str): Target reference frame
            alias (str): Alternative name for the object

        Returns:
            PoseStamped: Transformed grasp pose
        """
        grasp_pose = self.assembly_database.get_grasp_pose(object_name, grasp_name)
        grasp_pose.header.frame_id = "move_group/" + object_name
        if alias:
            return self.get_transformed_collision_object_pose(alias, grasp_pose, target_frame)
        else:
            return self.get_transformed_collision_object_pose(object_name, grasp_pose, target_frame)

    def get_transformed_collision_object_pose(self, object_name, object_pose=None, target_frame="tray_center"):
        """
        Get the pose of a MoveIt CollisionObject in the target frame.

        Args:
            object_name (str): Name of the object
            object_pose: Pose of the object (optional)
            target_frame (str): Target reference frame

        Returns:
            PoseStamped: Transformed pose or False if transformation failed
        """
        # Create default pose if none provided
        obj_pose = object_pose if object_pose else conversions.to_pose_stamped(
            "move_group/" + object_name, [0, 0, 0, 0, 0, 0]
        )

        return transform_pose(self.listener.buffer, target_frame, obj_pose)

    def simple_pick(self, robot_name, object_pose, grasp_height=0.0, speed_fast=1.0, speed_slow=0.4,
                    gripper_command="close", gripper_force=40.0, grasp_width=0.140,
                    minimum_grasp_width=0.0, maximum_grasp_width=1.0, approach_height=0.05,
                    item_id_to_attach="", lift_up_after_pick=True, acc_slow=.1,
                    gripper_velocity=.1, axis="x", sign=+1, retreat_height=None,
                    approach_with_move_lin=False, attach_with_collisions=False,
                    allow_collision_with_tray=False):
        """
        Perform a simple pick operation.

        This method executes a sequence of movements to pick an object:
        1. Approach the object from above
        2. Move down to grasp position
        3. Close gripper
        4. Lift up (optional)

        Args:
            robot_name (str): Name of the robot to use
            object_pose (PoseStamped): Pose of the object to pick
            grasp_height (float): Height offset for grasp position
            speed_fast (float): Speed for fast movements
            speed_slow (float): Speed for slow movements
            gripper_command (str): Command for gripper ("close" or "do_nothing")
            gripper_force (float): Force to apply when closing gripper
            grasp_width (float): Width to open gripper for grasping
            minimum_grasp_width (float): Minimum acceptable gripper width after grasp
            maximum_grasp_width (float): Maximum acceptable gripper width after grasp
            approach_height (float): Height for approach position
            item_id_to_attach (str): ID of item to attach to robot after grasp
            lift_up_after_pick (bool): Whether to lift up after picking
            acc_slow (float): Acceleration for slow movements
            gripper_velocity (float): Velocity for gripper movements
            axis (str): Axis for approach/retreat ("x", "y", or "z")
            sign (int): Direction sign (+1 or -1)
            retreat_height (float): Height for retreat position (defaults to approach_height)
            approach_with_move_lin (bool): Whether to use linear movement for approach
            attach_with_collisions (bool): Whether to attach with collisions
            allow_collision_with_tray (bool): Whether to allow collisions with tray

        Returns:
            bool: True if pick was successful, False otherwise
        """
        # Allow collisions if needed
        if allow_collision_with_tray:
            self.allow_collisions_with_robot_hand("tray", robot_name)

        rospy.loginfo("Entered simple_pick")

        if item_id_to_attach:
            self.allow_collisions_with_robot_hand(item_id_to_attach, robot_name)

        # Create sequence
        seq = []
        robot = self.active_robots[robot_name]

        # Open gripper if needed
        if gripper_command != "do_nothing":
            seq.append(to_sequence_gripper("open", gripper_opening_width=grasp_width,
                                           gripper_velocity=1.0, wait=False))

        # Create approach pose
        approach_pose = copy.deepcopy(object_pose)
        op = conversions.from_point(object_pose.pose.position)
        op[get_direction_index(axis)] += approach_height * sign
        approach_pose.pose.position = conversions.to_point(op)

        # Add approach movement to sequence
        seq.append(to_sequence_item(approach_pose, speed=speed_fast, acc=0.7,
                                    linear=approach_with_move_lin))

        rospy.logdebug(f"Going to height {op[get_direction_index(axis)]}")

        # Create grasp pose
        rospy.logdebug("Moving down to object")
        grasp_pose = copy.deepcopy(object_pose)
        op = conversions.from_point(object_pose.pose.position)
        op[get_direction_index(axis)] = max(op[get_direction_index(axis)] + grasp_height * sign, 0.0)
        rospy.loginfo(f"actual grasp height = {op[get_direction_index(axis)]}")
        grasp_pose.pose.position = conversions.to_point(op)
        rospy.logdebug(f"Going to height {op[get_direction_index(axis)]}")

        # Add grasp movement to sequence
        seq.append(to_sequence_item(grasp_pose, speed=speed_fast))

        # Close gripper if needed
        if gripper_command != "do_nothing":
            def post_cb():
                if item_id_to_attach:
                    robot.gripper.attach_object(object_to_attach=item_id_to_attach,
                                                with_collisions=attach_with_collisions)
            seq.append(to_sequence_gripper("close", gripper_velocity=gripper_velocity,
                                           gripper_force=gripper_force, post_callback=post_cb))

        # Execute sequence
        if not self.execute_sequence(robot_name, seq, "simple_pick"):
            rospy.logerr("Fail to simple pick with sequence")
            if allow_collision_with_tray:
                self.allow_collisions_with_robot_hand("tray", robot_name, False)
            return False

        # Check grasp success
        success = True
        if minimum_grasp_width > robot.gripper.opening_width and self.use_real_robot:
            rospy.logerr(f"Gripper opening width after pick less than minimum ({minimum_grasp_width}): {robot.gripper.opening_width}. Return False.")
            robot.gripper.open(opening_width=grasp_width)
            robot.gripper.forget_attached_item()
            if allow_collision_with_tray:
                self.allow_collisions_with_robot_hand("tray", robot_name, False)
            success = False

        if maximum_grasp_width < robot.gripper.opening_width and self.use_real_robot:
            rospy.logerr(f"Gripper opening width after pick more than allowed ({maximum_grasp_width}): {robot.gripper.opening_width}. Return False.")
            robot.gripper.open(opening_width=grasp_width)
            robot.gripper.forget_attached_item()
            if allow_collision_with_tray:
                self.allow_collisions_with_robot_hand("tray", robot_name, False)
            success = False

        # Lift up if needed
        if lift_up_after_pick:
            rospy.logdebug("Going back up")
            if retreat_height is None:
                retreat_height = approach_height

            retreat_pose = copy.deepcopy(object_pose)
            op = conversions.from_point(object_pose.pose.position)
            op[get_direction_index(axis)] += retreat_height * sign
            retreat_pose.pose.position = conversions.to_point(op)

            rospy.logdebug(f"Going to height {retreat_pose.pose.position.z}")

            if not robot.go_to_pose_goal(retreat_pose, speed=speed_slow, acceleration=acc_slow, move_lin=True):
                rospy.logerr("Fail to go to lift_up_pose. Opening.")
                robot.gripper.open(grasp_width)
                if allow_collision_with_tray:
                    self.allow_collisions_with_robot_hand("tray", robot_name, False)
                return False

        # Disallow collisions if needed
        if allow_collision_with_tray:
            self.allow_collisions_with_robot_hand("tray", robot_name, False)

        return success

    def simple_place(self, robot_name, object_pose, place_height=0.05, speed_fast=1.0, speed_slow=0.3,
                     gripper_command="open", gripper_opening_width=0.14, approach_height=0.05,
                     axis="x", sign=+1, item_id_to_detach="", lift_up_after_place=True,
                     acc_fast=0.6, acc_slow=0.15, move_lin=False):
        """
        Perform a simple place operation.

        This method executes a sequence of movements to place an object:
        1. Approach the place position from above
        2. Move down to place position
        3. Open gripper
        4. Lift up (optional)

        Args:
            robot_name (str): Name of the robot to use
            object_pose (PoseStamped): Pose where to place the object
            place_height (float): Height offset for place position
            speed_fast (float): Speed for fast movements
            speed_slow (float): Speed for slow movements
            gripper_command (str): Command for gripper ("open" or "do_nothing")
            gripper_opening_width (float): Width to open gripper for releasing
            approach_height (float): Height for approach position
            axis (str): Axis for approach/retreat ("x", "y", or "z")
            sign (int): Direction sign (+1 or -1)
            item_id_to_detach (str): ID of item to detach from robot after place
            lift_up_after_place (bool): Whether to lift up after placing
            acc_fast (float): Acceleration for fast movements
            acc_slow (float): Acceleration for slow movements
            move_lin (bool): Whether to use linear movement

        Returns:
            bool: True if place was successful, False otherwise
        """
        # Create sequence
        seq = []

        # Create approach pose
        rospy.loginfo("Going above place target")
        approach_pose = copy.deepcopy(object_pose)
        op = conversions.from_point(object_pose.pose.position)
        op[get_direction_index(axis)] += approach_height * sign
        approach_pose.pose.position = conversions.to_point(op)
        seq.append(to_sequence_item(approach_pose, speed=speed_fast, acc=acc_fast, linear=move_lin))

        # Create place pose
        rospy.loginfo("Moving to place target")
        place_pose = copy.deepcopy(object_pose)
        op = conversions.from_point(object_pose.pose.position)
        op[get_direction_index(axis)] += place_height * sign
        place_pose.pose.position = conversions.to_point(op)
        seq.append(to_sequence_item(place_pose, speed=speed_slow, acc=acc_slow, linear=move_lin))

        # Get robot reference
        robot = self.active_robots[robot_name]

        # Open gripper if needed
        if gripper_command != "do_nothing":
            seq.append(to_sequence_gripper('open', gripper_velocity=1.0,
                                           gripper_opening_width=gripper_opening_width))

        # Detach object if needed
        if item_id_to_detach:
            robot.robot_group.detach_object(item_id_to_detach)

        # Lift up if needed
        if lift_up_after_place:
            rospy.loginfo("Moving back up")
            seq.append(to_sequence_item(approach_pose, speed=speed_fast, acc=acc_fast, linear=move_lin))

        # Execute sequence
        if not self.execute_sequence(robot_name, seq, "simple place"):
            rospy.logerr("fail to simple place")
            return False

        return True
