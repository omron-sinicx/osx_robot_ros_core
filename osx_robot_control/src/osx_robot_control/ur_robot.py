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

import moveit_commander
from osx_robot_control.robot_base import RobotBase
import rospy
import time

import osx_msgs.msg
import controller_manager_msgs.msg
import std_srvs.srv
from ur_control.constants import ExecutionResult
from ur_control import conversions
import ur_dashboard_msgs.srv
import ur_msgs.srv

from std_msgs.msg import Bool

from osx_robot_control.ur_fzi_force_control import URForceController
from osx_robot_control.robotiq_gripper import RobotiqGripper
from osx_robot_control import helpers


class URRobot(RobotBase):
    """ Universal Robots specific implementation of RobotBase
        This class provides access to useful hardware specific services.
        Access to force control and gripper is also defined here.
    """

    def __init__(self, namespace, tf_listener, markers_scene, tcp_link='gripper_tip_link'):
        """
        namespace should be "a_bot" or "b_bot".
        use_real_robot is a boolean
        """
        RobotBase.__init__(self, group_name=namespace, tf_listener=tf_listener)

        self.use_real_robot = rospy.get_param("use_real_robot", False)
        self.use_gazebo_sim = rospy.get_param("use_gazebo_sim", False)

        self.ns = namespace
        self.marker_counter = 0
        self.ur_ros_control_running_on_robot = False

        if self.use_gazebo_sim or self.use_real_robot:
            try:
                self.force_controller = URForceController(robot_name=namespace, listener=tf_listener, tcp_link=tcp_link)
            except rospy.ROSException as e:
                self.force_controller = None
                rospy.logwarn("No force control capabilities since controller could not be instantiated: " + str(e))
        else:
            self.force_controller = None

        self.gripper_group = moveit_commander.MoveGroupCommander(self.ns+"_robotiq_85")

        self.ur_dashboard_clients = {
            "get_loaded_program": rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/get_loaded_program' % self.ns, ur_dashboard_msgs.srv.GetLoadedProgram),
            "program_running":    rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/program_running' % self.ns, ur_dashboard_msgs.srv.IsProgramRunning),
            "load_program":       rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/load_program' % self.ns, ur_dashboard_msgs.srv.Load),
            "play":               rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/play' % self.ns, std_srvs.srv.Trigger),
            "stop":               rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/stop' % self.ns, std_srvs.srv.Trigger),
            "quit":               rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/quit' % self.ns, std_srvs.srv.Trigger),
            "connect":            rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/connect' % self.ns, std_srvs.srv.Trigger),
            "close_popup":        rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/close_popup' % self.ns, std_srvs.srv.Trigger),
            "unlock_protective_stop": rospy.ServiceProxy("/%s/ur_hardware_interface/dashboard/unlock_protective_stop" % self.ns, std_srvs.srv.Trigger),
            "is_in_remote_control": rospy.ServiceProxy("/%s/ur_hardware_interface/dashboard/is_in_remote_control" % self.ns, ur_dashboard_msgs.srv.IsInRemoteControl),
            "get_program_state":   rospy.ServiceProxy('/%s/ur_hardware_interface/dashboard/program_state' % self.ns, ur_dashboard_msgs.srv.GetProgramState),
        }

        self.set_payload_srv = rospy.ServiceProxy('/%s/ur_hardware_interface/set_payload' % self.ns, ur_msgs.srv.SetPayload)
        self.speed_slider = rospy.ServiceProxy("/%s/ur_hardware_interface/set_speed_slider" % self.ns, ur_msgs.srv.SetSpeedSliderFraction)

        self.set_io = rospy.ServiceProxy('/%s/ur_hardware_interface/set_io' % self.ns, ur_msgs.srv.SetIO)

        self.sub_status_ = rospy.Subscriber("/%s/ur_hardware_interface/robot_program_running" % self.ns, Bool, self.ros_control_status_callback)
        self.service_proxy_list = rospy.ServiceProxy("/" + self.ns + "/controller_manager/list_controllers", controller_manager_msgs.srv.ListControllers)
        self.service_proxy_switch = rospy.ServiceProxy("/" + self.ns + "/controller_manager/switch_controller", controller_manager_msgs.srv.SwitchController)

        self.sub_robot_safety_mode = rospy.Subscriber("/%s/ur_hardware_interface/safety_mode" % self.ns, ur_dashboard_msgs.msg.SafetyMode, self.safety_mode_callback)

        self.robot_safety_mode = None
        self.robot_status = dict()

        self.gripper = RobotiqGripper(namespace=self.ns, gripper_group=self.gripper_group, markers_scene=markers_scene)

    def safety_mode_callback(self, msg):
        self.robot_safety_mode = msg.mode

    def ros_control_status_callback(self, msg):
        self.ur_ros_control_running_on_robot = msg.data

    def is_running_normally(self):
        """
        Returns true if the robot is running (no protective stop, not turned off etc).
        """
        return self.robot_safety_mode == 1 or self.robot_safety_mode == 2  # Normal / Reduced

    def is_protective_stopped(self):
        """
        Returns true if the robot is in protective stop.
        """
        return self.robot_safety_mode == 3

    def unlock_protective_stop(self):
        if not self.use_real_robot:
            return True

        service_client = self.ur_dashboard_clients["unlock_protective_stop"]
        request = std_srvs.srv.TriggerRequest()
        start_time = time.time()
        rospy.loginfo("Attempting to unlock protective stop of " + self.ns)
        while not rospy.is_shutdown():
            response = service_client.call(request)
            if time.time() - start_time > 20.0:
                rospy.logerr("Timeout of 20s exceeded in unlock protective stop")
                break
            if response.success:
                break
            rospy.sleep(0.2)
        self.ur_dashboard_clients["stop"].call(std_srvs.srv.TriggerRequest())
        if not response.success:
            rospy.logwarn("Could not unlock protective stop of " + self.ns + "!")
        return response.success

    def get_status_from_param_server(self):
        self.robot_status = osx_msgs.msg.RobotStatus()
        self.robot_status.carrying_object = rospy.get_param(self.ns + "/carrying_object", False)
        self.robot_status.carrying_tool = rospy.get_param(self.ns + "/carrying_tool", False)
        self.robot_status.held_tool_id = rospy.get_param(self.ns + "/held_tool_id", "")
        return self.robot_status

    def publish_robot_status(self):
        rospy.set_param(self.ns + "/carrying_object", self.robot_status.carrying_object)
        rospy.set_param(self.ns + "/carrying_tool",   self.robot_status.carrying_tool)
        rospy.set_param(self.ns + "/held_tool_id",    self.robot_status.held_tool_id)

    @helpers.check_for_real_robot
    def set_speed_scale(self, scale):
        try:
            self.speed_slider(ur_msgs.srv.SetSpeedSliderFractionRequest(speed_slider_fraction=scale))
        except:
            rospy.logerr("Failed to communicate with Dashboard when setting speed slider")
            return False

    @helpers.check_for_real_robot
    def set_payload(self, mass, center_of_gravity):
        """
            mass float
            center_of_gravity list[3]
        """
        self.activate_ros_control_on_ur()
        try:
            payload = ur_msgs.srv.SetPayloadRequest()
            payload.payload = mass
            payload.center_of_gravity = conversions.to_vector3(center_of_gravity)
            self.set_payload_srv(payload)
            return True
        except Exception as e:
            rospy.logerr("Exception trying to set payload: %s" % e)
        return False

    def call_service(self, service_name, wait_time=0.2, retry=True):
        try:
            response = self.ur_dashboard_clients[service_name].call()
            rospy.logdebug(f"{service_name} {response=}")
            rospy.sleep(wait_time)
            return response
        except rospy.ServiceException as e:
            if retry and "Failed to send request to dashboard server" in e.args[0]:
                rospy.logwarn("Call to service failed, retrying connection to dashboard")
                if self.reset_connection():
                    return self.call_service(service_name, retry=False)
            rospy.logerr("Unable to automatically activate robot. Manually activate the robot by pressing 'play' in the polyscope or turn ON the remote control mode.")
            raise e

    @helpers.check_for_real_robot
    def wait_for_control_status_to_turn_on(self, wait_time):
        start = rospy.Time.now()
        elapsed = rospy.Time.now() - start
        while elapsed < rospy.Duration(wait_time) and not rospy.is_shutdown():
            elapsed = rospy.Time.now() - start
            rospy.logdebug(f'{self.ur_ros_control_running_on_robot=}')
            if self.ur_ros_control_running_on_robot:
                response = self.call_service('get_program_state')
                if response.success:
                    if response.state.state == 'PLAYING':
                        return True
                    else:
                        self.call_service('stop')
                        self.call_service('play')
            rospy.sleep(.1)
        return False

    @helpers.check_for_real_robot
    def reset_connection(self):
        try:
            rospy.logdebug("Try to quit before connecting.")
            response = self.ur_dashboard_clients["quit"].call()
        except:
            # Ignore failures trying to reset if we cannot communicate with dashboard
            pass

        try:
            rospy.logdebug("Try to connect to dashboard service.")
            response = self.ur_dashboard_clients["connect"].call()
            return response.success
        except Exception as e:
            rospy.logerr("Unable to reset connection...")
            return False

    @helpers.check_for_real_robot
    def restart_program(self):
        rospy.logdebug("Try to stop program.")
        response = self.call_service('stop')
        rospy.sleep(1)
        if response.success:
            rospy.logdebug("Try to play program.")
            response = self.call_service('play')
            rospy.sleep(1)
        return response.success

    @helpers.check_for_real_robot
    def activate_ros_control_on_ur(self, recursion_depth=0):
        if not self.use_real_robot:
            return True

        # 1. check that the controller is working fine first
        # a. check that the robot_program_running is True
        # b. check that the get_program_state is PLAYING

        # Failure recovery
        # 1. if calling any of the services does not work, reset the connection and check if the controller is fine
        # 2. check what program is running and update if necessary and check if the controller is fine
        # 3. if the robot_program_running if true but the robot state is PAUSED then stop and play

        # Check if URCap is already running on UR
        if self.wait_for_control_status_to_turn_on(1.0):
            return True
        else:
            rospy.loginfo("Robot program not running for " + self.ns)

        try:
            response = self.call_service('is_in_remote_control')
            if not response.success or not response.in_remote_control:
                rospy.logerr(">> Unable to automatically activate robot. Manually activate the robot by pressing 'play' in the polyscope or turn ON the remote control mode.")
                return False
        except:
            pass

        rospy.logwarn(f"Attempt to reconnect # {recursion_depth+1}")

        if recursion_depth > 10:
            rospy.logerr("Tried too often. Breaking out.")
            rospy.logerr("Could not start UR ROS control.")
            raise Exception("Could not activate ROS control on robot " + self.ns + ". Breaking out. Is the UR in Remote Control mode and program installed with correct name?")

        if rospy.is_shutdown():
            return False

        program_loaded = self.check_loaded_program()

        if not program_loaded:
            rospy.logwarn("Could not load.")
        else:
            # Run the program
            rospy.loginfo("Running the program (play)")
            self.restart_program()

        if self.wait_for_control_status_to_turn_on(2.0):
            if self.check_for_dead_controller_and_force_start():
                rospy.loginfo("Successfully activated ROS control on robot " + self.ns)
                self.set_speed_scale(scale=1.0)  # Set speed to max always
                return True
        else:
            rospy.logwarn("Failed to start program")
            self.reset_connection()
            return self.activate_ros_control_on_ur(recursion_depth=recursion_depth+1)

    @helpers.check_for_real_robot
    def check_loaded_program(self):
        try:
            # Load program if it not loaded already
            response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
            if response.program_name == '/programs/ROS_external_control.urp':
                return True
            else:
                rospy.loginfo("Currently loaded program was:  " + response.program_name)
                rospy.loginfo("Loading ROS control on robot " + self.ns)
                request = ur_dashboard_msgs.srv.LoadRequest()
                request.filename = "ROS_external_control.urp"
                response = self.ur_dashboard_clients["load_program"].call(request)
                if response.success:  # Try reconnecting to dashboard
                    return True
                else:
                    rospy.logerr("Could not load the ROS_external_control.urp URCap. Is the UR in Remote Control mode and program installed with correct name?")
                for i in range(10):
                    rospy.sleep(0.2)
                    # rospy.loginfo("After-load check nr. " + str(i))
                    response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
                    # rospy.loginfo("Received response: " + response.program_name)
                    if response.program_name == '/programs/ROS_external_control.urp':
                        break
        except:
            rospy.logwarn("Dashboard service did not respond!")
        return False

    @helpers.check_for_real_robot
    def check_for_dead_controller_and_force_start(self):
        list_req = controller_manager_msgs.srv.ListControllersRequest()
        switch_req = controller_manager_msgs.srv.SwitchControllerRequest()
        rospy.loginfo("Checking for dead controllers for robot " + self.ns)
        list_res = self.service_proxy_list.call(list_req)
        for c in list_res.controller:
            if c.name == "scaled_pos_joint_traj_controller":
                if c.state == "stopped":
                    # Force restart
                    rospy.logwarn("Force restart of controller")
                    switch_req.start_controllers = ['scaled_pos_joint_traj_controller']
                    switch_req.strictness = 1
                    switch_res = self.service_proxy_switch.call(switch_req)
                    rospy.sleep(1)
                    return switch_res.ok
                else:
                    rospy.loginfo("Controller state is " + c.state + ", returning True.")
                    return True

    @helpers.check_for_real_robot
    def load_and_execute_program(self, program_name="", recursion_depth=0, skip_ros_activation=False):
        if not skip_ros_activation:
            self.activate_ros_control_on_ur()
        if not self.load_program(program_name, recursion_depth):
            return False
        return self.execute_loaded_program()

    @helpers.check_for_real_robot
    def load_program(self, program_name="", recursion_depth=0):
        if not self.use_real_robot:
            return True

        if recursion_depth > 10:
            rospy.logerr("Tried too often. Breaking out.")
            rospy.logerr("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
            return False

        load_success = False
        try:
            # Try to stop running program
            self.ur_dashboard_clients["stop"].call(std_srvs.srv.TriggerRequest())
            rospy.sleep(.5)

            # Load program if it not loaded already
            response = self.ur_dashboard_clients["get_loaded_program"].call(ur_dashboard_msgs.srv.GetLoadedProgramRequest())
            # print("response:")
            # print(response)
            if response.program_name == '/programs/' + program_name:
                return True
            else:
                rospy.loginfo("Loaded program is different %s. Attempting to load new program %s" % (response.program_name, program_name))
                request = ur_dashboard_msgs.srv.LoadRequest()
                request.filename = program_name
                response = self.ur_dashboard_clients["load_program"].call(request)
                if response.success:  # Try reconnecting to dashboard
                    load_success = True
                    return True
                else:
                    rospy.logerr("Could not load " + program_name + ". Is the UR in Remote Control mode and program installed with correct name?")
        except:
            rospy.logwarn("Dashboard service did not respond to load_program!")
        if not load_success:
            rospy.logwarn("Waiting and trying again")
            rospy.sleep(3)
            try:
                if recursion_depth > 0:  # If connect alone failed, try quit and then connect
                    response = self.ur_dashboard_clients["quit"].call()
                    rospy.logerr("Program could not be loaded on UR: " + program_name)
                    rospy.sleep(.5)
            except:
                rospy.logwarn("Dashboard service did not respond to quit! ")
                pass
            response = self.ur_dashboard_clients["connect"].call()
            rospy.sleep(.5)
            return self.load_program(program_name=program_name, recursion_depth=recursion_depth+1)

    @helpers.check_for_real_robot
    def execute_loaded_program(self):
        # Run the program
        try:
            response = self.ur_dashboard_clients["play"].call(std_srvs.srv.TriggerRequest())
            if not response.success:
                rospy.logerr("Could not start program. Is the UR in Remote Control mode and program installed with correct name?")
                return False
            else:
                rospy.loginfo("Successfully started program on robot " + self.ns)
                return True
        except Exception as e:
            rospy.logerr(str(e))
            return False

    @helpers.check_for_real_robot
    def close_ur_popup(self):
        # Close a popup on the teach pendant to continue program execution
        response = self.ur_dashboard_clients["close_popup"].call(std_srvs.srv.TriggerRequest())
        if not response.success:
            rospy.logerr("Could not close popup.")
            return False
        else:
            rospy.loginfo("Successfully closed popup on teach pendant of robot " + self.ns)
            return True

    def set_up_move_group(self, speed, acceleration, planner="OMPL"):
        self.activate_ros_control_on_ur()
        return RobotBase.set_up_move_group(self, speed, acceleration, planner)

    # ------ Force control functions

    def force_control(self, *args, **kwargs):
        if self.force_controller:
            self.activate_ros_control_on_ur()
            return self.force_controller.force_control(*args, **kwargs)
        return ExecutionResult.TERMINATION_CRITERIA

    def execute_circular_trajectory(self, *args, **kwargs):
        if self.force_controller:
            self.activate_ros_control_on_ur()
            return self.force_controller.execute_circular_trajectory(*args, **kwargs)
        return True

    def execute_spiral_trajectory(self, *args, **kwargs):
        if self.force_controller:
            self.activate_ros_control_on_ur()
            return self.force_controller.execute_spiral_trajectory(*args, **kwargs)
        return True

    def linear_push(self, *args, **kwargs):
        if self.force_controller:
            self.activate_ros_control_on_ur()
            return self.force_controller.linear_push(*args, **kwargs)
        return True

    def do_insertion(self, *args, **kwargs):
        if self.force_controller:
            self.activate_ros_control_on_ur()
            return self.force_controller.do_insertion(*args, **kwargs)
        return ExecutionResult.TERMINATION_CRITERIA
