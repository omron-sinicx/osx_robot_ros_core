#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Pose
import signal
import sys
import yaml
import os
import csv
import datetime
import threading

from matplotlib import pyplot as plt

from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control.constants import GripperType
from ur_control import transformations
from ur_control.math_utils import quaternion_from_axis_angle, quaternions_orientation_error
from osx_claw.claw_controller import ClawController
from bendlabs.bendlabs_recorder import BendLabsRecorder


def quaternion_to_euler(quaternion):
    """Convert quaternion to Euler angles in radians."""
    return transformations.euler_from_quaternion(quaternion)


def euler_to_quaternion(euler):
    """Convert Euler angles in radians to quaternion."""
    return transformations.quaternion_from_euler(euler[0], euler[1], euler[2])


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class DeformationMeasurement:
    """
    A class to perform deformation measurements on robotic systems.
    
    This class controls a robot arm to perform deformation tests in various directions,
    measuring forces and positions during the test. The data is saved to CSV files
    and visualized in plots.
    """
    
    # ==================== CONFIGURATION CONSTANTS ====================
    
    # Test parameters
    MOVEMENT_STEP_SIZE = 0.001      # 1mm per step
    SLEEP_RATE = 0.1               # Sleep between movements (seconds)
    
    # Displacement thresholds for different directions
    DISPLACEMENT_THRESHOLDS = {
        'x': 0.018,                 # 20mm in X direction
        'y': 0.018,                 # 20mm in Y direction  
        'z': 0.012,                 # 20mm in Z direction
        'xy_pos45': 0.018,          # 20mm magnitude in XY plane
        'xy_neg45': 0.018,          # 20mm magnitude in XY plane
        'yaw': 10.0,                # 20 degrees in yaw
        'pitch': 10.0,              # 20 degrees in pitch
        'roll': 10.0,               # 20 degrees in roll
        'roll_pitch_pos45': 10.0,   # 20 degrees magnitude in roll-pitch diagonal
        'roll_pitch_neg45': 10.0,   # 20 degrees magnitude in roll-pitch diagonal
    }
    
    # Rotation parameters
    YAW_STEP_SIZE = 0.01           # ~0.57 degrees per step
    ROLL_STEP_SIZE = 0.01          # ~0.57 degrees per step
    PITCH_STEP_SIZE = 0.01         # ~0.57 degrees per step
    
    # Default test configuration
    DEFAULT_REPETITIONS = 5
    
    # ================================================================
    
    def __init__(self):
        rospy.init_node("deformation_measurement", anonymous=True)

        # Load robot configuration
        self.use_default_config()
        
        # Calculate derived parameters
        self.dt = 1.0 / self.control_frequency

        # Initialize publishers for real-time data streaming
        self.force_pub = rospy.Publisher("/deformation/force_data", Float32MultiArray, queue_size=10)
        self.pose_pub = rospy.Publisher("/deformation/pose", Pose, queue_size=10)

        # Initialize CompliantController
        self.arm = CompliantController(
            namespace=None,
            joint_names_prefix=None,
            ee_link="tool0",
            ft_topic="wrench",
            gripper_type=None,
        )

        # Set control mode
        self.arm.set_position_control_mode(False)
        self.arm.set_control_mode(mode="spring-mass-damper")

        # Set solver params, initial stiffness, selection matrix, gains
        self.arm.set_solver_parameters(error_scale=self.error_scale, iterations=self.iterations)
        self.arm.update_stiffness(self.stiffness)
        self.arm.update_selection_matrix(np.array(self.selection_matrix))
        self.arm.update_pd_gains(self.p_gains, d_gains=self.d_gains)

        # Initialize error vectors and control flags  
        self.error_translation = np.zeros(3)
        self.error_rotation = np.zeros(4)
        self.manual_stop_requested = False

    def use_default_config(self):
        """Set default configuration values."""
        self.control_frequency = 5
        self.p_gains = [0.02, 0.02, 0.02, 1.5, 1.5, 1.5]
        self.d_gains = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.stiffness = [3000, 3000, 3000, 100, 100, 100]
        self.selection_matrix = [1, 1, 1, 1, 1, 1]
        self.error_scale = 1.5
        self.iterations = 1
        self.duration = 0.1
        self.max_force_torque = [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
        self.max_delta_translation = 0.10
        self.max_delta_rotation = np.deg2rad([10, 10, 30.0])
        self.initial_ur = np.radians([75.59, -85.07, 114.92, -120.95, -90.70, -13.37])
        # self.initial_ur = np.radians([90, -90, 90, -90, -90, 90])
        # rospy.init_node was already called in __init__, so avoid re-initializing here
        self.gripper = ClawController(init_node=False)

        # Bend-sensor recording. Subscribes to sensor0..sensorN-1 (TwoAxesAngles);
        # get_bend() returns a flat [twist0, bend0, twist1, bend1, ...] array. If the
        # sensors aren't publishing, get_bend() falls back to NaNs so the test still runs.
        self.num_bend_sensors = 4
        try:
            self.bend_recorder = BendLabsRecorder(init_node=False,
                                                  num_sensors=self.num_bend_sensors)
        except Exception as e:
            rospy.logwarn(f"Could not start BendLabsRecorder: {e}. Bend data will be NaN.")
            self.bend_recorder = None

    # ==================== SENSOR & DATA METHODS ====================
    
    def get_ft(self):
        """Get force/torque readings from the sensor."""
        return self.arm.get_wrench(base_frame_control=True)

    def get_bend(self):
        """Get bend-sensor readings as a flat array [twist0, bend0, twist1, bend1, ...].

        Returns NaNs (of the expected length) if the recorder is unavailable or the
        sensors haven't produced any data yet, so recording never blocks the test."""
        if self.bend_recorder is not None and self.bend_recorder.sensors_ready():
            return self.bend_recorder.get_angles()
        return np.full(2 * self.num_bend_sensors, np.nan, dtype=np.float32)
    
    def calculate_displacement(self, initial_pose, current_pose, direction):
        """
        Calculate displacement in the specified direction.
        
        Args:
            initial_pose: Initial end effector pose [x, y, z, qx, qy, qz, qw]
            current_pose: Current end effector pose [x, y, z, qx, qy, qz, qw]
            direction: Direction being tested
            
        Returns:
            float: Displacement value (absolute) in appropriate units
        """
        if direction == 'x':
            return abs(current_pose[0] - initial_pose[0])
        elif direction == 'y':
            return abs(current_pose[1] - initial_pose[1])
        elif direction == 'z':
            return abs(current_pose[2] - initial_pose[2])
        elif direction in ['xy_pos45', 'xy_neg45']:
            # Calculate XY plane displacement magnitude
            dx = current_pose[0] - initial_pose[0]
            dy = current_pose[1] - initial_pose[1]
            return np.sqrt(dx**2 + dy**2)
        elif direction in ['yaw', 'pitch', 'roll', 'roll_pitch_pos45', 'roll_pitch_neg45']:
            # Calculate angular displacement in degrees
            initial_quat = initial_pose[3:7]
            current_quat = current_pose[3:7]
            
            if direction in ['roll_pitch_pos45', 'roll_pitch_neg45']:
                # For roll-pitch diagonal, use axis-angle approach to avoid Euler singularities
                # Calculate quaternion difference
                initial_quat_normalized = initial_quat / np.linalg.norm(initial_quat)
                current_quat_normalized = current_quat / np.linalg.norm(current_quat)
                
                # Calculate relative quaternion (current relative to initial)
                # q_rel = q_current * q_initial^-1
                initial_quat_inv = np.array([-initial_quat_normalized[0], -initial_quat_normalized[1], 
                                           -initial_quat_normalized[2], initial_quat_normalized[3]])
                relative_quat = transformations.quaternion_multiply(current_quat_normalized, initial_quat_inv)
                
                # Convert relative quaternion to axis-angle
                relative_axis_angle = transformations.axis_angle_from_quaternion(relative_quat)
                
                # Calculate magnitude of axis-angle (total rotation in radians, then convert to degrees)
                rotation_magnitude = np.linalg.norm(relative_axis_angle)
                return np.rad2deg(rotation_magnitude)
            else:
                # Convert to Euler angles for single axis rotations
                initial_euler = transformations.euler_from_quaternion(initial_quat)
                current_euler = transformations.euler_from_quaternion(current_quat)
                
                if direction == 'yaw':
                    # Handle yaw angle wrapping
                    yaw_diff = current_euler[2] - initial_euler[2]
                    yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    return abs(np.rad2deg(yaw_diff))
                elif direction == 'pitch':
                    # Handle pitch angle wrapping
                    pitch_diff = current_euler[1] - initial_euler[1]
                    pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    return abs(np.rad2deg(pitch_diff))
                elif direction == 'roll':
                    # Handle roll angle wrapping
                    roll_diff = current_euler[0] - initial_euler[0]
                    roll_diff = (roll_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    return abs(np.rad2deg(roll_diff))
        
        return 0.0

    def get_eef_pos(self, orientation_representation="axis_angle"):
        """Get end effector position and orientation."""
        # Get current end effector pose [x,y,z] + [quat(4)]
        arm_pose = self.arm.end_effector()
        rotation = None

        if orientation_representation == "axis_angle":
            # Convert quaternion to axis angle: [axis_angle(3)]
            rotation = transformations.axis_angle_from_quaternion(arm_pose[3:])
        elif orientation_representation == "ortho6":
            # Convert quaternion to axis angle: [ortho6(6)]
            rotation = transformations.ortho6_from_quaternion(arm_pose[3:])
        else:
            raise ValueError(
                f"Unsupported orientation_representation: {orientation_representation}"
            )

        return np.concatenate([arm_pose[:3], rotation])
    
    def publish_force_data(self, force_data):
        """Publish force data to ROS topic for real-time monitoring."""
        msg = Float32MultiArray()
        msg.data = force_data
        self.force_pub.publish(msg)

    def publish_pose_data(self, pose_data):
        """Publish pose data to ROS topic for real-time monitoring."""
        msg = Pose()
        msg.position.x = pose_data[0]
        msg.position.y = pose_data[1]
        msg.position.z = pose_data[2]
        msg.orientation.x = pose_data[3]
        msg.orientation.y = pose_data[4]
        msg.orientation.z = pose_data[5]
        msg.orientation.w = pose_data[6]
        self.pose_pub.publish(msg)
        
    # ==================== ROBOT CONTROL METHODS ====================

    def deactivate_compliance_control(self):
        """Deactivate compliance control and switch to joint trajectory control."""
        self.arm.zero_ft_sensor()
        self.arm.activate_joint_trajectory_controller()

    def activate_compliance_control(self):
        """Activate compliance control."""
        self.arm.zero_ft_sensor()
        self.arm.activate_cartesian_controller()

    def initialize_robot_position(self):
        """Move robot to initial position."""
        self.deactivate_compliance_control()

        rospy.loginfo("Moving robot to initial position...")
        self.arm.set_joint_positions(
            target_time=3, positions=self.initial_ur, wait=True
        )

        self.arm.zero_ft_sensor()
        self.activate_compliance_control()

        rospy.loginfo("Robot initialized to starting position")
        return True

    def do_compliant_move(self, target_pose):

        current_pose = self.arm.end_effector()

        is_safe, partial_step = self.check_safety(
            target_position=target_pose[:3],
            target_orientation=target_pose[3:],
            current_pose=current_pose,
        )

        if is_safe:
            self.arm.set_cartesian_target_pose(target_pose)
        else:
            rospy.logwarn_throttle(2, "Movement too large; using partial step.")
            new_pose = np.zeros(7)
            # current + partial
            new_pose[:3] = current_pose[:3] + partial_step[:3]
            new_pose[3:] = partial_step[3:]
            self.arm.set_cartesian_target_pose(new_pose)

    def check_safety(self, target_position, target_orientation, current_pose):
        """
        Check if the target pose is safe to move to.
        Return (is_safe, partial_step).
            'partial_step' is a 7D [dx, dy, dz, qx, qy, qz, qw].
        """
        curr_pos = current_pose[:3]
        curr_quat = current_pose[3:]

        # Position error
        error_translation = target_position - curr_pos

        # Orientation error in Euler
        curr_euler = np.array(quaternion_to_euler(curr_quat))
        target_euler = np.array(quaternion_to_euler(target_orientation))
        error_euler = target_euler - curr_euler
        # Wrap to [-pi, pi]
        error_euler = (error_euler + np.pi) % (2 * np.pi) - np.pi

        unsafe_translation = np.any(
            np.abs(error_translation) > self.max_delta_translation
        )
        unsafe_rotation = np.any(np.abs(error_euler) > self.max_delta_rotation)

        if (not unsafe_translation) and (not unsafe_rotation):
            return True, np.zeros(7)

        partial_step = np.zeros(7)

        # Clamp translation
        clamped_translation = np.clip(
            error_translation, -self.max_delta_translation, self.max_delta_translation
        )
        # Clamp rotation
        clamped_euler_error = np.clip(
            error_euler, -self.max_delta_rotation, self.max_delta_rotation
        )
        new_euler = curr_euler + clamped_euler_error
        new_quat = euler_to_quaternion(new_euler)

        partial_step[:3] = clamped_translation
        partial_step[3:] = new_quat
        return False, partial_step

    def move_force(self, arm):
        """ Linear push. Move until the target force is felt and stop. """
        arm.zero_ft_sensor()

        arm.set_control_mode("parallel")
        selection_matrix = [1, 1, 0, 1, 1, 1]
        arm.update_selection_matrix(selection_matrix)

        # arm.set_control_mode("spring-mass-damper")

        arm.set_solver_parameters(error_scale=0.5, iterations=1)
        arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])

        p_gains = [0.05, 0.05, 0.1, 1.5, 1.5, 1.5]
        d_gains = [0.005, 0.005, 0.005, 0, 0, 0]
        arm.update_pd_gains(p_gains, d_gains)

        ee = arm.end_effector()

        target_force = np.array([0, 0, -1, 0, 0, 0])  # express in the end_effector_link
        # account for the direction of the force?
        stop_at_wrench = np.copy(target_force)
        stop_at_wrench *= -1

        # Initialize data collection arrays
        force_values = []
        eef_values = []

        # Set up a rate for data collection

        # Start compliance control in a separate thread
        import threading
        result = {"completed": False, "res": None}

        def run_compliance():
            result["res"] = arm.execute_compliance_control(
                ee, target_wrench=target_force,
                max_force_torque=[30., 30., 30., 4., 4., 4.],
                duration=30,
                stop_at_wrench=stop_at_wrench,
                stop_on_target_force=True
            )
            result["completed"] = True

        # Start the compliance control thread
        compliance_thread = threading.Thread(target=run_compliance)
        compliance_thread.start()

        # Collect data while compliance control is running
        try:
            while not result["completed"] and not rospy.is_shutdown():
                # Record force data
                force_values.append(self.get_ft())

                # Record end effector position
                eef_values.append(self.get_eef_pos())

                # Publish data for visualization
                self.publish_force_data(force_values[-1])
                self.publish_pose_data(arm.end_effector())

        except Exception as e:
            rospy.logerr(f"Error during data collection: {e}")
        finally:
            # Wait for compliance thread to finish if it hasn't already
            if compliance_thread.is_alive():
                compliance_thread.join(timeout=1.0)

        # Convert to numpy arrays for easier processing
        force_values = np.array(force_values)
        eef_values = np.array(eef_values)

        # Save the collected data
        self.save_data_to_csv(force_values, eef_values, folder_name="soft_wrist_lock_x")

        # Plot the results
        self.plot_force_position_data(force_values, eef_values)

        print(result["res"])
        print("EE change", ee - arm.end_effector())

        return force_values, eef_values

    # ==================== VISUALIZATION METHODS ====================

    def plot_force_position_data(self, force_values, eef_values):
        """Plot force vs position data."""
        if len(force_values) == 0 or len(eef_values) == 0:
            rospy.logwarn("No data to plot")
            return

        # Create three separate plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Force X vs Position Z
        ax1.scatter(force_values[:, 0], eef_values[:, 2], marker='o', s=3)
        ax1.set_title('Force X vs Z Position')
        ax1.set_xlabel('Force X (N)')
        ax1.set_ylabel('Position Z (m)')

        # Plot 2: Force Y vs Position Z
        ax2.scatter(force_values[:, 1], eef_values[:, 2], marker='x', s=3)
        ax2.set_title('Force Y vs Z Position')
        ax2.set_xlabel('Force Y (N)')
        ax2.set_ylabel('Position Z (m)')

        # Plot 3: Force Z vs Position Z
        ax3.scatter(force_values[:, 2], eef_values[:, 2], marker='^', s=3)
        ax3.set_title('Force Z vs Z Position')
        ax3.set_xlabel('Force Z (N)')
        ax3.set_ylabel('Position Z (m)')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "../force_deformation_data/force_position_plot.png"))
        plt.show()

    def run_deformation_test(self, direction="z"):
        """
        Run a single deformation test in the specified direction.
        
        Args:
            direction (str): Direction to test ("x", "y", "z", "yaw", "roll", etc.)
            
        Returns:
            tuple: (force_values, eef_values) - arrays of measured data
        """
        # Use class constants for configuration
        rate = self.SLEEP_RATE
        d = self.MOVEMENT_STEP_SIZE
        displacement_threshold = self.DISPLACEMENT_THRESHOLDS.get(direction, 0.020)  # Default 20mm or 20°
        yaw_step = self.YAW_STEP_SIZE
        roll_step = self.ROLL_STEP_SIZE
        pitch_step = self.PITCH_STEP_SIZE
        
        # Reset flag for manual stop at the beginning of each test
        self.manual_stop_requested = False
        
        def wait_for_enter():
            """Thread function to wait for Enter key press"""
            try:
                input("Press Enter to stop the compression and start returning...\n")
                self.manual_stop_requested = True
            except:
                pass  # Handle any input errors gracefully

        # Initialize robot
        # self.initialize_robot_position()
        self.arm.zero_ft_sensor()
        self.activate_compliance_control()
        self.arm.set_position_control_mode(True)

        # Close the claw so it grasps the specimen before the compression starts.
        rospy.loginfo("Closing gripper...")
        self.gripper.close_gripper()
        rospy.sleep(1.0)

        # Automatically start test without waiting for user input
        rospy.loginfo(f"Starting deformation test in {direction} direction...")
        self.arm.zero_ft_sensor()

        # Get initial position
        initial_pose = self.arm.end_effector()
        rospy.loginfo(f"Initial position: {initial_pose[:3]}")



        # Start the thread to listen for Enter key press
        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()
        
        rospy.loginfo("Compression phase started. Press Enter to stop and return to initial position.")
        
        # Get displacement threshold units for logging
        threshold_units = "mm" if direction in ['x', 'y', 'z', 'xy_pos45', 'xy_neg45'] else "degrees"
        threshold_value = displacement_threshold * 1000 if threshold_units == "mm" else displacement_threshold
        rospy.loginfo(f"Displacement threshold: {threshold_value:.1f}{threshold_units} in {direction} direction")

        # Example: Move down 5cm and measure forces
        force_values = []
        eef_values = []
        bend_values = []

        i = 0
        while True:
            # Get current pose and calculate displacement
            current_pose = self.arm.end_effector()
            current_displacement = self.calculate_displacement(initial_pose, current_pose, direction)
            
            # Check for displacement threshold or manual stop
            displacement_exceeded = current_displacement >= displacement_threshold
            
            # Progress logging every 10 steps
            if i % 10 == 0:
                current_display = current_displacement * 1000 if threshold_units == "mm" else current_displacement
                rospy.loginfo(f"Progress: {current_display:.1f}/{threshold_value:.1f}{threshold_units} moved ({current_display/threshold_value*100:.1f}%)")
            
            if displacement_exceeded or self.manual_stop_requested:
                rospy.sleep(0.5)
                if displacement_exceeded:
                    final_display = current_displacement * 1000 if threshold_units == "mm" else current_displacement
                    rospy.loginfo(f"Displacement threshold reached: {final_display:.1f}{threshold_units} after {i} steps. Moving back to initial position.")
                else:
                    rospy.loginfo(f"Manual stop requested after {i} steps. Moving back to initial position.")
                
                # Return to the initial position step by step
                for step in range(i):
                    target_pose = self.arm.end_effector().copy()
                        
                    if direction == "z":
                        target_pose[2] += d  # Move 1mm up in Z direction
                    elif direction == "y":
                        target_pose[1] += d  # Move 1mm up in Y direction
                    elif direction == "x":
                        target_pose[0] += d  # Move 1mm up in X direction
                    elif direction == "yaw":
                        # Rotate back in opposite direction
                        current_quat = target_pose[3:7]
                        # Create a small rotation about z-axis (axis=[0,0,1], angle=-yaw_step)
                        rot_quat = quaternion_from_axis_angle([0, 0, -yaw_step])
                        # Apply rotation
                        new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                        target_pose[3:7] = new_quat
                    elif direction == "pitch":
                        # Rotate back in opposite direction
                        current_quat = target_pose[3:7]
                        # Create a small rotation about y-axis (axis=[0,1,0], angle=-pitch_step)
                        rot_quat = quaternion_from_axis_angle([0, -pitch_step, 0])
                        # Apply rotation
                        new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                        target_pose[3:7] = new_quat
                    elif direction == "roll":
                        # Rotate back in opposite direction using offset pose
                        current_quat = target_pose[3:7]
                        # Create a small rotation about x-axis (axis=[1,0,0], angle=-roll_step)
                        rot_quat = quaternion_from_axis_angle([-roll_step, 0, 0])
                        # Apply rotation
                        new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                        target_pose[3:7] = new_quat
                    elif direction == "roll_pitch_pos45":
                        # Rotate back in opposite direction (45 degrees between roll and pitch) using offset pose
                        current_quat = target_pose[3:7]
                        # Create a small rotation about [1/√2, 1/√2, 0] axis (45° between x and y)
                        axis = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0])
                        rot_quat = quaternion_from_axis_angle(axis * (-roll_step))
                        # Apply rotation
                        new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                        target_pose[3:7] = new_quat
                    elif direction == "roll_pitch_neg45":
                        # Rotate back in opposite direction (-45 degrees between roll and pitch) using offset pose
                        current_quat = target_pose[3:7]
                        # Create a small rotation about [1/√2, -1/√2, 0] axis (-45° between x and y)
                        axis = np.array([1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0])
                        rot_quat = quaternion_from_axis_angle(axis * (-roll_step))
                        # Apply rotation
                        new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                        target_pose[3:7] = new_quat
                    # Handle 45-degree directions in XY plane
                    elif direction == "xy_pos45":
                        # Move in positive x and positive y (diagonal)
                        target_pose[0] += d/np.sqrt(2)
                        target_pose[1] += d/np.sqrt(2)
                    elif direction == "xy_neg45":
                        # Move in positive x and negative y (diagonal)
                        target_pose[0] += d/np.sqrt(2)
                        target_pose[1] -= d/np.sqrt(2)

                    self.do_compliant_move(target_pose)
                    force_values.append(self.get_ft())
                    eef_values.append(self.get_eef_pos())
                    bend_values.append(self.get_bend())
                    rospy.sleep(rate)

                # Release the specimen now that the arm is back at the start pose.
                rospy.sleep(1.0)
                rospy.loginfo("Opening gripper...")
                self.gripper.open_gripper()
                rospy.sleep(1.0)
                break

            target_pose = self.arm.end_effector().copy()

            if direction == "z":
                target_pose[2] -= d  # Move 1mm down in Z direction
            elif direction == "y":
                target_pose[1] -= d  # Move 1mm down in Y direction
            elif direction == "x":
                target_pose[0] -= d  # Move 1mm down in X direction
            elif direction == "yaw":
                # Apply small yaw rotation
                current_quat = target_pose[3:7]
                # Create a small rotation about z-axis (axis=[0,0,1], angle=yaw_step)
                rot_quat = quaternion_from_axis_angle([0, 0, yaw_step])
                # Apply rotation
                new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                target_pose[3:7] = new_quat
            elif direction == "pitch":
                # Apply small pitch rotation
                current_quat = target_pose[3:7]
                # Create a small rotation about y-axis (axis=[0,1,0], angle=pitch_step)
                rot_quat = quaternion_from_axis_angle([0, pitch_step, 0])
                # Apply rotation
                new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                target_pose[3:7] = new_quat
            elif direction == "roll":
                # Apply small roll rotation
                current_quat = target_pose[3:7]
                # Create a small rotation about x-axis (axis=[1,0,0], angle=roll_step)
                rot_quat = quaternion_from_axis_angle([roll_step, 0, 0])
                # Apply rotation
                new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                target_pose[3:7] = new_quat
            elif direction == "roll_pitch_pos45":
                # Apply small rotation 45 degrees between roll and pitch
                current_quat = target_pose[3:7]
                # Create a small rotation about [1/√2, 1/√2, 0] axis (45° between x and y)
                axis = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0])
                rot_quat = quaternion_from_axis_angle(axis * roll_step)
                # Apply rotation
                new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                target_pose[3:7] = new_quat
            elif direction == "roll_pitch_neg45":
                # Apply small rotation -45 degrees between roll and pitch
                current_quat = target_pose[3:7]
                # Create a small rotation about [1/√2, -1/√2, 0] axis (-45° between x and y)
                axis = np.array([1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0])
                rot_quat = quaternion_from_axis_angle(axis * roll_step)
                # Apply rotation
                new_quat = transformations.quaternion_multiply(rot_quat, current_quat)
                target_pose[3:7] = new_quat
            # Handle 45-degree directions in XY plane
            elif direction == "xy_pos45":
                # Move in negative x and negative y (diagonal)
                target_pose[0] -= d/np.sqrt(2)
                target_pose[1] -= d/np.sqrt(2)
            elif direction == "xy_neg45":
                # Move in negative x and positive y (diagonal)
                target_pose[0] -= d/np.sqrt(2)
                target_pose[1] += d/np.sqrt(2)

            self.do_compliant_move(target_pose)
            rospy.sleep(rate)
            current_forces = self.get_ft()
            force_values.append(current_forces)
            eef_values.append(self.get_eef_pos())
            bend_values.append(self.get_bend())

            # Print current forces, torques, displacement, and euler angles periodically for user feedback
            if i % 50 == 0:  # Print every 50 steps
                # Get current end effector pose and convert to euler angles
                current_eef_pose = self.arm.end_effector()
                current_quat = current_eef_pose[3:7]  # [qx, qy, qz, qw]
                current_euler = transformations.euler_from_quaternion(current_quat)
                current_euler_deg = np.rad2deg(current_euler)  # Convert to degrees
                
                # Calculate current displacement for detailed logging
                step_displacement = self.calculate_displacement(initial_pose, current_eef_pose, direction)
                step_display = step_displacement * 1000 if threshold_units == "mm" else step_displacement
                
                rospy.loginfo(f"Step {i}: Displacement = {step_display:.2f}{threshold_units}, "
                             f"Forces = [{current_forces[0]:.2f}, {current_forces[1]:.2f}, {current_forces[2]:.2f}] N, "
                             f"Torques = [{current_forces[3]:.2f}, {current_forces[4]:.2f}, {current_forces[5]:.2f}] Nm, "
                             f"Euler = [{current_euler_deg[0]:.1f}, {current_euler_deg[1]:.1f}, {current_euler_deg[2]:.1f}] deg")
            
            i += 1

        force_values = np.array(force_values)
        eef_values = np.array(eef_values)
        bend_values = np.array(bend_values)

        # Reset the manual stop flag for next test
        self.manual_stop_requested = False

        rospy.loginfo("Deformation test completed.")

        return force_values, eef_values, bend_values

    def plot_deformation_data(self, force_values, eef_values, direction, folder_name):
        """Plot force vs position data for a specific direction test."""
        if len(force_values) == 0 or len(eef_values) == 0:
            rospy.logwarn("No data to plot")
            return

        # Create a figure with four subplots (2x2 grid)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Determine which position axis to plot based on test direction
        if direction in ["x", "y", "z"]:
            pos_idx = 0 if direction == "x" else (1 if direction == "y" else 2)
            pos_label = f"Position {direction.upper()} (m)"
            plot_data = eef_values[:, pos_idx]
        elif direction == "yaw":
            # For yaw, we'll plot against the yaw angle in degrees
            # Extract yaw from orientation (assuming axis-angle representation)
            yaw_angles = np.array([transformations.euler_from_quaternion(
                transformations.quaternion_from_axis_angle(eef_values[i, 3:6]))[2]
                for i in range(len(eef_values))])
            yaw_degrees = np.rad2deg(yaw_angles)
            pos_label = "Yaw Angle (degrees)"
            plot_data = yaw_degrees
        elif direction == "pitch":
            # For pitch, we'll plot against the pitch angle in degrees
            # Extract pitch from orientation (assuming axis-angle representation)
            pitch_angles = np.array([transformations.euler_from_quaternion(
                transformations.quaternion_from_axis_angle(eef_values[i, 3:6]))[1]
                for i in range(len(eef_values))])
            pitch_degrees = np.rad2deg(pitch_angles)
            pos_label = "Pitch Angle (degrees)"
            plot_data = pitch_degrees
        elif direction == "roll":
            # For roll, we'll plot against the roll angle in degrees
            # Extract roll from orientation (assuming axis-angle representation)
            roll_angles = np.array([transformations.euler_from_quaternion(
                transformations.quaternion_from_axis_angle(eef_values[i, 3:6]))[0]
                for i in range(len(eef_values))])
            roll_degrees = np.rad2deg(roll_angles)
            pos_label = "Roll Angle (degrees)"
            plot_data = roll_degrees
        elif direction == "roll_pitch_pos45":
            # For roll-pitch +45°, we'll plot against the combined angle
            # Calculate the angle around the [1/√2, 1/√2, 0] axis
            indices = np.arange(len(eef_values))
            # Approximate combined angle based on step progression (0.01 rad = 0.57 degrees)
            combined_angles = indices * np.rad2deg(0.01)
            pos_label = "Roll-Pitch +45° Angle (degrees)"
            plot_data = combined_angles
        elif direction == "roll_pitch_neg45":
            # For roll-pitch -45°, we'll plot against the combined angle
            # Calculate the angle around the [1/√2, -1/√2, 0] axis
            indices = np.arange(len(eef_values))
            # Approximate combined angle based on step progression (0.01 rad = 0.57 degrees)
            combined_angles = indices * np.rad2deg(0.01)
            pos_label = "Roll-Pitch -45° Angle (degrees)"
            plot_data = combined_angles
        # Handle 45-degree directions in XY plane
        elif direction in ["xy_pos45", "xy_neg45"]:
            # Calculate the Euclidean distance from the starting point in XY plane
            distances = np.sqrt((eef_values[:, 0] - eef_values[0, 0])**2 +
                                (eef_values[:, 1] - eef_values[0, 1])**2)
            pos_label = "Distance in XY Plane (m)"
            plot_data = distances

        # Plot 1: Force X vs Position/Angle
        ax1.scatter(force_values[:, 0], plot_data, marker='o', s=3)
        ax1.set_title(f'Force X vs {direction}')
        ax1.set_xlabel('Force X (N)')
        ax1.set_ylabel(pos_label)

        # Plot 2: Force Y vs Position/Angle
        ax2.scatter(force_values[:, 1], plot_data, marker='x', s=3)
        ax2.set_title(f'Force Y vs {direction}')
        ax2.set_xlabel('Force Y (N)')
        ax2.set_ylabel(pos_label)

        # Plot 3: Force Z vs Position/Angle
        ax3.scatter(force_values[:, 2], plot_data, marker='^', s=3)
        ax3.set_title(f'Force Z vs {direction}')
        ax3.set_xlabel('Force Z (N)')
        ax3.set_ylabel(pos_label)

        # Plot 4: Force vs Index (time)
        indices = np.arange(len(force_values))
        ax4.plot(indices, force_values[:, 0], 'r-', label='Fx')
        ax4.plot(indices, force_values[:, 1], 'b-', label='Fy')
        ax4.plot(indices, force_values[:, 2], 'g-', label='Fz')
        ax4.set_title('Forces vs Index')
        ax4.set_xlabel('Index (time step)')
        ax4.set_ylabel('Force (N)')
        ax4.legend()

        plt.tight_layout()

        # Create directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../force_deformation_data", folder_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        plt.savefig(os.path.join(data_dir, f"force_position_plot_{direction}.png"))
        plt.close()

    # ==================== DEFORMATION TESTING METHODS ====================

    def get_available_directions(self):
        """
        Get all available test directions organized by category.
        
        Returns:
            dict: Dictionary of direction categories and their available tests
        """
        return {
            'translational': {
                'x': 'X-axis linear movement',
                'y': 'Y-axis linear movement', 
                'z': 'Z-axis linear movement'
            },
            'diagonal': {
                'xy_pos45': 'XY diagonal +45° movement',
                'xy_neg45': 'XY diagonal -45° movement'
            },
            'rotational': {
                'yaw': 'Yaw rotation (Z-axis)',
                'pitch': 'Pitch rotation (Y-axis)',
                'roll': 'Roll rotation (X-axis)',
                'roll_pitch_pos45': 'Roll-Pitch +45° diagonal rotation',
                'roll_pitch_neg45': 'Roll-Pitch -45° diagonal rotation'
            }
        }
    
    def select_test_directions_interactive(self):
        """
        Interactive direction selection with organized menu.
        
        Returns:
            list: Selected directions to test
        """
        available_directions = self.get_available_directions()
        
        print("\n" + "="*60)
        print("🎯 DIRECTION SELECTION MENU")
        print("="*60)
        
        # Display available directions by category
        all_directions = []
        direction_map = {}  # Map numbers to direction keys
        counter = 1
        
        for category, directions in available_directions.items():
            print(f"\n📋 {category.upper()} TESTS:")
            for direction, description in directions.items():
                print(f"  {counter:2d}. {direction:20s} - {description}")
                direction_map[counter] = direction
                all_directions.append(direction)
                counter += 1
        
        # Add preset options
        print(f"\n🎛️  PRESET OPTIONS:")
        print(f"  {counter:2d}. all_translational    - All linear movement tests (x,y,z)")
        preset_all_trans = counter
        counter += 1
        print(f"  {counter:2d}. all_rotational       - All rotational tests (yaw,roll,diagonals)") 
        preset_all_rot = counter
        counter += 1
        print(f"  {counter:2d}. roll_only            - Roll tests only (roll + diagonals)")
        preset_roll_only = counter
        counter += 1
        print(f"  {counter:2d}. basic_suite          - Basic tests (x,y,z,roll)")
        preset_basic = counter
        counter += 1
        print(f"  {counter:2d}. all_directions       - Complete test suite")
        preset_all = counter
        
        print(f"\n" + "="*60)
        print("📝 SELECTION OPTIONS:")
        print("  • Enter individual numbers (e.g., 1,3,5)")
        print("  • Enter ranges (e.g., 1-4)")  
        print("  • Use preset numbers")
        print("  • Type 'default' for default roll tests")
        print("="*60)
        
        while True:
            try:
                selection = input("\nEnter your selection: ").strip().lower()
                
                if selection == 'default' or selection == '':
                    return ["roll", "roll_pitch_pos45", "roll_pitch_neg45"]
                
                selected_directions = []
                
                # Handle preset selections
                if selection == str(preset_all_trans):
                    selected_directions = list(available_directions['translational'].keys())
                elif selection == str(preset_all_rot):
                    selected_directions = list(available_directions['rotational'].keys())
                elif selection == str(preset_roll_only):
                    selected_directions = ["roll", "roll_pitch_pos45", "roll_pitch_neg45"]
                elif selection == str(preset_basic):
                    selected_directions = ["x", "y", "z", "roll"]
                elif selection == str(preset_all):
                    selected_directions = all_directions
                else:
                    # Parse individual selections
                    parts = selection.replace(' ', '').split(',')
                    
                    for part in parts:
                        if '-' in part:
                            # Handle ranges like "1-4"
                            start, end = map(int, part.split('-'))
                            for i in range(start, end + 1):
                                if i in direction_map:
                                    selected_directions.append(direction_map[i])
                        else:
                            # Handle individual numbers
                            num = int(part)
                            if num in direction_map:
                                selected_directions.append(direction_map[num])
                
                # Remove duplicates while preserving order
                selected_directions = list(dict.fromkeys(selected_directions))
                
                if selected_directions:
                    print(f"\n✅ Selected directions: {', '.join(selected_directions)}")
                    confirm = input("Confirm selection? (y/N): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected_directions
                    else:
                        continue
                else:
                    print("❌ No valid directions selected. Please try again.")
                    
            except (ValueError, KeyError) as e:
                print(f"❌ Invalid selection. Please enter valid numbers or presets.")
            except KeyboardInterrupt:
                print("\n🛑 Selection cancelled.")
                return ["roll", "roll_pitch_pos45", "roll_pitch_neg45"]  # Default
    
    def get_test_directions(self):
        """
        Get the list of directions to test. 
        This method can be overridden or will return default directions.
        """
        # Default test directions for non-interactive use
        return ["roll", "roll_pitch_pos45", "roll_pitch_neg45"]

    def run_all_deformation_tests(self, custom_folder_name=None, repetitions=1, selected_directions=None):
        """
        Run deformation tests in multiple directions with repetitions.
        
        Args:
            custom_folder_name (str): Custom name for data folder, auto-generated if None
            repetitions (int): Number of repetitions per direction
            selected_directions (list): List of directions to test, uses default if None
        
        Returns:
            str: Name of the folder where data was saved
        """
        # Create a timestamp for this test session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use custom folder name if provided, otherwise use timestamp
        if custom_folder_name:
            folder_name = custom_folder_name
        else:
            folder_name = f"deformation_test_{timestamp}"

        # Create the main folder
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../force_deformation_data", folder_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Use provided directions or get default directions
        if selected_directions is None:
            selected_directions = self.get_test_directions()
        rospy.loginfo(f"Testing directions: {selected_directions}")

        # Run all repetitions for each direction before moving to next direction
        for direction in selected_directions:
            rospy.loginfo(f"Starting all repetitions for {direction} direction")
            
            for rep in range(1, repetitions + 1):
                rospy.loginfo(f"Starting deformation test in {direction} direction (repetition {rep} of {repetitions})")
                force_values, eef_values, bend_values = self.run_deformation_test(direction)

                # Save data to CSV with direction and repetition in filename
                csv_filename = f"deformation_data_{direction}_rep{rep}_{timestamp}.csv"
                self.save_data_to_csv(force_values, eef_values,
                                      folder_name=folder_name,
                                      filename=csv_filename,
                                      bend_values=bend_values)

                # Print Euler angle ranges for this repetition
                self.print_euler_angle_ranges(eef_values, direction, rep)

                # Plot the data
                self.plot_deformation_data(force_values, eef_values, direction, folder_name)

                # Give the robot a moment to stabilize between repetitions
                rospy.sleep(1.0)
            
            # Longer pause between different directions to allow for any manual adjustments
            if direction != selected_directions[-1]:  # Don't pause after the last direction
                rospy.loginfo(f"Completed all {repetitions} repetitions for {direction}. Preparing for next direction...")
                rospy.sleep(2.0)

        # Create combined visualization
        self.create_combined_visualization(folder_name)
        
        # Print overall test summary
        self.print_test_summary(folder_name, selected_directions, repetitions)

        return folder_name

    # ==================== DATA SAVING & PROCESSING METHODS ====================

    def save_data_to_csv(self, force_values, eef_values, folder_name="force_deformation_data", filename=None, bend_values=None):
        """
        Save force and position data to a CSV file.

        Args:
            force_values: numpy array of force/torque measurements [Fx, Fy, Fz, Tx, Ty, Tz]
            eef_values: numpy array of end effector positions and orientations
            folder_name: name of the folder to save data in
            filename: optional specific filename to use
            bend_values: optional numpy array of bend-sensor readings, one row per sample,
                each row [twist0, bend0, twist1, bend1, ...]. Columns are appended to the CSV.
        """
        # Create a timestamp for the filename if not provided
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deformation_data_{timestamp}.csv"

        # Create directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../force_deformation_data", folder_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filepath = os.path.join(data_dir, filename)

        # Determine whether we have bend data to append (and how many columns)
        has_bend = bend_values is not None and len(bend_values) > 0
        n_bend_cols = bend_values.shape[1] if has_bend else 0

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz',
                      'Pos_X', 'Pos_Y', 'Pos_Z', 'Orient_1', 'Orient_2', 'Orient_3']
            # One (twist, bend) pair per sensor: S0_twist, S0_bend, S1_twist, ...
            for s in range(n_bend_cols // 2):
                header += [f'S{s}_twist', f'S{s}_bend']
            writer.writerow(header)

            # Write data rows
            for i in range(len(force_values)):
                row = list(force_values[i]) + list(eef_values[i])
                if has_bend and i < len(bend_values):
                    row += list(bend_values[i])
                writer.writerow(row)

        rospy.loginfo(f"Data saved to {filepath}")
        return filepath

    def print_euler_angle_ranges(self, eef_values, direction, repetition):
        """
        Print the range of Euler angles achieved during a test repetition.
        
        Args:
            eef_values: Array of end effector poses [position + axis_angle_orientation]
            direction: Test direction name
            repetition: Repetition number
        """
        if len(eef_values) == 0:
            rospy.logwarn(f"No EEF data available for {direction} rep {repetition}")
            return
        
        try:
            # Extract orientation data (axis-angle format in columns 3:6)
            orientations = eef_values[:, 3:6]  # axis-angle representation
            
            # Convert axis-angle to Euler angles for each pose
            euler_angles = []
            for i in range(len(orientations)):
                # Convert axis-angle to quaternion, then to Euler angles
                axis_angle = orientations[i]
                quaternion = transformations.quaternion_from_axis_angle(axis_angle)
                euler = transformations.euler_from_quaternion(quaternion)
                euler_degrees = np.rad2deg(euler)  # Convert to degrees
                euler_angles.append(euler_degrees)
            
            euler_angles = np.array(euler_angles)
            
            # Calculate ranges for each Euler angle
            roll_range = [np.min(euler_angles[:, 0]), np.max(euler_angles[:, 0])]
            pitch_range = [np.min(euler_angles[:, 1]), np.max(euler_angles[:, 1])]
            yaw_range = [np.min(euler_angles[:, 2]), np.max(euler_angles[:, 2])]
            
            # Calculate total range (max - min) for each axis
            roll_total = roll_range[1] - roll_range[0]
            pitch_total = pitch_range[1] - pitch_range[0]
            yaw_total = yaw_range[1] - yaw_range[0]
            
            # Print the results
            print(f"\n📊 EULER ANGLE RANGES - {direction.upper()} (Rep {repetition}):")
            print(f"   Roll:  [{roll_range[0]:6.2f}° to {roll_range[1]:6.2f}°] = {roll_total:6.2f}° range")
            print(f"   Pitch: [{pitch_range[0]:6.2f}° to {pitch_range[1]:6.2f}°] = {pitch_total:6.2f}° range")
            print(f"   Yaw:   [{yaw_range[0]:6.2f}° to {yaw_range[1]:6.2f}°] = {yaw_total:6.2f}° range")
            
            # Highlight the primary motion axis for rotational tests
            if direction == "roll" or "roll" in direction:
                print(f"   🎯 Primary axis (Roll): {roll_total:.2f}° range")
            elif direction == "yaw":
                print(f"   🎯 Primary axis (Yaw): {yaw_total:.2f}° range")
            elif direction == "pitch" or "pitch" in direction:
                print(f"   🎯 Primary axis (Pitch): {pitch_total:.2f}° range")
                
        except Exception as e:
            rospy.logerr(f"Error calculating Euler ranges for {direction} rep {repetition}: {e}")

    def print_test_summary(self, folder_name, directions, repetitions):
        """
        Print a summary of all completed tests.
        
        Args:
            folder_name: Name of the data folder
            directions: List of test directions
            repetitions: Number of repetitions per direction
        """
        print(f"\n" + "="*60)
        print(f"🎉 DEFORMATION TEST SUMMARY")
        print(f"="*60)
        print(f"📁 Data folder: {folder_name}")
        print(f"🔄 Total tests: {len(directions)} directions × {repetitions} reps = {len(directions) * repetitions} tests")
        print(f"📋 Directions tested: {', '.join(directions)}")
        print(f"⚙️  Configuration:")
        print(f"   • Step size: {self.MOVEMENT_STEP_SIZE*1000:.1f}mm")
        print(f"   • Sleep rate: {self.SLEEP_RATE}s")
        print(f"   • Displacement thresholds: {self.DISPLACEMENT_THRESHOLDS}")
        print(f"   • Rotation steps: {np.rad2deg(self.ROLL_STEP_SIZE):.2f}° (roll), {np.rad2deg(self.PITCH_STEP_SIZE):.2f}° (pitch), {np.rad2deg(self.YAW_STEP_SIZE):.2f}° (yaw)")
        print(f"\n📊 Data files generated:")
        
        # List the expected CSV files
        import os
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../force_deformation_data", folder_name)
        
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            for csv_file in sorted(csv_files):
                print(f"   • {csv_file}")
        
        print(f"\n📈 Visualizations created:")
        print(f"   • Individual plots for each test")
        print(f"   • Combined force-displacement curves")
        print(f"   • Stiffness comparison chart")
        print(f"="*60)

    def create_combined_visualization(self, folder_name):
        """Create combined visualizations of all test data."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../force_deformation_data", folder_name)

        # Find all CSV files in the folder
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        if not csv_files:
            rospy.logwarn("No CSV files found for combined visualization")
            return

        # Create a figure for combined force-displacement curves
        plt.figure(figsize=(15, 10))

        # Define colors and markers for different directions
        direction_styles = {
            'x': {'color': 'r', 'marker': 'o', 'label': 'X Direction'},
            'y': {'color': 'g', 'marker': 's', 'label': 'Y Direction'},
            'z': {'color': 'b', 'marker': '^', 'label': 'Z Direction'},
            'yaw': {'color': 'm', 'marker': 'x', 'label': 'Yaw Rotation'},
            'pitch': {'color': 'darkgreen', 'marker': '<', 'label': 'Pitch Rotation'},
            'roll': {'color': 'orange', 'marker': 'v', 'label': 'Roll Rotation'},
            'roll_pitch_pos45': {'color': 'purple', 'marker': 'd', 'label': 'Roll-Pitch +45°'},
            'roll_pitch_neg45': {'color': 'brown', 'marker': 'h', 'label': 'Roll-Pitch -45°'},
            'xy_pos45': {'color': 'c', 'marker': '+', 'label': 'XY +45°'},
            'xy_neg45': {'color': 'y', 'marker': '*', 'label': 'XY -45°'}
        }

        # Group files by direction
        all_directions = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'roll_pitch_pos45', 'roll_pitch_neg45', 'xy_pos45', 'xy_neg45']
        direction_files = {direction: [] for direction in all_directions}

        for csv_file in csv_files:
            for direction in all_directions:
                if f"_{direction}_" in csv_file:
                    direction_files[direction].append(csv_file)

        # Plot data from each direction (averaging across repetitions)
        for direction, files in direction_files.items():
            if not files:
                continue

            all_displacements = []
            all_forces = []

            for csv_file in files:
                # Load data
                data = np.loadtxt(os.path.join(data_dir, csv_file), delimiter=',', skiprows=1)

                # Extract forces and positions
                forces = data[:, :3]  # Fx, Fy, Fz
                positions = data[:, 6:9]  # Pos_X, Pos_Y, Pos_Z

                # Calculate resultant force
                resultant_force = np.sqrt(np.sum(forces**2, axis=1))

                # For translational directions, calculate displacement
                if direction in ['x', 'y', 'z']:
                    dir_idx = 0 if direction == 'x' else (1 if direction == 'y' else 2)
                    # Calculate displacement from initial position
                    displacement = positions[:, dir_idx] - positions[0, dir_idx]
                    all_displacements.append(np.abs(displacement))
                    all_forces.append(resultant_force)
                # For 45-degree directions, calculate Euclidean distance in XY plane
                elif direction in ['xy_pos45', 'xy_neg45']:
                    displacement = np.sqrt((positions[:, 0] - positions[0, 0])**2 +
                                           (positions[:, 1] - positions[0, 1])**2)
                    all_displacements.append(displacement)
                    all_forces.append(resultant_force)
                # For yaw, we need to calculate angular displacement
                elif direction == 'yaw':
                    # This is simplified - in practice you'd need to extract yaw angles
                    indices = np.arange(len(resultant_force))
                    normalized_indices = indices / max(indices) * 30  # Assuming ~30 degrees max rotation
                    all_displacements.append(normalized_indices)
                    all_forces.append(resultant_force)
                # For pitch, similar to yaw
                elif direction == 'pitch':
                    # This is simplified - in practice you'd need to extract pitch angles
                    indices = np.arange(len(resultant_force))
                    normalized_indices = indices / max(indices) * 30  # Assuming ~30 degrees max rotation
                    all_displacements.append(normalized_indices)
                    all_forces.append(resultant_force)
                # For roll, similar to yaw
                elif direction == 'roll':
                    # This is simplified - in practice you'd need to extract roll angles
                    indices = np.arange(len(resultant_force))
                    normalized_indices = indices / max(indices) * 30  # Assuming ~30 degrees max rotation
                    all_displacements.append(normalized_indices)
                    all_forces.append(resultant_force)
                # For roll-pitch +45°
                elif direction == 'roll_pitch_pos45':
                    # This is simplified - in practice you'd need to extract combined roll-pitch angles
                    indices = np.arange(len(resultant_force))
                    normalized_indices = indices / max(indices) * 30  # Assuming ~30 degrees max rotation
                    all_displacements.append(normalized_indices)
                    all_forces.append(resultant_force)
                # For roll-pitch -45°
                elif direction == 'roll_pitch_neg45':
                    # This is simplified - in practice you'd need to extract combined roll-pitch angles
                    indices = np.arange(len(resultant_force))
                    normalized_indices = indices / max(indices) * 30  # Assuming ~30 degrees max rotation
                    all_displacements.append(normalized_indices)
                    all_forces.append(resultant_force)

            # Plot each repetition with slight transparency
            for i, (disp, force) in enumerate(zip(all_displacements, all_forces)):
                plt.plot(disp, force,
                         color=direction_styles[direction]['color'],
                         marker=direction_styles[direction]['marker'],
                         alpha=0.3,
                         markersize=4, markevery=10,
                         label=f"{direction_styles[direction]['label']} (Rep {i+1})" if i == 0 else "")

            # Calculate and plot the average if we have multiple repetitions
            if len(all_displacements) > 1:
                # Interpolate to common x values for averaging
                max_len = max(len(d) for d in all_displacements)
                common_x = np.linspace(0, 1, max_len)

                interp_forces = []
                for disp, force in zip(all_displacements, all_forces):
                    if len(disp) > 1:  # Need at least 2 points for interpolation
                        # Normalize displacement to [0,1] for interpolation
                        norm_disp = disp / np.max(disp) if np.max(disp) > 0 else disp
                        interp_force = np.interp(common_x, norm_disp, force)
                        interp_forces.append(interp_force)

                if interp_forces:
                    avg_force = np.mean(interp_forces, axis=0)
                    # Scale x back to original range
                    avg_x = common_x * np.mean([np.max(d) for d in all_displacements if len(d) > 0])

                    plt.plot(avg_x, avg_force,
                             color=direction_styles[direction]['color'],
                             linewidth=3,
                             label=f"{direction_styles[direction]['label']} (Average)")

        plt.title('Force-Displacement Curves for All Directions')
        plt.xlabel('Displacement (m) / Rotation (deg)')
        plt.ylabel('Resultant Force (N)')
        plt.legend()
        plt.grid(True)

        # Save the combined visualization
        plt.savefig(os.path.join(data_dir, "combined_force_displacement.png"))
        plt.close()

        # Create a second visualization for stiffness comparison
        plt.figure(figsize=(10, 8))

        # Calculate and plot stiffness for each direction
        bar_data = {direction: [] for direction in all_directions}

        for direction, files in direction_files.items():
            for csv_file in files:
                # Load data
                data = np.loadtxt(os.path.join(data_dir, csv_file), delimiter=',', skiprows=1)

                # Extract forces and positions
                forces = data[:, :3]  # Fx, Fy, Fz
                positions = data[:, 6:9]  # Pos_X, Pos_Y, Pos_Z

                # Calculate resultant force
                resultant_force = np.sqrt(np.sum(forces**2, axis=1))

                # Calculate stiffness (force/displacement)
                if direction in ['x', 'y', 'z']:
                    dir_idx = 0 if direction == 'x' else (1 if direction == 'y' else 2)
                    displacement = positions[:, dir_idx] - positions[0, dir_idx]

                    # Use the maximum force and corresponding displacement
                    max_force_idx = np.argmax(resultant_force)
                    if max_force_idx > 0 and abs(displacement[max_force_idx]) > 0.001:
                        stiffness = resultant_force[max_force_idx] / abs(displacement[max_force_idx])
                        bar_data[direction].append(stiffness)

                # For 45-degree directions
                elif direction in ['xy_pos45', 'xy_neg45']:
                    displacement = np.sqrt((positions[:, 0] - positions[0, 0])**2 +
                                           (positions[:, 1] - positions[0, 1])**2)

                    # Use the maximum force and corresponding displacement
                    max_force_idx = np.argmax(resultant_force)
                    if max_force_idx > 0 and displacement[max_force_idx] > 0.001:
                        stiffness = resultant_force[max_force_idx] / displacement[max_force_idx]
                        bar_data[direction].append(stiffness)

                elif direction == 'yaw':
                    # For yaw, we'd need a different calculation based on angular displacement
                    max_force = np.max(resultant_force)
                    # Assuming 30 degrees max rotation and linear relationship
                    stiffness = max_force / np.deg2rad(30)
                    bar_data[direction].append(stiffness / 10)  # Scale down for visualization
                elif direction == 'pitch':
                    # For pitch, similar calculation based on angular displacement
                    max_force = np.max(resultant_force)
                    # Assuming 30 degrees max rotation and linear relationship
                    stiffness = max_force / np.deg2rad(30)
                    bar_data[direction].append(stiffness / 10)  # Scale down for visualization
                elif direction == 'roll':
                    # For roll, similar calculation based on angular displacement
                    max_force = np.max(resultant_force)
                    # Assuming 30 degrees max rotation and linear relationship
                    stiffness = max_force / np.deg2rad(30)
                    bar_data[direction].append(stiffness / 10)  # Scale down for visualization
                elif direction == 'roll_pitch_pos45':
                    # For roll-pitch +45°, similar calculation based on angular displacement
                    max_force = np.max(resultant_force)
                    # Assuming 30 degrees max rotation and linear relationship
                    stiffness = max_force / np.deg2rad(30)
                    bar_data[direction].append(stiffness / 10)  # Scale down for visualization
                elif direction == 'roll_pitch_neg45':
                    # For roll-pitch -45°, similar calculation based on angular displacement
                    max_force = np.max(resultant_force)
                    # Assuming 30 degrees max rotation and linear relationship
                    stiffness = max_force / np.deg2rad(30)
                    bar_data[direction].append(stiffness / 10)  # Scale down for visualization

        # Calculate average stiffness for each direction
        avg_stiffness = []
        std_stiffness = []
        bar_labels = []

        for direction in all_directions:
            if bar_data[direction]:
                avg_stiffness.append(np.mean(bar_data[direction]))
                std_stiffness.append(np.std(bar_data[direction]))
                bar_labels.append(direction)

        # Create bar chart with error bars
        colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
        plt.bar(bar_labels, avg_stiffness,
                color=colors[:len(bar_labels)],
                yerr=std_stiffness,
                capsize=10)
        plt.title('Stiffness Comparison Across Directions')
        plt.ylabel('Stiffness (N/m or N·m/rad)')
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)

        # Save the stiffness comparison
        plt.savefig(os.path.join(data_dir, "stiffness_comparison.png"))
        plt.close()

        rospy.loginfo(f"Combined visualizations created in {data_dir}")

    # ==================== MAIN EXECUTION METHODS ====================

    def run(self):
        """
        Main run loop for interactive deformation testing.
        
        This method prompts the user for test configuration and runs the tests.
        """
        try:
            # Get user configuration
            print("\n" + "="*50)
            print("🤖 DEFORMATION MEASUREMENT SYSTEM")
            print("="*50)
            
            # Step 1: Select test directions
            selected_directions = self.select_test_directions_interactive()
            
            # Step 2: Get folder name
            print("\n" + "="*50)
            print("📁 FOLDER CONFIGURATION")
            print("="*50)
            custom_folder_name = input("Enter folder name (leave blank for auto-generated): ").strip()
            if not custom_folder_name:
                custom_folder_name = None

            # Step 3: Get repetitions
            print("\n" + "="*50)
            print("🔄 REPETITION CONFIGURATION")
            print("="*50)
            try:
                repetitions_input = input(f"Enter number of repetitions (default {self.DEFAULT_REPETITIONS}): ").strip()
                repetitions = int(repetitions_input) if repetitions_input else self.DEFAULT_REPETITIONS
            except ValueError:
                repetitions = self.DEFAULT_REPETITIONS
                rospy.logwarn(f"Invalid input, using default {self.DEFAULT_REPETITIONS} repetitions")

            # Step 4: Final confirmation
            print("\n" + "="*50)
            print("📋 FINAL TEST CONFIGURATION")
            print("="*50)
            print(f"  📁 Folder: {custom_folder_name or 'auto-generated'}")
            print(f"  🔄 Repetitions: {repetitions}")
            print(f"  🎯 Directions: {', '.join(selected_directions)}")
            print(f"  ⚡ Total tests: {len(selected_directions)} × {repetitions} = {len(selected_directions) * repetitions}")
            print(f"  ⚙️ Configuration:")
            print(f"     • Displacement thresholds: {self.DISPLACEMENT_THRESHOLDS}")
            print(f"     • Step size: {self.MOVEMENT_STEP_SIZE*1000:.1f}mm")
            print(f"     • Sleep rate: {self.SLEEP_RATE}s")
            print("="*50)
            
            confirm = input("\n🚀 Start tests? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                rospy.loginfo("Tests cancelled by user")
                return

            # Step 5: Run the tests
            rospy.loginfo("Starting deformation tests...")
            folder_name = self.run_all_deformation_tests(
                custom_folder_name=custom_folder_name,
                repetitions=repetitions,
                selected_directions=selected_directions
            )
            print(f"\n✓ All tests completed successfully!")
            print(f"  Data saved in folder: {folder_name}")

        except KeyboardInterrupt:
            rospy.loginfo("\nTests interrupted by user (Ctrl+C)")
        except Exception as e:
            rospy.logerr(f"Error during tests: {e}")
        finally:
            # Clean up
            self.deactivate_compliance_control()
            rospy.loginfo("Deformation measurement system shutdown complete")




def main():
    node = DeformationMeasurement()
    rospy.loginfo("Starting deformation measurement node...")
    node.run()


if __name__ == "__main__":
    main()