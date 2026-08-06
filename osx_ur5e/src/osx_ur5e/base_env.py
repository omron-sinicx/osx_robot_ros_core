import rospy
import numpy as np
from omegaconf import DictConfig

from osx_ur5e.timestep import TimeStep, STEP_FIRST

from osx_ur5e.observation_assembler import ObservationAssembler
from osx_ur5e.sample_feeders import RosSampleFeeder

from ur_control.arm import Arm
from ur_control.constants import ExecutionResult
from ur_control.fzi_cartesian_compliance_controller import CompliantController

ORIENTATION_REPRESENTATIONS = ['axis_angle', 'ortho6']

HOME_JOINT_TOLERANCE = 0.005  # rad, per joint
HOME_JOINT_SPEED = 0.3  # rad/s, used to scale the trajectory duration with the distance left
HOME_MIN_TRAJECTORY_TIME = 0.5  # s
HOME_SETTLE_TIME = 0.5  # s, grace after a trajectory before re-issuing it
HOME_POLL_RATE = 50  # Hz


class BaseEnv:
    """
    Forward Dynamics Compliant Control Environment
    """

    def __init__(self, config: DictConfig):
        self.load_params(config)

        if self.config.controller.type == "fdcc":
            self.arm = CompliantController(gripper_type=None,
                                           base_link=config.controller.base_link,
                                           ee_link=config.controller.ee_link)
        elif self.config.controller.type == "joint_velocity":
            self.arm = Arm(gripper_type=None,
                           base_link=config.controller.base_link,
                           ee_link=config.controller.ee_link,
                           use_velocity_interface=config.controller.use_velocity_interface,
                           robot_version="ur5e")
        else:
            raise ValueError(f"Unsupported controller type: {self.config.controller.type}")
        self.arm.dashboard_services.activate_ros_control_on_ur()

        self.target_wrench = np.zeros(6)
        # Shared observation contract: the same feeder+assembler pair the
        # offline bag converter uses, fed by live topics (raw per-camera
        # topics, hardware stamps - no /sync remapping).
        self.feeder = RosSampleFeeder(camera_names=list(self.cam_names))
        self.assembler = ObservationAssembler(self.arm.kdl, list(self.cam_names))
        self.feeder.wait_until_fresh(max_age_s=1.0, timeout_s=5.0)
        # comet's eval runner and artifacts reach the camera feed through
        # this attribute (feeder.get_images() keeps that API).
        self.image_recorder = self.feeder

        self.rate = rospy.Rate(self.control_frequency)

        self.position_control_dims = np.ones(6)
        self.force_control_dims = np.zeros(6)
        self.reference_trajectory = None

    def load_params(self, config):
        self.config = config
        self.initial_config = config.controller.init_qpos

    def go_home(self, tolerance=HOME_JOINT_TOLERANCE, target_time=5.0, max_time=30.0):
        """Move to ``initial_config``, returning only once the robot is actually there.

        Returns as soon as every joint is within ``tolerance`` rather than waiting for the
        whole trajectory, and re-issues the motion as many times as needed if the robot
        falls short of the goal. ``target_time`` is the duration of a full-swing trajectory;
        shorter moves get a proportionally shorter one.

        Raises RuntimeError if the configuration is not reached within ``max_time``.
        """
        assert self.arm.dashboard_services.activate_ros_control_on_ur(), "Failed to activate ROS control on UR"

        target = np.asarray(self.initial_config, dtype=float)
        deadline = rospy.get_time() + max_time
        poll = rospy.Rate(HOME_POLL_RATE)

        while not rospy.is_shutdown():
            joint_error = np.max(np.abs(self.arm.joint_angles() - target))
            if joint_error < tolerance:
                return
            if rospy.get_time() >= deadline:
                raise RuntimeError(
                    f"Failed to reach the home joint configuration for robot '{self.arm.ns}' within "
                    f"{max_time}s: worst joint error {joint_error:.4f} rad > {tolerance} rad")

            trajectory_time = float(np.clip(joint_error / HOME_JOINT_SPEED,
                                            HOME_MIN_TRAJECTORY_TIME, target_time))
            self.arm.set_joint_positions(target_time=trajectory_time, positions=target, wait=False)

            segment_deadline = min(rospy.get_time() + trajectory_time + HOME_SETTLE_TIME, deadline)
            while rospy.get_time() < segment_deadline and not rospy.is_shutdown():
                if np.max(np.abs(self.arm.joint_angles() - target)) < tolerance:
                    return
                poll.sleep()

    def get_images(self):
        return self.feeder.get_images()

    def get_observation(self):
        """Latest observation via the shared assembler (grab-latest semantics).

        Images are excluded here - the policy path pulls them itself through
        format_real_robot_observations, and nothing downstream consumes
        TimeStep.observation images.
        """
        samples = self.feeder.get_latest(["joint_states", "wrench"])
        return self.assembler.assemble_observation(
            samples, tick_time=rospy.get_time(), include_images=False)

    def get_reward(self):
        return 0

    def reset(self, move_robot=True):
        if move_robot:
            # Move away from contact
            success = self.arm.move_relative(target_time=1.0, transformation=[0, 0, -0.05, 0, 0, 0], relative_to_tcp=True, wait=True)
            assert success == ExecutionResult.DONE, f"Failed to move to initial configuration for robot '{self.arm.ns}' : {self.initial_config} {success}"
            # Move to the initial configuration
            success = self.arm.set_joint_positions(target_time=self.config.controller.reset_time, positions=self.initial_config, wait=True)
            assert success == ExecutionResult.DONE, f"Failed to move to initial configuration for robot '{self.arm.ns}' : {self.initial_config} {success}"

        return TimeStep(
            step_type=STEP_FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        raise NotImplementedError("Step method not implemented for base environment")

    def check_contact_force_limits(self):
        """
            Check that contact force limits are not violated.
            Returns False if the limits are violated, 
            otherwise return True
        """
        # Safety limits: max force
        current_wrench = self.arm.get_wrench()
        force_norm = np.linalg.norm(current_wrench)
        torque_norm = np.linalg.norm(current_wrench[3:])
        if force_norm > self.max_force_torque[0] or torque_norm > self.max_force_torque[1]:
            rospy.logerr(f'Maximum force/torque exceeded {force_norm}/{self.max_force_torque[0]} {torque_norm}/{self.max_force_torque[1]}')
            return False
        return True

    def zero_ft_sensor(self):
        self.arm.zero_ft_sensor()
