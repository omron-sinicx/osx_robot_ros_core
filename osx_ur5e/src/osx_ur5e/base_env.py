import collections

import rospy
import numpy as np
from omegaconf import DictConfig

from osx_ur5e.timestep import TimeStep, STEP_FIRST


from osx_gym_env.utils import ImageRecorder

from ur_control.arm import Arm
from ur_control import transformations
from ur_control.constants import ExecutionResult
from ur_control.fzi_cartesian_compliance_controller import CompliantController

ORIENTATION_REPRESENTATIONS = ['axis_angle', 'ortho6']


class BaseEnv:
    """
    Forward Dynamics Compliant Control Environment
    """

    def __init__(self, config: DictConfig, use_torch_for_cameras=False):
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
        self.image_recorder = ImageRecorder(init_node=False, camera_names=self.cam_names, use_torch=use_torch_for_cameras)

        self.rate = rospy.Rate(self.control_frequency)

        self.position_control_dims = np.ones(6)
        self.force_control_dims = np.zeros(6)
        self.reference_trajectory = None

    def load_params(self, config):
        self.config = config
        self.initial_config = config.controller.init_qpos

    def go_home(self):
        assert self.arm.dashboard_services.activate_ros_control_on_ur(), "Failed to activate ROS control on UR"
        self.arm.set_joint_positions(target_time=5.0, positions=self.initial_config, wait=True)

    def get_eef_components(self):
        eef_pos = self.get_eef_pose(orientation_representation='quaternion')
        rot_ortho6 = transformations.ortho6_from_quaternion(eef_pos[3:])
        axis_angle = transformations.axis_angle_from_quaternion(eef_pos[3:])
        return {
            "eef_pos.position": eef_pos[:3],
            "eef_pos.quaternion": eef_pos[3:],
            "eef_pos.rotation_axis_angle": axis_angle,
            "eef_pos.rotation_ortho6": rot_ortho6,
            "eef_pos.wrench": self.arm.get_wrench(),
            "eef_pos.velocity": self.arm.end_effector_velocity(),
        }

    def get_eef_pose(self, orientation_representation='axis_angle'):
        # get current end effector pose [x,y,z] + [quat(4)]
        arm_pose = self.arm.end_effector()
        rotation = None

        if orientation_representation == 'axis_angle':
            # convert quaternion to axis angle: [axis_angle(3)]
            rotation = transformations.axis_angle_from_quaternion(arm_pose[3:])
        elif orientation_representation == 'ortho6':
            # convert quaternion to axis angle: [ortho6(6)]
            rotation = transformations.ortho6_from_quaternion(arm_pose[3:])
        elif orientation_representation == 'quaternion':
            rotation = arm_pose[3:]
        else:
            raise ValueError(f'Unsupported orientation_representation: {orientation_representation}')

        return np.concatenate([arm_pose[:3], rotation])

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs.update(self.get_eef_components())
        obs['qpos'] = self.arm.joint_angles()
        obs['qvel'] = self.arm.joint_velocities()
        return obs

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
