import collections

import numpy as np
from omegaconf import DictConfig

from osx_ur5e.image_recorder import ImageRecorder
from osx_ur5e.ros_node import RosRuntime
from osx_ur5e.timestep import TimeStep, STEP_FIRST

from ur_control import utils, transformations
from ur_control.arm import Arm
from ur_control.constants import ExecutionResult
from ur_control.fzi_cartesian_compliance_controller import CompliantController

ORIENTATION_REPRESENTATIONS = ["axis_angle", "ortho6"]


class BaseEnv:
    """Forward Dynamics Compliant Control environment base class."""

    def __init__(self, config: DictConfig, node=None, runtime=None, use_torch_for_cameras=False):
        self._owns_runtime = runtime is None and node is None
        if node is None:
            runtime = runtime or RosRuntime()
            node = runtime.node
        self.runtime = runtime
        self.node = node

        self.load_params(config)

        arm_kwargs = {
            "node": self.node,
            "gripper_type": None,
            "base_link": config.controller.base_link,
            "ee_link": config.controller.ee_link,
        }
        if self.config.controller.type == "fdcc":
            self.arm = CompliantController(**arm_kwargs)
        elif self.config.controller.type == "joint_velocity":
            self.arm = Arm(
                **arm_kwargs,
                use_velocity_interface=config.controller.use_velocity_interface,
                robot_version="ur5e",
            )
        else:
            raise ValueError(f"Unsupported controller type: {self.config.controller.type}")

        self.arm.dashboard_services.activate_ros_control_on_ur()

        self.target_wrench = np.zeros(6)
        self.image_recorder = ImageRecorder(
            node=self.node,
            camera_names=self.cam_names,
            use_torch=use_torch_for_cameras,
        )

        self.rate = utils.Rate(self.control_frequency)

        self.position_control_dims = np.ones(6)
        self.force_control_dims = np.zeros(6)
        self.reference_trajectory = None

    def shutdown(self):
        if self._owns_runtime and self.runtime is not None:
            self.runtime.shutdown()

    def load_params(self, config):
        self.config = config
        self.initial_config = config.controller.init_qpos
        self.control_frequency = config.dataset.fps
        self.cam_names = config.dataset.cameras.keys()

    def go_home(self):
        assert self.arm.dashboard_services.activate_ros_control_on_ur(), (
            "Failed to activate ROS control on UR"
        )
        self.arm.set_joint_positions(
            target_time=5.0, positions=self.initial_config, wait=True
        )

    def get_eef_components(self):
        eef_pos = self.get_eef_pose(orientation_representation="quaternion")
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

    def get_eef_pose(self, orientation_representation="axis_angle"):
        arm_pose = self.arm.end_effector()
        rotation = None

        if orientation_representation == "axis_angle":
            rotation = transformations.axis_angle_from_quaternion(arm_pose[3:])
        elif orientation_representation == "ortho6":
            rotation = transformations.ortho6_from_quaternion(arm_pose[3:])
        elif orientation_representation == "quaternion":
            rotation = arm_pose[3:]
        else:
            raise ValueError(
                f"Unsupported orientation_representation: {orientation_representation}"
            )

        return np.concatenate([arm_pose[:3], rotation])

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs.update(self.get_eef_components())
        obs["qpos"] = self.arm.joint_angles()
        obs["qvel"] = self.arm.joint_velocities()
        return obs

    def get_reward(self):
        return 0

    def reset(self, move_robot=True):
        if move_robot:
            success = self.arm.move_relative(
                target_time=1.0,
                transformation=[0, 0, -0.05, 0, 0, 0],
                relative_to_tcp=True,
                wait=True,
            )
            assert success == ExecutionResult.DONE, (
                f"Failed to move away from contact for robot '{self.arm.ns}'"
            )
            success = self.arm.set_joint_positions(
                target_time=self.config.controller.reset_time,
                positions=self.initial_config,
                wait=True,
            )
            assert success == ExecutionResult.DONE, (
                f"Failed to move to initial configuration for robot '{self.arm.ns}' "
                f": {self.initial_config} {success}"
            )

        return TimeStep(
            step_type=STEP_FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action):
        raise NotImplementedError("Step method not implemented for base environment")

    def check_contact_force_limits(self):
        current_wrench = self.arm.get_wrench()
        force_norm = np.linalg.norm(current_wrench)
        torque_norm = np.linalg.norm(current_wrench[3:])
        if force_norm > self.max_force_torque[0] or torque_norm > self.max_force_torque[1]:
            self.node.get_logger().error(
                f"Maximum force/torque exceeded "
                f"{force_norm}/{self.max_force_torque[0]} "
                f"{torque_norm}/{self.max_force_torque[1]}"
            )
            return False
        return True

    def zero_ft_sensor(self):
        self.arm.zero_ft_sensor()
