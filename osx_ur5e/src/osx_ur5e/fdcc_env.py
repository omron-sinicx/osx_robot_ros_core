import dm_env
import rospy
import numpy as np
from omegaconf import DictConfig

from osx_robot_control import math_utils
from osx_ur5e.base_env import ORIENTATION_REPRESENTATIONS, BaseEnv
from ur_control import transformations


STIFFNESS_REPRESENTATIONS = ['cholesky', 'diag']


class FDCCEnv(BaseEnv):
    """
    Forward Dynamics Compliant Control Environment
    """

    def __init__(self, config: DictConfig, use_torch_for_cameras=False):
        super().__init__(config, use_torch_for_cameras)

        self.last_stiffness_command_stamp = 0.0
        self.last_stiffness_params = np.zeros(6)
        self.current_waypoint_index = 0

    def load_params(self, config):
        super().load_params(config)

        # Parameters
        self.control_frequency = config.control_frequency
        self.dt = 1. / self.control_frequency

        self.cam_names = config.camera_names
        rospy.loginfo(f"Cameras to record from: {self.cam_names}")
        self.max_force_torque = config.controller.max_force_torque
        self.translation_stiffness_limits = config.controller.stiffness_limits.translation
        self.rotation_stiffness_limits = config.controller.stiffness_limits.rotation
        self.max_delta_translation = config.controller.max_delta_translation
        self.max_delta_rotation = config.controller.max_delta_rotation
        self.controller_config = config.controller
        self.initial_config = config.task.trajectory.init_qpos


        self.task_config = config.task

        self.actions_as_deltas = config.controller.actions_as_deltas
        self.stiffness_representation = config.controller.stiffness_representation
        self.translation_scale = config.controller.translation_scale
        self.rotation_scale = config.controller.rotation_scale
        assert self.stiffness_representation in STIFFNESS_REPRESENTATIONS, (
            "Error: unsupported stiffness representation"
            "Inputted : {}, Supported modes: {}".format(self.stiffness_representation, STIFFNESS_REPRESENTATIONS)
        )
        self.orientation_representation = config.controller.orientation_representation
        assert self.orientation_representation in ORIENTATION_REPRESENTATIONS, (
            "Error: unsupported orientation representation"
            "Inputted : {}, Supported modes: {}".format(self.orientation_representation, ORIENTATION_REPRESENTATIONS)
        )

    def set_controller_parameters(self):
        p_gains = self.controller_config['p_gains']
        d_gains = self.controller_config['d_gains']
        error_scale = self.controller_config['error_scale']
        iterations = self.controller_config['iterations']

        self.arm.set_control_mode(self.config.controller.mode)
        self.arm.update_pd_gains(p_gains, d_gains)
        self.arm.set_position_control_mode(enable=True)
        self.arm.update_selection_matrix(self.config.controller.selection_matrix)
        self.arm.set_solver_parameters(error_scale=error_scale, iterations=iterations, publish_state_feedback=True)
        self.arm.auto_switch_controllers = False
        self.arm.async_mode = True
        self.arm.zero_ft_sensor()

    def deactivate_compliance_control(self):
        self.arm.zero_ft_sensor()
        self.arm.activate_joint_trajectory_controller()

    def activate_compliance_control(self):
        self.arm.zero_ft_sensor()
        self.arm.activate_cartesian_controller()

    def reset(self, move_robot=True):
        if move_robot:
            assert self.reference_trajectory is not None, "Reference trajectory not set"
            self.set_controller_parameters()
            self.arm.activate_joint_trajectory_controller()

        return super().reset(move_robot=move_robot)

    def step(self, action) -> dm_env.TimeStep:
        # Check force/torque limits here and if needed return StepType.LAST to end episode.
        if not self.check_contact_force_limits():
            self.deactivate_compliance_control()
            return dm_env.TimeStep(
                step_type=dm_env.StepType.LAST,
                reward=self.get_reward(),
                discount=None,
                observation=self.get_observation())

        self.set_compliant_control_action(action)

        self.rate.sleep()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def set_compliant_control_action(self, action):
        """
            actions: dictionary of actions for each robot
        """

        if self.stiffness_representation == 'cholesky':
            stiffness_trans_matrix = math_utils.cholesky_vector_to_spd(action['action.stiffness_cholesky'][:6])
            stiffness_rot_matrix = math_utils.cholesky_vector_to_spd(action['action.stiffness_cholesky'][6:])
            # Only diagonal values are supported for now.
            stiff_trans = np.clip(np.diag(stiffness_trans_matrix), *self.translation_stiffness_limits)
            stiff_rot = np.clip(np.diag(stiffness_rot_matrix), *self.rotation_stiffness_limits)
        elif self.stiffness_representation == 'diag':
            stiff_trans = action['action.stiffness_diag'][:3]
            stiff_rot = action['action.stiffness_diag'][3:]

        stiff_trans = np.interp(stiff_trans, [-1, 1], self.translation_stiffness_limits)
        stiff_rot = np.interp(stiff_rot, [-1, 1], self.rotation_stiffness_limits)

        stiff_act = np.concatenate([stiff_trans, stiff_rot]).astype(np.int64)
        
        # Cap the bandwidth to change the controller's parameters to 40hz and only if the change is significant
        # TODO: make this more robust
        if rospy.get_time() - self.last_stiffness_command_stamp > 0.025 and \
                not np.all(np.isclose(self.last_stiffness_params, stiff_act, atol=5.0)):
            self.arm.update_stiffness(stiff_act)
            self.last_stiffness_params = np.copy(stiff_act)
            self.last_stiffness_command_stamp = rospy.get_time()

        action_translation = action['action.position']
        action_rotation = action['action.orientation']
        current_pose = self.arm.end_effector()
        if self.actions_as_deltas:
            assert len(action_rotation) == 3, "Rotation actions must be 3D for delta actions"
            delta_translation = self.clip_delta_actions(action_translation * self.translation_scale)
            delta_orientation = self.clip_delta_actions(action_rotation * self.rotation_scale)
        else: # actions are absolute positions
            assert len(action_rotation) == 6, "Rotation actions must be 6D for absolute actions"
            action_rotation_quaternion = transformations.quaternion_from_ortho6(action_rotation)
            delta_translation = self.clip_delta_actions(action_translation - current_pose[:3])
            delta_orientation = self.clip_delta_actions(transformations.quaternions_orientation_error(action_rotation_quaternion, current_pose[3:]))
        
        target_position = current_pose[:3] + delta_translation
        target_orientation = transformations.rotate_quaternion_by_rpy(*delta_orientation, current_pose[3:])

        target_pose = np.concatenate([target_position, target_orientation])
        self.arm.set_cartesian_target_pose(target_pose)

        #  # slow traffic to gripper controller
        # if rospy.get_time() - self.last_gripper_command_stamp > 0.1 and \
        #         not math.isclose(action['action.gripper'], self.last_gripper_command, abs_tol=0.05):
        #     self.arm.gripper.percentage_command(value=action['action.gripper'], wait=False)
        #     self.last_gripper_command_stamp = rospy.get_time()
        #     self.last_gripper_command = action['action.gripper']

    def clip_delta_actions(self, delta_translation, delta_orientation):
        """
            Make sure that the delta translation and delta orientation are not too large and won't cause jumps in motion 

            returns the clipped delta translation and delta orientation
        """

        clipped_delta_translation = np.clip(delta_translation, -self.max_delta_translation, self.max_delta_translation)
        clipped_delta_orientation = np.clip(delta_orientation, -self.max_delta_rotation, self.max_delta_rotation)

        return clipped_delta_translation, clipped_delta_orientation
