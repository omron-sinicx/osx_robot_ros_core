from comet.common.utils.vt_utils import (
    process_factored_action_dict,
    compute_directional_stiffness_diagonal,
)
from comet.common.datasets.postprocess_utils import load_characteristic_length
import rospy
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from osx_ur5e.base_env import BaseEnv
from osx_ur5e.timestep import TimeStep, STEP_MID, STEP_LAST
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

        self.last_compliance_stiffness = 0.0

    def load_params(self, config):
        super().load_params(config)

        # Parameters
        self.control_frequency = config.dataset.dataset.fps
        self.dt = 1. / self.control_frequency

        self.cam_names = config.dataset.cameras.keys()
        rospy.loginfo(f"Cameras to record from: {self.cam_names}")
        self.controller_config = config.controller
        self.max_force_torque = self.controller_config.safety_parameters.max_force_torque
        self.translation_stiffness_limits = self.controller_config.safety_parameters.stiffness_limits.translation
        self.rotation_stiffness_limits = self.controller_config.safety_parameters.stiffness_limits.rotation
        self.max_delta_translation = self.controller_config.safety_parameters.max_delta_translation
        self.max_delta_rotation = np.deg2rad(self.controller_config.safety_parameters.max_delta_rotation)
        self.initial_config = self.controller_config.init_qpos

        self.actions_as_deltas = self.controller_config.actions_as_deltas

        self.load_characteristic_length(config)

    def load_characteristic_length(self, config):
        """Read characteristic_length from the dataset's info.json, with fallback."""
        try:
            from lerobot.datasets.utils import load_info
            dataset_dir = Path(config.dataset.dataset.dir) / config.dataset.dataset.repo_id[0]
            info = load_info(dataset_dir)
            self.characteristic_length = load_characteristic_length(info)
            rospy.loginfo(f"characteristic_length={self.characteristic_length} (from dataset info.json)")
        except Exception as e:
            self.characteristic_length = 0.1
            rospy.logwarn(f"Could not read characteristic_length from dataset: {e}. "
                          f"Using default={self.characteristic_length}")

    def set_controller_parameters(self):
        p_gains = self.controller_config['p_gains']
        d_gains = self.controller_config['d_gains']
        error_scale = self.controller_config['error_scale']
        iterations = self.controller_config['iterations']

        self.arm.set_control_mode(self.controller_config.mode)
        self.arm.update_pd_gains(p_gains, d_gains)
        self.arm.update_selection_matrix(self.controller_config.selection_matrix)
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
            self.set_controller_parameters()
            self.arm.activate_joint_trajectory_controller()

        return super().reset(move_robot=move_robot)

    def step(self, action) -> TimeStep:
        # Check force/torque limits here and if needed return StepType.LAST to end episode.
        if not self.check_contact_force_limits():
            self.deactivate_compliance_control()
            return TimeStep(
                step_type=STEP_LAST,
                reward=self.get_reward(),
                discount=None,
                observation=self.get_observation())

        controller_action = self.prepare_action(action)
        self.set_compliant_control_action(controller_action)

        self.rate.sleep()

        return TimeStep(
            step_type=STEP_MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def prepare_action(self, action):
        """
            Prepare the action for the controller.
        """
        if "action.contact_direction" in action:  # FVT mode
            self.last_compliance_stiffness = action["action.estimated_stiffness"].item()
            fvt_action = process_factored_action_dict(action,
                                                      default_stiffness=self.controller_config.stiffness,
                                                      default_stiffness_rot=self.controller_config.stiffness,
                                                      characteristic_length=self.characteristic_length,
                                                      use_isotropic_stiffness=False,
                                                      controller_type="variable_kp",
                                                      orientation_representation="quaternion")
            controller_action = {
                "action.position": fvt_action[0:3],
                "action.orientation": fvt_action[3:7],
                "action.stiffness_diag": fvt_action[7:13],
            }
        elif "action.virtual_target_position" in action:  # VT mode
            self.last_compliance_stiffness = action["action.estimated_stiffness"].item()
            vt_pos = action["action.virtual_target_position"]
            vt_rot_ortho6 = action["action.virtual_target_rotation"]
            estimated_stiffness = action["action.estimated_stiffness"]

            stiffness_val = float(estimated_stiffness[0]) if np.ndim(estimated_stiffness) > 0 else float(estimated_stiffness)

            if "action.ref_position" in action:
                ref_position = action["action.ref_position"]
                ref_rotation_ortho6 = action["action.ref_rotation_ortho6"]
            else:
                current_eef = self.arm.end_effector()
                ref_position = current_eef[:3]
                ref_rotation_ortho6 = transformations.ortho6_from_quaternion(current_eef[3:])

            stiffness_matrix = compute_directional_stiffness_diagonal(
                position=ref_position,
                virtual_target_position=vt_pos,
                estimated_stiffness=stiffness_val,
                default_stiffness=self.controller_config.stiffness,
                default_stiffness_rot=self.controller_config.stiffness,
                rotation_ortho6=ref_rotation_ortho6,
                virtual_target_rotation_ortho6=vt_rot_ortho6,
                full_stiffness_matrix=True,
            )

            vt_quat = transformations.quaternion_from_ortho6(vt_rot_ortho6)

            controller_action = {
                "action.position": vt_pos,
                "action.orientation": vt_quat,
                "action.stiffness_diag": stiffness_matrix,
            }
        elif "action.position" in action and "action.orientation" in action:  # raw_actions mode
            self.last_compliance_stiffness = action["action.stiffness_diag"][0]
            controller_action = action
        else:
            raise ValueError(f"Invalid action: {action}")
        return controller_action

    def set_compliant_control_action(self, action):
        """
            actions: dictionary of actions for each robot
        """
        stiffness_matrix = action['action.stiffness_diag']
        if len(stiffness_matrix) == 6:
            stiffness_diag = stiffness_matrix
        elif len(stiffness_matrix) == 36:
            stiffness_diag = np.diag(stiffness_matrix)
        else:
            raise ValueError(f"Invalid stiffness matrix length: {len(stiffness_matrix)}")

        # Only update the stiffness if the change is significant
        if not np.all(np.isclose(self.last_stiffness_params, stiffness_diag, atol=5.0)):
            # Cap the bandwidth to change the controller's parameters to 40hz and only if the change is significant
            # TODO: make this more robust
            if True:  # rospy.get_time() - self.last_stiffness_command_stamp > 0.025:
                self.arm.update_stiffness(stiffness_matrix)
                self.last_stiffness_params = np.copy(stiffness_diag)
                self.last_stiffness_command_stamp = rospy.get_time()

        action_translation = action['action.position']
        action_rotation = action['action.orientation']
        current_pose = self.arm.end_effector()
        if self.actions_as_deltas:
            assert len(action_rotation) == 3, "Rotation actions must be 3D for delta actions"
            delta_translation, delta_orientation = self.clip_delta_actions(action_translation * self.translation_scale, action_rotation * self.rotation_scale)
        else:  # actions are absolute positions
            assert len(action_rotation) == 4, "Rotation actions must be quaternion (4D) for absolute actions"
            delta_translation, delta_orientation = self.clip_delta_actions(action_translation - current_pose[:3], transformations.quaternions_orientation_error(action_rotation, current_pose[3:]))

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
