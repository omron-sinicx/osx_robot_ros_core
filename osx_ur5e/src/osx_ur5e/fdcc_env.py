import rospy
import numpy as np
from omegaconf import DictConfig

from osx_ur5e.base_env import BaseEnv
from osx_ur5e.timestep import TimeStep, STEP_MID, STEP_LAST
from ur_control import transformations

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

        self.control_frequency = config.dataset.fps
        self.dt = 1. / self.control_frequency

        self.cam_names = config.dataset.cameras.keys()
        rospy.loginfo(f"Cameras to record from: {self.cam_names}")

        safety = config.controller.safety_parameters
        self.max_force_torque = safety.max_force_torque
        self.translation_stiffness_limits = safety.stiffness_limits.translation
        self.rotation_stiffness_limits = safety.stiffness_limits.rotation
        self.max_delta_translation = safety.max_delta_translation
        self.max_delta_rotation = np.deg2rad(safety.max_delta_rotation)
        self.controller_config = config.controller
        self.actions_as_deltas = config.controller.actions_as_deltas

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
            from comet.common.utils.vt_utils import process_factored_action_dict
            fvt_action = process_factored_action_dict(action,
                                                      default_stiffness=self.controller_config.stiffness,
                                                      default_stiffness_rot=self.controller_config.stiffness,
                                                      characteristic_length=0.1,
                                                      use_isotropic_stiffness=False,
                                                      controller_type="variable_kp",
                                                      orientation_representation="quaternion")
            controller_action = {
                "action.position": fvt_action[0:3],
                "action.orientation": fvt_action[3:7],
                "action.stiffness_diag": fvt_action[7:13],
            }
        elif "action.position" in action and "action.orientation" in action:  # raw_actions mode
            controller_action = action
        else:  # TODO implement VT mode
            raise ValueError(f"Invalid action: {action}")
        return controller_action

    def set_compliant_control_action(self, action):
        """
            actions: dictionary of actions for each robot
        """
        stiff_trans = action['action.stiffness_diag'][:3]
        stiff_rot = action['action.stiffness_diag'][3:]

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
