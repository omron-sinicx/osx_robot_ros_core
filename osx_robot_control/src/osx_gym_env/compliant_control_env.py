import collections
import dm_env
import math
import rospy
import yaml

import matplotlib.pyplot as plt
import numpy as np

from osx_robot_control import math_utils
from osx_robot_control.core import OSXCore
from osx_robot_control.ur_fzi_force_control import URForceController
import vive_tracking_ros

from osx_gym_env.utils import ImageRecorder, compute_eef_velocity

from ur_control import transformations
import vive_tracking_ros.math_utils

ORIENTATION_REPRESENTATIONS = ['axis_angle', 'ortho6']
STIFFNESS_REPRESENTATIONS = ['cholesky', 'diag']


class CompliantControlEnv:
    """
    Environment for real robot manipulation for one or many robots with active compliant control method:
    Action space:   [
                    stiffness_translation  # cholesky vector (6) or diagonal (3) vector for translation
                    stiffness_rotation     # cholesky vector (6) or diagonal (3) vector for orientation
                    position (3)           # absolute Cartesian pose (x,y,z)
                    orientation            # Axis angle (3) or Orthogonal 6D (6) representation
                    gripper_positions (1), # normalized gripper position (0: close, 1: open)
                    ] * number of robots

    Observation space: {"eef_pos": [ position (3)           # absolute Cartesian pose (x,y,z)
                                     orientation            # Axis angle (3) or Orthogonal 6D (6) representation
                                     gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                   ] * number of robots
                        "eef_vel": [eef_vel (6),
                                    gripper_velocity (1),
                                   ] * number of robots
                        "images": {"a_bot_inside_camera": (480x640x3),  # h, w, c, dtype='uint8'
                                   "b_bot_inside_camera": (480x640x3),  # h, w, c, dtype='uint8'
                                   "extra_camera": (480x640x3),         # h, w, c, dtype='uint8'
                                  }
    """

    def __init__(self, config_filepath, use_torch_for_cameras=False):
        self.load_params(config_filepath)

        self.osx = OSXCore()

        self.target_wrench = np.zeros(6)
        self.image_recorder = ImageRecorder(init_node=False, camera_names=self.cam_names, use_torch=use_torch_for_cameras)

        self.rate = rospy.Rate(self.control_frequency)

        if self.stiffness_representation == 'diag':
            stiffness_dim = 6
        elif self.stiffness_representation == 'cholesky':
            stiffness_dim = 12

        if self.orientation_representation == 'axis_angle':
            orientation_dim = 3
        elif self.orientation_representation == 'ortho6':
            orientation_dim = 6

        position_dim = 3
        gripper_dim = 1
        self.num_robots = len(self.active_robot_names)
        self.obs_dim = (position_dim + orientation_dim + gripper_dim) * self.num_robots
        self.ft_dim = 6 * self.num_robots
        self.action_dim = (stiffness_dim + position_dim + orientation_dim + gripper_dim) * self.num_robots

        self.active_robots: dict[str, URForceController] = {}
        self.last_gripper_command = {}
        self.last_gripper_command_stamp = {}
        self.last_stiffness_command_stamp = {}
        self.last_stiffness_params = {}

        for robot in self.active_robot_names:
            self.active_robots[robot] = self.osx.active_robots[robot].force_controller
            self.last_gripper_command[robot] = 0.0
            self.last_gripper_command_stamp[robot] = 0.0
            self.last_stiffness_command_stamp[robot] = 0.0
            self.last_stiffness_params[robot] = np.zeros(6)

    def load_params(self, config_filepath):
        if isinstance(config_filepath, dict):
            config = config_filepath
        else:
            with open(config_filepath, 'r') as f:
                config = yaml.safe_load(f)

        # Parameters
        self.active_robot_names = config['active_robots']
        self.delta_actions = config['delta_actions']
        self.control_frequency = config.get('control_frequency', 20)
        self.dt = 1. / self.control_frequency

        self.cam_names = config.get('camera_names', [])
        rospy.loginfo(f"Cameras to record from: {self.cam_names}")
        self.max_force_torque = config.get('max_force_torque', np.array([50, 50, 50, 5, 5, 5], dtype=np.float32))
        self.translation_stiffness_limits = config['compliant_controller']['stiffness_limits']['translation']
        self.rotation_stiffness_limits = config['compliant_controller']['stiffness_limits']['rotation']
        self.controller_config = config['compliant_controller']
        self.initial_config = config['initial_configuration']
        # self.pic_config = config['pic_config']

        # Delta limits from robot's current pose to target pose
        self.max_delta_translation = config['safety']['max_delta_translation']
        self.max_delta_rotation = np.deg2rad(config['safety']['max_delta_rotation'])

        self.task_config = config['task_parameters']
        print(f"{self.task_config=}")
        self.vr_config = config.get('vr_config', None)

        self.stiffness_configuration = self.task_config.get('stiffness_configuration', None)
        self.stiffness_representation = self.task_config.get('stiffness_representation', 'cholesky')
        assert self.stiffness_representation in STIFFNESS_REPRESENTATIONS, (
            "Error: unsupported stiffness representation"
            "Inputted : {}, Supported modes: {}".format(self.stiffness_representation, STIFFNESS_REPRESENTATIONS)
        )
        self.orientation_representation = self.task_config.get('orientation_representation', 'axis_angle')
        assert self.orientation_representation in ORIENTATION_REPRESENTATIONS, (
            "Error: unsupported orientation representation"
            "Inputted : {}, Supported modes: {}".format(self.orientation_representation, ORIENTATION_REPRESENTATIONS)
        )

    def get_eef_components(self):
        position = []
        rotation_ortho6 = []
        rotation_axis_angle = []
        gripper = []
        eef_pos_axis_angle = []
        eef_pos_ortho6 = []
        for robot_name in self.active_robot_names:
            arm_pose = self.active_robots[robot_name].end_effector()
            gripper_qpos = [self.active_robots[robot_name].gripper.get_opening_percentage()]

            position.append(arm_pose[:3])
            rotation_axis_angle.append(transformations.axis_angle_from_quaternion(arm_pose[3:]))
            rotation_ortho6.append(transformations.ortho6_from_quaternion(arm_pose[3:]))
            gripper.append(gripper_qpos)
            eef_pos_axis_angle.append(np.concatenate([arm_pose[:3], transformations.axis_angle_from_quaternion(arm_pose[3:]), gripper_qpos]))
            eef_pos_ortho6.append(np.concatenate([arm_pose[:3], transformations.ortho6_from_quaternion(arm_pose[3:]), gripper_qpos]))

        return {
            "eef_pos.position": np.ravel(position),
            "eef_pos.rotation_axis_angle": np.ravel(rotation_axis_angle),
            "eef_pos.rotation_ortho6": np.ravel(rotation_ortho6),
            "eef_pos.gripper": np.ravel(gripper),
            "eef_pos_ortho6": np.ravel(eef_pos_ortho6),
            "eef_pos_axis_angle": np.ravel(eef_pos_axis_angle),
            "eef_pos": np.ravel(eef_pos_axis_angle),
        }

    def get_eef_pos(self, robot_name, orientation_representation='axis_angle'):
        # get current end effector pose [x,y,z] + [quat(4)]
        arm_pose = self.active_robots[robot_name].end_effector()
        rotation = None

        if orientation_representation == 'axis_angle':
            # convert quaternion to axis angle: [axis_angle(3)]
            rotation = transformations.axis_angle_from_quaternion(arm_pose[3:])
        elif orientation_representation == 'ortho6':
            # convert quaternion to axis angle: [ortho6(6)]
            rotation = transformations.ortho6_from_quaternion(arm_pose[3:])
        else:
            raise ValueError(f'Unsupported orientation_representation: {orientation_representation}')

        # gripper positions
        gripper_qpos = [self.active_robots[robot_name].gripper.get_opening_percentage()]
        return np.concatenate([arm_pose[:3], rotation, gripper_qpos])

    def get_eef_vel(self, robot_name):
        if not hasattr(self, 'previous_poses') or robot_name not in self.previous_time:
            self.previous_poses = {}
            self.previous_eef_velocities = {}
            self.previous_time = {}
            self.previous_poses[robot_name] = self.get_eef_pos(robot_name)
            self.previous_eef_velocities[robot_name] = np.zeros(14)
            self.previous_time[robot_name] = rospy.get_time()
            return self.previous_eef_velocities[robot_name]

        current_poses = self.get_eef_pos(robot_name)
        # Use the previous poses to get the present velocity.
        dt = rospy.get_time() - self.previous_time[robot_name]
        if math.isclose(dt, 0.0):
            rospy.logwarn("dt is close to zero!")
            dt = self.dt
        eef_velocity = compute_eef_velocity(current_poses[:6], self.previous_poses[robot_name][:6], dt)

        self.previous_poses = current_poses

        gripper_qpos = [self.active_robots[robot_name].gripper.get_velocity()]
        return np.concatenate([eef_velocity, gripper_qpos])

    def get_qpos(self, robot_name):
        arm_qpos = self.active_robots[robot_name].joint_angles()
        gripper_qpos = [self.active_robots[robot_name].gripper.get_opening_percentage()]  # Normalize
        return np.concatenate([arm_qpos, gripper_qpos])

    def get_qvel(self, robot_name):
        arm_qvel = self.active_robots[robot_name].joint_velocities()
        gripper_qvel = [self.active_robots[robot_name].gripper.get_velocity()]
        return np.concatenate([arm_qvel, gripper_qvel])

    def get_effort(self, robot_name):
        robot_effort = self.active_robots[robot_name].joint_efforts()
        gripper_effort = [0.0]
        return np.concatenate([robot_effort, gripper_effort])

    def get_ft(self, robot_name):
        return self.active_robots[robot_name].get_wrench(base_frame_control=True)

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self):
        obs = collections.OrderedDict()
        obs.update(self.get_eef_components())
        # obs['eef_vel'] = np.concatenate([self.get_eef_vel(robot_name) for robot_name in self.active_robot_names])
        obs['qpos'] = np.concatenate([self.get_qpos(robot_name) for robot_name in self.active_robot_names])
        obs['qvel'] = np.concatenate([self.get_qvel(robot_name) for robot_name in self.active_robot_names])
        obs['ft'] = np.concatenate([self.get_ft(robot_name) for robot_name in self.active_robot_names])
        # obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def move_to_initial_configuration(self, robot_name: str, initial_pose):
        if isinstance(initial_pose, str):
            success = self.osx.active_robots[robot_name].go_to_named_pose(initial_pose)
        else:
            success = self.osx.active_robots[robot_name].move_joints(joint_pose_goal=initial_pose, speed=0.2)
        assert success, f"Failed to move to initial configuration for robot '{self.osx.active_robots[robot_name].ns}' : {initial_pose}"

    def set_controller_parameters(self, robot_name):
        p_gains = self.controller_config['p_gains']
        d_gains = self.controller_config['d_gains']
        error_scale = self.controller_config['error_scale']
        iterations = self.controller_config['iterations']

        self.active_robots[robot_name].set_control_mode("spring-mass-damper")
        self.active_robots[robot_name].update_pd_gains(p_gains, d_gains)
        self.active_robots[robot_name].set_solver_parameters(error_scale=error_scale, iterations=iterations, publish_state_feedback=True)
        self.active_robots[robot_name].auto_switch_controllers = False
        self.active_robots[robot_name].async_mode = True
        self.active_robots[robot_name].zero_ft_sensor()

    def reset(self, move_robot=True):
        if move_robot:
            # TODO: define actions for gripper. probably no action.
            for robot_name in self.active_robot_names:
                self.set_controller_parameters(robot_name)
                self.osx.active_robots[robot_name].force_controller.activate_joint_trajectory_controller()

            for robot_name in self.initial_config:
                self.move_to_initial_configuration(robot_name, self.initial_config[robot_name])

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def deactivate_compliance_control(self):
        for robot_name in self.active_robot_names:
            self.active_robots[robot_name].zero_ft_sensor()
            self.active_robots[robot_name].activate_joint_trajectory_controller()

    def activate_compliance_control(self):
        for robot_name in self.active_robot_names:
            self.active_robots[robot_name].zero_ft_sensor()
            self.active_robots[robot_name].activate_cartesian_controller()

    def step(self, action):
        """
            action_dim: 38
                6 stiffness vector (Cholesky) x 2(translation, rotation) x 2 arms
                3 delta translation (x,y,z) x 3 delta rotation (axis-angle) x 2 arms
                1 joint x 2 grippers [Normalized]
        """
        start_time = rospy.get_time()

        # Check force/torque limits here and if needed return StepType.LAST to end episode.
        for robot_name in self.active_robot_names:
            if not self.check_contact_force_limits(robot_name):
                self.deactivate_compliance_control()
                return dm_env.TimeStep(
                    step_type=dm_env.StepType.LAST,
                    reward=self.get_reward(),
                    discount=None,
                    observation=self.get_observation())

        state_len = int(len(action) / self.num_robots)

        for i, robot_name in enumerate(self.active_robot_names):
            robot_action = action[i * state_len:(i+1) * state_len]
            safe_step = self.set_compliant_control_action(robot_action, robot_name)
            if not safe_step:
                return dm_env.TimeStep(
                    step_type=dm_env.StepType.LAST,
                    reward=self.get_reward(),
                    discount=None,
                    observation=self.get_observation())

        duration = rospy.get_time() - start_time

        # if duration < self.dt:
        #     rospy.loginfo_throttle(.5, f"duration {round(duration, 4)}")
        #     # rospy.sleep(self.dt - duration)
        # else:
        #     rospy.logwarn(f"Longer than expected execution: {self.dt} + {round(duration - self.dt, 4)}")
        # self.rate.sleep()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def check_safety(self, target_position, target_orientation, current_pose):
        """
            Make sure that the absolute error between the current pose of the robot and the target pose
            given by the policy is not huge and won't cause jumps in motion 
            
            returns if the action step is safe to use or not 
        """
        safe = True

        # distance from the robot's current pose to the controller's current pose
        error_translation = target_position - current_pose[:3]
        error_rotation = vive_tracking_ros.math_utils.orientation_error_as_euler(target_orientation, current_pose[3:])*2

        if np.any(np.abs(error_translation) > self.max_delta_translation) \
                or np.any(np.abs(error_rotation) > self.max_delta_rotation):
            safe = False

            # apply limits
            # error_translation = np.clip(error_translation, -self.max_delta_translation, self.max_delta_translation)
            # error_rotation = np.clip(error_rotation, -self.max_delta_rotation, self.max_delta_rotation)

        return safe
    
    def set_compliant_control_action(self, actions, robot_name):
        """
            action_dim: 19
                stiffness_translation 0:6
                stiffness_rotation 6:12
                delta_translation 12:15
                delta_rotation 15:18
                gripper_action 18
        """

        safe_step = True

        if self.stiffness_representation == 'cholesky':
            stiffness_trans_matrix = math_utils.cholesky_vector_to_spd(actions[:6])
            stiffness_rot_matrix = math_utils.cholesky_vector_to_spd(actions[6:12])
            # Only diagonal values are supported for now.
            stiff_trans = np.clip(np.diag(stiffness_trans_matrix), *self.translation_stiffness_limits)
            stiff_rot = np.clip(np.diag(stiffness_rot_matrix), *self.rotation_stiffness_limits)
            pos_start_at = 12
        elif self.stiffness_representation == 'diag':
            stiff_trans = actions[:3]
            stiff_rot = actions[3:6]
            pos_start_at = 6

        stiff_act = np.concatenate([stiff_trans, stiff_rot]).astype(np.int64)

        # Cap the bandwidth to change the controller's parameters to 40hz and only if the change is significant
        if rospy.get_time() - self.last_gripper_command_stamp[robot_name] > 0.025 and \
                not np.all(np.isclose(self.last_stiffness_params[robot_name], stiff_act, atol=5.0)):
            self.active_robots[robot_name].update_stiffness(stiff_act)
            self.last_stiffness_params[self.active_robots[robot_name].ns] = np.copy(stiff_act)

        current_pose = self.active_robots[robot_name].end_effector()

        if self.delta_actions:
            delta_translation = actions[pos_start_at:pos_start_at+3]
            delta_rotation = actions[pos_start_at+3:pos_start_at+6]  # always 3 dim for deltas

            # print("Current pose", np.round(current_pose, 4))
            target_position = current_pose[:3] + delta_translation
            target_orientation = transformations.rotate_quaternion_by_rpy(*delta_rotation, current_pose[3:])

            safe_step = self.check_safety(target_position, target_orientation, current_pose)

            target_pose = np.concatenate([target_position, target_orientation])
        else:
            target_position = actions[pos_start_at:pos_start_at+3]

            current_pose = self.active_robots[robot_name].end_effector() # to get the delta 

            rot_start_at = pos_start_at+3
            if self.orientation_representation == 'axis_angle':
                target_orientation = transformations.quaternion_from_axis_angle(actions[rot_start_at:rot_start_at+3])
            elif self.orientation_representation == 'ortho6':
                target_orientation = transformations.quaternion_from_ortho6(actions[rot_start_at:rot_start_at+6])

            safe_step = self.check_safety(target_position, target_orientation, current_pose)

            target_pose = np.concatenate([target_position, target_orientation])

         # slow traffic to gripper controller
        if rospy.get_time() - self.last_gripper_command_stamp[robot_name] > 0.1 and \
                not math.isclose(actions[-1], self.last_gripper_command[robot_name], abs_tol=0.05):
            # print(robot_name, f"gripper action: {actions[-1]:0.02f}")
            self.active_robots[robot_name].gripper.percentage_command(value=actions[-1], wait=False)
            self.last_gripper_command_stamp[robot_name] = rospy.get_time()
            self.last_gripper_command[robot_name] = actions[-1]

        # print("Target pose", np.round(target_pose, 4))
        # print("stiff_act", np.round(stiff_act, 3))
        if safe_step:
            self.active_robots[robot_name].set_cartesian_target_pose(target_pose)
        else:
            print("the step is not safe will shutdown") 
            self.deactivate_compliance_control()

        return safe_step


    def check_contact_force_limits(self, robot_name):
        """
            Check that contact force limits are not violated.
            Returns False if the limits are violated, 
            otherwise return True
        """
        # Safety limits: max force
        current_wrench = self.active_robots[robot_name].get_wrench()
        if np.any(np.greater(np.abs(current_wrench), self.max_force_torque)):
            rospy.logerr('Maximum force/torque exceeded {}'.format(np.round(current_wrench, 3)))
            return False
        return True

    def zero_ft_sensor(self):
        for robot_name in self.active_robot_names:
            self.active_robots[robot_name].zero_ft_sensor()


def get_actions():
    stiffness_trans = np.diag([500, 500, 500.])
    stiffness_rot = np.diag([20, 20, 20.])
    cholesky_vector_trans = math_utils.spd_to_cholesky_vector(stiffness_trans)
    cholesky_vector_rot = math_utils.spd_to_cholesky_vector(stiffness_rot)
    actions = np.concatenate([
        # a_bot
        cholesky_vector_trans,
        cholesky_vector_rot,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1],
        # b_bot
        cholesky_vector_trans,
        cholesky_vector_rot,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1],
    ])
    return actions


def test_real_control():
    """
    Test real control for dual arm in cartesian space with
    compliant control.

    A mock action is obtain from `get_actions()`.
    Then use it as actions to step the environment.
    The environment returns full observations including images.
    """

    rospy.init_node("test_real_control")

    # test parameters
    onscreen_render = False
    render_cam = 'wrist_camera'
    num_steps = 100

    # setup the environment
    config_filepath = '/root/osx-ur/catkin_ws/src/osx_robot_control/config/gym/dual_arm_simple.yaml'
    env = CompliantControlEnv(config_filepath)
    env.delta_actions = True

    input("Press any key to continue: `reset env`")
    ts = env.reset()

    episode = [ts]

    # setup visualization
    if onscreen_render:
        plt.ion()
        fig = plt.figure()
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])

    input("Press any key to continue: `start episode`")

    start_time = rospy.get_time()
    for t in range(num_steps):
        action = get_actions()
        ts = env.step(action)
        episode.append(ts)
        if ts.last():
            break

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            fig.canvas.draw()
            fig.canvas.flush_events()
    print("Total env time:", round(rospy.get_time() - start_time, 2), "expected:", num_steps*env.dt)

    env.a_bot.activate_joint_trajectory_controller()
    env.b_bot.activate_joint_trajectory_controller()


if __name__ == '__main__':
    test_real_control()
