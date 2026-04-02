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

from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations


class CompliantControlEnv:
    """
    Environment for real robot manipulation for one or many robots with active compliant control method:
    Action space:   [
                    position (3)             # absolute Cartesian pose (x,y,z)
                    orientation (3           # Axis angle (3)
                    gripper_positions (1),   # normalized gripper position (0: close, 1: open)
                    ] 

    Observation space: {"eef_pos": 6
                        "eef_vel": 6
                        "gripper": 1
                        "force": 3
                        "torque": 3
                        "images": {"a_bot_inside_camera": (480x640x3),  # h, w, c, dtype='uint8'
                                   "b_bot_inside_camera": (480x640x3),  # h, w, c, dtype='uint8'
                                   "extra_camera": (480x640x3),         # h, w, c, dtype='uint8'
                                  }
    """

    def __init__(self, config_filepath, use_torch_for_cameras=False):
        self.load_params(config_filepath)

        self.arm = CompliantController(ee_link='tool0',  # FIXME: what is the corresponding one with robosuite?
                                       ft_topic='wrench',
                                       gripper_type=None)

        # obs
        self.arm.end_effector()
        self.arm.end_effector_velocity()
        self.arm.get_wrench()  # Force/Torque sensor

        # actions
        self.arm.set_cartesian_target_pose(target_pose)
        self.arm.gripper.percentage_command(value=0, wait=False)

        # 33

        self.target_wrench = np.zeros(6)
        # ['wrist_camera', 'front_camera']
        self.image_recorder = ImageRecorder(init_node=False, camera_names=self.cam_names, use_torch=use_torch_for_cameras)

        self.rate = rospy.Rate(self.control_frequency)

        self.active_robots: dict[str, URForceController] = {}
        self.last_gripper_command = {}
        self.last_gripper_command_stamp = {}

        self.active_robots = self.osx.active_robots.force_controller
        self.last_gripper_command = 0.0
        self.last_gripper_command_stamp = 0.0

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

    # def get_eef_components(self):
    #     position = []
    #     rotation_ortho6 = []
    #     rotation_axis_angle = []
    #     rotation_euler = []
    #     gripper = []
    #     eef_pos_axis_angle = []
    #     eef_pos_ortho6 = []
    #     for robot_name in self.active_robot_names:
    #         arm_pose = self.active_robots[robot_name].end_effector()
    #         gripper_qpos = [self.active_robots[robot_name].gripper.get_opening_percentage()]

    #         position.append(arm_pose[:3])
    #         # rotation_axis_angle.append(transformations.axis_angle_from_quaternion(arm_pose[3:]))
    #         # rotation_ortho6.append(transformations.ortho6_from_quaternion(arm_pose[3:]))
    #         rotation_euler.append(transformations.euler_from_quaternion(arm_pose[3:]))
    #         gripper.append(gripper_qpos)
    #         # eef_pos_axis_angle.append(np.concatenate([arm_pose[:3], transformations.axis_angle_from_quaternion(arm_pose[3:]), gripper_qpos]))
    #         # eef_pos_ortho6.append(np.concatenate([arm_pose[:3], transformations.ortho6_from_quaternion(arm_pose[3:]), gripper_qpos]))

    #     return {
    #         "eef_pos.position": np.ravel(position),
    #         "eef_pos.rotation_euler": np.ravel(rotation_euler),
    #         # "eef_pos.rotation_axis_angle": np.ravel(rotation_axis_angle),
    #         # "eef_pos.rotation_ortho6": np.ravel(rotation_ortho6),
    #         "eef_pos.gripper": np.ravel(gripper),
    #         "eef_pos_ortho6": np.ravel(eef_pos_ortho6),
    #         # "eef_pos_axis_angle": np.ravel(eef_pos_axis_angle),
    #         # "eef_pos": np.ravel(eef_pos_axis_angle),
    #     }

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
        if orientation_representation == 'euler':
            # convert quaternion to axis angle: [axis_angle(3)]
            rotation = transformations.euler_from_quaternion(arm_pose[3:])
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
            success = self.osx.active_robots[robot_name].set_joint_position_goal(joint_pose_goal=initial_pose, speed=0.2)
        assert success, f"Failed to move to initial configuration for robot '{self.osx.active_robots[robot_name].ns}' : {initial_pose}"

    def set_controller_parameters(self, robot_name):
        p_gains = self.controller_config['p_gains']
        d_gains = self.controller_config['d_gains']
        error_scale = self.controller_config['error_scale']
        iterations = self.controller_config['iterations']

        # Compliance control needed?
        # self.active_robots[robot_name].set_control_mode("spring-mass-damper")
        self.active_robots[robot_name].set_control_mode("parallel")
        self.active_robots[robot_name].update_pd_gains(p_gains, d_gains)
        self.active_robots[robot_name].update_selection_matrix(np.ones(6))
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
            action_dim: 6
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

        if duration < self.dt:
            rospy.loginfo_throttle(.5, f"duration {round(duration, 4)}")
        else:
            rospy.logwarn(f"Longer than expected execution: {self.dt} + {round(duration - self.dt, 4)}")
        self.rate.sleep()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def set_compliant_control_action(self, actions, robot_name):
        """
            action_dim: 19
                stiffness_translation 0:6
                stiffness_rotation 6:12
                delta_translation 12:15
                delta_rotation 15:18
                gripper_action 18
        """

        current_pose = self.active_robots[robot_name].end_effector()

        if self.delta_actions:
            delta_translation = actions[:3]
            delta_rotation = actions[3:6]  # always 3 dim for deltas

            # print("Current pose", np.round(current_pose, 4))
            target_position = current_pose[:3] + delta_translation
            target_orientation = transformations.rotate_quaternion_by_rpy(*delta_rotation, current_pose[3:])

            target_pose = np.concatenate([target_position, target_orientation])

         # slow traffic to gripper controller
        if rospy.get_time() - self.last_gripper_command_stamp[robot_name] > 0.1 and \
                not math.isclose(actions[-1], self.last_gripper_command[robot_name], abs_tol=0.05):
            # print(robot_name, f"gripper action: {actions[-1]:0.02f}")
            self.active_robots[robot_name].gripper.percentage_command(value=actions[-1], wait=False)
            self.last_gripper_command_stamp[robot_name] = rospy.get_time()
            self.last_gripper_command[robot_name] = actions[-1]

        self.active_robots[robot_name].set_cartesian_target_pose(target_pose)

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
