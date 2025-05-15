#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import rospy
from ur3e_openai.task_envs.force_control.slicing import UR3eSlicingEnv
from ur3e_openai.common import load_environment, clear_gym_params, load_ros_params
from tf2rl.algos.sac import SAC
import sys
import signal
from datetime import datetime
import os
import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow logging disabled


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class SlicingRLPolicy():
    def __init__(self, policy_directory, arm):
        start_time = timeit.default_timer()

        self.policy_directory = policy_directory

        self.use_real_robot = rospy.get_param("use_real_robot", True)
        self.use_gazebo_sim = rospy.get_param("use_gazebo_sim", False)

        clear_gym_params('ur3e_gym')
        clear_gym_params('ur3e_force_control')

        assert os.path.exists(self.policy_directory)

        self.env: UR3eSlicingEnv = self._load_environment_()

        actor_class = rospy.get_param("ur3e_gym/actor_class", "default")
        rospy.set_param("ur3e_gym/update_initial_conditions", False)

        self.policy = SAC(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.high.size,
            max_action=self.env.action_space.high[0],
            actor_class=actor_class,
        )

        checkpoint = tf.train.Checkpoint(policy=self.policy)
        _latest_path_ckpt = tf.train.latest_checkpoint(self.policy_directory)
        checkpoint.restore(_latest_path_ckpt).expect_partial()

        rospy.logerr(f"SlicingRLPolicy loading time: {timeit.default_timer()-start_time:.2f} secs")

    def _load_environment_(self):
        gyv_envs_params = self.policy_directory + '/ros_gym_env_params.yaml'

        assert os.path.exists(gyv_envs_params)

        rospy.set_param("ur3e_gym/rand_init_interval", 1)
        if self.use_real_robot and not self.use_gazebo_sim:
            load_ros_params(rospackage_name="ur3e_rl",
                            rel_path_from_package_to_file="config",
                            yaml_file_name="real/slicing_3d.yaml")
            rospy.loginfo("loading config file: %s" % "real/slicing_3d.yaml")
        else:
            load_ros_params(rospackage_name="ur3e_rl",
                            rel_path_from_package_to_file="config",
                            yaml_file_name="simulation/slicing_3d.yaml")
            rospy.loginfo("loading config file: %s" % "simulation/slicing_3d.yaml")

        steps_per_episode = rospy.get_param("ur3e_gym/steps_per_episode", 200)

        return load_environment(rospy.get_param('ur3e_gym/env_id'), max_episode_steps=steps_per_episode)

    def execute_policy(self, target_pose, record=False):
        steps_per_episode = rospy.get_param("ur3e_gym/steps_per_episode", 200)
        total_steps = 0
        episode_return = 0.

        obs_data = []

        # Set target pose
        rospy.set_param("ur3e_gym/target_pose", target_pose.tolist())
        print(f"target pose: {np.round(target_pose, 4).tolist()}")
        obs = self.env.reset()

        self.env.ur3e_arm.set_cartesian_target_pose(self.env.ur3e_arm.end_effector(tip_link="b_bot_knife_center"))
        self.env.ur3e_arm.activate_cartesian_controller()
        self.env.ur3e_arm.set_solver_parameters(error_scale=2.5)
        try:
            for j in range(steps_per_episode):
                action = self.policy.get_action(obs, test=True)
                next_obs, reward, done, info = self.env.step(action)
                episode_return += reward
                obs = next_obs

                if record:
                    obs_data.append(obs)

                if done:
                    total_steps = (j+1)
                    break

            if record:
                now = datetime.now()
                timesuffix = now.strftime("%Y-%m-%d_%H:%M:%S_%f")
                filename = self.policy_directory + "/test_obs_%s.npy" % timesuffix
                np.save(filename, obs_data)

            rospy.loginfo("Episode Steps: {0: 5} Return: {1: 5.4f}".format(total_steps, episode_return))
        finally:
            self.env.ur3e_arm.activate_joint_trajectory_controller()
