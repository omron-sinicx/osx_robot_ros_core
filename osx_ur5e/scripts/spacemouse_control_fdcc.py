# license included since this code was adapted from cambel UR3 repo

# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import sys  # noqa
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'robosuite_meta/learning'))  # noqa

from import_weights import import_weights
from utils import get_logger
from encoder_utils import encoder_prediction, load_policy
from models import build_encoder_model
import colorsys
from matplotlib import pyplot as plt
import multiprocessing as mp
import rospy
import sys
import cv2
import signal
import argparse
import time
import random
from datetime import datetime

from collections import deque
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch
np.set_printoptions(precision=6, linewidth=2000)
torch.set_printoptions(precision=6, linewidth=2000)
torch.set_float32_matmul_precision('medium')


try:
    from ur_control.exceptions import InverseKinematicsException
    from ur_control.constants import GripperType
    from ur_control import transformations
    from ur_control.mouse_6d import Mouse6D
    from ur_control.arm import Arm
    from ur_control.fzi_cartesian_compliance_controller import CompliantController
except Exception as e:
    pass


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

log = get_logger(__name__)
np.set_printoptions(suppress=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f'Using device {device}')

axes = "rxyz"

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
POLICY_CKPT_DIR = MODELS_DIR / 'old/85_rigid_fdcc_basic_shapes'
# POLICY_CKPT_DIR = MODELS_DIR / 'old/policy_rigid_fdcc_basic_shapes'
# ENCODER_CKPT_DIR = MODELS_DIR / 'encoder_np_soft_basic_50hz'
ENCODER_CKPT_DIR = MODELS_DIR / 'old/encoder_np_rigid_basic_shape_v2'
# ENCODER_CKPT_DIR = MODELS_DIR / 'old/encoder_np_rigid_basic_shape/2025_12_05_13_22_27'

# Helper functions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def e2q(e):
    return transformations.quaternion_from_euler(e[0], e[1], e[2], axes=axes)


def xya2df(X, Y, A):
    X, Y, A = [np.array(v) for v in [X, Y, A]]
    df = pd.DataFrame({
        **{f'non_priv-{i}': X[:, i] for i in range(X.shape[1])},
        **{f'priv-{i}': Y[:, i] for i in range(Y.shape[1])},
        **{f'action-{i}': A[:, i] for i in range(A.shape[1])},
    })
    return df


def load_models(cfg):
    """Load policy and encoder models"""
    policy = import_weights(Path(cfg.policy_path).parent.parent)
    log.info(f'Loaded policy: {cfg.policy.model.name} from {cfg.policy_path}')

    encoder = build_encoder_model(
        cfg.encoder, cfg.encoder.non_priv_dim, cfg.encoder.priv_dim,
        seq_len=cfg.encoder.input_horizon,
    ).to(device)
    encoder.load_state_dict(torch.load(
        cfg.encoder_path, map_location=device, weights_only=True))
    encoder = torch.compile(encoder)  # compile for faster inference
    encoder.eval()
    log.info(f"Loading encoder weights from {cfg.encoder_path}")

    return policy, encoder


class SpaceMouseFDCCController:
    """FDCC controller with spacemouse teleoperation and policy inference"""

    def __init__(self, args):
        """Initialize controller with all config and hardware"""
        self.args = args

        # Load configs
        self.control_cfg = OmegaConf.load(
            Path(__file__).resolve().parent.parent / 'config/control.yaml'
        )
        self.policy_cfg = OmegaConf.load(str(ENCODER_CKPT_DIR / 'config.yaml'))

        # Load and merge policy config from checkpoint dir
        policy_ckpt_cfg = OmegaConf.load(str(POLICY_CKPT_DIR / 'config.yaml'))
        self.policy_cfg = OmegaConf.merge(self.policy_cfg, policy_ckpt_cfg)

        self.policy_cfg.policy_path = POLICY_CKPT_DIR / 'best_ckpt/best_model.zip'
        # self.policy_cfg.policy_path = POLICY_CKPT_DIR / 'SAC.zip'
        self.policy_cfg.encoder_path = ENCODER_CKPT_DIR / 'ckpt/best.pt'

        # Set scaling constants
        self._init_scaling_constants()

        # Initialize hardware
        self._init_robot_controllers()

        # Load models
        set_seed(self.policy_cfg.seed)
        self.policy, self.encoder = load_models(self.policy_cfg)

        # State
        self.non_priv_queue = deque(maxlen=self.policy_cfg.encoder.input_horizon)
        self.X = []
        self.Y = []
        self.A = []
        self.step = 0
        self.is_begin = True

    def _control_spring_mass_damper(self):
        """Switch to spring-mass-damper control mode"""
        cfg = self.control_cfg.robot_config.fdcc
        self.arm.set_control_mode("spring-mass-damper")
        self.arm.update_stiffness(np.array(cfg.stiffness))
        self.arm.update_pd_gains(cfg.pd_gains, cfg.d_gains)

    def _control_parallel(self):
        """Switch to parallel control mode"""
        cfg = self.control_cfg.robot_config.osc_position
        self.arm.set_control_mode("parallel")
        self.arm.update_selection_matrix(np.ones(6))
        self.arm.update_stiffness(np.array(cfg.stiffness))
        self.arm.update_pd_gains(cfg.pd_gains, cfg.d_gains)

    def _init_scaling_constants(self):
        """Set scaling based on gripper type"""
        gripper_type = self.policy_cfg.policy.env.gripper_type

        # TODO: set these in the control.yaml config instead of hardcoding here.
        if gripper_type == 'Robotiq85Gripper':
            self.position_scale = 0.00035
            self.rotation_scale = 0.001
            self.stiffness_min = np.array([5, 5, 5, 5, 5, 5])
            self.stiffness_max = np.array([1000, 1000, 1000, 200, 200, 200])
            log.info(f'Using RIGID gripper scaling (position={self.position_scale}, rotation={self.rotation_scale})')
        elif gripper_type == 'Robotiq85GripperSoft':
            self.position_scale = 0.0022
            self.rotation_scale = 0.005
            self.stiffness_min = np.array([100, 100, 100, 20, 20, 20])
            self.stiffness_max = np.array([500, 500, 500, 100, 100, 100])
            log.info(f'Using SOFT gripper scaling (position={self.position_scale}, rotation={self.rotation_scale})')
        else:
            raise ValueError(f"Unknown gripper: {gripper_type}")

    def _init_robot_controllers(self):
        """Initialize arm and mouse"""
        self.arm = CompliantController(gripper_type=None)

        controller_type = self.policy_cfg.policy.env.control_space

        if controller_type == 'FDCC':
            self._control_spring_mass_damper()
        else:
            self._control_parallel()

        if not self.arm.dashboard_services.activate_ros_control_on_ur():
            raise RuntimeError("Failed to activate ROS control")

        self.mouse6d = Mouse6D()
        log.info('Initialized arm and mouse6D')

    def _get_obs(self, rs_sub=None, x=None, q=None):
        """Get observations and check collision"""
        x = self.arm.end_effector() if x is None else x
        q = self.arm.joint_angles() if q is None else q

        wrist_pos = np.array(x[:3])
        wrist_pos_rel = wrist_pos - self.control_cfg.task.hole_pos_base
        scaled_wrist_pos_rel = wrist_pos_rel / self.policy_cfg.policy.env.obs_pose_scale

        wrist_wrench = self.arm.get_wrench()
        wrist_force = wrist_wrench[:3]
        scaled_wrist_force = wrist_force / self.policy_cfg.policy.env.obs_force_scale
        wrist_torque = wrist_wrench[3:]
        scaled_wrist_torque = wrist_torque / self.policy_cfg.policy.env.obs_torque_scale

        obs_non_priv = np.hstack((
            scaled_wrist_pos_rel,
            scaled_wrist_force,
            scaled_wrist_torque,
        ))

        if rs_sub:
            # vision data
            rgb = np.array(rs_sub.get_rgb())
            raw_rgb = rgb.copy()
        else:
            rgb = None
            raw_rgb = None

        ret = {
            'non_priv': obs_non_priv,
            'rgb': rgb,
            'raw_rgb': raw_rgb,
            'depth': None,
        }

        # Check collision
        collision = (np.linalg.norm(wrist_force) > self.control_cfg.collision.force_threshold or
                     np.linalg.norm(wrist_torque) > self.control_cfg.collision.torque_threshold)
        if collision:
            print("Force", np.linalg.norm(wrist_force), 'Torque', np.linalg.norm(wrist_torque))
        return ret, collision

    def _check_success(self, x):
        """Check if task condition met"""
        wrist_pos = np.array(x[:3])
        pos_diff = wrist_pos - np.array(self.control_cfg.task.hole_pos_base)
        return (np.abs(pos_diff[0]) < self.control_cfg.success.xy_tolerance and
                np.abs(pos_diff[1]) < self.control_cfg.success.xy_tolerance and
                np.abs(pos_diff[2]) < self.control_cfg.success.z_tolerance)

    def run(self):
        """Main control loop"""
        # Get constants
        SPEED_ACTION_SCALE = self.control_cfg.control.speed_action_scale
        SPACEMOUSE_SENSITIVITY = self.control_cfg.control.spacemouse_sensitivity
        CONTROL_FREQ = self.control_cfg.control.control_freq
        HORIZON = self.control_cfg.control.horizon
        controller_type = self.policy_cfg.policy.env.control_space

        HOLE_POS_BASE = np.array(self.control_cfg.task.hole_pos_base)
        HOME_POS = HOLE_POS_BASE + np.array(self.control_cfg.task.home_offset)

        Q_HOME = tuple(self.control_cfg.task.q_home)
        INITIAL_ORIENTATION = np.array(self.control_cfg.task.initial_orientation)
        dt = 1.0 / CONTROL_FREQ
        rate = rospy.Rate(CONTROL_FREQ)

        log.info(f'Set seed to {self.policy_cfg.seed}')

        if self.args.viz or self.args.use_seg:
            self.policy_cfg.policy.env.use_rgb = True
            self.policy_cfg.policy.env.use_depth = True

        if CONTROL_FREQ != self.policy_cfg.policy.env.control_freq:
            log.warning(
                f"The control freq is different the config. Expect {self.policy_cfg.policy.env.control_freq}, got {CONTROL_FREQ}")

        assert Path(self.policy_cfg.encoder_path).exists(), f"Encoder not found: {self.policy_cfg.encoder_path}"
        assert Path(self.policy_cfg.policy_path).exists(), f"Policy not found: {self.policy_cfg.policy_path}"

        # Initialize SAM2 process
        self.args.use_seg = self.args.use_seg or self.policy_cfg.policy.env.seg_mode
        if self.args.use_seg:
            seg_sub = SegSubscriber()

        rs_sub = None  # Placeholder for RealSense

        # Rerun setup
        if self.args.viz:
            rr.init("traj")
            rr.spawn(memory_limit='50%')
            rr.log(
                "raw_camera",
                rr.Pinhole(
                    resolution=[640, 480],
                    image_from_camera=[[605.71160889,   0., 330.10424805],
                                       [0., 605.06787109, 244.28578186],
                                       [0.,   0.,   1.]],
                    camera_xyz=rr.ViewCoordinates.RDF
                )
            )

        try:
            x_des = self.arm.end_effector().copy()
            self.arm.zero_ft_sensor()

            prev_time = time.time()
            action = np.zeros(3)

            self.arm.activate_cartesian_controller()

            log.info("Start Moving Now!")
            while not rospy.is_shutdown():
                if self.arm.dashboard_services.is_protective_stopped():
                    rospy.logerr_throttle(1, "Protective stop triggered")
                    rospy.logerr(f"last action {action.tolist()}")
                    self.arm.joint_traj_controller.stop()
                    break

                # Observation & Logic
                x = self.arm.end_effector()
                q = self.arm.joint_angles()
                x_des = x.copy()

                obs, collision = self._get_obs(rs_sub, x=x, q=q)

                if collision:
                    log.info(f'Number of step {self.step}')
                    log.info(f'Collision!!')

                    self.X = []
                    self.Y = []
                    self.A = []
                    self.is_begin = True
                    self.step = 0
                    self.arm.set_cartesian_target_pose(self.arm.end_effector())
                    input("continue?")

                # Process SAM2
                if self.args.use_seg:
                    obs['depth'] = seg_sub.get_seg_depth()

                # Tensorify
                for k, v in obs.items():
                    if v is not None:
                        obs[k] = torch.from_numpy(v.astype(np.float32))
                self.non_priv_queue.append(obs['non_priv'])

                if self.is_begin:
                    rospy.loginfo_throttle(1, 'Setup at beginning...')

                    if self.args.pos_var:
                        xy_var = np.random.uniform(-0.02, 0.02, size=2)
                        pos_var = np.array([*xy_var, 0])
                        target_pos = HOME_POS + pos_var
                        rospy.loginfo_throttle(1, f'sample xy var: {xy_var}')
                    else:
                        target_pos = HOME_POS.copy()

                    with torch.inference_mode():
                        policy_obs, pred_priv = encoder_prediction(
                            self.encoder, obs, self.non_priv_queue, device=device)
                        policy_obs = policy_obs.cpu().numpy()
                        self.policy.predict(policy_obs, deterministic=True)

                    # wait for data to be available
                    if (len(self.non_priv_queue) >= self.policy_cfg.encoder.input_horizon) \
                            and (not self.args.use_seg or (self.args.use_seg and depth.any())):
                        rospy.loginfo_throttle(1, 'Setup done')
                        self.is_begin = False

                # Control Logic
                if self.mouse6d.joy_buttons[0] == 1 and self.mouse6d.joy_buttons[1] == 1:
                    rospy.loginfo_throttle(1, 'Moving to gripper home')

                    EEF_TO_WRIST_OFFSET = np.array(
                        [0, 0, - GRIP_TO_WRIST - PEG_Z_SIZE - COMPENSATE_FOR_HOLE])
                    x_pos = x[:3].copy()
                    delta = (x_pos + EEF_TO_WRIST_OFFSET) - target_pos

                    to_home_error = np.linalg.norm(delta)
                    if to_home_error < 1e-4:
                        log.info(f'Moved to home')
                    else:
                        if to_home_error > POLICY_ACTION_SCALE:
                            delta = delta / to_home_error * POLICY_ACTION_SCALE
                        x_des[:3] = x_pos - delta
                        x_des[3:] = INITIAL_ORIENTATION
                        self.arm.set_cartesian_target_pose(x_des)

                elif self.mouse6d.joy_buttons[1] == 1:
                    self.arm.zero_ft_sensor()
                    rospy.loginfo_throttle(1, 'Zero the FT sensor')
                    rospy.loginfo_throttle(1, f'Current q pose {q=}')
                    self.X = []
                    self.Y = []
                    self.A = []
                    self.is_begin = True
                    self.step = 0
                    rospy.loginfo_throttle(1, 'Reset states')
                else:
                    if self.mouse6d.joy_buttons[0] == 1:
                        if self.is_begin:
                            rospy.loginfo_throttle(1, 'Wait for begin')
                        else:
                            is_success = self._check_success(x)
                            if is_success or self.step >= HORIZON:
                                log.info(f'Step {self.step}, Success: {is_success}')

                                self._control_parallel()

                                if self.args.save:
                                    df = xya2df(self.X, self.Y, self.A)
                                    output_path = Path(
                                        f'trajectories/{datetime.now().strftime("%m%d-%H%M%S")}.csv')
                                    output_path.parent.mkdir(exist_ok=True)
                                    df.to_csv(output_path, index=False)
                                    log.info(f'Saved: {output_path}')

                                self.X = []
                                self.Y = []
                                self.A = []
                                self.is_begin = True
                                self.step = 0
                                self.arm.set_cartesian_target_pose(self.arm.end_effector())
                                input("continue?")

                            # Inference
                            with torch.inference_mode():
                                policy_obs, pred_priv = encoder_prediction(
                                    self.encoder, obs, self.non_priv_queue, device=device)
                                policy_obs = policy_obs.cpu().numpy()
                                action, _ = self.policy.predict(policy_obs, deterministic=True)
                            print("action", action)
                            action = action[0]

                            if controller_type == 'FDCC':
                                # FDCC: 12D action = [pos(3), ori(3), stiffness(6)]
                                delta_pos = action[0:3]
                                delta_ori = action[3:6]  # axis-angle
                                stiffness_raw = action[6:12]

                                # Apply variable stiffness
                                stiffness = self.stiffness_min + (stiffness_raw + 1.0) / 2.0 * (self.stiffness_max - self.stiffness_min)
                                stiffness = np.clip(stiffness, self.stiffness_min, self.stiffness_max)
                                self.arm.update_stiffness(stiffness)

                                # Apply position delta
                                x_des[:3] += delta_pos * self.position_scale * SPEED_ACTION_SCALE

                                # Apply orientation delta
                                scaled_delta_ori = delta_ori * self.rotation_scale * SPEED_ACTION_SCALE

                                # Apply orientation delta
                                if np.linalg.norm(scaled_delta_ori) > 1e-6:
                                    angle = np.linalg.norm(scaled_delta_ori)
                                    axis = scaled_delta_ori / angle
                                    delta_quat = transformations.quaternion_about_axis(angle, axis)
                                    x_des[3:] = transformations.quaternion_multiply(x_des[3:], delta_quat)
                            else:
                                x_des[:3] += action * self.position_scale * SPEED_ACTION_SCALE

                            self.step += 1

                            self.X.append(obs['non_priv'].cpu().numpy())
                            self.Y.append(pred_priv.cpu().numpy()[0])
                            self.A.append(action)

                            # Viz code omitted for brevity unless enabled
                            if self.args.viz:
                                rr.log('wrist_pos', rr.Points3D(obs['non_priv'][:3]))
                                if 'raw_rgb' in obs:
                                    raw_rgb = obs['raw_rgb'].type(torch.uint8)
                                    rr.log('raw_camera/raw_rgb',
                                           rr.Image(raw_rgb).compress(jpeg_quality=80))
                                if obs.get('depth') is not None:
                                    d = obs['depth']
                                    rr.log('obs/depth',
                                           rr.DepthImage(d.cpu().numpy(), meter=1000.0))
                    else:
                        # Switch to parallel (stiff) mode for manual spacemouse control
                        self._control_parallel()

                        # Spacemouse
                        spacenav = np.array(self.mouse6d.twist)
                        spacenav_normalized = spacenav.copy() / 0.68359375
                        spacenav_normalized[:2] *= -1.0
                        spacenav_normalized[3:5] *= -1.0

                        x_des[:3] += spacenav_normalized[:3] * SPACEMOUSE_SENSITIVITY
                        x_des[3:] = transformations.rotate_quaternion_by_rpy(
                            *(spacenav_normalized[3:] * np.deg2rad(1.0)), x_des[-4:])

                x_des[3:] = INITIAL_ORIENTATION
                try:
                    self.arm.set_cartesian_target_pose(x_des)
                except InverseKinematicsException:
                    pass

                rate.sleep()
                now_time = time.time()
                current_fps = np.round(1 / (now_time - prev_time), 4)
                prev_time = now_time
                if round(current_fps) < CONTROL_FREQ:
                    rospy.logwarn_throttle(
                        1, f'{current_fps=} is lower than expected {CONTROL_FREQ}')

        finally:
            # Return to joint trajectory control for safe shutdown
            try:
                self.arm.activate_joint_trajectory_controller()
            except AttributeError:
                log.warning("Controller doesn't support joint trajectory activation")
            # Clean up
            cv2.destroyAllWindows()
            if self.args.use_seg:
                pass


def main():
    """Main entry point"""
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument('-seg', '--use-seg', action='store_true')
    parser.add_argument('-pv', '--pos-var', action='store_true')
    parser.add_argument('-viz', '--viz', action='store_true')
    parser.add_argument('-save', '--save', action='store_true')
    parser.add_argument('--log_level', '-log', type=str, default='info')

    args = parser.parse_args(rospy.myargv()[1:])
    log.setLevel(args.log_level.upper())

    rospy.init_node("joint_position_keyboard")
    log.info('Initialized ROS node')

    # Create and run controller
    controller = SpaceMouseFDCCController(args)
    controller.run()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
