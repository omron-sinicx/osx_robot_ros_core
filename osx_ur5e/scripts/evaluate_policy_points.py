#!/usr/bin/env python3
"""Evaluate a LeRobot ACT policy trained on live-tracked object keypoints.

Counterpart to evaluate_policy.py, for policies trained on datasets produced
by point_policy/robot_utils/ur5e/convert_pkl_to_lerobot.py:
    observation.state             -- EE cartesian pose (pos + ortho6d) + gripper (10,)
    observation.environment_state -- tracked object keypoints, world-frame 3D (N*3,)
    action                        -- next EE cartesian pose + gripper (10,)
No images are fed to the policy at all (use_videos=False) -- instead this
script consumes the offline pipeline's tracked-point signal live, produced by
a separate tracking_node.py ROS node (DIFT localization + causal TAPIR
tracking + two-camera triangulation). This script itself does no DIFT/TAPIR
work and never touches the cameras -- it only calls tracking_node's services
and subscribes to its topics.

Requires tracking_node.py to already be running for the matching task:
    rosrun osx_ur5e tracking_node.py --task_name pick_place_00

Usage:
    python evaluate_policy_points.py \
        --policy_path outputs/train/act_pick_place_00

The gripper is whichever one dataset.gripper_type names -- "claw" (osx_claw's
Dynamixel ClawController, the default) or "robotiq" (a Robotiq C-model through
robotiq_control's gripper_action_controller, the driver connect_real_robot.launch
already starts). Both are driven through the same normalized [0, 1] command
(1 = open, 0 = closed), so every other gripper option (gripper_home,
start_gripper, gripper_slowdown_*, grasp_*) applies unchanged:
    python evaluate_policy_points.py --policy_path ... dataset.gripper_type=robotiq

Every rollout is labelled with its outcome, and its video, FT trace and
pose-error plot are saved as rollout<NN>_<outcome>.* so results can be sorted
by outcome afterwards. Judging it is the operator's job:

    Enter  - before the rollout: confirm start (after the reset prompt).
             LIVE DURING the rollout: it worked -- mark this rollout SUCCESS
             and end it here. Counted in the step statistics like a rollout
             that ran to max_timesteps. Files: rollout<NN>_success.*
    f      - live during the rollout (no Enter needed): it failed -- mark this
             rollout FAIL, stop the arm where it is, abandon the episode, and
             reset: freeze the current pose so the compliance controller stops
             chasing the policy's last target, open the gripper, retract
             dataset.reset_lift_distance_m (default 7 cm) straight up, then
             ease back to the start pose. Also the panic key for a rollout
             heading somewhere dangerous. The episode is dropped from the step
             statistics but still counts against --num_rollouts.
             Files: rollout<NN>_fail.*
    Both keys need a TTY (ignored under roslaunch/pipes).

A rollout that runs all the way to max_timesteps is not labelled yet, so the
script asks once it finishes -- Enter for success, 'f' then Enter for fail --
and names its files from the answer. It ends where the policy left it, exactly
as before; only the label comes from the prompt. With no TTY to ask, the files
stay unlabelled instead.

The same abandon-and-reset also happens by itself when the force/torque sensor
reports a force magnitude above dataset.ft_force_limit_n (default 30 N) --
the safety stop, so a rollout pressing into the fixture does not need a human
watching. Files are labelled rollout<NN>_ft_limit.*, and the FT trace records
where it tripped. The reset opens the gripper before it moves
(dataset.reset_open_gripper) rather than retracting while still clamped onto
whatever went wrong.

Every rollout also records its sensors alongside the video, under the same
outcome-labelled name:
    ft/rollout<NN>_<outcome>.{npz,png}    the UR5e's force/torque sensor, at
        the topic's own rate (dataset.record_ft_data / record_ft_plot /
        ft_topic)
    bend/rollout<NN>_<outcome>.{npz,png}  the flexible (BendLabs) sensors, one
        sample per control step (dataset.record_bend_data / record_bend_plot);
        needs uart_bridge_node.py running
Both carry ROS timestamps off the same clock, and the FT npz's step_marks map
control steps to its samples, so the two traces and the per-step pose-error
plot all line up.

Optional continuous (linear-ramp) stiffness during rollouts, driven by the
live MLP-estimated wrist deflection (off by default):
    python evaluate_policy_points.py --policy_path ... \
        dataset.enable_pose_stiffness=true

When enabled, PoseStiffnessModulator subscribes to the live MLP-estimated
wrist deflection (bendlabs/scripts/estimate_pose_node.py, /estimated_pose by
default -- requires uart_bridge_node.py + estimate_pose_node.py running
alongside this rollout) and ramps the controller's stiffness linearly between
a max value near the origin and a min value past dataset.pose_stiffness_*_far,
instead of the single fixed controller_cfg.stiffness_trans/rot value the
rollout otherwise runs with throughout. Runs every control step, independent
of tracking staleness. Tune via dataset.pose_stiffness_* overrides
(pose_topic, trans_near/far, rot_near/far, trans_min/max, rot_min/max,
min_change, pose_timeout) -- see PoseStiffnessModulator's constructor.
"""

import argparse
import dataclasses
import faulthandler
import json
import logging
import os
import select
import signal
import sys
import tempfile
import termios
import time
import timeit
import tty
from collections import Counter, deque, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import
import matplotlib.pyplot as plt

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import rospy
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import lerobot_policy_custom_act  # noqa: F401 -- registers "custom_act" with draccus

from osx_claw.claw_controller import ClawController
from osx_ur5e.compliance_utils import quat_to_rotvec_abs, distance_to_stiffness
from osx_ur5e import pose_offset
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations
from bendlabs.bendlabs_recorder import BendLabsRecorder

# Reuse replay_pkl_all.py's null/list/"0-9,20,30-35"-range-string episode
# selection parsing for the sequential start-pose-idx pool, rather than
# reimplementing it (see pick_start_pose_episode_idx below). Also reuse
# collect_image_data.py's mp4 writer for VisualizationRecorder below, so the
# two scripts can't drift in how frames get encoded.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_pkl_all import parse_episode_spec  # noqa: E402
from collect_image_data import write_video  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def install_stack_dumper() -> None:
    """SIGUSR1 -> dump every thread's stack to stderr, without disturbing the run.

    For diagnosing a hang. This script spends most of its life in one of two
    places that can block with nothing printed: an input() prompt between
    rollouts, or an unbounded rospy service call into tracking_node
    (reset_and_track/stop_track take that node's lock, which its 30 Hz _tick
    also holds -- a wedged tick there blocks the caller here indefinitely).
    A stack dump says which one it is:

        kill -USR1 $(pgrep -f evaluate_policy_points.py)

    Repeat it a few seconds apart: an unchanged stack is stuck, a moving one
    is merely slow. chain=False because nothing else wants SIGUSR1, and the
    handler must not also raise.
    """
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, ValueError, OSError) as e:
        # Not fatal -- the run just loses the hang diagnostic.
        logger.warning(f"SIGUSR1 stack dump unavailable ({e}).")


# Pressed during a rollout -> stop the arm, abandon the episode, reset, and
# record it as a failure. Also accepted at the end-of-rollout outcome prompt.
FAIL_KEYS = ("f",)
# Pressed during a rollout -> end it here and record it as a success. Both
# characters: tty.setcbreak leaves ICRNL on, so Enter normally arrives as "\n",
# but that is a terminal setting rather than a guarantee.
SUCCESS_KEYS = ("\r", "\n")

# How a rollout ended -> the suffix its video / FT / pose-error files get, so
# results can be sorted by outcome afterwards. Every rollout gets one of these:
# 'f' or Enter end it early, the force limit ends it by itself, and one that
# runs to max_timesteps is labelled by the operator at the prompt afterwards
# (see prompt_rollout_outcome). The empty fallback only applies when there is
# no TTY to ask (roslaunch, piped input).
OUTCOME_SUFFIXES = {
    "success": "_success",     # the task worked
    "fail": "_fail",           # it did not
    "ft_limit": "_ft_limit",   # force-magnitude safety limit tripped
}


class KeyPoller:
    """Non-blocking single-keypress reader for the timed rollout loop.

    The rollout has a per-step budget of 1/fps seconds, so it cannot afford a
    blocking input(). Instead stdin is switched to cbreak mode (characters
    delivered as typed, no Enter, no line editing) for the duration of the
    loop and polled with a zero-timeout select() once per step, which costs
    nothing measurable against the step budget.

    start()/stop() rather than a context manager on purpose: cbreak must cover
    ONLY the rollout, because the reset prompts between rollouts call input(),
    whose line editing (backspace, ^U) needs canonical mode back. stop() is
    idempotent and also runs from main()'s outer finally, so an exception or
    ^C mid-rollout cannot leave the user's terminal in cbreak.

    A no-op when stdin is not a TTY (piped input, roslaunch): poll() then
    always returns None and the rollout behaves exactly as it did before.
    """

    def __init__(self):
        self._fd = None
        self._saved = None
        self._warned = False   # the setup warning is worth saying once, not per rollout

    @property
    def enabled(self) -> bool:
        return self._fd is not None

    def start(self) -> bool:
        """Put stdin in cbreak mode. Returns whether key polling is live."""
        if self._fd is not None:
            return True
        try:
            if sys.stdin is None or not sys.stdin.isatty():
                return False
            fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            # setcbreak leaves ECHO on; drop it, so a keypress does not print
            # itself into the middle of the rich progress bar.
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            # Discard anything typed before now (impatient keys during the
            # reset motion), so the rollout cannot start already "skipped".
            termios.tcflush(fd, termios.TCIFLUSH)
            self._fd = fd
        except (termios.error, ValueError, OSError) as e:
            # Not fatal -- the rollout just runs without the live keys.
            if not self._warned:
                logger.warning(f"Live keyboard control unavailable ({e}); "
                               "the in-rollout fail/success keys are disabled.")
                self._warned = True
            self._fd, self._saved = None, None
            return False
        return True

    def stop(self) -> None:
        """Restore the terminal. Safe to call repeatedly / when never started."""
        if self._fd is None:
            return
        fd, saved, self._fd, self._saved = self._fd, self._saved, None, None
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, ValueError, OSError):
            pass  # best-effort -- never mask the real error that got us here

    def poll(self):
        """Return the most recent key pressed since the last poll (lowercased),
        or None. Drains everything buffered, so a held-down key or a burst of
        typing acts once now rather than queuing up to fire on later steps.
        Reads the raw fd, not sys.stdin, so Python's text buffering cannot hold
        a character back that select() has already reported as ready."""
        if self._fd is None:
            return None
        key = None
        while select.select([self._fd], [], [], 0)[0]:
            try:
                ch = os.read(self._fd, 1)
            except OSError:
                break
            if not ch:
                break
            key = ch.decode("utf-8", errors="ignore").lower()
        return key


class EnvironmentStateSubscriber:
    """Subscribes to tracking_node's ~environment_state (std_msgs/Float64MultiArray)
    and records the wall-clock time each message was received, so the 30Hz
    control loop can detect a stalled/dead tracking node and stop commanding
    the arm instead of acting on an indefinitely-aging tracked point (same
    role SharedValue/TrackingWorker played when tracking ran in-process).
    `seq` counts messages received, so a caller can detect "at least one new
    message since I last checked" (see wait loops in main()). No lock
    needed -- single writer (rospy's subscriber callback thread), same
    GIL-atomic-attribute-assignment reasoning as RawImageRecorder used."""

    def __init__(self, topic: str):
        self._value = None
        self._timestamp = None
        self.seq = 0
        rospy.Subscriber(topic, Float64MultiArray, self._callback)

    def _callback(self, msg: Float64MultiArray) -> None:
        self._value = np.array(msg.data, dtype=np.float32)
        self._timestamp = time.time()
        self.seq += 1

    def get(self):
        """Returns (env_state, age_seconds). age_seconds is inf until the
        first message ever arrives."""
        if self._timestamp is None:
            return self._value, float("inf")
        return self._value, time.time() - self._timestamp


class VisualizationRecorder:
    """Buffers tracking_node's ~visualization_image (sensor_msgs/Image, bgr8)
    frames during a rollout and writes them to an mp4 on stop_and_save().

    No cv_bridge -- manual decode, same pattern as tracking_node.py's
    RawImageRecorder (and already bgr8, so no channel-order conversion is
    needed before handing frames to cv2.VideoWriter). Subscribes for this
    object's whole lifetime; start()/stop_and_save() just gate whether
    incoming frames are buffered, so one instance covers every rollout.

    fps is derived from actual elapsed wall-clock time / frame count at
    stop_and_save() time rather than hardcoded: tracking_node's publish rate
    (--tracking_fps, default 5 Hz) is a launch-time CLI arg on that separate
    node, not something visible here, and frames only arrive when tracking
    actually produces a new estimate (so the real rate can vary run to run).
    """

    def __init__(self, topic: str):
        self._recording = False
        self._frames = []
        self._start_time = None
        rospy.Subscriber(topic, Image, self._callback)

    def _callback(self, msg: Image) -> None:
        if not self._recording:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
        self._frames.append(arr)

    def start(self) -> None:
        self._frames = []
        self._start_time = time.perf_counter()
        self._recording = True

    def stop(self) -> None:
        """Stop capturing without saving -- best-effort cleanup on an
        abnormal exit (see main()'s finally block), so a mid-rollout crash
        doesn't leave the callback buffering frames nobody will write out."""
        self._recording = False

    def stop_and_save(self, path: Path) -> bool:
        """Stop recording and write buffered frames to `path` as mp4.

        Returns False (no file written) when fewer than 2 frames were
        captured -- e.g. tracking_node wasn't actually publishing viz frames
        -- rather than writing a degenerate near-empty video.
        """
        self._recording = False
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        n = len(self._frames)
        if n < 2:
            logger.warning(
                f"Visualization recording: only {n} frame(s) captured -- skipping mp4 save "
                f"(is tracking_node publishing {path.name}'s source topic?)."
            )
            return False
        fps = n / elapsed if elapsed > 0 else 5.0
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(path, self._frames, fps)
        logger.info(f"Saved rollout visualization video: {path} ({n} frames @ {fps:.1f} fps)")
        return True


# A latched force-limit breach: which sample tripped it, when, and how hard.
FTTrip = namedtuple("FTTrip", ["index", "stamp", "force"])


def prompt_rollout_outcome(rollout_id: int, steps: int) -> str:
    """Ask the operator whether a rollout that ran to the end succeeded.

    Only for rollouts nobody labelled live: pressing Enter or 'f' during the
    rollout already says the answer and ends it there. Blocks until answered,
    so an unattended run stops here -- deliberately, since an unlabelled
    rollout is the thing this exists to prevent.

    Returns "success" or "fail". Returns None when there is no terminal to ask
    (piped input / roslaunch, the same condition that disables KeyPoller), in
    which case the rollout's files stay unlabelled rather than the run dying.
    """
    prompt = (f"\n  Rollout {rollout_id} ran to the end ({steps} steps) -- press Enter if it "
              f"SUCCEEDED, or 'f' then Enter if it FAILED: ")
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            logger.warning(
                f"Rollout {rollout_id}: no terminal to ask for the outcome -- leaving its "
                "files unlabelled."
            )
            return None
        if answer == "":
            return "success"
        if answer in FAIL_KEYS:
            return "fail"
        print(f"  Unrecognized '{answer}' -- Enter for success, 'f' for fail.")


class WrenchRecorder:
    """Buffers the UR5e's force/torque sensor (geometry_msgs/WrenchStamped, /wrench
    by default) during a rollout and saves it to an npz on stop_and_save().

    Recorded from the topic at its own full rate (~500 Hz off the UR driver),
    not sampled once per control step: contact transients are the interesting
    part of an FT trace and a 30 Hz sample would alias them away. A rollout's
    samples are still mappable back to control steps via mark_step() -- see
    step_marks in the saved arrays -- so the trace can be lined up with the
    pose-error plot and the visualization video.

    Same lifecycle as VisualizationRecorder above: subscribes once for this
    object's whole lifetime, start()/stop_and_save() only gate whether incoming
    messages are buffered, so one instance covers every rollout.

    Doubles as the safety stop (force_limit, dataset.ft_force_limit_n): the
    same callback watches the force magnitude and latches a trip for the
    control loop to pick up via check_trip(). It lives here rather than in a
    second subscriber precisely because this one is already armed exactly when
    it should be -- during policy execution and not during the resets and
    homing moves in between, where pushing against something is normal.
    """

    def __init__(self, topic: str, force_limit: float = None, limit_consecutive: int = 3):
        self._topic = topic
        self._force_limit = float(force_limit) if force_limit else None
        self._limit_consecutive = max(1, int(limit_consecutive))
        self._recording = False
        self._samples = []      # (stamp_s, fx, fy, fz, tx, ty, tz)
        self._step_marks = []   # (step_index, len(self._samples) at that step)
        self._over_count = 0    # consecutive samples above _force_limit
        self._trip = None       # FTTrip, latched on the FIRST breach of a rollout
        rospy.Subscriber(topic, WrenchStamped, self._callback)

    def _callback(self, msg: WrenchStamped) -> None:
        if not self._recording:
            return
        # Simulated/replayed sources can publish a zero stamp; fall back to
        # receipt time so t_rel stays monotonic instead of collapsing to 0.
        stamp = msg.header.stamp.to_sec() or rospy.get_time()
        f, tq = msg.wrench.force, msg.wrench.torque
        self._samples.append((stamp, f.x, f.y, f.z, tq.x, tq.y, tq.z))

        if self._force_limit is None:
            return
        # Force-magnitude safety limit, evaluated at the sensor's own rate.
        # _limit_consecutive samples must breach in a row before it counts: a
        # single spiky reading on the raw topic should not abandon a good
        # rollout, while real contact holds for far longer than the few ms this
        # costs. Only the first breach of a rollout is latched -- the trip is
        # what ends the rollout, so later samples are of no further interest.
        magnitude = float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
        if magnitude <= self._force_limit:
            self._over_count = 0
            return
        self._over_count += 1
        if self._over_count >= self._limit_consecutive and self._trip is None:
            self._trip = FTTrip(index=len(self._samples) - 1, stamp=stamp, force=magnitude)

    def start(self) -> None:
        self._samples = []
        self._step_marks = []
        self._over_count = 0
        self._trip = None       # re-arm the force limit for this rollout
        self._recording = True

    def mark_step(self, step: int) -> None:
        """Note that control step `step` begins at the current sample count."""
        self._step_marks.append((step, len(self._samples)))

    def check_trip(self):
        """The latched FTTrip if the force limit has been breached during this
        rollout, else None. Detection happens in the subscriber callback at the
        sensor's rate; the control loop polls this once per step, so the
        reaction lands up to 1/fps after the breach -- the same latency the
        fail key has."""
        return self._trip

    def stop(self) -> None:
        """Stop capturing without saving -- best-effort cleanup on an abnormal
        exit, mirroring VisualizationRecorder.stop()."""
        self._recording = False

    def stop_and_save(self, path: Path = None):
        """Stop recording and return the trace as a dict of arrays:
            stamp       (N,)   absolute ROS time (s)
            t_rel       (N,)   seconds since the first sample of this rollout
            wrench      (N, 6) [fx, fy, fz, tx, ty, tz] in the sensor frame,
                               exactly as published (no filtering or rotation)
            step_marks  (M, 2) [control step, sample index where it began]
        Also writes them to `path` as an npz when `path` is given.

        Returns None (nothing saved) when fewer than 2 samples arrived -- e.g.
        the FT topic isn't publishing -- rather than saving a useless file.
        """
        self._recording = False
        n = len(self._samples)
        if n < 2:
            logger.warning(
                f"FT recording: only {n} sample(s) captured on {self._topic} -- skipping save "
                f"(is the topic publishing? check `rostopic hz {self._topic}`)."
            )
            return None

        arr = np.asarray(self._samples, dtype=np.float64)
        data = {
            "stamp": arr[:, 0],
            "t_rel": arr[:, 0] - arr[0, 0],
            "wrench": arr[:, 1:].astype(np.float32),
            "step_marks": np.asarray(self._step_marks, dtype=np.int64).reshape(-1, 2),
            "topic": np.asarray(self._topic),
            # Where the force limit tripped, if it did: -1/nan when it didn't
            # (or when no limit was armed), so the trace records why the
            # rollout ended. See check_trip.
            "trip_index": np.asarray(self._trip.index if self._trip else -1),
            "trip_force": np.asarray(self._trip.force if self._trip else np.nan),
            "trip_t_rel": np.asarray(
                self._trip.stamp - arr[0, 0] if self._trip else np.nan),
            "force_limit": np.asarray(
                self._force_limit if self._force_limit is not None else np.nan),
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **data)
            duration = data["t_rel"][-1]
            logger.info(
                f"Saved rollout FT data: {path} ({n} samples, {duration:.1f} s, "
                f"{n / duration if duration > 0 else float('nan'):.0f} Hz, "
                f"max |F|={np.linalg.norm(data['wrench'][:, :3], axis=1).max():.1f} N)"
            )
        return data


def save_ft_visualization(ft_data: dict, out_path: Path, rollout_id: int) -> bool:
    """Plot a rollout's FT trace (WrenchRecorder.stop_and_save output) to PNG.

    Top row: the three force and three torque components against time. Bottom
    row: their magnitudes, with the peak annotated -- the quickest read on
    whether a rollout ran into the fixture or squeezed the object harder than
    the demonstrations did. Values are in the sensor frame, exactly as
    published on the recorded topic.

    Returns False (no file written) when fewer than 2 samples were recorded.
    """
    wrench = np.asarray(ft_data["wrench"])
    t = np.asarray(ft_data["t_rel"])
    if len(t) < 2:
        logger.warning(f"Rollout {rollout_id}: only {len(t)} FT sample(s) -- skipping FT plot.")
        return False

    force, torque = wrench[:, :3], wrench[:, 3:]
    force_mag = np.linalg.norm(force, axis=1)
    torque_mag = np.linalg.norm(torque, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, vals, labels, unit, title in (
        (axes[0, 0], force, ["Fx", "Fy", "Fz"], "N", "Force"),
        (axes[0, 1], torque, ["Tx", "Ty", "Tz"], "Nm", "Torque"),
    ):
        for i, (label, color) in enumerate(zip(labels, ["#377eb8", "#4daf4a", "#e41a1c"])):
            ax.plot(t, vals[:, i], color=color, linewidth=1.0, label=label)
        ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax, mag, unit, title, color in (
        (axes[1, 0], force_mag, "N", "Force magnitude", "#ff7f00"),
        (axes[1, 1], torque_mag, "Nm", "Torque magnitude", "#984ea3"),
    ):
        ax.plot(t, mag, color=color, linewidth=1.2,
                label=f"max={mag.max():.2f} {unit}, mean={mag.mean():.2f} {unit}")
        ax.axhline(mag.max(), color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Mark where the force limit tripped, if it did (see WrenchRecorder).
    trip_t = float(ft_data.get("trip_t_rel", np.nan))
    trip_note = ""
    if np.isfinite(trip_t):
        trip_force = float(ft_data["trip_force"])
        for ax in axes.flat:
            ax.axvline(trip_t, color="#e41a1c", linestyle="--", linewidth=1.4, alpha=0.9)
        trip_note = (f" -- FORCE LIMIT TRIPPED at {trip_t:.2f} s "
                     f"(|F|={trip_force:.1f} N > {float(ft_data['force_limit']):.1f} N)")

    fig.suptitle(
        f"Rollout {rollout_id}: FT sensor ({ft_data['topic']}, {len(t)} samples, "
        f"{t[-1]:.1f} s){trip_note}", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Saved rollout FT plot: {out_path} (max |F|={force_mag.max():.1f} N, "
        f"max |T|={torque_mag.max():.2f} Nm)"
    )
    return True


def save_bend_data(bend_log: list, baseline, rewind_threshold, out_path: Path,
                   rollout_id: int):
    """Save a rollout's flexible-sensor (BendLabs) trace and return it as a dict.

    bend_log: list of (ros_stamp_s, step, twist0, bend0, twist1, bend1, ...) --
    one entry per control step, sampled from BendLabsRecorder.get_angles(),
    which is already a moving average of a source that publishes smoothed,
    zeroed angles (uart_bridge_node). Recorded per step rather than per message
    because the sensors publish asynchronously on one topic each: a per-step
    snapshot is a single aligned matrix, indexes the same way the pose-error
    plot does, and matches the step_marks the FT trace stores. The stamps are
    ROS time, the same clock the FT trace uses, so the two overlay exactly.

    Saved arrays:
        stamp     (T,)     ROS time of each step's reading
        t_rel     (T,)     seconds since this rollout's first reading
        step      (T,)     control step index
        angles    (T, 2N)  [twist, bend] per sensor, degrees
        baseline  (2N,)    the reading at the rollout's start -- what the
                           bend-sensor rewind measures deviation against
        labels    (2N,)    "twist0", "bend0", "twist1", ...
        rewind_threshold   scalar, nan when the rewind reaction is off

    Returns None (nothing saved) when fewer than 2 steps were sampled.
    """
    if len(bend_log) < 2:
        logger.warning(
            f"Rollout {rollout_id}: only {len(bend_log)} flexible-sensor sample(s) -- "
            "skipping save (are the BendLabs sensors publishing?)."
        )
        return None

    arr = np.asarray(bend_log, dtype=np.float64)
    n_sensors = (arr.shape[1] - 2) // 2
    data = {
        "stamp": arr[:, 0],
        "t_rel": arr[:, 0] - arr[0, 0],
        "step": arr[:, 1].astype(np.int64),
        "angles": arr[:, 2:].astype(np.float32),
        "baseline": np.asarray(baseline if baseline is not None
                               else np.full(2 * n_sensors, np.nan), dtype=np.float32),
        "labels": np.asarray([f"{axis}{i}" for i in range(n_sensors)
                              for axis in ("twist", "bend")]),
        "rewind_threshold": np.asarray(rewind_threshold if rewind_threshold is not None
                                       else np.nan),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        peak = np.abs(data["angles"] - data["baseline"]).max()
        logger.info(
            f"Saved rollout flexible-sensor data: {out_path} ({len(bend_log)} steps, "
            f"{n_sensors} sensors, peak deviation {peak:.2f} deg)"
        )
    return data


def save_bend_visualization(bend_data: dict, out_path: Path, rollout_id: int) -> bool:
    """Plot a rollout's flexible-sensor trace (save_bend_data output) to PNG.

    Top row: the raw twist and bend angle of every sensor against the control
    step. Bottom row: the same as absolute deviation from the rollout's opening
    reading -- the quantity the bend-sensor rewind actually thresholds -- with
    that threshold drawn in when it is armed.

    Returns False (no file written) when fewer than 2 steps were sampled.
    """
    angles = np.asarray(bend_data["angles"])
    step = np.asarray(bend_data["step"])
    if len(step) < 2:
        logger.warning(f"Rollout {rollout_id}: only {len(step)} flexible-sensor sample(s) "
                       "-- skipping plot.")
        return False

    labels = [str(l) for l in np.asarray(bend_data["labels"])]
    baseline = np.asarray(bend_data["baseline"])
    deviation = np.abs(angles - baseline)
    threshold = float(bend_data["rewind_threshold"])
    n_sensors = angles.shape[1] // 2
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, axis in enumerate(("twist", "bend")):
        idx = [i for i, l in enumerate(labels) if l.startswith(axis)]
        for k, i in enumerate(idx):
            axes[0, col].plot(step, angles[:, i], color=colors[k % 10], linewidth=1.2,
                              label=labels[i])
            axes[1, col].plot(step, deviation[:, i], color=colors[k % 10], linewidth=1.2,
                              label=f"{labels[i]} (max {deviation[:, i].max():.2f})")
        axes[0, col].set_title(f"{axis.capitalize()} angle (deg)", fontsize=11,
                               fontweight="bold")
        axes[1, col].set_title(f"|{axis} deviation from rollout start| (deg)", fontsize=11,
                               fontweight="bold")
        if np.isfinite(threshold):
            axes[1, col].axhline(threshold, color="#e41a1c", linestyle="--", linewidth=1.2,
                                 label=f"rewind threshold ({threshold:g})")
        for row in (0, 1):
            axes[row, col].set_xlabel("Step")
            axes[row, col].legend(fontsize=8, loc="upper right")
            axes[row, col].grid(True, alpha=0.3)

    fig.suptitle(
        f"Rollout {rollout_id}: flexible sensors ({n_sensors} sensors, {len(step)} steps, "
        f"peak deviation {deviation.max():.2f} deg)", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rollout flexible-sensor plot: {out_path} "
                f"(peak deviation {deviation.max():.2f} deg)")
    return True


def save_pose_error_visualization(pose_error_log: list, out_path: Path, rollout_id: int) -> bool:
    """Plot commanded-vs-actual cartesian pose over a rollout and save to PNG.

    pose_error_log: list of {"t": int, "commanded": pose7, "actual": pose7}
    with pose7 = [x, y, z, qx, qy, qz, qw]. "commanded" is next_target --
    the pose actually handed to arm.set_cartesian_target_pose, i.e. AFTER
    execute_cartesian_target's per-step delta/workspace clip, not the
    policy's raw predicted pose (see execute_cartesian_target's return).
    "actual" is arm.end_effector() read right after issuing that command, so
    this shows how far the compliant controller lags/deviates from what it
    was just told, not a steady-state tracking error -- the arm hasn't had a
    full step to converge yet.

    Top row: X/Y/Z commanded vs actual (mm), gap shaded -- same idiom as
    replay_pkl_episode.py's save_visualization. Bottom-right: combined
    position-error magnitude (mm) and orientation-error magnitude (deg) over
    time, the latter via transformations.quaternions_orientation_error (the
    same function execute_cartesian_target itself uses for its rotation
    clip, so the error reported here is consistent with what actually
    governed clipping).

    Returns False (no file written) when fewer than 2 samples were logged.
    """
    if len(pose_error_log) < 2:
        logger.warning(
            f"Rollout {rollout_id}: only {len(pose_error_log)} pose-error sample(s) -- "
            "skipping pose-error plot."
        )
        return False

    t_arr = np.array([e["t"] for e in pose_error_log])
    commanded = np.array([e["commanded"] for e in pose_error_log])  # (T, 7)
    actual = np.array([e["actual"] for e in pose_error_log])        # (T, 7)

    pos_error_mm = np.linalg.norm(commanded[:, :3] - actual[:, :3], axis=1) * 1000.0
    orient_error_deg = np.array([
        np.degrees(np.linalg.norm(transformations.quaternions_orientation_error(c[3:], a[3:])))
        for c, a in zip(commanded, actual)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axis_labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axes.flat[:3]):
        cmd_mm = commanded[:, i] * 1000.0
        act_mm = actual[:, i] * 1000.0
        ax.plot(t_arr, cmd_mm, color="#377eb8", linewidth=1.6, label="commanded", zorder=3)
        ax.plot(t_arr, act_mm, color="#e41a1c", linewidth=1.2, linestyle="--", alpha=0.8,
                label="actual", zorder=4)
        ax.fill_between(t_arr, cmd_mm, act_mm, alpha=0.15, color="orange")
        ax.set_title(f"{axis_labels[i]} (mm)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    ax = axes.flat[3]
    ax.plot(t_arr, pos_error_mm, color="#4daf4a", linewidth=1.6,
            label=f"position error (max={pos_error_mm.max():.1f} mm)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position error (mm)", color="#4daf4a")
    ax.tick_params(axis="y", labelcolor="#4daf4a")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_arr, orient_error_deg, color="#984ea3", linewidth=1.2, linestyle="--",
             label=f"orientation error (max={orient_error_deg.max():.1f} deg)")
    ax2.set_ylabel("Orientation error (deg)", color="#984ea3")
    ax2.tick_params(axis="y", labelcolor="#984ea3")
    ax.set_title("Combined pose error", fontsize=11, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle(f"Rollout {rollout_id}: commanded vs actual pose", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Saved rollout pose-error plot: {out_path} (max pos err {pos_error_mm.max():.1f} mm, "
        f"max orient err {orient_error_deg.max():.1f} deg)"
    )
    return True


def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        if not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(file_handler)


def load_hydra_config(config_path: str, config_name: str):
    """Load the Hydra config used by data_collection.py / replay_episode.py / evaluate_policy.py."""
    config_dir_abs = str((Path(__file__).resolve().parent / config_path).resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_policy_config_compat(policy_path: Path):
    """PreTrainedConfig.from_pretrained, tolerant of newer-lerobot config.json keys.

    fluxtrain trains under lerobot >=0.6.1 while this container runs 0.4.4, so its
    checkpoints carry fields (e.g. pretrained_revision) that draccus rejects here --
    it decodes config.json strictly, so an unknown field is a hard DecodingError
    rather than an ignore. Those fields are HuggingFace Hub metadata and mean
    nothing for a local checkpoint directory, so drop them -- but log which ones,
    since the version skew itself is worth knowing about.

    Only config.json needs this; the pre/post-processor JSONs parse as-is, so the
    temp dir holds just the rewritten config and the weights still load from the
    real checkpoint directory (pass the returned config to from_pretrained).
    """
    raw = json.loads((policy_path / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(get_policy_class(raw["type"]).config_class)}
    dropped = sorted(k for k in raw if k not in known and k != "type")
    if not dropped:
        return PreTrainedConfig.from_pretrained(str(policy_path))

    logger.warning(
        f"Checkpoint config.json has {len(dropped)} field(s) unknown to the installed "
        f"lerobot ({', '.join(dropped)}) -- it was trained under a newer version. "
        "Ignoring them."
    )
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "config.json").write_text(
            json.dumps({k: v for k, v in raw.items() if k in known or k == "type"})
        )
        return PreTrainedConfig.from_pretrained(td)


def gripper_state_to_normalized(gripper_state: float) -> float:
    """-1 (open) / 1 (closed) -> ClawController's normalized [0, 1] (1=open, 0=closed).
    Same convention as replay_pkl_episode.py -- used to command the predicted
    gripper action."""
    return float(np.clip((1.0 - gripper_state) / 2.0, 0.0, 1.0))


def gripper_normalized_to_state(gripper_norm: float) -> float:
    """Inverse of the above -- ClawController's normalized [0, 1] -> raw -1/1
    training encoding. Used to read the *current* gripper position into
    observation.state, matching how the training data encoded it."""
    return float(1.0 - 2.0 * gripper_norm)


# ClawController normalized positions (see gripper_state_to_normalized).
# Same convention as replay_pkl_episode.py.
GRIPPER_OPEN_NORM = 1.0    # == gripper_state -1
GRIPPER_CLOSED_NORM = 0.0  # == gripper_state +1


# ---------------------------------------------------------------------------
# Gripper backends (dataset.gripper_type)
# ---------------------------------------------------------------------------
# Nothing below this point cares which gripper is bolted to the wrist: the
# whole script drives it through three methods only --
# set_normalized_position / get_normalized_command / get_normalized_position,
# all in ClawController's [0, 1] convention (1 = open, 0 = closed). osx_claw's
# ClawController speaks that natively; RobotiqGripperAdapter below teaches it
# to ur_control's RobotiqGripper. make_gripper() picks between them.
GRIPPER_TYPES = ("claw", "robotiq")


class RobotiqGripperAdapter:
    """ClawController-compatible view of a Robotiq C-model gripper.

    Wraps ur_control.grippers.RobotiqGripper (robotiq_control's
    gripper_action_controller -- the driver connect_real_robot.launch already
    brings up), which is nearly compatible already: its percentage_command()
    and get_opening_percentage() use the same 0 = closed / 1 = open
    normalization ClawController does. Two gaps are closed here:

      * there is no "what did I last command?" getter, which read_grasp() needs
        to spot a commanded-shut / measured-open gap (= jaws blocked by an
        object), so the last command is remembered here;
      * commands default to wait=True, i.e. blocking on the action result for
        up to 5 s -- unusable at fps. Sends are issued with wait=False instead,
        which is safe because the driver's default action callback
        (cmodel_action_controller.py's _execute_no_delay_cb) is fire-and-forget:
        it publishes one CModelCommand to the gripper and returns.

    open_fraction rescales the whole normalized range onto the first fraction of
    the physical stroke: normalized 1.0 ("open") commands open_fraction * max_gap
    metres rather than the full jaw width, and measured positions are divided
    back out by the same factor. It exists so a Robotiq can be given the CLAW's
    opening width -- a policy's gripper channel then means the same physical gap
    it meant in training, and the grasp_* thresholds keep their claw-tuned
    meaning too, since a held object occupies the same share of the (shortened)
    normalized range. 1.0 = use the full stroke. Closed is still 0.0 either way;
    only the open end moves.
    """

    def __init__(self, namespace: str = "", timeout: float = 5.0,
                 open_fraction: float = 1.0):
        # Imported lazily so a claw-only machine never needs robotiq_msgs built
        # (ur_control.grippers swallows that import error at module level, then
        # trips over the missing name only when a RobotiqGripper is built).
        from ur_control.grippers import RobotiqGripper

        if not 0.0 < open_fraction <= 1.0:
            raise ValueError(
                f"dataset.robotiq_open_fraction must be in (0, 1], got {open_fraction} "
                f"(it divides the measured position, and 0 would command a gripper that "
                f"can never open)"
            )
        self._open_fraction = float(open_fraction)

        # RobotiqGripper wants a trailing slash ("gripper/"), or "" for the root
        # namespace -- which is where connect_real_robot.launch puts the driver.
        ns = namespace.strip("/")
        self._ns = f"{ns}/" if ns else ""
        try:
            # prefix="" explicitly: GripperControllerBase does `prefix +
            # joint_name`, which raises on the None default.
            self._gripper = RobotiqGripper(namespace=self._ns, prefix="", timeout=timeout)
        except NameError as e:
            raise RuntimeError(
                "Robotiq support needs the robotiq_msgs package, which is not importable "
                "(ur_control.grippers logged 'Robotiq gripper can't be load' at import). "
                "Build underlay_ws/src/third_party/robotiq-cri and source the workspace, "
                "or set dataset.gripper_type=claw."
            ) from e

        # RobotiqGripper only logs an error when its action client fails to
        # connect -- it does not raise -- and SimpleActionClient.send_goal() to
        # a server that isn't there just publishes into the void. Without this
        # check the rollout runs to completion with a gripper that never moves.
        # (The first wait already burned `timeout` inside RobotiqGripper, so
        # this one only has to confirm the outcome.)
        if not self._gripper.gripper.wait_for_server(rospy.Duration(0.5)):
            raise RuntimeError(
                f"No action server on /{self._ns}gripper_action_controller -- commands would "
                f"be silently dropped.\n"
                f"  The driver node can be running and publishing /{self._ns}status, "
                f"/{self._ns}gripper_status and joint states while still having NO action "
                f"server: cmodel_action_controller.py gives the gripper ~6s to report "
                f"activation (gSTA==3) at startup and, if it does not, returns before "
                f"starting its servers -- look for 'Failed to activate gripper' in its log.\n"
                f"  Check the gripper itself:   rostopic echo -n1 /{self._ns}status\n"
                f"  (gACT=1, gSTA=3, gFLT=0 means it IS activated and the node just gave up "
                f"too early.) If so, restart the node -- it comes up immediately once the "
                f"gripper is already activated:\n"
                f"    rosnode kill /cmodel_action_controller\n"
                f"    rosrun robotiq_control cmodel_action_controller.py"
            )

        # RobotiqGripper.percentage_command() branches on gripper_type -- which
        # is really the driver's gripper_action_controller/joint_name param --
        # and falls off the end doing NOTHING for any other value, so an
        # unrecognized one would leave the jaws frozen with no error anywhere.
        # Fail loudly instead. The cmodel_action_controller_{85,140,hande}
        # configs connect_real_robot.launch selects all use "finger_joint".
        if self._gripper.gripper_type not in ("finger_joint", "hand-e"):
            raise RuntimeError(
                f"ur_control's RobotiqGripper cannot command joint "
                f"{self._gripper.gripper_type!r} (from /{self._ns}gripper_action_controller/"
                f"joint_name) -- it only handles 'finger_joint' and 'hand-e', and would "
                f"silently ignore every command otherwise. Load one of robotiq_control's "
                f"cmodel_action_controller_{{85,140,hande}}.yaml configs, which all set "
                f"joint_name: finger_joint."
            )

        # opening_width starts at 0.0 and is overwritten by the driver's first
        # gripper_status message -- so 0.0 is ambiguous ("closed" vs "nothing
        # heard yet"). None is not: the callback only ever writes a real
        # measurement over it.
        self._gripper.opening_width = None
        self._cmd_norm = GRIPPER_OPEN_NORM
        self._wait_for_status(timeout)
        if self._gripper.opening_width is not None:
            # Seed the remembered command from where the jaws actually are, so
            # the first read_grasp() doesn't see a phantom commanded-vs-measured
            # gap before anything has been commanded.
            self._cmd_norm = self.get_normalized_position()
        logger.info(
            f"Robotiq gripper ready (ns='/{self._ns}', max_gap={self._gripper._max_gap:.3f} m, "
            f"position={self._cmd_norm:.2f} normalized). Normalized 1.0 = "
            f"{self._open_fraction:.3f} of the stroke = "
            f"{self._open_fraction * self._gripper._max_gap * 1000:.1f} mm"
            + (" (full jaw width)" if self._open_fraction >= 1.0 else
               f" instead of {self._gripper._max_gap * 1000:.1f} mm")
        )

    def _wait_for_status(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        while self._gripper.opening_width is None and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                logger.warning(
                    f"No message on /{self._ns}gripper_status within {timeout:.1f}s -- is "
                    f"cmodel_action_controller running? Falling back to reporting the last "
                    f"commanded position as the measured one (grasp detection will never "
                    f"see a gap, so dataset.grasp_binary_enabled will always read 'empty')."
                )
                return
            rospy.sleep(0.05)

    def set_normalized_position(self, value: float) -> None:
        """Command the gripper from a [0.0, 1.0] value (0=closed, 1=open).
        1.0 opens to open_fraction of the physical stroke, not the full jaw."""
        self._cmd_norm = float(np.clip(value, 0.0, 1.0))
        self._gripper.percentage_command(self._cmd_norm * self._open_fraction, wait=False)

    def get_normalized_command(self) -> float:
        """Last commanded position, normalized to [0.0, 1.0] (0=closed, 1=open)."""
        return self._cmd_norm

    def get_normalized_position(self) -> float:
        """Measured jaw opening, normalized to [0.0, 1.0] (0=closed, 1=open) --
        in the same open_fraction-scaled range set_normalized_position takes, so
        commanded and measured stay directly comparable (read_grasp subtracts
        them). Jaws wider than open_fraction * max_gap -- which is where they
        sit before the first command -- simply read 1.0. Falls back to the last
        command while the driver is silent."""
        if self._gripper.opening_width is None:
            return self._cmd_norm
        return float(np.clip(
            self._gripper.get_opening_percentage() / self._open_fraction, 0.0, 1.0))


def resolve_gripper_type(ds_cfg) -> str:
    """dataset.gripper_type, validated. Defaults to "claw" (previous behavior),
    including when there is no hydra config at all."""
    gripper_type = str(ds_cfg.get("gripper_type", "claw")) if ds_cfg is not None else "claw"
    if gripper_type not in GRIPPER_TYPES:
        raise ValueError(
            f"Invalid dataset.gripper_type: {gripper_type!r} "
            f"(expected one of {', '.join(repr(g) for g in GRIPPER_TYPES)})"
        )
    return gripper_type


def make_gripper(ds_cfg):
    """Build the gripper backend named by dataset.gripper_type -- either
    osx_claw's ClawController or a RobotiqGripperAdapter. Both expose the same
    normalized [0, 1] interface, so the caller never branches again."""
    gripper_type = resolve_gripper_type(ds_cfg)
    if gripper_type == "claw":
        logger.info("Initializing ClawController (dataset.gripper_type=claw)...")
        return ClawController(init_node=False)

    namespace = str(ds_cfg.get("robotiq_namespace", "")) if ds_cfg is not None else ""
    timeout = float(ds_cfg.get("robotiq_timeout", 5.0)) if ds_cfg is not None else 5.0
    open_fraction = float(ds_cfg.get("robotiq_open_fraction", 1.0)) if ds_cfg is not None else 1.0
    logger.info(
        f"Initializing Robotiq gripper (dataset.gripper_type=robotiq, "
        f"namespace='/{namespace.strip('/')}', open_fraction={open_fraction})..."
    )
    return RobotiqGripperAdapter(namespace=namespace, timeout=timeout,
                                 open_fraction=open_fraction)


# Config for deriving the gripper element of observation.state from the gap
# between the gripper's commanded and measured position (see get_current_state).
GraspConfig = namedtuple("GraspConfig", ["enabled", "cmd_close_threshold", "gap_threshold"])

# A single grasp reading: whether the gripper is holding something, plus the raw
# normalized numbers it was derived from (handy for logging / threshold tuning).
GraspReading = namedtuple("GraspReading", ["holding", "cmd_norm", "act_norm", "gap"])


def read_grasp(gripper, grasp_cfg: GraspConfig) -> GraspReading:
    """Read the gripper and decide whether it is holding something: commanded
    (near) shut while the measured jaws remain significantly more open -- i.e.
    blocked by an object. Both readings are ClawController-normalized to [0, 1]
    (1=open, 0=closed), so a *positive* (measured - commanded) gap means the
    jaws are more open than commanded."""
    cmd_norm = gripper.get_normalized_command()
    act_norm = gripper.get_normalized_position()
    gap = act_norm - cmd_norm
    holding = bool(cmd_norm <= grasp_cfg.cmd_close_threshold and gap >= grasp_cfg.gap_threshold)
    return GraspReading(holding, cmd_norm, act_norm, gap)


def get_current_state(arm, gripper, grasp_cfg: GraspConfig = None, offset=None, yaw: float = 0.0) -> np.ndarray:
    """Current EE cartesian pose (pos + ortho6d) + gripper -> (10,), matching
    convert_pkl_to_lerobot.py's observation.state encoding exactly.

    The gripper element uses the training encoding (-1 open / +1 closed). When
    grasp_cfg is enabled, that element is a *binary* grasp signal derived from
    the commanded-vs-measured gripper gap (see _detect_holding): +1 (closed) while
    holding an object, -1 (open) otherwise. This matches the binary gripper
    channel the policy was trained on (convert_pkl_to_lerobot.py writes demo
    gripper_states as -1/+1) better than the raw continuous measured position,
    which the object-blocked jaws would otherwise report as a mid-range value
    the policy never saw in training. When grasp_cfg is None (or disabled), the
    raw measured position is used (previous behavior).

    offset/yaw undo dataset.eval_offset/eval_yaw_offset_deg (see main()): the
    arm is physically commanded [x, y, z] further and yaw degrees rotated
    along/about its own local axes (rotated into world frame by its current
    orientation -- so "+z" always means "further out of the gripper" and "+y"
    always the same finger, regardless of which way the gripper is currently
    facing), so that same local-frame offset/yaw is subtracted back out here
    before the policy ever sees it -- keeping the policy's world consistent
    with training while the real arm operates offset away from its nominal
    pose."""
    pose = arm.end_effector().copy()  # [x, y, z, qx, qy, qz, qw]
    if offset is not None:
        pose = pose_offset.undo_offset(pose, offset, yaw)
    ortho6 = transformations.ortho6_from_quaternion(pose[3:])
    if gripper is None:
        gripper_raw = gripper_normalized_to_state(1.0)
    elif grasp_cfg is not None and grasp_cfg.enabled:
        gripper_raw = 1.0 if read_grasp(gripper, grasp_cfg).holding else -1.0
    else:
        gripper_raw = gripper_normalized_to_state(gripper.get_normalized_position())
    return np.concatenate([pose[:3], ortho6, [gripper_raw]]).astype(np.float32)


def execute_cartesian_target(target_pose: np.ndarray, arm, gripper, gripper_norm, safety_cfg) -> np.ndarray:
    """Clip target_pose to a safe delta from the current pose and command it.
    Identical to replay_pkl_episode.py's function of the same name, except
    this copy also returns next_target -- the pose actually handed to
    arm.set_cartesian_target_pose, i.e. what's really commanded this step
    after the delta/workspace clip, as opposed to the policy's raw
    target_pose argument (used by the rollout loop's pose-error logging)."""
    current_pose = arm.end_effector()

    delta_translation = target_pose[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:])

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_translation = np.clip(delta_translation,
                                  -safety_cfg.max_delta_translation,
                                  safety_cfg.max_delta_translation)
    clipped_rotation = np.clip(delta_rotation, -max_delta_rotation, max_delta_rotation)

    next_pos = current_pose[:3] + clipped_translation
    next_pos[0] = np.clip(next_pos[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
    next_pos[1] = np.clip(next_pos[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
    next_pos[2] = np.clip(next_pos[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
    next_orient = transformations.rotate_quaternion_by_rpy(*clipped_rotation, current_pose[3:])
    next_target = np.concatenate([next_pos, next_orient])

    arm.set_cartesian_target_pose(pose=next_target)
    if gripper is not None:
        gripper.set_normalized_position(gripper_norm)
    return next_target


def move_to_pose_via_cartesian_control(
    target_pose: np.ndarray, arm, gripper, gripper_norm, safety_cfg, fps: float,
    pos_tol: float = 0.01, rot_tol: float = 0.05, timeout_s: float = 15.0,
) -> None:
    """Ease into target_pose by repeatedly calling execute_cartesian_target
    against the same target, instead of jumping there via IK + joint
    trajectory controller. Each call only moves by safety_cfg's clipped
    per-step delta, so this just re-issues the same target every tick until
    the arm converges (or timeout_s elapses). Same helper as
    replay_pkl_episode.py's function of the same name -- used here to
    retract straight up between rollouts (see main())."""
    step_duration_s = 1.0 / fps
    deadline = time.perf_counter() + timeout_s

    while True:
        step_start = time.perf_counter()

        current_pose = arm.end_effector()
        pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
        rot_err = np.linalg.norm(
            transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:])
        )
        if pos_err < pos_tol and rot_err < rot_tol:
            logger.info(f"Reached target pose (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)")
            return
        if time.perf_counter() > deadline:
            logger.warning(
                f"Timed out easing into target pose after {timeout_s}s "
                f"(pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad) -- proceeding anyway."
            )
            return

        execute_cartesian_target(target_pose, arm, gripper, gripper_norm, safety_cfg)

        dt_s = time.perf_counter() - step_start
        remaining = step_duration_s - dt_s
        if remaining > 0:
            time.sleep(remaining)


def rewind_through_poses(recorded, arm, gripper, safety_cfg, step_duration_s, slow_factor) -> None:
    """Back the arm off by replaying `recorded` (a list of EE pose7 arrays) in
    reverse (newest -> oldest) via execute_cartesian_target, each waypoint held
    for slow_factor * step_duration_s so the retreat is visibly slower than
    normal execution. The gripper is commanded OPEN (normalized 1.0) throughout
    the retreat -- a "let go and back off" safety reaction. execute_cartesian_target's
    per-step delta-clip already bounds motion, and the recorded poses are dense
    (one per control step at fps) so the clip won't bite here."""
    for pose7 in reversed(recorded):
        execute_cartesian_target(np.asarray(pose7, dtype=float), arm, gripper, 1.0, safety_cfg)
        precise_sleep(slow_factor * step_duration_s)


def move_to_home(arm, cfg, gripper=None) -> None:
    """Same as evaluate_policy.py's function of the same name."""
    if cfg is None:
        logger.warning("Hydra config unavailable; skipping arm homing.")
        return
    ds_cfg = cfg.dataset
    home = ds_cfg.get("home_position", None)
    if home is None:
        logger.warning("dataset.home_position not set; skipping arm homing.")
        return
    home = np.array(home, dtype=float)

    jitter = float(ds_cfg.get("home_randomization", 0.0))
    if jitter > 0.0:
        home = home + np.random.uniform(-jitter, jitter, size=home.shape)

    if gripper is not None:
        # Release the gripper before moving, not after -- otherwise the arm
        # drags whatever it's still holding all the way back to home first.
        gripper_home = float(ds_cfg.get("gripper_home", 0.0))
        gripper.set_normalized_position(float(np.clip(gripper_home, 0.0, 1.0)))
        time.sleep(1.0)  # let the gripper finish releasing before the arm moves

    target_time = float(ds_cfg.get("home_target_time_s", 4.0))
    logger.info(f"Homing arm to {home} ({target_time:.1f}s)")
    arm.activate_joint_trajectory_controller()
    arm.set_joint_positions(positions=home, target_time=target_time, wait=True)


def load_start_pose_dataset(cfg):
    """Load the LeRobotDataset that rollout start poses are drawn from, once.

    Split out of resolve_dataset_start_pose() so the (relatively expensive)
    dataset/metadata load happens a single time per evaluation run, while the
    per-rollout episode pick (see pick_start_pose_episode_idx) and pose decode
    (resolve_dataset_start_pose) can vary every rollout.

    Returns (dataset, repo_id), or None when disabled/unavailable (caller then
    just uses joint homing).
    """
    if cfg is None:
        logger.warning("Hydra config unavailable; falling back to joint-space homing only.")
        return None
    ds_cfg = cfg.dataset
    if not bool(ds_cfg.get("use_dataset_start_pose", True)):
        logger.info("dataset.use_dataset_start_pose=false -- using joint-space homing only.")
        return None

    # Same repo_id convention as replay_pkl_episode.py, with start_pose_repo_id
    # taking priority: the start pose should come from the dataset the policy
    # was TRAINED on, which is not necessarily the one dataset.repo_id points
    # at (test_task.yaml's repo_id is used by the recording/replay scripts).
    repo_id = ds_cfg.get("start_pose_repo_id", None)
    if repo_id is None:
        task_name = ds_cfg.get("task_name", None)
        if task_name is not None:
            repo_id = f"{task_name}_lerobot"
        else:
            repo_id = ds_cfg.get("repo_id", None)
            if isinstance(repo_id, (list, ListConfig)):
                repo_id = str(repo_id[0])
    if repo_id is None:
        logger.warning(
            "No dataset.start_pose_repo_id / task_name / repo_id set -- "
            "falling back to joint-space homing only."
        )
        return None

    dataset_root = Path(ds_cfg.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Start-pose dataset not found: {dataset_root} (from dataset.start_pose_repo_id="
            f"{repo_id!r}). Set dataset.use_dataset_start_pose=false to home to "
            "dataset.home_position instead."
        )

    logger.info(f"Reading start pose from dataset {repo_id} ({dataset_root})")
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)
    return dataset, repo_id


def pick_start_pose_episode_idx(ds_cfg, dataset, rollout_id):
    """Which episode's first frame this rollout should ease into.

    "fixed" (default) always returns the same dataset.start_pose_episode_idx,
    matching the pre-existing behavior (one repeatable start pose for the
    whole run). "sequential" cycles through dataset.start_pose_episode_idxs
    -- null means every episode in the start-pose dataset -- one per rollout,
    wrapping around if num_rollouts exceeds the pool size.
    """
    mode = ds_cfg.get("start_pose_episode_idx_mode", "fixed")
    if mode == "fixed":
        return int(ds_cfg.get("start_pose_episode_idx", ds_cfg.get("episode_idx", 0)))
    if mode == "sequential":
        pool = parse_episode_spec(ds_cfg.get("start_pose_episode_idxs", None), dataset.meta.total_episodes)
        return pool[rollout_id % len(pool)]
    raise ValueError(
        f"Invalid dataset.start_pose_episode_idx_mode: {mode!r} (expected 'fixed' or 'sequential')"
    )


def resolve_dataset_start_pose(dataset, repo_id, episode_idx, offset, yaw: float = 0.0):
    """Cartesian pose the demonstrations start from, for a rollout to ease into.

    dataset.home_position is a joint configuration picked by hand, so the arm
    used to begin every rollout somewhere the policy never saw at t=0. Instead
    read observation.state of the first frame of episode_idx -- the pose the
    demos actually started from -- and go straight there, skipping the
    joint-space homing entirely. (observation.state, not action: action[t] ==
    state[t+1], so the first action is already one step in.)

    eval_offset/eval_yaw_offset_deg are applied here the same way
    execute_cartesian_target applies them to policy actions: along/about the
    pose's own local axes, so the physical arm sits offset while the pose the
    policy is asked to continue from still matches training. get_current_state
    subtracts them straight back out.

    Returns (pose7, gripper_norm, label) with pose7 = [x, y, z, qx, qy, qz, qw].
    """
    if not (0 <= episode_idx < dataset.meta.total_episodes):
        raise ValueError(
            f"dataset start-pose episode_idx={episode_idx} out of range "
            f"({repo_id} has {dataset.meta.total_episodes} episodes)"
        )
    ep_start = int(dataset.meta.episodes[episode_idx]["dataset_from_index"])

    state = dataset[ep_start]["observation.state"].numpy()  # [x,y,z, ortho6d(6), gripper]
    pose = np.concatenate(
        [state[:3], transformations.quaternion_from_ortho6(state[3:9])]
    )
    gripper_norm = gripper_state_to_normalized(float(state[9]))

    pose = pose_offset.apply_offset(pose, offset, yaw)

    label = f"{repo_id} episode {episode_idx} (frame {ep_start})"
    logger.info(
        f"Dataset start pose from {label}: pos={np.round(pose[:3], 4)} "
        f"quat={np.round(pose[3:], 4)} gripper_norm={gripper_norm:.2f}"
        + (f" (includes eval_offset {pose_offset.describe(offset, yaw)})"
           if pose_offset.is_enabled(offset, yaw) else "")
    )
    return pose, gripper_norm, label


def apply_random_start_pose_offset(dataset_start_pose, magnitude_m, rng):
    """Nudge a resolved start pose by magnitude_m in a random world-frame 3D direction.

    Unlike dataset.eval_offset (a fixed local-frame TCP correction that
    get_current_state subtracts back out so the policy never sees it), this
    perturbation is REAL as far as the policy is concerned -- it lands in
    observation.state at t=0 like any other change of start position -- so
    repeated rollouts probe the checkpoint's sensitivity to where it begins.
    The direction is uniform over the full 3D sphere (x, y, and z all vary),
    at the fixed magnitude_m radius -- same treatment on z as on x/y.

    `rng` is a np.random.Generator seeded per-rollout by the caller (see
    get_dataset_start_pose), NOT the global numpy RNG -- so the offset is
    deterministic: the same --seed reproduces the exact same direction for a
    given rollout_id every run, independent of whatever else in the script
    draws from the global RNG. Returns dataset_start_pose unchanged when it
    is None or magnitude_m is 0 (disabled).
    """
    if dataset_start_pose is None or magnitude_m == 0.0:
        return dataset_start_pose
    pose, gripper_norm, label = dataset_start_pose
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    offset = magnitude_m * direction
    pose = pose.copy()
    pose[:3] += offset
    logger.info(
        f"Start pose random offset: {magnitude_m * 1000:.1f} mm along "
        f"[{direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f}] "
        f"(dx={offset[0]:+.4f} dy={offset[1]:+.4f} dz={offset[2]:+.4f})"
    )
    return pose, gripper_norm, f"{label} + random offset {magnitude_m * 1000:.1f}mm"


def ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps) -> None:
    """Move to the demos' start pose, holding whatever the gripper was last told to do.

    Called once per rollout, and when enabled it REPLACES the joint-space
    move_to_home reset -- the arm goes straight from wherever it is to where
    the demonstrations began, via the same clipped-delta cartesian control the
    rollout itself uses (so it is a straight-line cartesian move, bounded by
    safety_cfg's per-step delta and workspace_range). Runs before the object is
    grasped (see main()), so the insert-by-hand step happens at the pose the
    policy actually starts from.

    The gripper is followed, not driven: whatever the caller last commanded is
    re-issued every step, so this never opens or closes the gripper itself.
    """
    if dataset_start_pose is None:
        return
    start_pose, _, label = dataset_start_pose

    if gripper is not None:
        # The *command*, not the measured position -- measurement lags a
        # just-issued open/close, and re-commanding a stale value here would
        # fight the gripper mid-travel.
        hold_norm = gripper.get_normalized_command()
    else:
        hold_norm = GRIPPER_OPEN_NORM

    arm.zero_ft_sensor()
    arm.activate_cartesian_controller()
    logger.info(f"Easing into dataset start pose [{label}]")
    move_to_pose_via_cartesian_control(start_pose, arm, gripper, hold_norm, safety_cfg, fps)


def stop_and_reset(arm, gripper, safety_cfg, fps, lift_distance_m,
                   dataset_start_pose, hydra_cfg, open_gripper: bool = True) -> bool:
    """Stop the arm and reset it -- for the in-rollout fail key (see KeyPoller)
    and for an FT force-limit trip (see WrenchRecorder.check_trip).

    1. FREEZE. Re-command the arm's own current pose as its cartesian target.
       Without this the compliance controller keeps servoing toward whatever
       the policy last asked for -- which, when the rollout was abandoned
       precisely because the arm is heading somewhere it should not, is the one
       place it must stop going. execute_cartesian_target's delta clip and
       workspace_range still apply, so the freeze target is a safe one.
    2. RELEASE the gripper (open_gripper, dataset.reset_open_gripper), before
       any further motion: retracting while still clamped onto an object that
       is jammed against the fixture -- a likely reason to be here at all --
       drags the whole fixture with it. Freezing first means the arm has already
       stopped chasing the policy's target before the jaws move, and
       ease_into_dataset_start_pose re-issues whatever the gripper was last
       commanded, so it stays open for the rest of the reset. Same "let go and
       back off" reaction rewind_through_poses has. Pass open_gripper=False to
       keep holding instead, on the argument that dropping a payload unasked is
       its own hazard.
    3. RETRACT lift_distance_m straight up (world +Z) to disengage, the same
       move the between-rollouts reset uses.
    4. EASE back to the demos' start pose (or home, when no start pose is
       configured), leaving the arm ready for the next rollout.

    Returns whether the gripper was actually released -- the caller needs to
    know, because a released object invalidates the next rollout's
    "still holding it from last time" assumption under start_gripper=closed.
    """
    hold_norm = gripper.get_normalized_command() if gripper is not None else GRIPPER_OPEN_NORM

    freeze_pose = arm.end_effector().copy()
    execute_cartesian_target(freeze_pose, arm, gripper, hold_norm, safety_cfg)
    logger.warning(f"STOPPED at pos={np.round(freeze_pose[:3], 4)} -- holding position.")
    time.sleep(0.5)   # let the controller settle onto the frozen target before moving again

    released = bool(open_gripper and gripper is not None)
    if released:
        logger.info("Opening the gripper before the reset motion...")
        gripper.set_normalized_position(GRIPPER_OPEN_NORM)
        hold_norm = GRIPPER_OPEN_NORM
        time.sleep(1.0)   # let the jaws finish opening before the arm moves

    logger.info(f"Retracting {lift_distance_m * 100:.0f} cm straight up before resetting...")
    lift_pose = arm.end_effector().copy()
    lift_pose[2] += lift_distance_m
    move_to_pose_via_cartesian_control(lift_pose, arm, gripper, hold_norm, safety_cfg, fps)

    if dataset_start_pose is None:
        move_to_home(arm, hydra_cfg, gripper=None)
    # No-op when dataset_start_pose is None -- move_to_home just handled it.
    ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps)
    return released


class PoseStiffnessModulator:
    """Optional continuous (linear-ramp) stiffness modulation during
    rollouts, driven by the live MLP-estimated wrist deflection
    (bendlabs/estimate_pose_node.py).

    Subscribes to pose_topic (a relative geometry_msgs/PoseStamped, position
    = displacement-from-start, orientation = rotation-since-start -- same
    convention as dynamic_compliance_test.py/stiffness_test.py). Call
    update() once per control step; it computes per-axis deflection, maps it
    through osx_ur5e.compliance_utils.distance_to_stiffness with
    profile="linear" (max stiffness at/below near, ramping linearly down to
    min stiffness at/above far -- no hysteresis/latching, unlike the 2-stage
    profile dynamic_compliance_test.py also supports), and pushes the result
    to arm.update_stiffness(...) whenever it changes by at least min_change.
    If the pose topic goes stale for more than pose_timeout seconds, forces
    max stiffness, so a dropped estimator never leaves the arm unexpectedly
    soft."""

    def __init__(self, arm, pose_topic, trans_near, trans_far, rot_near, rot_far,
                trans_min, trans_max, rot_min, rot_max, min_change, pose_timeout):
        self.arm = arm
        self.trans_near, self.trans_far = trans_near, trans_far
        self.rot_near, self.rot_far = rot_near, rot_far
        self.trans_min, self.trans_max = trans_min, trans_max
        self.rot_min, self.rot_max = rot_min, rot_max
        self.min_change = min_change
        self.pose_timeout = pose_timeout

        self.max_stiffness = np.array([trans_max] * 3 + [rot_max] * 3)
        self.last_pose = None  # (pos: np.array(3), quat: np.array(4))
        self.last_pose_stamp = None
        self.last_sent = None

        rospy.Subscriber(pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        logger.info(
            f"PoseStiffnessModulator: tracking {pose_topic} for linear stiffness "
            f"(trans {trans_min}-{trans_max} N/m over {trans_near}-{trans_far} m, "
            f"rot {rot_min}-{rot_max} Nm/rad over {rot_near}-{rot_far} rad)"
        )

    def _pose_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.last_pose = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        self.last_pose_stamp = rospy.get_time()

    def _send(self, stiffness):
        if self.last_sent is not None and np.all(np.abs(stiffness - self.last_sent) < self.min_change):
            return
        self.arm.update_stiffness(stiffness.astype(np.int64))
        self.last_sent = stiffness
        self.arm.current_stiffness = stiffness  # keep in sync so any generic "log current stiffness" reads this

    def update(self) -> None:
        """Call once per control step."""
        if self.last_pose is None:
            return  # nothing received yet -- don't command anything

        if rospy.get_time() - self.last_pose_stamp > self.pose_timeout:
            logger.warning(
                f"PoseStiffnessModulator: no pose for >{self.pose_timeout}s -- forcing max stiffness (fail-safe)"
            )
            self._send(self.max_stiffness)
            return

        pos, quat = self.last_pose
        d_trans = np.abs(pos)
        d_rot = quat_to_rotvec_abs(*quat)

        stiff_trans = distance_to_stiffness(
            d_trans, self.trans_near, self.trans_far, self.trans_min, self.trans_max, "linear")
        stiff_rot = distance_to_stiffness(
            d_rot, self.rot_near, self.rot_far, self.rot_min, self.rot_max, "linear")
        self._send(np.concatenate([stiff_trans, stiff_rot]))

    def status_str(self) -> str:
        """Deflection only -- the caller already logs current stiffness."""
        if self.last_pose is None:
            return "  (no pose received yet)"
        pos, quat = self.last_pose
        return (f"  defl_pos={np.round(np.abs(pos), 4)} defl_rot={np.round(quat_to_rotvec_abs(*quat), 3)}")


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot ACT policy trained on live-tracked object keypoints")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the trained policy checkpoint directory")
    parser.add_argument("--tracking_node_ns", type=str, default="tracking_node",
                        help="Namespace tracking_node.py was started in -- selects which "
                             "services/topics to talk to (<ns>/reset_and_track, "
                             "<ns>/stop_track, <ns>/environment_state, <ns>/visualization_image, "
                             "<ns>/num_points).")
    parser.add_argument("--num_rollouts", type=int, default=60)
    parser.add_argument("--max_timesteps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use_gripper", action="store_true", default=True)
    parser.add_argument("--fps", type=int, default=None,
                        help="Control frequency in Hz. If not given, read from dataset.fps "
                             "in the hydra config (falls back to 30 if unavailable).")
    parser.add_argument("--hydra_config_path", type=str, default="../config/hydra")
    parser.add_argument("--hydra_config_name", type=str, default="test_task")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tracking_staleness_s", type=float, default=1.0,
                        help="If tracking_node hasn't produced a fresh environment_state "
                             "within this many seconds (stalled, crashed, or not running), "
                             "stop commanding new cartesian targets (hold at the last "
                             "commanded pose) until tracking recovers.")
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01,
                        help="Enable ACT temporal ensembling at inference with this "
                             "exponential weighting coefficient (the original ACT work "
                             "uses 0.01; 0 weighs all overlapping predictions equally, "
                             "positive favours older ones, negative favours newer). This "
                             "is inference-only -- no retraining needed -- but it forces "
                             "n_action_steps=1, so the policy runs a full forward pass "
                             "every control step instead of once per chunk. Omit to keep "
                             "the checkpoint's own setting (queue mode for most ACTs).")
    parser.add_argument("--drop_env_keypoints", nargs="+", type=int, default=[],
                        help="Keypoint INDICES (0-based) that tracking_node publishes but this "
                             "policy was NOT trained on -- i.e. whatever was passed to "
                             "convert_pkl_to_lerobot.py's --drop_keypoints when the dataset was "
                             "exported. tracking_node always publishes every annotated keypoint, "
                             "so without this the policy-dim-vs-tracker-dim check aborts and, "
                             "worse, an unchecked mismatch would silently feed the wrong "
                             "keypoint into the wrong input slot. Must match the export exactly.")
    parser.add_argument("--env_state_offset", type=float, nargs="+", default=None,
                        metavar="V",
                        help="TEMPORARY WORKAROUND. Add this offset (metres) to the tracked "
                             "observation.environment_state before it reaches the policy, to "
                             "compensate for the scene having physically moved since the demos "
                             "were recorded. Give 3 values (dx dy dz, applied to every kept "
                             "keypoint -- the right form when the whole fixture shifted) or one "
                             "value per kept dimension. Applied AFTER --drop_env_keypoints, so "
                             "it matches the vector the policy actually consumes. This papers "
                             "over a real mismatch: the honest fixes are to restore the rig or "
                             "retrain. Because these keypoints are near-static, their training "
                             "std can be well under a millimetre, so a few mm of drift is worth "
                             "tens of sigma -- run with it and check the logged z-scores.")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override how many actions of each predicted chunk are "
                             "executed before the policy re-runs inference. Capped at the "
                             "checkpoint's chunk_size (the decoder's output length, which "
                             "is architectural -- going beyond it needs retraining). "
                             "Inference-only, no retraining. Lower = the policy re-reads "
                             "observation.environment_state more often (tighter feedback, "
                             "less stale-action drift within a chunk); higher = longer "
                             "open-loop stretches that lean less on the tracker. Mutually "
                             "exclusive with --temporal_ensemble_coeff, which pins it to 1. "
                             "Omit to keep the checkpoint's own value.")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Resolve to absolute up front: rosrun inherits the calling shell's cwd,
    # which is easy to lose track of, and PreTrainedConfig.from_pretrained
    # silently treats any path that doesn't resolve to a real directory as a
    # HuggingFace Hub repo id instead -- producing a confusing
    # HFValidationError with no mention of the actual (missing) local path.
    policy_path = Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(
            f"Policy checkpoint directory not found: {policy_path} "
            f"(from --policy_path={args.policy_path!r}, resolved relative to cwd={Path.cwd()})"
        )
    if not (policy_path / "config.json").exists():
        raise FileNotFoundError(
            f"{policy_path} exists but has no config.json -- is this a valid "
            "policy checkpoint directory (not e.g. its parent outputs/train/ dir)?"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/eval") / policy_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "evaluation.log")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Results will be saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Hydra config (controller/safety parameters + fps/home fallback)
    # ------------------------------------------------------------------
    hydra_cfg = None
    try:
        hydra_cfg = load_hydra_config(args.hydra_config_path, args.hydra_config_name)
        logger.info(f"Loaded hydra config '{args.hydra_config_name}'")
    except Exception as e:
        logger.warning(f"Could not load hydra config '{args.hydra_config_name}': {e}")

    dataset_fps = int(hydra_cfg.dataset.fps) if hydra_cfg is not None else None
    controller_cfg = hydra_cfg.controller if hydra_cfg is not None else None
    safety_cfg = controller_cfg.safety_parameters if controller_cfg is not None else None

    if args.fps is not None:
        fps = args.fps
    elif dataset_fps is not None:
        fps = dataset_fps
        logger.info(f"--fps not given; using dataset.fps={fps} from hydra config.")
    else:
        fps = 30
        logger.warning(f"Hydra config unavailable; falling back to fps={fps}.")

    # On a large step-to-step change in the policy's predicted gripper
    # command, stretch out (not freeze) the next gripper_slowdown_duration_s
    # of real time by multiplying the per-step sleep by
    # gripper_slowdown_factor -- gives the slow gripper a head start before the
    # arm moves far. Same mechanism as replay_pkl.py, adapted for a
    # continuous per-step command instead of a recorded discrete array.
    ds_cfg_for_slowdown = hydra_cfg.dataset if hydra_cfg is not None else None
    gripper_slowdown_enabled = bool(ds_cfg_for_slowdown.get("gripper_slowdown_enabled", True)) \
        if ds_cfg_for_slowdown is not None else True
    gripper_slowdown_factor = float(ds_cfg_for_slowdown.get("gripper_slowdown_factor", 4.0)) \
        if ds_cfg_for_slowdown is not None else 4.0
    gripper_slowdown_duration_s = float(ds_cfg_for_slowdown.get("gripper_slowdown_duration_s", 2.0)) \
        if ds_cfg_for_slowdown is not None else 2.0
    gripper_slowdown_change_threshold = float(ds_cfg_for_slowdown.get("gripper_slowdown_change_threshold", 0.1)) \
        if ds_cfg_for_slowdown is not None else 0.1

    # When any BendLabs bend/twist sensor deviates from its start-of-episode
    # baseline by more than bendlab_rewind_threshold, back the arm off
    # rewind_steps control steps (replaying recorded poses in reverse at
    # rewind_slow_factor x normal speed, opening the gripper during the
    # retreat), clear the policy's action queue, and re-run inference from the
    # backed-off pose. After rewind_max_attempts consecutive rewinds without
    # rewind_clear_steps clean steps in between, hold position and abort the
    # rollout. Independent of the gripper_slowdown_* trigger above.
    bendlab_rewind_enabled = bool(ds_cfg_for_slowdown.get("bendlab_rewind_enabled", True)) \
        if ds_cfg_for_slowdown is not None else True
    bendlab_rewind_threshold = float(ds_cfg_for_slowdown.get("bendlab_rewind_threshold", 5.0)) \
        if ds_cfg_for_slowdown is not None else 5.0
    rewind_steps = int(ds_cfg_for_slowdown.get("rewind_steps", 15)) \
        if ds_cfg_for_slowdown is not None else 15
    rewind_slow_factor = float(ds_cfg_for_slowdown.get("rewind_slow_factor", 4.0)) \
        if ds_cfg_for_slowdown is not None else 4.0
    rewind_max_attempts = int(ds_cfg_for_slowdown.get("rewind_max_attempts", 3)) \
        if ds_cfg_for_slowdown is not None else 3
    rewind_clear_steps = int(ds_cfg_for_slowdown.get("rewind_clear_steps", 15)) \
        if ds_cfg_for_slowdown is not None else 15
    num_bendlabs_sensors = int(ds_cfg_for_slowdown.get("num_bendlabs_sensors", 4)) \
        if ds_cfg_for_slowdown is not None else 4

    # Feed the policy a *binary* gripper signal (open=holding nothing,
    # closed=holding something) derived from the gap between the gripper's
    # commanded and measured position, instead of the raw measured position.
    # grasp_cmd_close_threshold: commanded position (normalized, 0=closed) at or
    # below which the gripper counts as "commanded shut". grasp_gap_threshold: how
    # much more open (normalized) the measured jaws must be than commanded
    # before it counts as "held open by an object". See get_current_state.
    grasp_binary_enabled = bool(ds_cfg_for_slowdown.get("grasp_binary_enabled", True)) \
        if ds_cfg_for_slowdown is not None else True
    grasp_cmd_close_threshold = float(ds_cfg_for_slowdown.get("grasp_cmd_close_threshold", 0.5)) \
        if ds_cfg_for_slowdown is not None else 0.5
    grasp_gap_threshold = float(ds_cfg_for_slowdown.get("grasp_gap_threshold", 0.15)) \
        if ds_cfg_for_slowdown is not None else 0.15
    grasp_cfg = GraspConfig(grasp_binary_enabled, grasp_cmd_close_threshold, grasp_gap_threshold)

    # Gripper position at the start of each rollout when start_gripper is
    # "auto" -- normally issued by move_to_home, which is skipped when the
    # rollout goes straight to the dataset start pose, so the reset block
    # issues it itself in that case.
    gripper_home = float(ds_cfg_for_slowdown.get("gripper_home", 0.0)) \
        if ds_cfg_for_slowdown is not None else 0.0

    # Force the gripper OPEN or CLOSED at the start of each rollout, instead of
    # the dataset.gripper_home value above. "closed" eases in with the gripper
    # OPEN, pauses for you to insert an object, then closes it before the
    # rollout starts -- same idea as replay_pkl_episode.py's start_gripper.
    start_gripper = ds_cfg_for_slowdown.get("start_gripper", "auto") \
        if ds_cfg_for_slowdown is not None else "auto"
    if start_gripper not in ("auto", "open", "closed"):
        raise ValueError(f"Invalid dataset.start_gripper: {start_gripper!r} (expected 'auto', 'open', or 'closed')")

    # start_gripper=="closed" only: re-run the open/insert-by-hand/close ritual
    # before EVERY rollout, instead of only the first one (and after a reset
    # that released the object). Use it when the object cannot simply stay in
    # the jaws between attempts -- it has to be re-seated, swapped, or put back
    # in the fixture by hand each time. The arm still retracts
    # reset_lift_distance_m straight up first, so the object is disengaged
    # before the jaws open. false (default) = previous behavior: keep holding
    # it from one rollout to the next.
    reinsert_every_rollout = bool(ds_cfg_for_slowdown.get("reinsert_every_rollout", False)) \
        if ds_cfg_for_slowdown is not None else False
    if reinsert_every_rollout and start_gripper != "closed":
        logger.warning(
            "dataset.reinsert_every_rollout is set but dataset.start_gripper is "
            f"{start_gripper!r}, not 'closed' -- there is no insert-by-hand step to repeat, "
            "so it has no effect."
        )

    # Which gripper is on the wrist -- validated here rather than at the
    # make_gripper() call below, so a typo aborts before the policy loads
    # instead of a minute later. See RobotiqGripperAdapter/make_gripper.
    resolve_gripper_type(ds_cfg_for_slowdown)

    # For start_gripper=="closed", only the very first rollout goes through the
    # open/insert-by-hand/close ritual above (unless reinsert_every_rollout is
    # set, in which case every one does). Every rollout after that assumes
    # the object is still held from the previous attempt, so instead of
    # releasing and re-inserting it, the arm retracts this far straight up
    # (world +Z) to disengage, then homes without touching the gripper --
    # see the reset block in main()'s rollout loop. The retract happens either
    # way; reinsert_every_rollout only decides whether the jaws then open.
    reset_lift_distance_m = float(ds_cfg_for_slowdown.get("reset_lift_distance_m", 0.07)) \
        if ds_cfg_for_slowdown is not None else 0.07

    # Shift the physically-commanded pose by [x, y, z] metres along, and
    # dataset.eval_yaw_offset_deg degrees about, the target's OWN local axes
    # (+z = tool0's approach/reach direction, yaw = spin about that same
    # axis) -- e.g. to correct a gripper whose real tool tip sits off-centre
    # and/or rotated from where the dataset's hand-derived geometry put it,
    # without retraining. Local, not world: whichever way the gripper is
    # currently facing, the offset/yaw always mean the same thing relative to
    # the gripper. Added to every commanded target (see target_pose below)
    # and subtracted back out of observation.state (see get_current_state),
    # so the policy always sees/plans in its original, un-offset frame while
    # the arm actually operates offset away. See osx_ur5e.pose_offset; the
    # legacy scalar dataset.eval_z_offset is still honoured as [0, 0, z].
    eval_offset = pose_offset.resolve_offset(ds_cfg_for_slowdown)
    eval_yaw_offset = pose_offset.resolve_yaw_offset(ds_cfg_for_slowdown)
    if pose_offset.is_enabled(eval_offset, eval_yaw_offset):
        logger.info(
            f"dataset.eval_offset: {pose_offset.describe(eval_offset, eval_yaw_offset)} -- "
            "physical pose will be shifted/rotated along the gripper's own axes relative to "
            "the policy's frame."
        )

    # Where each rollout starts from: the demos' own first-frame EE pose rather
    # than the hand-picked dataset.home_position joint config. The dataset
    # itself is loaded once here; which episode's start pose to use is instead
    # re-picked every rollout (see the rollout loop below), so
    # dataset.start_pose_episode_idx_mode="sequential" can vary it per
    # rollout. eval_offset is already folded in.
    start_pose_dataset = load_start_pose_dataset(hydra_cfg)

    # Extra domain-randomization nudge on top of the resolved start pose: a
    # fixed-magnitude, random-direction (world XY) offset. Independent of
    # eval_offset above -- see apply_random_start_pose_offset's docstring for
    # why. 0.01 m = 1 cm default; set dataset.start_pose_random_offset_m: 0.0
    # to disable. Deterministic: seeded per-rollout from --seed below (not
    # the global RNG), so re-running with the same --seed reproduces the
    # exact same offset for a given rollout_id.
    start_pose_random_offset_m = float(ds_cfg_for_slowdown.get("start_pose_random_offset_m", 0.01)) \
        if ds_cfg_for_slowdown is not None else 0.01

    def get_dataset_start_pose(rollout_id):
        if start_pose_dataset is None:
            return None
        dataset, repo_id = start_pose_dataset
        episode_idx = pick_start_pose_episode_idx(hydra_cfg.dataset, dataset, rollout_id)
        pose = resolve_dataset_start_pose(dataset, repo_id, episode_idx, eval_offset, eval_yaw_offset)
        rng = np.random.default_rng(args.seed + rollout_id)
        return apply_random_start_pose_offset(pose, start_pose_random_offset_m, rng)

    dataset_start_pose = get_dataset_start_pose(0)

    # Optional continuous (linear-ramp) stiffness during rollouts, driven by
    # the live MLP-estimated wrist deflection -- see PoseStiffnessModulator
    # for the full mechanics. Off by default.
    enable_pose_stiffness = bool(ds_cfg_for_slowdown.get("enable_pose_stiffness", False)) \
        if ds_cfg_for_slowdown is not None else False
    pose_stiffness_pose_topic = ds_cfg_for_slowdown.get("pose_stiffness_pose_topic", "/estimated_pose") \
        if ds_cfg_for_slowdown is not None else "/estimated_pose"
    pose_stiffness_trans_near = float(ds_cfg_for_slowdown.get("pose_stiffness_trans_near", 0.003)) \
        if ds_cfg_for_slowdown is not None else 0.003
    pose_stiffness_trans_far = float(ds_cfg_for_slowdown.get("pose_stiffness_trans_far", 0.01)) \
        if ds_cfg_for_slowdown is not None else 0.01
    pose_stiffness_rot_near = float(ds_cfg_for_slowdown.get("pose_stiffness_rot_near", 0.05)) \
        if ds_cfg_for_slowdown is not None else 0.05
    pose_stiffness_rot_far = float(ds_cfg_for_slowdown.get("pose_stiffness_rot_far", 0.5)) \
        if ds_cfg_for_slowdown is not None else 0.5
    pose_stiffness_trans_min = float(ds_cfg_for_slowdown.get("pose_stiffness_trans_min", 300.0)) \
        if ds_cfg_for_slowdown is not None else 300.0
    pose_stiffness_trans_max = float(ds_cfg_for_slowdown.get("pose_stiffness_trans_max", 3000.0)) \
        if ds_cfg_for_slowdown is not None else 3000.0
    pose_stiffness_rot_min = float(ds_cfg_for_slowdown.get("pose_stiffness_rot_min", 10.0)) \
        if ds_cfg_for_slowdown is not None else 10.0
    pose_stiffness_rot_max = float(ds_cfg_for_slowdown.get("pose_stiffness_rot_max", 150.0)) \
        if ds_cfg_for_slowdown is not None else 150.0
    pose_stiffness_min_change = float(ds_cfg_for_slowdown.get("pose_stiffness_min_change", 5.0)) \
        if ds_cfg_for_slowdown is not None else 5.0
    pose_stiffness_pose_timeout = float(ds_cfg_for_slowdown.get("pose_stiffness_pose_timeout", 0.5)) \
        if ds_cfg_for_slowdown is not None else 0.5

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    logger.info(f"Loading policy from: {policy_path}")
    config = load_policy_config_compat(policy_path)

    # Temporal ensembling is a pure inference-time aggregation of the overlapping
    # action chunks predicted at successive steps -- training never sees it, so it
    # can be switched on for any ACT checkpoint without retraining. It does require
    # n_action_steps=1 (ACTConfig.__post_init__ enforces this): every chunk has to
    # be re-predicted each step for there to be anything to ensemble.
    if args.temporal_ensemble_coeff is not None:
        if not hasattr(config, "temporal_ensemble_coeff"):
            raise ValueError(
                f"--temporal_ensemble_coeff given, but policy type '{config.type}' has no "
                "temporal_ensemble_coeff field (it is an ACT-family option)."
            )
        config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        config.n_action_steps = 1
        logger.info(
            f"Temporal ensembling ENABLED (coeff={args.temporal_ensemble_coeff}, "
            f"chunk_size={config.chunk_size}); n_action_steps forced to 1, so the "
            "policy runs a full forward pass every control step."
        )

    # How much of each predicted chunk gets executed before re-inferring. Purely
    # an inference-time schedule -- the decoder still emits chunk_size actions
    # either way -- so this needs no retraining, but chunk_size is a hard ceiling
    # (it IS the decoder's output length). Rejected rather than silently clamped:
    # a value above chunk_size means the caller expected more lookahead than the
    # checkpoint can produce, and quietly capping it would hide that.
    if args.n_action_steps is not None:
        if args.temporal_ensemble_coeff is not None:
            raise ValueError(
                "--n_action_steps conflicts with --temporal_ensemble_coeff, which "
                "requires n_action_steps=1. Pass only one."
            )
        if args.n_action_steps < 1:
            raise ValueError(f"--n_action_steps must be >= 1, got {args.n_action_steps}")
        if args.n_action_steps > config.chunk_size:
            raise ValueError(
                f"--n_action_steps {args.n_action_steps} exceeds this checkpoint's "
                f"chunk_size {config.chunk_size}; the policy only predicts "
                f"{config.chunk_size} actions per forward pass. Retrain with a larger "
                "chunk_size to go beyond it."
            )
        previous = config.n_action_steps
        config.n_action_steps = args.n_action_steps
        logger.info(
            f"n_action_steps overridden {previous} -> {args.n_action_steps} "
            f"(chunk_size={config.chunk_size}): re-running inference every "
            f"{args.n_action_steps} control step(s)."
        )

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=config)
    policy.cuda()
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    logger.info(f"Input features:  {list(config.input_features.keys())}")
    logger.info(f"Output features: {list(config.output_features.keys())}")

    env_feature = config.input_features.get("observation.environment_state")
    if env_feature is None:
        raise ValueError(
            "Policy has no observation.environment_state input feature -- "
            "was it trained on tracked points (convert_pkl_to_lerobot.py output)?"
        )

    # ------------------------------------------------------------------
    # Connect to tracking_node.py: sanity-check N against the policy, wait
    # for its services, subscribe to its topics.
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy_points", anonymous=False)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    logger.info("ROS node initialized")
    # After init_node, so rospy's own signal setup can't clobber it.
    install_stack_dumper()
    logger.info(f"PID {os.getpid()} -- if this run ever stops responding, "
                f"`kill -USR1 {os.getpid()}` dumps every thread's stack to stderr "
                "(see install_stack_dumper).")

    ns = args.tracking_node_ns.rstrip("/")
    num_points_param = f"{ns}/num_points"
    logger.info(f"Waiting for tracking_node ('{ns}') to report {num_points_param}...")
    n_points = None
    deadline = time.perf_counter() + 15.0
    while n_points is None and time.perf_counter() < deadline and not rospy.is_shutdown():
        if rospy.has_param(num_points_param):
            n_points = rospy.get_param(num_points_param)
        else:
            rospy.sleep(0.2)
    if n_points is None:
        raise RuntimeError(
            f"Timed out waiting for rosparam {num_points_param} -- is tracking_node.py "
            f"running (rosrun osx_ur5e tracking_node.py --task_name <task>)?"
        )
    expected_dim = env_feature.shape[0]
    dropped = sorted(set(args.drop_env_keypoints))
    bad = [i for i in dropped if not (0 <= i < n_points)]
    if bad:
        raise ValueError(
            f"--drop_env_keypoints {bad} out of range: tracking_node publishes "
            f"{n_points} keypoint(s) (valid indices 0..{n_points - 1})."
        )
    env_keep_idx = [i for i in range(n_points) if i not in set(dropped)]
    if not env_keep_idx:
        raise ValueError("--drop_env_keypoints would discard every tracked keypoint.")
    if expected_dim != len(env_keep_idx) * 3:
        raise ValueError(
            f"Policy expects observation.environment_state of dim {expected_dim}, but "
            f"tracking_node reports {n_points} point(s) and --drop_env_keypoints={dropped} "
            f"leaves {len(env_keep_idx)} ({len(env_keep_idx) * 3} dims). Either "
            "tracking_node.py was started with a --task_name that doesn't match this "
            "policy's task, or --drop_env_keypoints doesn't match the --drop_keypoints "
            "used when the dataset was exported."
        )
    if dropped:
        logger.info(
            f"tracking_node reports {n_points} tracked point(s); dropping {dropped} to match "
            f"the policy -- feeding keypoints {env_keep_idx} ({expected_dim} dims)."
        )
    else:
        logger.info(f"tracking_node reports {n_points} tracked point(s) -- matches policy.")

    # Flat indices into the published (N*3,) vector, precomputed once.
    env_keep_flat = np.array([3 * i + c for i in env_keep_idx for c in range(3)], dtype=int)

    env_state_offset = None
    if args.env_state_offset is not None:
        off = np.asarray(args.env_state_offset, dtype=float)
        n_dims = len(env_keep_flat)
        if off.size == 3:
            off = np.tile(off, n_dims // 3)       # same xyz shift for every keypoint
        elif off.size != n_dims:
            raise ValueError(
                f"--env_state_offset needs 3 values (a per-keypoint xyz shift) or "
                f"{n_dims} (one per kept dimension); got {off.size}."
            )
        env_state_offset = off
        logger.warning(
            "TEMPORARY --env_state_offset active: adding %s (mm) to the tracked "
            "environment_state. This compensates for a scene shift rather than fixing "
            "it; remove it once the rig is restored or the policy retrained.",
            np.round(off * 1000, 2).tolist(),
        )

    try:
        rospy.wait_for_service(f"{ns}/reset_and_track", timeout=15.0)
        rospy.wait_for_service(f"{ns}/reset_and_track_manual", timeout=15.0)
        rospy.wait_for_service(f"{ns}/stop_track", timeout=15.0)
    except rospy.ROSException as e:
        raise RuntimeError(f"tracking_node services not available: {e}")
    reset_and_track_srv = rospy.ServiceProxy(f"{ns}/reset_and_track", Trigger)
    reset_and_track_manual_srv = rospy.ServiceProxy(f"{ns}/reset_and_track_manual", Trigger)
    stop_track_srv = rospy.ServiceProxy(f"{ns}/stop_track", Trigger)

    env_state_sub = EnvironmentStateSubscriber(f"{ns}/environment_state")

    # dataset.record_rollout_video: save each rollout's tracking_node
    # visualization (the annotated two-camera panel, same one the reset
    # prompts point at rqt_image_view for) as an mp4 under
    # output_dir/videos/. Default on; set false to skip it.
    record_rollout_video = bool(
        hydra_cfg.dataset.get("record_rollout_video", True) if hydra_cfg is not None else True
    )
    viz_recorder = VisualizationRecorder(f"{ns}/visualization_image") if record_rollout_video else None

    # dataset.record_pose_error_plot: save a per-step commanded-vs-actual
    # cartesian pose plot per rollout under output_dir/pose_error/. Default
    # on; set false to skip it. See save_pose_error_visualization.
    record_pose_error_plot = bool(
        hydra_cfg.dataset.get("record_pose_error_plot", True) if hydra_cfg is not None else True
    )

    # dataset.record_ft_data / record_ft_plot: record the UR5e's force/torque
    # sensor for the duration of each rollout, saved alongside that rollout's
    # video as output_dir/ft/rollout<NN>.npz (raw samples) and .png (plot).
    # Captured at the topic's own rate off dataset.ft_topic, so it is
    # independent of the control loop -- a stalled or skipped rollout still
    # yields whatever the sensor saw. See WrenchRecorder/save_ft_visualization.
    record_ft_data = bool(
        hydra_cfg.dataset.get("record_ft_data", True) if hydra_cfg is not None else True
    )
    record_ft_plot = bool(
        hydra_cfg.dataset.get("record_ft_plot", True) if hydra_cfg is not None else True
    )
    ft_topic = str(
        hydra_cfg.dataset.get("ft_topic", "/wrench") if hydra_cfg is not None else "/wrench"
    )
    # dataset.ft_force_limit_n: safety stop. When the force magnitude on
    # ft_topic exceeds this many newtons (for ft_force_limit_consecutive samples
    # in a row), the rollout is abandoned exactly as if 'f' had been pressed --
    # freeze, retract, reset -- and the next one starts. null/0 = no limit.
    ft_force_limit_n = (
        hydra_cfg.dataset.get("ft_force_limit_n", 30.0) if hydra_cfg is not None else 30.0
    )
    ft_force_limit_n = float(ft_force_limit_n) if ft_force_limit_n else 0.0
    ft_force_limit_consecutive = int(
        hydra_cfg.dataset.get("ft_force_limit_consecutive", 3) if hydra_cfg is not None else 3
    )

    # Also needed when only the safety limit is on -- it is detected in the
    # recorder's subscriber callback, so the recorder has to exist either way.
    ft_recorder = (
        WrenchRecorder(ft_topic, force_limit=ft_force_limit_n,
                       limit_consecutive=ft_force_limit_consecutive)
        if (record_ft_data or record_ft_plot or ft_force_limit_n) else None
    )
    if ft_recorder is not None:
        logger.info(f"Recording FT sensor from {ft_topic} for each rollout.")
    if ft_force_limit_n:
        logger.info(
            f"FT safety stop ARMED: rollout is abandoned when |F| on {ft_topic} exceeds "
            f"{ft_force_limit_n:.1f} N for {ft_force_limit_consecutive} consecutive samples."
        )
    else:
        logger.warning("FT safety stop DISABLED (dataset.ft_force_limit_n is null/0).")

    # dataset.record_bend_data / record_bend_plot: record the flexible
    # (BendLabs) sensors for the duration of each rollout, saved alongside the
    # video and FT trace as output_dir/bend/rollout<NN>.npz / .png. Sampled once
    # per control step from the BendLabsRecorder the bend-sensor rewind already
    # uses. See save_bend_data / save_bend_visualization.
    record_bend_data = bool(
        hydra_cfg.dataset.get("record_bend_data", True) if hydra_cfg is not None else True
    )
    record_bend_plot = bool(
        hydra_cfg.dataset.get("record_bend_plot", True) if hydra_cfg is not None else True
    )
    record_bend = record_bend_data or record_bend_plot

    # dataset.reset_open_gripper: open the gripper before stop_and_reset's
    # retract, rather than retracting while still clamped onto the object.
    reset_open_gripper = bool(
        hydra_cfg.dataset.get("reset_open_gripper", True) if hydra_cfg is not None else True
    )

    # ------------------------------------------------------------------
    # Initialize hardware
    # ------------------------------------------------------------------
    arm = CompliantController(gripper_type=None)
    if controller_cfg is not None:
        arm.set_control_mode(controller_cfg.mode)
        arm.update_pd_gains(
            OmegaConf.to_container(controller_cfg.p_gains),
            OmegaConf.to_container(controller_cfg.d_gains),
        )
        arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
        arm.set_solver_parameters(
            error_scale=controller_cfg.error_scale,
            iterations=controller_cfg.iterations,
        )
        arm.current_stiffness = np.array(
            [controller_cfg.stiffness_trans] * 3 + [controller_cfg.stiffness_rot] * 3)
        arm.update_stiffness(np.diag(arm.current_stiffness))
        logger.info(f"Initial stiffness: trans={arm.current_stiffness[:3].round(0)} N/m  "
                    f"rot={arm.current_stiffness[3:].round(0)} Nm/rad")
        arm.auto_switch_controllers = False
        arm.async_mode = True
    arm.zero_ft_sensor()

    gripper = None
    if args.use_gripper:
        gripper = make_gripper(ds_cfg_for_slowdown)

    # Needed for the bend-sensor rewind reaction, for recording the flexible
    # sensors, or both -- one recorder serves either purpose.
    bendlabs = None
    if bendlab_rewind_enabled or record_bend:
        wanted = " + ".join(
            w for w, on in (("rewind", bendlab_rewind_enabled), ("recording", record_bend)) if on
        )
        logger.info(f"Initializing BendLabsRecorder ({num_bendlabs_sensors} sensors, {wanted})...")
        bendlabs = BendLabsRecorder(init_node=False, num_sensors=num_bendlabs_sensors)
        bendlabs_deadline = time.perf_counter() + 10.0
        while not bendlabs.sensors_ready() and not rospy.is_shutdown():
            if time.perf_counter() > bendlabs_deadline:
                logger.warning(
                    "Timed out waiting for BendLabs sensors -- is uart_bridge_node running? "
                    f"Disabling bend-sensor {wanted} for this run."
                )
                bendlabs = None
                break
            rospy.sleep(0.1)
        if bendlabs is not None:
            logger.info("BendLabs sensors ready.")

    stiffness_modulator = None
    if enable_pose_stiffness:
        stiffness_modulator = PoseStiffnessModulator(
            arm=arm,
            pose_topic=pose_stiffness_pose_topic,
            trans_near=pose_stiffness_trans_near,
            trans_far=pose_stiffness_trans_far,
            rot_near=pose_stiffness_rot_near,
            rot_far=pose_stiffness_rot_far,
            trans_min=pose_stiffness_trans_min,
            trans_max=pose_stiffness_trans_max,
            rot_min=pose_stiffness_rot_min,
            rot_max=pose_stiffness_rot_max,
            min_change=pose_stiffness_min_change,
            pose_timeout=pose_stiffness_pose_timeout,
        )

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    num_rollouts = args.num_rollouts
    max_timesteps = args.max_timesteps
    step_duration_s = 1.0 / fps
    total_steps_per_episode = []
    rollout_outcomes = []     # (rollout_id, outcome) for every rollout attempted
    eval_start_time = timeit.default_timer()
    total_steps_completed = 0
    # Set when a reset released the gripper, so the next rollout re-runs the
    # insert-by-hand ritual instead of assuming the object is still held
    # (start_gripper=="closed" only). See stop_and_reset's return value.
    reinsert_needed = False

    logger.info(f"Control frequency: {fps} Hz | step budget: {step_duration_s * 1000:.1f} ms")

    # Live keyboard control during rollouts -- 's' stops the arm, abandons the
    # episode and resets (see KeyPoller / stop_and_reset); Enter ends the
    # rollout and records it as a success. One instance for the whole
    # evaluation; it only holds the terminal while a rollout is running.
    keys = KeyPoller()

    def make_step_description(rollout_id: int, steps: int) -> str:
        elapsed = timeit.default_timer() - eval_start_time
        actual_fps = steps / elapsed if elapsed > 0 else 0.0
        return f"Rollout {rollout_id + 1}/{num_rollouts} | FPS: {actual_fps:.1f}"

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=80),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    try:
        with Progress(*progress_columns, console=console) as progress:
            rollout_task = progress.add_task("Evaluation", total=num_rollouts)

            for rollout_id in range(num_rollouts):
                dataset_start_pose = get_dataset_start_pose(rollout_id)

                step_task = progress.add_task(
                    make_step_description(rollout_id, total_steps_completed), total=max_timesteps,
                )

                progress.stop()

                # Not the first rollout, and nothing has released the object
                # since (reinsert_needed) -- it is presumably still held from
                # the previous attempt, wherever the policy left the arm. Get
                # it out of the fixture before anything else moves, whether or
                # not the jaws are about to open (reinsert_every_rollout).
                still_holding = (start_gripper == "closed" and rollout_id > 0
                                 and not reinsert_needed)

                if still_holding:
                    logger.info(
                        f"Retracting {reset_lift_distance_m * 100:.0f} cm straight up "
                        "before resetting to the start pose..."
                    )
                    arm.zero_ft_sensor()
                    arm.activate_cartesian_controller()
                    lift_pose = arm.end_effector().copy()
                    lift_pose[2] += reset_lift_distance_m
                    hold_gripper_norm = gripper.get_normalized_position() if gripper is not None else GRIPPER_CLOSED_NORM
                    move_to_pose_via_cartesian_control(lift_pose, arm, gripper, hold_gripper_norm, safety_cfg, fps)

                if still_holding and not reinsert_every_rollout:
                    # Keep the object: skip the open/insert-by-hand/close
                    # ritual below and reset without touching the gripper
                    # (still closed, still holding it).
                    if dataset_start_pose is None:
                        move_to_home(arm, hydra_cfg, gripper=None)
                    # Object is still held from the previous rollout, so there
                    # is nothing to grasp -- go straight to the start pose
                    # (the lift above already disengaged it from the fixture).
                    ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps)
                    input(f"\n  Reset. Press Enter to start rollout {rollout_id + 1}/{num_rollouts}...")
                else:
                    if dataset_start_pose is None:
                        move_to_home(arm, hydra_cfg, gripper)
                    elif gripper is not None and start_gripper == "auto":
                        # Not homing, so nothing else issues move_to_home's
                        # gripper_home release -- do it here, before the arm
                        # moves, so it doesn't drag a still-held object along.
                        # ("open"/"closed" force OPEN just below anyway.)
                        gripper.set_normalized_position(float(np.clip(gripper_home, 0.0, 1.0)))
                        time.sleep(1.0)  # let the gripper finish releasing first

                    if still_holding and gripper is not None:
                        # reinsert_every_rollout: the jaws are about to open
                        # with the object still in them. The retract above
                        # already lifted it clear of the fixture, so it drops
                        # from there -- same as what reset_open_gripper does on
                        # an abandoned rollout, but say so, because here it is
                        # a rollout that ended normally.
                        logger.info(
                            "reinsert_every_rollout: releasing the object at the retracted "
                            "pose -- it will be re-inserted by hand at the start pose."
                        )

                    if gripper is not None and start_gripper != "auto":
                        # Both "open" and "closed" start the reset phase with
                        # the gripper OPEN (for "closed", so an object can be
                        # inserted by hand next).
                        gripper.set_normalized_position(GRIPPER_OPEN_NORM)

                    # Go straight to the demos' start pose, and do it BEFORE
                    # grasping: the object is then inserted with the gripper
                    # already where the episode begins, so both the grasp
                    # geometry and the policy's first observation match
                    # training. Also puts the arm in its true starting
                    # position before the tracking reset below, so DIFT
                    # localizes the scene as the policy will see it.
                    ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps)

                    at_pose = "start pose" if dataset_start_pose is not None else "home pose"
                    if start_gripper == "closed":
                        input(f"\n  At {at_pose} — insert the object into the gripper, then press Enter to close...")
                        if gripper is not None:
                            gripper.set_normalized_position(GRIPPER_CLOSED_NORM)
                            time.sleep(1.0)  # let the gripper finish closing before the arm moves
                        input(f"\n  Closed. Press Enter to start rollout {rollout_id + 1}/{num_rollouts}...")
                    else:
                        input(f"\n  Reset scene to start position, then press Enter to start rollout "
                              f"{rollout_id + 1}/{num_rollouts}...")
                    # The object (if any) has just been placed by hand, so the
                    # "still held from last time" assumption is valid again.
                    reinsert_needed = False
                progress.start()

                arm.zero_ft_sensor()
                arm.activate_cartesian_controller()

                # Ask tracking_node to re-run DIFT localization on the current
                # frames and (re)start publishing tracked points -- the object
                # may have moved between episodes, same reason
                # points_class.reset_episode() ran every rollout before. Lets
                # the user retry as many times as needed (e.g. after checking
                # tracking_node's ~visualization_image in rqt_image_view and
                # deciding the tracked point looks wrong) before committing to
                # running the policy.
                progress.stop()
                skip_rollout = False
                use_manual = False
                while True:
                    try:
                        resp = (reset_and_track_manual_srv if use_manual else reset_and_track_srv)()
                    except rospy.ServiceException as e:
                        resp = None
                        logger.warning(f"reset_and_track{'_manual' if use_manual else ''} call failed: {e}")

                    if resp is None or not resp.success:
                        logger.warning(
                            "tracking_node localization failed"
                            + (f" ({resp.message})" if resp is not None else "") + "."
                        )
                        choice = input(
                            "  Press Enter to retry localization, 'm' to pick the point(s) "
                            "manually, or 's' to skip this rollout: "
                        ).strip().lower()
                        if choice == "s":
                            skip_rollout = True
                            break
                        use_manual = choice == "m"
                        continue

                    env_seq_before_reset = env_state_sub.seq
                    choice = input(
                        "  Tracking (re)initialized -- check "
                        f"{ns}/visualization_image in rqt_image_view to verify the tracked "
                        "point looks right. Press Enter to proceed with this rollout, "
                        "'r' to reset tracking again, 'm' to pick the point(s) manually, "
                        "or 's' to skip this rollout: "
                    ).strip().lower()
                    if choice in ("r", "m"):
                        use_manual = choice == "m"
                        continue
                    if choice == "s":
                        skip_rollout = True
                    break
                progress.start()

                if skip_rollout:
                    logger.warning(f"Rollout {rollout_id + 1}: skipped by user.")
                    stop_track_srv()
                    progress.remove_task(step_task)
                    progress.update(rollout_task, advance=1)
                    continue

                policy.reset()
                if viz_recorder is not None:
                    viz_recorder.start()
                if ft_recorder is not None:
                    ft_recorder.start()

                # Block here (not inside the timed per-step loop) until
                # tracking_node has published at least one real estimate
                # since our reset_and_track call -- normally near-instant,
                # only slow on tracking_node's very first-ever tracking tick
                # (one-time cuDNN warm-up, see tracking_node.py's TrackingNode
                # docstring). Absorbing that here means it shows up as this
                # one log line instead of a string of "stale" warnings during
                # the timed rollout.
                progress.stop()
                logger.info("Waiting for tracking_node's first estimate (only slow the very first time)...")
                warmup_deadline = time.perf_counter() + 10.0
                while (env_state_sub.seq <= env_seq_before_reset
                       and time.perf_counter() < warmup_deadline
                       and not rospy.is_shutdown()):
                    time.sleep(0.05)
                if env_state_sub.seq <= env_seq_before_reset:
                    logger.warning(
                        "tracking_node produced no estimate within 10s; starting rollout anyway "
                        "(the staleness check will hold position until it catches up)."
                    )
                progress.start()

                last_stale_warn_time = 0.0
                # Reset against the gripper_home value move_to_home just
                # actually commanded, so t=0 doesn't spuriously trigger (or
                # fail to trigger) a slowdown against a stale prior rollout's
                # last command.
                prev_gripper_norm = float(np.clip(
                    hydra_cfg.dataset.get("gripper_home", 0.0) if hydra_cfg is not None else 0.0,
                    0.0, 1.0))
                slowdown_until = -1.0
                bendlab_baseline = bendlabs.get_angles().copy() if bendlabs is not None else None
                was_slowing = False
                prev_holding = None      # for logging grasp-state transitions fed to the policy
                last_grasp_log_time = 0.0  # throttles the periodic grasp heartbeat log
                last_stiffness_log_time = 0.0  # throttles the periodic pose-stiffness heartbeat log

                # Rolling history of the last rewind_steps executed EE poses, so
                # a bend-sensor trigger can back the arm off along the path it
                # just travelled. Plus the retry bookkeeping: rewind_attempts
                # accumulates across consecutive rewinds and only resets after
                # rewind_clear_steps clean (non-triggering) steps of real
                # forward progress -- so thrashing at one spot escalates to an
                # abort, but a genuinely-cleared contact later in the episode
                # does not.
                recent_poses = deque(maxlen=rewind_steps + 1)
                rewind_attempts = 0
                clean_streak = 0

                # Commanded (post-clip next_target) vs actual EE pose, one
                # entry per step -- see save_pose_error_visualization.
                pose_error_log = []

                # Flexible-sensor (BendLabs) angles, one entry per step --
                # see save_bend_data.
                bend_log = []

                # How this rollout ended -- a key of OUTCOME_SUFFIXES. Still
                # None after the loop means it ran to max_timesteps unlabelled,
                # and the operator is asked (see prompt_rollout_outcome).
                outcome = None
                # Whether it was cut short with the arm mid-motion ('f' or the
                # force limit), which is what calls for stop_and_reset --
                # separate from the label, since a rollout can be labelled
                # "fail" at the prompt after ending perfectly normally.
                abandoned = False

                # Live keys, armed for the timed loop only -- the reset prompts
                # below/above use input() and need canonical mode back.
                if keys.start() and rollout_id == 0:
                    logger.info("During a rollout: press 'f' to mark it FAILED, stop the arm, "
                                "abandon that episode and reset; press Enter to mark it "
                                "SUCCEEDED and end it there. A rollout that runs to the end "
                                "asks you afterwards.")

                for t in range(max_timesteps):
                    step_start = time.perf_counter()
                    if ft_recorder is not None:
                        # Where this step falls in the FT trace, so the two can
                        # be lined up afterwards (see WrenchRecorder.mark_step).
                        ft_recorder.mark_step(t)

                    # One poll per step: poll() drains everything buffered, so a
                    # second call this step would see nothing.
                    key = keys.poll()
                    if key in FAIL_KEYS:
                        logger.warning(
                            f"t={t:03d}  'f' pressed -- rollout marked FAIL; stopping the arm "
                            "and abandoning this episode."
                        )
                        outcome, abandoned = "fail", True
                        break
                    if key in SUCCESS_KEYS:
                        logger.info(
                            f"t={t:03d}  Enter pressed -- rollout marked SUCCESS, ending here."
                        )
                        outcome = "success"
                        break

                    # Force-magnitude safety limit. Checked before anything is
                    # commanded this step, and handled exactly like the fail
                    # key: freeze, abandon the episode, reset (see stop_and_reset).
                    ft_trip = ft_recorder.check_trip() if ft_recorder is not None else None
                    if ft_trip is not None:
                        logger.error(
                            f"t={t:03d}  FT LIMIT: |F|={ft_trip.force:.1f} N > "
                            f"{ft_force_limit_n:.1f} N -- stopping the arm and abandoning "
                            "this episode."
                        )
                        outcome, abandoned = "ft_limit", True
                        break

                    if bendlabs is not None:
                        # One read per step, used for both the recording and the
                        # rewind check below (get_angles averages its window, so
                        # reading it twice would just redo that work).
                        bend_angles = bendlabs.get_angles()
                        if record_bend:
                            bend_log.append((rospy.get_time(), t, *bend_angles))

                    if bendlabs is not None and bendlab_rewind_enabled:
                        bendlab_deviations = np.abs(bend_angles - bendlab_baseline)
                        exceeded = {
                            i: float(d) for i, d in enumerate(bendlab_deviations)
                            if d > bendlab_rewind_threshold
                        }
                        if exceeded:
                            rewind_attempts += 1
                            clean_streak = 0
                            if rewind_attempts > rewind_max_attempts:
                                logger.error(
                                    f"t={t:03d}  BendLabs threshold exceeded {exceeded} "
                                    f"after {rewind_max_attempts} rewind attempts -- holding "
                                    "position, aborting rollout."
                                )
                                break
                            logger.warning(
                                f"t={t:03d}  BendLabs threshold exceeded (> {bendlab_rewind_threshold}) "
                                f"{exceeded} -- rewinding {len(recent_poses)} steps and replanning "
                                f"(attempt {rewind_attempts}/{rewind_max_attempts})."
                            )
                            rewind_through_poses(
                                list(recent_poses), arm, gripper, safety_cfg,
                                step_duration_s, rewind_slow_factor,
                            )
                            recent_poses.clear()
                            # Force select_action to re-infer a fresh chunk from
                            # the backed-off pose instead of resuming the queued
                            # (unsafe) one.
                            policy.reset()
                            total_steps_completed += 1
                            progress.update(
                                step_task, advance=1,
                                description=make_step_description(rollout_id, total_steps_completed),
                            )
                            continue
                        else:
                            clean_streak += 1
                            if clean_streak >= rewind_clear_steps:
                                rewind_attempts = 0

                    env_state, env_age_s = env_state_sub.get()
                    if env_state is not None and len(env_keep_flat) != len(env_state):
                        env_state = np.asarray(env_state)[env_keep_flat]
                    if env_state is not None and env_state_offset is not None:
                        # After keypoint selection, so the offset lines up with the
                        # vector the policy consumes (and with the training stats a
                        # user would have compared against).
                        env_state = np.asarray(env_state, dtype=float) + env_state_offset
                    state = get_current_state(arm, gripper, grasp_cfg, offset=eval_offset, yaw=eval_yaw_offset)

                    if stiffness_modulator is not None:
                        # Runs every step regardless of tracking staleness --
                        # wrist-deflection-driven stiffness is independent of
                        # the object-tracking pipeline.
                        stiffness_modulator.update()

                    if step_start - last_stiffness_log_time > 1.0:
                        # Unconditional: logs the static startup stiffness
                        # even when enable_pose_stiffness is off, or before
                        # PoseStiffnessModulator has received its first pose.
                        current_stiffness = getattr(arm, "current_stiffness", None)
                        if current_stiffness is not None:
                            extra = stiffness_modulator.status_str() if stiffness_modulator is not None else ""
                            logger.info(
                                f"t={t:03d}  stiffness: trans={np.round(current_stiffness[:3], 0)} N/m  "
                                f"rot={np.round(current_stiffness[3:], 0)} Nm/rad{extra}"
                            )
                        last_stiffness_log_time = step_start

                    if grasp_binary_enabled and gripper is not None:
                        gr = read_grasp(gripper, grasp_cfg)
                        nums = f"cmd={gr.cmd_norm:.2f} meas={gr.act_norm:.2f} gap={gr.gap:+.2f} (thr {grasp_cfg.gap_threshold:.2f})"
                        if prev_holding is not None and gr.holding != prev_holding:
                            # State changed: log immediately regardless of throttle.
                            logger.info(
                                f"t={t:03d}  Gripper grasp signal -> "
                                f"{'HOLDING (closed, +1)' if gr.holding else 'EMPTY (open, -1)'}  {nums}"
                            )
                            last_grasp_log_time = step_start
                        elif step_start - last_grasp_log_time > 1.0:
                            # Otherwise a ~1Hz heartbeat so the current state is always visible.
                            logger.info(
                                f"t={t:03d}  Gripper: {'HOLDING' if gr.holding else 'empty '}  {nums}"
                            )
                            last_grasp_log_time = step_start
                        prev_holding = gr.holding

                    if env_age_s > args.max_tracking_staleness_s:
                        if step_start - last_stale_warn_time > 1.0:
                            logger.warning(
                                f"t={t:03d}  tracked point is {env_age_s:.2f}s stale "
                                f"(> {args.max_tracking_staleness_s}s) -- holding position, "
                                "not commanding new targets until tracking recovers."
                            )
                            last_stale_warn_time = step_start
                    else:
                        obs = {
                            "observation.state": torch.tensor(state, dtype=torch.float32),
                            "observation.environment_state": torch.tensor(env_state, dtype=torch.float32),
                        }
                        policy_obs = preprocessor(obs)

                        with torch.inference_mode():
                            action = policy.select_action(policy_obs)
                        action = postprocessor(action).squeeze().cpu().numpy()  # (10,)

                        target_pose = np.concatenate(
                            [action[:3], transformations.quaternion_from_ortho6(action[3:9])]
                        )
                        # Offset/yaw along/about the target's own local axes,
                        # not world -- so it tracks wherever the gripper is
                        # pointing.
                        target_pose = pose_offset.apply_offset(target_pose, eval_offset, eval_yaw_offset)
                        gripper_norm = gripper_state_to_normalized(action[9])

                        if gripper is not None and abs(gripper_norm - prev_gripper_norm) > gripper_slowdown_change_threshold:
                            if gripper_slowdown_enabled:
                                slowdown_until = step_start + gripper_slowdown_duration_s
                        prev_gripper_norm = gripper_norm

                        next_target = execute_cartesian_target(target_pose, arm, gripper, gripper_norm, safety_cfg)
                        # Record the pose actually reached this step so a later
                        # bend trigger can retreat back along this path.
                        actual_pose = arm.end_effector().copy()
                        recent_poses.append(actual_pose)
                        if record_pose_error_plot:
                            pose_error_log.append({"t": t, "commanded": next_target, "actual": actual_pose})

                    dt_s = time.perf_counter() - step_start
                    target_step_duration = step_duration_s
                    slowdown_reasons = []
                    if step_start < slowdown_until:
                        target_step_duration = max(target_step_duration, step_duration_s * gripper_slowdown_factor)
                        slowdown_reasons.append("gripper")

                    is_slowing = target_step_duration > step_duration_s
                    if is_slowing and not was_slowing:
                        logger.info(
                            f"t={t:03d}  Slowing down execution ({'+'.join(slowdown_reasons)}, "
                            f"{target_step_duration / step_duration_s:.1f}x)"
                        )
                    elif was_slowing and not is_slowing:
                        logger.info(f"t={t:03d}  Resuming normal speed")
                    was_slowing = is_slowing

                    sleep_s = target_step_duration - dt_s
                    if sleep_s < 0:
                        logger.warning(
                            f"Step {t} overran budget: {dt_s * 1000:.1f} ms > {target_step_duration * 1000:.1f} ms "
                            f"({1 / dt_s:.1f} Hz actual vs {1.0 / target_step_duration:.1f} Hz target)."
                        )
                    precise_sleep(max(sleep_s, 0.0))

                    total_steps_completed += 1
                    progress.update(
                        step_task, advance=1,
                        description=make_step_description(rollout_id, total_steps_completed),
                    )

                keys.stop()   # canonical mode back before the next input() prompt
                stop_track_srv()

                # Ran to max_timesteps without anyone saying how it went -- ask,
                # before the files are named after the answer.
                if outcome is None:
                    progress.stop()
                    outcome = prompt_rollout_outcome(rollout_id, t + 1)
                    progress.start()

                # One stem for every artifact of this rollout, labelled with how
                # it ended (see OUTCOME_SUFFIXES) so the videos and FT traces
                # can be sorted by result afterwards.
                rollout_stem = f"rollout{rollout_id:02d}{OUTCOME_SUFFIXES.get(outcome, '')}"

                if viz_recorder is not None:
                    viz_recorder.stop_and_save(output_dir / "videos" / f"{rollout_stem}.mp4")

                if record_pose_error_plot:
                    save_pose_error_visualization(
                        pose_error_log, output_dir / "pose_error" / f"{rollout_stem}.png",
                        rollout_id)

                if ft_recorder is not None:
                    ft_data = ft_recorder.stop_and_save(
                        output_dir / "ft" / f"{rollout_stem}.npz" if record_ft_data else None
                    )
                    if ft_data is not None and record_ft_plot:
                        save_ft_visualization(ft_data, output_dir / "ft" / f"{rollout_stem}.png",
                                              rollout_id)

                if record_bend and bendlabs is not None:
                    bend_data = save_bend_data(
                        bend_log, bendlab_baseline,
                        bendlab_rewind_threshold if bendlab_rewind_enabled else None,
                        output_dir / "bend" / f"{rollout_stem}.npz" if record_bend_data else None,
                        rollout_id,
                    )
                    if bend_data is not None and record_bend_plot:
                        save_bend_visualization(
                            bend_data, output_dir / "bend" / f"{rollout_stem}.png", rollout_id)

                # Cut short with the arm mid-motion ('f' or the force limit):
                # stop and reset. A rollout that ended on its own -- whichever
                # way it is labelled -- takes the normal completion path below,
                # where the next rollout's reset block does the honours.
                if abandoned:
                    reason = ("the FT force limit" if outcome == "ft_limit" else "the user")
                    progress.stop()
                    released = stop_and_reset(arm, gripper, safety_cfg, fps,
                                              reset_lift_distance_m, dataset_start_pose,
                                              hydra_cfg, open_gripper=reset_open_gripper)
                    reinsert_needed = reinsert_needed or released
                    progress.start()
                    rollout_outcomes.append((rollout_id, outcome))
                    logger.warning(
                        f"Rollout {rollout_id + 1}: abandoned by {reason} at t={t}, reset "
                        "complete -- excluded from the step statistics, but it still counts "
                        "against --num_rollouts."
                    )
                    progress.remove_task(step_task)
                    progress.update(rollout_task, advance=1)
                    continue

                arm.activate_joint_trajectory_controller()
                steps_taken = t + 1
                total_steps_per_episode.append(steps_taken)
                rollout_outcomes.append((rollout_id, outcome))
                logger.info(
                    f"Rollout {rollout_id} complete | steps: {steps_taken} | "
                    f"{(outcome or 'unlabelled').upper()}"
                )

                progress.remove_task(step_task)
                progress.update(rollout_task, advance=1)
    finally:
        keys.stop()   # idempotent; the terminal must not stay in cbreak on ^C/crash
        if viz_recorder is not None:
            viz_recorder.stop()   # discard any in-flight recording, don't try to save it
        if ft_recorder is not None:
            ft_recorder.stop()
        try:
            stop_track_srv()
        except Exception:
            pass  # best-effort -- don't mask an earlier exception with a cleanup failure

    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(f"Rollouts with steps recorded: {len(total_steps_per_episode)} "
                f"({num_rollouts - len(total_steps_per_episode)} abandoned)")

    # Outcome tally. Successes and failures are the operator's verdict (live
    # Enter/'f' or the end-of-rollout prompt); ft_limit rollouts were stopped by
    # the safety limit, which is a failure of a different kind and counted
    # separately rather than folded in.
    counts = Counter(outcome for _, outcome in rollout_outcomes)
    labelled = counts["success"] + counts["fail"]
    logger.info(
        "Outcomes: "
        + ", ".join(f"{counts[k]} {label}" for k, label in
                    (("success", "success"), ("fail", "fail"), ("ft_limit", "FT limit")))
        + (f", {counts[None]} unlabelled" if counts[None] else "")
    )
    if labelled:
        logger.info(f"Success rate (of {labelled} judged): "
                    f"{100.0 * counts['success'] / labelled:.0f}%")
    successes = [i for i, outcome in rollout_outcomes if outcome == "success"]
    if successes:
        logger.info(f"Succeeded: rollouts {', '.join(str(i) for i in successes)}")
    if total_steps_per_episode:
        logger.info(f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
        logger.info(f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    else:
        logger.info("Mean/std steps per episode: n/a -- every rollout was abandoned")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
