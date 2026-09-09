"""Live Gello->robot teleoperation loop, decoupled from recording.

Runs free at the Gello device's native rate (100 Hz) in its own thread. It
does no dataset writing: the commanded target is captured by recording the
stamped ``target_frame`` topic that ``set_cartesian_target_pose`` publishes,
and this loop additionally publishes the action-side sources that only exist
in Python:

    /data_collection/gello_joints       sensor_msgs/JointState (calibrated)
    /data_collection/stiffness_command  osx_msgs/Float64ArrayStamped
        (on change + 1 Hz keepalive, so every episode's first tick has a
        causal stiffness sample)
"""

import threading

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from osx_msgs.msg import Float64ArrayStamped
from ur_control import transformations


class GelloTeleop:
    """Free-running teleop: Gello joints -> clipped Cartesian target."""

    def __init__(self, arm, gello, safety_cfg, rate_hz: float = 100.0):
        self.arm = arm
        self.gello = gello
        self.safety_cfg = safety_cfg
        self.rate_hz = rate_hz

        self._thread = None
        self._stop = threading.Event()
        self._enabled = threading.Event()

        self._gello_pub = rospy.Publisher(
            "/data_collection/gello_joints", JointState, queue_size=1)
        self._stiffness_pub = rospy.Publisher(
            "/data_collection/stiffness_command", Float64ArrayStamped, queue_size=1, latch=True)
        self._last_stiffness = None
        self._last_stiffness_pub_t = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the loop thread (paused; call resume() to drive the robot)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gello_teleop", daemon=True)
        self._thread.start()

    def resume(self):
        self._enabled.set()

    def pause(self):
        """Stop sending targets (robot holds its last commanded pose)."""
        self._enabled.clear()

    def shutdown(self):
        self._stop.set()
        self._enabled.set()  # unblock the wait
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def active(self) -> bool:
        return self._enabled.is_set() and self._thread is not None

    # ------------------------------------------------------------------
    # Stiffness
    # ------------------------------------------------------------------

    def set_stiffness(self, values):
        """Update controller stiffness and publish the stamped command."""
        values = np.asarray(values, dtype=np.float64)
        self.arm.update_stiffness(values)
        self.arm.current_stiffness = values.copy()
        self._publish_stiffness(values)

    def _publish_stiffness(self, values):
        msg = Float64ArrayStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = list(np.asarray(values, dtype=np.float64))
        self._stiffness_pub.publish(msg)
        self._last_stiffness = np.asarray(values, dtype=np.float64)
        self._last_stiffness_pub_t = rospy.get_time()

    def _keepalive_stiffness(self):
        """Republish at 1 Hz so bags always contain a causal sample."""
        now = rospy.get_time()
        if now - self._last_stiffness_pub_t < 1.0:
            return
        values = (
            self._last_stiffness
            if self._last_stiffness is not None
            else np.asarray(self.arm.current_stiffness, dtype=np.float64)
        )
        self._publish_stiffness(values)

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------

    def step(self):
        """One teleop step: Gello joints -> clipped Cartesian target."""
        safety_cfg = self.safety_cfg
        # UR convention: the two URDFs disagree on lift, elbow and wrist_3
        gello_joints = self.gello.joint_angles_ur()
        current_pose = self.arm.end_effector()
        target_pose = self.arm.end_effector(joint_angles=gello_joints)

        delta_translation = target_pose[:3] - current_pose[:3]
        delta_rotation = transformations.quaternions_orientation_error(
            target_pose[3:], current_pose[3:])

        max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
        clipped_delta_translation = np.clip(
            delta_translation, -safety_cfg.max_delta_translation, safety_cfg.max_delta_translation)
        clipped_delta_orientation = np.clip(
            delta_rotation, -max_delta_rotation, max_delta_rotation)

        next_position = current_pose[:3] + clipped_delta_translation
        next_position[0] = np.clip(next_position[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
        next_position[1] = np.clip(next_position[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
        next_position[2] = np.clip(next_position[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
        next_orientation = transformations.rotate_quaternion_by_rpy(
            *clipped_delta_orientation, current_pose[3:])

        # Publishes the stamped target_frame that the bag records as the action.
        self.arm.set_cartesian_target_pose(
            pose=np.concatenate([next_position, next_orientation]))

        gello_msg = JointState()
        gello_msg.header.stamp = rospy.Time.now()
        gello_msg.position = list(gello_joints)
        self._gello_pub.publish(gello_msg)

    def _run(self):
        rate = rospy.Rate(self.rate_hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            if self._enabled.is_set():
                try:
                    self.step()
                except Exception as e:
                    rospy.logerr_throttle(1.0, "GelloTeleop step failed: %s", e)
                self._keepalive_stiffness()
                rate.sleep()
            else:
                self._enabled.wait(timeout=0.1)
