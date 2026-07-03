"""Shared ROS 2 node + executor bootstrap for osx_ur5e (matches ur_control pattern)."""

import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class RosRuntime:
    """Owns a shared node spun by a background MultiThreadedExecutor.

    Library classes (BaseEnv, FDCCEnv, ImageRecorder) receive ``runtime.node``.
    Application scripts create one RosRuntime for the process lifetime.
    """

    def __init__(self, node_name="osx_ur5e", namespace=""):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name, namespace=namespace)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def shutdown(self):
        if not rclpy.ok():
            return
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        rclpy.shutdown()
