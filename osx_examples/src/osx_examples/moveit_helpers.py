"""Shared helpers for MoveIt 2 example scripts."""

import time

import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit.core.robot_state import RobotState
from rclpy.duration import Duration
from rclpy.time import Time

from ur_control import conversions, transformations

ARM_GROUP = "ur_manipulator"
EE_LINK = "tool0"
BASE_LINK = "base_link"

UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def confirm_to_proceed(message: str) -> None:
    input("%s — press Enter to continue..." % message)


def plan_and_execute(moveit, planning_component, logger, sleep_time=1.0):
    planning_component.set_start_state_to_current_state()
    logger.info("Planning trajectory")
    plan_result = planning_component.plan()
    if not plan_result:
        logger.error("Planning failed")
        return False

    logger.info("Executing plan")
    moveit.execute(plan_result.trajectory, controllers=[])
    time.sleep(sleep_time)
    return True


def set_joint_position_goal(planning_component, moveit, joint_values, logger):
    robot_model = moveit.get_robot_model()
    robot_state = RobotState(robot_model)
    robot_state.joint_positions = {
        name: float(value) for name, value in zip(UR_JOINT_NAMES, joint_values)
    }
    constraint = construct_joint_constraint(
        robot_state=robot_state,
        joint_model_group=robot_model.get_joint_model_group(ARM_GROUP),
    )
    planning_component.set_start_state_to_current_state()
    planning_component.set_goal_state(motion_plan_constraints=[constraint])
    return plan_and_execute(moveit, planning_component, logger)


def set_pose_goal(planning_component, moveit, pose_stamped: PoseStamped, logger):
    planning_component.set_start_state_to_current_state()
    planning_component.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=EE_LINK)
    return plan_and_execute(moveit, planning_component, logger)


def set_named_pose_goal(planning_component, moveit, name: str, logger):
    planning_component.set_start_state_to_current_state()
    planning_component.set_goal_state(configuration_name=name)
    return plan_and_execute(moveit, planning_component, logger)


def lookup_tool0_pose(tf_buffer) -> np.ndarray:
    transform = tf_buffer.lookup_transform(
        BASE_LINK, EE_LINK, Time(), timeout=Duration(seconds=2.0)
    )
    matrix = conversions.from_transform(transform.transform)
    quat = transformations.quaternion_from_matrix(matrix)
    return np.concatenate([matrix[:3, 3], quat])


def set_relative_motion_goal(
    planning_component,
    moveit,
    tf_buffer,
    relative_translation,
    logger,
    relative_to_tcp=False,
):
    current_pose = lookup_tool0_pose(tf_buffer)

    if relative_to_tcp:
        delta = np.array(list(relative_translation) + [0.0, 0.0, 0.0])
        target_pose = transformations.transform_pose(current_pose, delta, rotated_frame=True)
    else:
        target_pose = current_pose.copy()
        target_pose[:3] += np.asarray(relative_translation, dtype=float)

    pose_stamped = conversions.to_pose_stamped(BASE_LINK, target_pose)
    return set_pose_goal(planning_component, moveit, pose_stamped, logger)
