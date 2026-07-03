"""Build MoveIt 2 config dict for ur_gripper_gz_moveit_config (matches ur_moveit.launch.py)."""

import os

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

PKG = "ur_gripper_gz_moveit_config"


def build_moveit_config(ur_type="ur5e", gripper="robotiq_2f85", load_gripper=True):
    share = get_package_share_directory(PKG)
    srdf_path = os.path.join(share, "srdf", "ur_gripper_gz.srdf.xacro")
    controllers = os.path.join(share, "config", "moveit_controllers_%s.yaml" % gripper)
    pipelines = ["ompl"]

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name=PKG)
        .robot_description_semantic(
            file_path=srdf_path,
            mappings={
                "name": ur_type,
                "gripper": gripper,
                "load_gripper": str(load_gripper).lower(),
            },
        )
        .robot_description_kinematics(file_path=os.path.join(share, "config", "kinematics.yaml"))
        .joint_limits(file_path=os.path.join(share, "config", "joint_limits.yaml"))
        .trajectory_execution(file_path=controllers)
        .planning_pipelines(pipelines=pipelines, default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    # MoveItPy's MoveItCpp reads pipeline names from "planning_pipelines.pipeline_names",
    # but MoveItConfigsBuilder.planning_pipelines() only sets the flat "planning_pipelines"
    # list consumed by move_group. Provide the nested key MoveItCpp actually looks up, plus
    # default plan_request_params so PlanningComponent.plan() has a pipeline/planner to use
    # when called without explicit parameters.
    moveit_config.moveit_cpp = {
        "planning_pipelines": {"pipeline_names": pipelines, "namespace": ""},
        "plan_request_params": {
            "planning_pipeline": "ompl",
            "planner_id": "RRTConnectkConfigDefault",
            "planning_attempts": 10,
            "planning_time": 5.0,
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        },
    }
    return moveit_config
