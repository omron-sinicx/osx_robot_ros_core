#!/usr/bin/env python
import argparse
from copy import deepcopy
from math import pi
from osx_robot_control.helpers import get_trajectory_joint_goal
import rospy
import sys
import signal

from geometry_msgs.msg import TwistStamped, Twist, Vector3Stamped, Vector3, PoseStamped
from std_msgs.msg import Header

from moveit.task_constructor import core, stages
from moveit_commander import PlanningSceneInterface
from py_binding_tools import roscpp_init

from ur_control import conversions, transformations


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    roscpp_init('osx_ur')

    # Task parameters
    # Specify object parameters
    planner = "RRTConnect"
    object_name = "grasp_object"

    # Grasp object properties
    object_pose = PoseStamped()
    object_pose.header.frame_id = "world"
    object_pose.pose.orientation.x = 1.0
    object_pose.pose.position.x = 0.2
    object_pose.pose.position.y = -0.1
    object_pose.pose.position.z = 0.8

    # Define the pose that the object should have after placing
    place_pose = deepcopy(object_pose)
    place_pose.pose.position.y += 0.2  # shift object by 20cm along y axis

    place_pose2 = deepcopy(object_pose)
    place_pose2.pose.position.y += 0.1  # shift object by 10cm along y axis

    # Specify robot parameters
    robot_name = "a_bot"

    # Setting environment
    # Start with a clear planning scene
    psi = PlanningSceneInterface(synchronous=True)
    psi.remove_world_object()

    # Add the grasp object to the planning scene
    psi.add_box(object_name, object_pose, size=[0.1, 0.05, 0.03])

    # Create a planner instance that is used to connect
    # the current state to the grasp approach pose
    # TODO (cambel): recycle pipeline/planners
    pipeline = core.PipelinePlanner()
    pipeline.planner = planner
    planners = [(robot_name, pipeline), ("b_bot", pipeline)]

    ############################################
    ######## Specific task definition ##########
    ############################################

    # 1. pick obj1 b_bot
    # 2. place obj1 b_bot p1
    # 3. pick obj1 b_bot
    # 4. place obj1 b_bot p2

    task = create_task("pick_and_place_and_pick")

    task.add(stages.CurrentState("current"))
    # Connect the two stages (current and pick)
    task.add(create_connect_state("connect1", planners))

    # Add the pick stage to the task's stage hierarchy
    grasp_generator = construct_grasp_generator(stage_name="grasp1", object_name=object_name, robot_name=robot_name, monitored_stage=task["current"])
    task.add(create_pick_stage(grasp_generator, robot_name, object_name))

    # Connect the Pick stage with the following Place stage
    task.add(create_connect_state("connect2", planners))

    # Add the place pipeline to the task's hierarchy
    place_generator = construct_place_generator(place_pose, object_name, monitored_stage=task["Pick"])
    task.add(create_place_stage(place_generator, robot_name, object_name))

    task.add(create_connect_state("connect3", planners))

    grasp_generator = construct_grasp_generator(stage_name="grasp1", object_name=object_name, robot_name=robot_name, monitored_stage=task["Place"])
    task.add(create_pick_stage(grasp_generator, robot_name, object_name, stage_name="Pick2"))

    task.add(create_connect_state("connect4", planners))

    # Add the place pipeline to the task's hierarchy
    place_generator2 = construct_place_generator(place_pose2, object_name, monitored_stage=task["Pick2"])
    task.add(create_place_stage(place_generator2, robot_name, object_name))

    ######################################################
    ######## Solve and execute if plan is found ##########
    ######################################################

    if task.plan(max_solutions=2):
        # print("plan succeeded", task.solutions[0].toMsg())
        print("plan succeeded", len(task.solutions))
        print("sol", get_trajectory_joint_goal(task.solutions[0].toMsg().sub_trajectory[-1].trajectory))
        task.publish(task.solutions[0])

        input("continue to execute")
        task.execute(task.solutions[0])
    else:
        print("plan failed?")
    rospy.sleep(100)


def create_task(name):
    # Create a task
    task = core.Task()
    task.name = name
    task.enableIntrospection()
    return task


def construct_grasp_generator(stage_name, robot_name, object_name, monitored_stage):
    # TODO (cambel): add necessary parameters (grasp name, desired orientation or something)
    grasp_pose = PoseStamped()
    grasp_pose.header.frame_id = object_name
    # grasp_generator = stages.GeneratePose("Custom Pose")
    grasp_generator = stages.LoadedGraspPoses(stage_name)
    grasp_generator.setMonitoredStage(monitored_stage)  # task["current"]
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"

    # Default orientation
    grasp_generator.addPose(grasp_pose)
    # Consider this orientation as well...
    grasp_pose.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, 0, pi/2))
    grasp_generator.addPose(grasp_pose)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    grasp_stage = stages.SimpleGrasp(grasp_generator, f"grasp_{stage_name}")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = robot_name + '_gripper_tip_link'
    ik_frame.pose.position.x = -0.01
    ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
    grasp_stage.setIKFrame(ik_frame)

    return grasp_stage


def create_connect_state(name, planners, max_distance=0.001):
    connect = stages.Connect(name, planners)
    connect.max_distance = max_distance
    return connect


def create_pick_stage(grasp_stage, robot_name, object_name, stage_name="Pick"):
    # Pick container comprises approaching, grasping (using SimpleGrasp stage), and lifting of object
    pick = stages.Pick(grasp_stage, stage_name)
    pick.eef = robot_name + '_tip'
    pick.object = object_name

    # Twist to approach the object
    # TODO (cambel): can we parametrized this? do we need to?
    approach = TwistStamped()
    approach.header.frame_id = "world"
    approach.twist.linear.z = -1.0
    pick.setApproachMotion(approach, 0.03, 0.1)

    # Twist to lift the object
    # TODO (cambel): can we parametrized this? do we need to?
    lift = TwistStamped()
    lift.header.frame_id = "world"
    lift.twist.linear.z = 1.0
    pick.setLiftMotion(lift, 0.03, 0.1)
    return pick


def construct_place_generator(place_pose, object_name, monitored_stage):
    # TODO: get the target pose from somewhere
    # Generate Cartesian place poses for the object
    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(monitored_stage)  # task["Pick"]
    place_generator.object = object_name
    place_generator.pose = place_pose
    place_generator.properties["pregrasp"] = "open"

    return place_generator


def create_place_stage(place_generator, robot_name, object_name, stage_name="Place"):

    # The SimpleUnGrasp container encapsulates releasing the object at the given Cartesian pose
    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    # Place container comprises placing, ungrasping, and retracting
    place = stages.Place(simpleUnGrasp, stage_name)
    place.eef = robot_name + '_tip'
    place.object = object_name
    place.eef_frame = robot_name + '_gripper_tip_link'

    # Twist to retract from the object
    # TODO (cambel): can we parametrized this? do we need to?
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    # TODO (cambel): can we parametrized this? do we need to?
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = "world"
    placeMotion.twist.linear.z = -1.0  # value does not matter, just the approach direction (sign)
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    return place


if __name__ == "__main__":
    main()
