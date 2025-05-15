#!/usr/bin/env python
import argparse
from math import pi
from osx_robot_control.helpers import get_trajectory_joint_goal, rotateQuaternionByRPY, rotateQuaternionByRPYInUnrotatedFrame
import rospy
import sys
import signal

from geometry_msgs.msg import TwistStamped, Twist, Vector3Stamped, Vector3, PoseStamped, Quaternion, Pose, Point
from std_msgs.msg import Header
from tf.listener import transformations

from moveit.task_constructor import core, stages
from moveit_commander import PlanningSceneInterface
from py_binding_tools import roscpp_init

from ur_control import conversions
from ur_control.math_utils import quaternion_normalize


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cartesian', action='store_true', help='Simple cartesian sequence')
    parser.add_argument('--pickplace', action='store_true', help='Simple pick and place sequence')
    parser.add_argument('--pick_then_place', action='store_true', help='Simple pick and place sequence')
    parser.add_argument('--place', action='store_true', help='Simple pick and place sequence')
    parser.add_argument('--move_to', action='store_true', help='Simple move to sequence')
    parser.add_argument('--compute_ik', action='store_true', help='Simple compute IK sequence')
    parser.parse_args()

    args = parser.parse_args(rospy.myargv()[1:])

    roscpp_init('osx_ur')

    # a_bot = RobotBase("a_bot", None)

    if args.cartesian:
        task = create_cartesian_task()
    elif args.pickplace:
        task = create_pick_and_place_task()
    elif args.place:
        task2 = create_place_task()
        if task2.plan(1):
            task2.publish(task2.solutions[0])
            task2.execute(task2.solutions[0])
        else:
            print("FAILED PLACE")
        return
    elif args.pick_then_place:
        task = create_pick_task()
        if task.plan(1):
            task.publish(task.solutions[0])
            task.execute(task.solutions[0])

            task2 = create_place_task()
            if task2.plan(1):
                task2.publish(task2.solutions[0])
                task2.execute(task2.solutions[0])
            else:
                print("FAILED PLACE")
        else:
            print("FAILED PICK")
        rospy.sleep(100)
        return
    elif args.move_to:
        task = create_move_to_task()
    elif args.compute_ik:
        task = create_compute_ik_task()
    else:
        raise ValueError("Choose one of the available options, see -h")

    print("constructed task, planning...")

    if task.plan(max_solutions=2):
        input("continue to publish")
        # print("plan succeeded", task.solutions[0].toMsg())
        print("plan succeeded", len(task.solutions))
        # print("sol", get_trajectory_joint_goal(task.solutions[0].toMsg().sub_trajectory[-1].trajectory))
        task.publish(task.solutions[0])

        input("continue to execute")
        task.execute(task.solutions[0])
    else:
        print("plan failed?")
    rospy.sleep(100)


def create_pick_task():
    # Specify robot parameters
    arm = "a_bot"
    eef = "a_bot_tip"

    # Specify object parameters
    object_name = "grasp_object"

    # Start with a clear planning scene
    psi = PlanningSceneInterface(synchronous=True)
    psi.remove_world_object()

    # Grasp object properties
    objectPose = PoseStamped()
    objectPose.header.frame_id = "world"
    objectPose.pose.orientation.x = 1.0
    objectPose.pose.position.x = 0.2
    objectPose.pose.position.y = -0.1
    objectPose.pose.position.z = 0.8

    # Add the grasp object to the planning scene
    psi.add_box(object_name, objectPose, size=[0.1, 0.05, 0.03])

    # Create a task
    task = core.Task()
    task.name = "PickPipelineExample"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create a planner instance that is used to connect
    # the current state to the grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))

    grasp_pose = PoseStamped()
    grasp_pose.header.frame_id = "grasp_object"
    # grasp_generator = stages.GeneratePose("Custom Pose")
    grasp_generator = stages.LoadedGraspPoses("Custom Pose")
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(grasp_pose)
    grasp_pose.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, 0, pi/2))
    grasp_generator.addPose(grasp_pose)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = "a_bot_gripper_tip_link"
    ik_frame.pose.position.x = -0.01
    ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
    simpleGrasp.setIKFrame(ik_frame)

    # Pick container comprises approaching, grasping (using SimpleGrasp stage), and lifting of object
    pick = stages.Pick(simpleGrasp, "Pick")
    pick.eef = eef
    pick.object = object_name

    # Twist to approach the object
    approach = TwistStamped()
    approach.header.frame_id = "world"
    approach.twist.linear.z = -1.0
    pick.setApproachMotion(approach, 0.03, 0.1)

    # Twist to lift the object
    lift = TwistStamped()
    lift.header.frame_id = "a_bot_gripper_tip_link"
    lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.03, 0.1)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    # avoid ClassLoader warning
    del pipeline
    del planners

    return task


def create_place_task():
    # Specify robot parameters
    arm = "a_bot"
    eef = "a_bot_tip"

    # Specify object parameters
    object_name = "grasp_object"

    # Grasp object properties
    objectPose = PoseStamped()
    objectPose.header.frame_id = "world"
    objectPose.pose.orientation.x = 1.0
    objectPose.pose.position.x = 0.2
    objectPose.pose.position.y = -0.1
    objectPose.pose.position.z = 0.8

    # Create a task
    task = core.Task()
    task.name = "Place pipeline"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create a planner instance that is used to connect
    # the current state to the grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))
    # # Connect the Pick stage with the following Place stage
    # task.add(stages.Connect("connect2", planners))

    # Define the pose that the object should have after placing
    placePose = objectPose
    placePose.pose.position.y += 0.2  # shift object by 20cm along y axis

    # Generate Cartesian place poses for the object
    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(task["current"])
    place_generator.object = object_name
    place_generator.pose = placePose
    place_generator.properties["pregrasp"] = "open"

    # The SimpleUnGrasp container encapsulates releasing the object at the given Cartesian pose
    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    # Place container comprises placing, ungrasping, and retracting
    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = object_name
    place.eef_frame = "a_bot_gripper_tip_link"

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = "a_bot_gripper_tip_link"
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    # Add the place pipeline to the task's hierarchy
    task.add(place)

    # avoid ClassLoader warning
    del pipeline
    del planners

    return task


def create_pick_and_place_task():
    # Specify robot parameters
    arm = "a_bot"
    eef = "a_bot_tip"

    # Specify object parameters
    object_name = "grasp_object"

    # Start with a clear planning scene
    psi = PlanningSceneInterface(synchronous=True)
    psi.remove_world_object()

    # Grasp object properties
    objectPose = PoseStamped()
    objectPose.header.frame_id = "world"
    objectPose.pose.orientation.x = 1.0
    objectPose.pose.position.x = 0.2
    objectPose.pose.position.y = -0.1
    objectPose.pose.position.z = 0.8

    # Add the grasp object to the planning scene
    psi.add_box(object_name, objectPose, size=[0.1, 0.05, 0.03])

    # Create a task
    task = core.Task()
    task.name = "PickPipelineExample"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create a planner instance that is used to connect
    # the current state to the grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    connect1 = stages.Connect("connect1", planners)
    connect1.max_distance = 0.001
    task.add(connect1)

    grasp_pose = PoseStamped()
    grasp_pose.header.frame_id = "grasp_object"
    # grasp_generator = stages.GeneratePose("Custom Pose")
    grasp_generator = stages.LoadedGraspPoses("Custom Pose")
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(grasp_pose)
    grasp_pose.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, 0, pi/2))
    grasp_generator.addPose(grasp_pose)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = "a_bot_gripper_tip_link"
    ik_frame.pose.position.x = -0.01
    ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
    simpleGrasp.setIKFrame(ik_frame)

    # Pick container comprises approaching, grasping (using SimpleGrasp stage), and lifting of object
    pick = stages.Pick(simpleGrasp, "Pick")
    pick.eef = eef
    pick.object = object_name

    # Twist to approach the object
    approach = TwistStamped()
    approach.header.frame_id = "world"
    approach.twist.linear.z = -1.0
    pick.setApproachMotion(approach, 0.03, 0.1)

    # Twist to lift the object
    lift = TwistStamped()
    lift.header.frame_id = "a_bot_gripper_tip_link"
    lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.03, 0.1)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    # Connect the Pick stage with the following Place stage
    connect2 = stages.Connect("connect2", planners)
    connect2.max_distance = 0.001
    task.add(connect2)

    # Define the pose that the object should have after placing
    placePose = objectPose
    placePose.pose.position.y += 0.2  # shift object by 20cm along y axis

    # Generate Cartesian place poses for the object
    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(task["Pick"])
    place_generator.object = object_name
    place_generator.pose = placePose
    place_generator.properties["pregrasp"] = "open"

    # The SimpleUnGrasp container encapsulates releasing the object at the given Cartesian pose
    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    # Place container comprises placing, ungrasping, and retracting
    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = object_name
    place.eef_frame = "a_bot_gripper_tip_link"

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = "a_bot_gripper_tip_link"
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    # Add the place pipeline to the task's hierarchy
    task.add(place)

    # avoid ClassLoader warning
    del pipeline
    del planners

    return task


def create_cartesian_task():
    group = "a_bot"

    # Cartesian and joint-space interpolation planners
    cartesian = core.CartesianPath()
    jointspace = core.JointInterpolationPlanner()

    task = core.Task()
    task.name = "Cartesian Task"

    # start from current robot state
    task.add(stages.CurrentState("current state"))

    # move along x
    move = stages.MoveRelative("y +0.2", cartesian)
    move.group = group
    header = Header(frame_id="world")
    move.setDirection(Vector3Stamped(header=header, vector=Vector3(0, 0.2, 0)))
    task.add(move)

    # move along y
    move = stages.MoveRelative("x +0.2", cartesian)
    move.group = group
    move.setDirection(Vector3Stamped(header=header, vector=Vector3(0.2, 0, 0)))
    task.add(move)

    # rotate about z
    move = stages.MoveRelative("rz +45°", cartesian)
    move.group = group
    move.setDirection(TwistStamped(header=header, twist=Twist(angular=Vector3(0, 0, pi / 4.0))))
    task.add(move)

    # Cartesian motion, defined as joint-space offset
    move = stages.MoveRelative("joint offset", cartesian)
    move.group = group
    move.setDirection(dict(a_bot_shoulder_pan_joint=pi / 6))
    task.add(move)

    # moveTo named posture, using joint-space interplation
    move = stages.MoveTo("moveTo ready", jointspace)
    move.group = group
    move.setGoal("home")
    task.add(move)

    return task


def create_compute_ik_task():
    group = "a_bot"

    ik_frame = PoseStamped(header=Header(frame_id="a_bot_outside_camera_color_frame"))
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "tray_center"
    orientation = quaternion_normalize([-0.006, 0.653, 0.002, 0.757])

    goal_pose.pose.orientation = Quaternion(*orientation)
    goal_pose.pose.position.x = 0.0
    goal_pose.pose.position.y = 0.0
    goal_pose.pose.position.z = 0.55

    task = core.Task()
    task.enableIntrospection()
    task.name = "MoveTo Task"

    task.add(stages.CurrentState("current state"))

    planner = core.PipelinePlanner()  # create default planning pipeline
    task.add(stages.Connect("connect", [(group, planner)]))  # operate on group

    # Add a Cartesian pose generator
    generator = stages.GeneratePose("cartesian pose")
    # Inherit PlanningScene state from "current state" stage
    generator.setMonitoredStage(task["current state"])
    # Configure target pose
    generator.pose = goal_pose

    # Wrap Cartesian generator into a ComputeIK stage to yield a joint pose
    computeIK = stages.ComputeIK("compute IK", generator)
    computeIK.group = group  # Use the group-specific IK solver
    # Which end-effector frame should reach the target?
    computeIK.ik_frame = ik_frame
    computeIK.min_solution_distance = 0.01
    computeIK.ignore_collisions = False
    computeIK.max_ik_solutions = 4  # Limit the number of IK solutions
    props = computeIK.properties
    # derive target_pose from child's solution
    props.configureInitFrom(core.Stage.PropertyInitializerSource.INTERFACE, ["target_pose"])

    # Add the stage to the task hierarchy
    task.add(computeIK)

    return task


def create_move_to_task():
    group = "a_bot"

    ik_frame = PoseStamped(header=Header(frame_id="a_bot_outside_camera_color_frame"))
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "tray_center"
    orientation = quaternion_normalize([-0.006, 0.653, 0.002, 0.757])

    goal_pose.pose.orientation = Quaternion(*orientation)
    goal_pose.pose.position.x = 0.0
    goal_pose.pose.position.y = 0.0
    goal_pose.pose.position.z = 0.55

    task = core.Task()
    task.enableIntrospection()
    task.name = "MoveTo Task"

    task.add(stages.CurrentState("current state"))

    planner = core.CartesianPath()
    planner.min_fraction = 0.04
    move = stages.MoveTo("move to PoseStamped", planner)
    move.group = group
    move.setGoal(goal_pose)

    if ik_frame:
        move.ik_frame = ik_frame

    task.add(move)
    return task


if __name__ == "__main__":
    main()
