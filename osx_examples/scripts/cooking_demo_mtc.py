#!/usr/bin/env python
import argparse
from math import pi

from osx_robot_control.common import OSXCommon
from osx_assembly_database.assembly_reader import AssemblyReader
from osx_robot_control.helpers import get_trajectory_joint_goal
import rospy
import sys
import signal

from geometry_msgs.msg import TwistStamped, PoseStamped
from tf.listener import transformations

from moveit.task_constructor import core, stages
from moveit_commander import PlanningSceneInterface
# from moveit.task_constructor.stages import Holding

from moveit_msgs.msg import RobotState

from ur_control import conversions
import numpy as np
from math import tau


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
    parser.parse_args()

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("mtc_example")

    osx = OSXCommon()

    osx.reset_scene_and_robots()
    osx.a_bot.go_to_named_pose("home")
    rospy.sleep(1)
    osx.b_bot.go_to_named_pose("home")
    osx.a_bot.gripper.open()  # Reset gripper
    osx.b_bot.gripper.open()

    # Load cooking objects
    osx.assembly_database = AssemblyReader("cooking")
    # osx.spawn_object("cucumber", conversions.to_pose_stamped("tray_center", [0.0, 0.05, 0.01] + np.deg2rad([0, 90., 0]).tolist()))

    osx.spawn_object("cucumber", conversions.to_pose_stamped("cutting_board_surface", [0.02, -0.03, 0.02, tau/4, tau/4, 0]))
    cucumber = "cucumber"
    knife = "knife"
    b_bot = "b_bot"
    a_bot = "a_bot"

    # osx.a_bot.go_to_named_pose("above_tray", speed=0.5)

    grasp_names = osx.assembly_database.get_grasp_names(cucumber)
    for grasp_name in grasp_names:
        print(grasp_name)

    grasp_pose_stamped = osx.assembly_database.get_grasp_pose(cucumber, "grasp_2")
    place_pose = conversions.to_pose_stamped("cutting_board_surface", [0.02, -0.03, 0.02, tau/4, tau/4, 0])
    equip_grasp_pose = conversions.to_pose_stamped(
        "knife_pickup_link", [0.039, -0.004, 0.017, -0.016, -0.000, -0.009, 1.000])
    holding_grasp_pose = conversions.to_pose_stamped(
        "move_group/cucumber/tip1", [0.01, -0.02, -0.009, 0, 0.299, 0, 0.954])
    holding_grasp_pose = conversions.to_pose_stamped("cucumber/tip1",
                                                     [0, 0, 0, 0.000, 0.273, -0.000, 0.962])
    # holding_grasp_pose = grasp_pose_stamped = osx.assembly_database.get_grasp_pose(cucumber, "grasp_1")

    # Slightly elevated to avoid collision
    return_pose = conversions.to_pose_stamped("tray_center", [0.0, 0.05, 0.05] + np.deg2rad([0, 90., 0]).tolist())

    unequip_pose = conversions.to_pose_stamped(
        "knife_pickup_link", [0.033, -0.004, 0.017, -0.016, -0.000, -0.009, 1.000])

    osx.define_tool_collision_objects()

    osx.planning_scene_interface.remove_attached_object("knife")
    osx.planning_scene_interface.remove_world_object("knife")
    osx.allow_collisions_with_robot_hand("tray", a_bot)

    osx.spawn_tool("knife")

    osx.allow_collisions_with_robot_hand("knife", "b_bot")
    osx.planning_scene_interface.allow_collisions("knife", "knife_holder")
    osx.allow_collisions_with_robot_hand("cutting_board_surface", a_bot)
    osx.allow_collisions_with_robot_hand("workplate", a_bot)
    osx.planning_scene_interface.allow_collisions("cucumber", "tray_center")

    ### PARAMETERS ###
    slice_width = 0.008  # meters (1cm is easier to pick)
    num_of_slices = 8  # this times the slice_width show be less than the distance between the tip of the cucumber and the finger of a_bot
    slice_height = 0.035  # should be a bit higher (+5mm) than the cucumber diameter
    ##################

    # Tasks needed for cooking demo
    # task = create_pick_task(a_bot, cucumber, grasp_pose_stamped)
    # task = create_equip_task(osx, b_bot, knife, equip_grasp_pose)
    task = create_hold_vegetable_task(a_bot, cucumber, holding_grasp_pose)
    # task = create_holding_vegetable_task(osx, a_bot, cucumber, holding_grasp_conf)
    # task = create_check_vegetable_extremity_task(osx, b_bot, pose)

    # task = create_equip_task(osx, b_bot, knife, equip_grasp_pose)

    print("constructed task, planning...")
    if task.plan(max_solutions=2):
        # print("plan succeeded", task.solutions[0].toMsg())
        print("plan succeeded", len(task.solutions))
        print("sol", get_trajectory_joint_goal(task.solutions[0].toMsg().sub_trajectory[-1].trajectory))
        task.publish(task.solutions[0])

        input("continue to execute")
        task.execute(task.solutions[0])
    else:
        print("plan failed?")

    rospy.spin()  # Give time to check plan / errors


def create_pick_task(robot_name, object_name, grasp_pose_stamped):
    psi = PlanningSceneInterface(synchronous=True)
    # psi.remove_world_object()

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    object_pose = PoseStamped()
    object_pose.header.frame_id = "world"
    object_pose.pose = psi.get_object_poses([object_name])[object_name]

    # print("Object Pose: ", object_pose)

    # Create a task
    task = core.Task()
    task.name = "PickTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance to connect current to grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))

    # grasp_generator = stages.GenerateGraspPose("Generate Grasp Pose")
    grasp_generator = stages.LoadedGraspPoses("Predefined Grasp Poses")
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(grasp_pose_stamped)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = arm_tcp_link
    # ik_frame.pose.position.z = 0.1034
    ik_frame.pose.position.x = -0.01
    # ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
    # simpleGrasp.setMaxIKSolutions(10)
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
    lift.header.frame_id = arm_tcp_link
    lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.03, 0.1)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    del pipeline
    del planners

    return task


def create_pick_and_place_task(robot_name, object_name, grasp_pose_stamped):
    psi = PlanningSceneInterface(synchronous=True)

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    object_pose = PoseStamped()
    object_pose.header.frame_id = "world"
    object_pose.pose = psi.get_object_poses([object_name])[object_name]

    print("Object Pose: ", object_pose)

    # Create a task
    task = core.Task()
    task.name = "PickAndPlacePipeline"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance to connect current to grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))

    grasp_generator = stages.LoadedGraspPoses("Predefined Grasp Poses")
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(grasp_pose_stamped)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = arm_tcp_link
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
    lift.header.frame_id = arm_tcp_link
    lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.03, 0.1)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    # Connect the Pick stage with the following Place stage
    task.add(stages.Connect("connect2", planners))

    # Define the pose that the object should have after placing
    placePose = conversions.to_pose_stamped("cutting_board_surface", [0.02, -0.03, 0.02, tau/4, tau/4, 0])

    print("Place pose: ", placePose)

    # Generate Cartesian place poses for the object
    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(task["Pick"])
    place_generator.object = object_name
    place_generator.pose = placePose

    # The SimpleUnGrasp container encapsulates releasing the object at the given Cartesian pose
    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    # Place container comprises placing, ungrasping, and retracting
    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = object_name
    place.eef_frame = arm_tcp_link

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = arm_tcp_link
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    # Add the place pipeline to the task's hierarchy
    task.add(place)

    # avoid ClassLoader warning
    del pipeline
    del planners

    return task


def create_place_task(robot_name, object_name, place_pose):
    # psi = PlanningSceneInterface(synchronous=True)

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Create a task
    task = core.Task()
    task.name = "PlaceTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect2", planners))

    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(task["current"])
    place_generator.object = object_name
    place_generator.pose = place_pose
    place_generator.properties["pregrasp"] = "open"

    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = object_name
    place.eef_frame = arm_tcp_link

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = arm_tcp_link
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    task.add(place)

    del pipeline
    del planners

    return task


def create_equip_task(osx, robot_name, object_name, equip_grasp_pose):
    psi = PlanningSceneInterface(synchronous=True)

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Grasp object properties
    object_pose = PoseStamped()
    object_pose.header.frame_id = "world"
    object_pose.pose = psi.get_object_poses([object_name])[object_name]

    # Create a task
    task = core.Task()
    task.name = "EquipTask"
    task.enableIntrospection()

    # Start with the current state
    task.add(stages.CurrentState("current"))

    # Create planner instance to connect current to grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))

    # grasp_generator = stages.GenerateGraspPose("Generate Grasp Pose")
    grasp_generator = stages.LoadedGraspPoses("Predefined Grasp Poses")
    # grasp_generator.angle_delta = 0.2
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(equip_grasp_pose)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = arm_tcp_link
    # ik_frame.pose.position.z = 0.009
    # ik_frame.pose.position.x = -0.01
    # ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
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
    lift.header.frame_id = arm_tcp_link
    lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.00, 0.1)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    # Always return to home after equipping
    jointspace = core.JointInterpolationPlanner()
    move = stages.MoveTo("moveTo home", jointspace)
    move.group = robot_name
    move.setGoal("home")

    # move.restrictDirection(stages.MoveTo.Direction.FORWARD)
    task.add(move)

    del pipeline
    del planners

    return task


def create_unequip_tool_task(osx, robot_name, tool_name, unequip_pose):
    psi = PlanningSceneInterface(synchronous=True)

    # osx.planning_scene_interface.allow_collisions("knife", "knife_holder")

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Create a task
    task = core.Task()
    task.name = "UnequipToolTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create a planner instance to connect current to place approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Return to home first before unequipping
    jointspace = core.JointInterpolationPlanner()
    move = stages.MoveTo("moveTo home", jointspace)
    move.group = robot_name
    move.setGoal("home")

    task.add(move)

    # Connect the two stages
    task.add(stages.Connect("connect2", planners))

    place_generator = stages.GeneratePlacePose("Generate Unequip Pose")
    place_generator.setMonitoredStage(task["current"])
    place_generator.object = tool_name
    place_generator.pose = unequip_pose
    place_generator.properties["pregrasp"] = "open"

    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")
    # Be careful with IK frame
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = arm_tcp_link
    simpleUnGrasp.setIKFrame(ik_frame)

    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = tool_name
    place.eef_frame = arm_tcp_link

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = arm_tcp_link
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    task.add(place)

    # Return home after unequipping
    move = stages.MoveTo("moveTo home", jointspace)
    move.group = robot_name
    move.setGoal("home")

    task.add(move)

    del pipeline
    del planners

    return task


def create_hold_vegetable_task(robot_name, object_name, grasp_pose):
    psi = PlanningSceneInterface(synchronous=True)

    # psi.allow_collisions("b_bot", "knife")
    # psi.allow_collisons("knife", "knife_holder")

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    object_pose = PoseStamped()
    object_pose.header.frame_id = "world"
    object_pose.pose = psi.get_object_poses([object_name])[object_name]

    # grasp_pose.pose.position.x = -0.1

    print("Object Pose: ", object_pose)

    # Create a task
    task = core.Task()
    task.name = "PickAndPlacePipeline"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance to connect current to grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect1", planners))

    # modifyPlanningScene = stages.ModifyPlanningScene("modify planning scene")
    # modifyPlanningScene.allowCollisions(arm_tcp_link, object_name)
    # task.add(modifyPlanningScene)

    # grasp_generator = stages.GenerateGraspPose("Generate Grasp Pose")
    grasp_generator = stages.LoadedGraspPoses("Predefined Grasp Poses")
    grasp_generator.setMonitoredStage(task["current"])
    grasp_generator.properties["grasp"] = "close"
    grasp_generator.properties["pregrasp"] = "open"
    grasp_generator.properties["eef"] = eef
    grasp_generator.addPose(grasp_pose)

    # SimpleGrasp container encapsulates IK calculation of arm pose as well as finger closing
    simpleGrasp = stages.SimpleGrasp(grasp_generator, "Grasp")
    # Set frame for IK calculation in the center between the fingers
    ik_frame = PoseStamped()
    ik_frame.header.frame_id = arm_tcp_link
    ik_frame.pose.position.x = -0.01
    # ik_frame.pose.orientation = conversions.to_quaternion(transformations.quaternion_from_euler(0, pi/2, 0))
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
    lift.header.frame_id = arm_tcp_link
    # lift.twist.linear.x = -1.0
    pick.setLiftMotion(lift, 0.0, 0.0)

    # Add the pick stage to the task's stage hierarchy
    task.add(pick)

    del pipeline
    del planners

    return task


def create_holding_vegetable_task(osx, robot_name, object_name, holding_grasp_conf):

    psi = PlanningSceneInterface()

    # Robot specifications
    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Create a task
    task = core.Task()
    task.name = "HoldVegetableTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # task.add(stages.Connect("connect1", planners))

    # generator = stages.LoadedGraspPoses("Predefined grasp pose")
    # generator.addPose(holding_grasp_pose)
    # generator.setMonitoredStage(task["current state"])

    target_state = RobotState()
    target_state.joint_state.name = osx.active_robots[robot_name].robot_group.get_active_joints()
    target_state.joint_state.position = holding_grasp_conf
    target_state.is_diff = True

    jointspace = core.JointInterpolationPlanner()
    move = stages.MoveTo("moveTo holding_grasp", jointspace)
    move.group = robot_name
    move.setGoal(target_state)

    task.add(move)

    del pipeline
    del planners

    return task


def create_check_vegetable_extremity_task(osx, robot_name, pose):
    psi = PlanningSceneInterface(synchronous=True)

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Create a task
    task = core.Task()
    task.name = "CheckExtremityTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance to connect current to grasp approach pose
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    extremity_conf = osx.active_robots[robot_name].compute_ik(pose)

    target_state = RobotState()
    target_state.joint_state.name = osx.active_robots[robot_name].robot_group.get_active_joints()
    target_state.joint_state.position = extremity_conf
    target_state.is_diff = True

    jointspace = core.JointInterpolationPlanner()

    move = stages.MoveTo("moveTo pose", jointspace)
    move.group = robot_name
    move.setGoal(target_state)

    task.add(move)

    del pipeline
    del planners

    return task


def create_cleanup_task(robot_name, object_name, place_pose):

    arm = robot_name
    eef = robot_name + "_tip"
    arm_tcp_link = robot_name + "_gripper_tip_link"

    # Create a task
    task = core.Task()
    task.name = "PlaceTask"
    task.enableIntrospection()

    task.add(stages.CurrentState("current"))

    # Create planner instance
    pipeline = core.PipelinePlanner()
    pipeline.planner = "RRTConnect"
    planners = [(arm, pipeline)]

    # Connect the two stages
    task.add(stages.Connect("connect2", planners))

    place_generator = stages.GeneratePlacePose("Generate Place Pose")
    place_generator.setMonitoredStage(task["current"])
    place_generator.object = object_name
    place_generator.pose = place_pose
    place_generator.properties["pregrasp"] = "open"

    simpleUnGrasp = stages.SimpleUnGrasp(place_generator, "UnGrasp")

    place = stages.Place(simpleUnGrasp, "Place")
    place.eef = eef
    place.object = object_name
    place.eef_frame = arm_tcp_link

    # Twist to retract from the object
    retract = TwistStamped()
    retract.header.frame_id = "world"
    retract.twist.linear.z = 1.0
    place.setRetractMotion(retract, 0.03, 0.1)

    # Twist to place the object
    placeMotion = TwistStamped()
    placeMotion.header.frame_id = arm_tcp_link
    placeMotion.twist.linear.x = 1.0
    place.setPlaceMotion(placeMotion, 0.03, 0.1)

    task.add(place)

    del pipeline
    del planners

    return task


# def create_check_extremity_task(osx, robot_name):

#     move_down_pose = conversions.to_pose_stamped("cutting_board_surface", [0.0, 0.14, 0.005, -tau/2, tau/4, 0])
#     res = osx.active_robots[robot_name].go_to_pose_goal(move_down_pose, end_effector_link="b_bot_knife_center", speed=0.2, move_lin=True, plan_only = True)

#     if res is None:
#         return None
#     extremity_trajectory = res[0]

#     target_conf = extremity_trajectory.joint_trajectory.points[-1].positions

#     jointspace = core.JointInterpolationPlanner()

#     move = stages.MoveTo("moveTo", jointspace)
#     move.group = robot_name
#     move.setGoal(target_conf)
#     task.add(move)


if __name__ == "__main__":
    main()
