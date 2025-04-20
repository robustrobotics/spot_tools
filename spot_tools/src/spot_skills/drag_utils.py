# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).


""" From arm grasp example.
"""
import argparse
import math
import sys
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand, block_for_trajectory_cmd)
from bosdyn.client.robot_state import RobotStateClient

""" From arm_impedance example
# """
# from arm_impedance_control_helpers import (apply_force_at_current_position,
#                                            get_impedance_mobility_params, get_root_T_ground_body)

from bosdyn.client.math_helpers import Quat, SE3Pose, Vec3

from bosdyn.util import seconds_to_duration

"""
From arm_freeze example
"""
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from spot_skills.grasp_utils import (get_user_grasp_input, add_grasp_constraint)
from spot_skills.arm_utils import (
    close_gripper,
    open_gripper,
    stow_arm,
)

import matplotlib as plt
###
g_image_click = None
g_image_display = None

# impedance 
ENABLE_STAND_HIP_ASSIST = True
ENABLE_STAND_YAW_ASSIST = False

# assume object already held (use grasp_utils)
# 1. YOLO / User input for image 

# view = "front_left_fisheye" # check
# image, img = spot.getimageRGB(view)
# # image request - info about camera, 
# # img - numpy array, actual image
# pixel_xy = get_user_grasp_input(spot, img)

# 2. Grasp (without stowing back)
def grasp_object(spot, image, xy, grasp_constraint):
    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client

    pick_vec = geometry_pb2.Vec2(x=xy[0], y=xy[1])

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(grasp_constraint, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request
    )

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        current_state = manipulation_api_pb2.ManipulationFeedbackState.Name(
            response.current_state
        )
        print(f"Current state: {current_state}")

        failed_states = [
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE,
        ]

        if response.current_state in failed_states:
            print("Grasp failed.")
            break

        if (
            response.current_state
            == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
        ):
            success = True
            break


# 3. Keep joint frozen, move into drag position (spinning)
def move_into_drag_mode(spot):
    # # relative_pose = math_helpers.Vec3(x=1, y=0, z=0)
    # rel_x = relative_pose.x
    # rel_y = relative_pose.y

    # Set up spot
    robot = spot.robot

    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    ## 1. Keep arm in place relative to world frame (odom)
    odom_T_hand = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "hand")
    arm_command = RobotCommandBuilder.arm_pose_command_from_pose(odom_T_hand.to_proto(),
                                                                ODOM_FRAME_NAME, seconds=2)
    # Construct a RobotCommand from the arm command
    command = RobotCommandBuilder.build_synchro_command(arm_command)
    
    # Send the request to the robot.
    cmd_id = command_client.robot_command(command)
    robot.logger.info('Keeping end-effector at location in odom front of and to the side of the robot.')

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)
    input("Is it frozen?")
    
    ## 2. Turn to the left 90 degrees (feel free to adjust the angle or rel x, y)
    turn_angle = np.deg2rad(90)
    mobility_params = mobility_params_for_slow_walk()
    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    0.75, 0, turn_angle,
    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    params=mobility_params)

    # Send the request
    end_time = 3
    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
    robot.logger.info('Walking with hand fixed relative to world.')

    # Wait until the body arrives at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)

    input("Did spot turn 90 degrees?")


    # # Need to adjust the input rel x, rel y to account for the arm being to off to the side
    # # Obtain transform from hand (arm) to body
    # body_T_hand = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, "body", "hand")
    # x_offset = body_T_hand.position.x
    # y_offset = body_T_hand.position.y
    # print(f"Hand to body offset: {x_offset:.2f} m, {y_offset:.2f} m")
    # rel_x = rel_x - x_offset # might need to add half of the body length?
    # rel_y = rel_y - y_offset 
    # print(f"Updated relative (x,y) = ({rel_x:.2f}, {rel_y:.2f})")

    ##3. Now fix arm relative to body (should be off to the side)
    body_T_hand = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, "body", "hand")
    arm_command = RobotCommandBuilder.arm_pose_command_from_pose(body_T_hand.to_proto(),
                                                                "body", seconds=2)
    # Construct a RobotCommand from the arm command
    command = RobotCommandBuilder.build_synchro_command(arm_command)

    # Send the request to the robot.
    cmd_id = command_client.robot_command(command)
    robot.logger.info(
        'Keeping end-effector at location relative to body to the side of the robot.')

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)

    ## Turn back to the right (reverse the original turn (?))
    mobility_params = mobility_params_for_slow_walk()
    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    0, 0, -turn_angle,
    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    params=mobility_params)

    # Send the request
    end_time = 3
    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
    robot.logger.info('Walking with hand fixed relative to world.')

    # Wait until the body arrives at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)
    input("Did spot turn  back?")

# 4. Move it relative pose
def navigate_to_relative_pose(spot, relative_pose, start_odom_T_body):
    ''' Navigates to relative pose while keep arm joint frozen'''
    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client
    command_client = spot.robot.ensure_client(RobotCommandClient.default_service_name)

    rel_x = relative_pose.x
    rel_y = relative_pose.y

    start_pos = start_odom_T_body.position  # x, y, z
    start_yaw = start_odom_T_body.rot.to_yaw()
    # robot = spot 

    # Create a joint freeze command
    joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()

    # Create a synchronized command with the joint_freeze_command and a walking command (mobility)
    # WALK_DIST = 1.0  # meters # previously 0.75 * WALK_DIST
    # 4/12 - relative x and y should be relative to robot's original position
    
    # compute new relative x and y
    # intermediate_robot_state = robot_state_client.get_robot_state()
    intermediate_robot_state = spot.get_state()
    # intermediate_vision_T_body = get_vision_tform_body(intermediate_robot_state.kinematic_state.transforms_snapshot)
    intermediate_odom_T_body = get_a_tform_b(intermediate_robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
    # start_tf = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, "vision", "body")
    intermediate_pos = intermediate_odom_T_body.position  # x, y, z
    intermediate_yaw = intermediate_odom_T_body.rot.to_yaw()
    print(f"Spot's intermediate position: {intermediate_pos.x:.2f} m, {intermediate_pos.y:.2f} m, {intermediate_pos.z:.2f} m, {intermediate_yaw:.2f} rad")
    new_rel_x = rel_x - (intermediate_pos.x - start_pos.x)
    new_rel_y = rel_y - (intermediate_pos.y - start_pos.y)
    new_rel_yaw = 0.0 - (intermediate_yaw - start_yaw)
    print(f"New relative (x,y) = ({new_rel_x:.2f}, {new_rel_y:.2f}), yaw = {new_rel_yaw:.2f}")
    # new_rel_y = start_pos.y + rel_y - intermediate_pos.y

    end_time = 10  # seconds
    mobility_params = mobility_params_for_slow_walk()
    
    joint_freeze_and_walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        new_rel_x, new_rel_y, new_rel_yaw,
        robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
        params=mobility_params, build_on_command=joint_freeze_command)

    # joint_freeze_and_walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    #     rel_x, rel_y, 0,
    #     robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    #     params=mobility_params, build_on_command=joint_freeze_command)

    # Send the request
    cmd_id = command_client.robot_command(joint_freeze_and_walk_command, end_time_secs=time.time() + end_time)
    spot.robot.logger.info('Walking with joint move to freeze.')

    # Wait until the body arrives at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)

# given relative x,y position 

# 1. usewith arm in front, move to left (or right) so arm to side front
# 2. 

# High level function to be called 
def drag_object(spot, relative_pose, grasp_constraint=None, debug=False):
    # S0 - Get start pose
    start_odom_T_body = get_a_tform_b(spot.get_state().kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
    start_pos = start_odom_T_body.position  # x, y, z
    start_yaw = start_odom_T_body.rot.to_yaw()
    print(f"Spot's starting position: {start_pos.x:.2f} m, {start_pos.y:.2f} m, {start_pos.z:.2f} m, {start_yaw:.2f} rad")

    # S1 - Get image and ask user to select object in image
    view = "frontleft_fisheye_image" # check
    image, img = spot.get_image_RGB(view) # # image request - info about camera, img - numpy array, actual image
    
    pixel_xy = get_user_grasp_input(spot, img) # returns pixel coordinates
    print(pixel_xy)

    # S2 - Grasp object based on pixel coordinates
    grasp_object(spot, image, pixel_xy, grasp_constraint)

    #S3 - Go into drag mode (freeze, and move) 
    move_into_drag_mode(spot)

    #S4 - Move to relative pose with frozen joint 
    navigate_to_relative_pose(spot,relative_pose,start_odom_T_body)
    #S5 - Release grip
    open_gripper(spot)

    spot.robot.logger("finished drag sequence")

    



# helper functions
def mobility_params_for_slow_walk():
    """
    Limit the x speed of the robot during the mobility phases of this example.  This is purely
      for aesthetic purposes to observe the hand's behavior longer while walking.
    """
    speed_limit = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=0.2, y=2), angular=1),
        min_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=-0.2, y=-2), angular=-1))
    return spot_command_pb2.MobilityParams(vel_limit=speed_limit)