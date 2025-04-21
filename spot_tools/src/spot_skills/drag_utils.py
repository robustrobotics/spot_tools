import argparse
import math
import sys
import time

import cv2
import numpy as np
import matplotlib as plt
from typing import Tuple

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME, get_vision_tform_body, get_a_tform_b, get_se2_a_tform_b, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand, block_for_trajectory_cmd)
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.math_helpers import Quat, SE3Pose, Vec3
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus

from bosdyn.util import seconds_to_duration
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from spot_skills.grasp_utils import (get_user_grasp_input, add_grasp_constraint)
from spot_skills.arm_utils import (
    close_gripper,
    open_gripper,
    stow_arm,
)

from spot_skills.skills_definitions_graspfeedback import (
    GraspFeedback,
)

"""
1. Pixel coordinate or user input
2. Grasp
3. Keep joint frozen, move into drag position - with arm on right or left
4. Move it relative pose
5. Move back to start
"""

##### Helper functions
def grasp_in_image(spot, image, xy, grasp_constraint):
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

def move_into_drag_mode(spot, arm_on_side="right"):

    robot = spot.robot
    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot.logger.info(f'Move arm to {arm_on_side} side')

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
    # input("Is it frozen?")
    
    ## 2. Turn robot 90 degrees (feel free to adjust the angle or rel x, y)
    if arm_on_side=="right": 
        turn_angle = np.deg2rad(90)
        x_offset = 0.5
        y_offset = 0.0
    else:
        turn_angle = -np.deg2rad(90)
        x_offset = 0.5
        y_offset = 0.0

    mobility_params = mobility_params_for_slow_walk()
    
    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    x_offset, 0, turn_angle,
    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    params=mobility_params)

    # Send the request
    end_time = 3
    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
    robot.logger.info('Walking with hand fixed relative to world.')

    # Wait until the body arrives at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=5)
    # input("Did spot turn 90 degrees?")

    ##3. Now fix arm relative to body (should be off to the side)
    body_T_hand = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, "body", "hand")
    arm_command = RobotCommandBuilder.arm_pose_command_from_pose(body_T_hand.to_proto(),
                                                                "body", seconds=2)
    # Construct a RobotCommand from the arm command
    command = RobotCommandBuilder.build_synchro_command(arm_command)

    # Send the request to the robot.
    cmd_id = command_client.robot_command(command)
    robot.logger.info(
        'Keeping end-effector at location relative to body to the {arm_side} side of the robot.')

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)

    ## Turn back to the right (reverse the original turn (?))
    if arm_on_side=="right": 
        # turn_angle = np.deg2rad(90)
        x_offset = 0.75
        y_offset = -0.75
    else:
        # turn_angle = -np.deg2rad(90)
        x_offset = 0.0
        y_offset = 0.6

    mobility_params = mobility_params_for_slow_walk()
    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    x_offset, y_offset, -turn_angle,
    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    params=mobility_params)

    # Send the request
    end_time = 3
    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
    robot.logger.info('Walking with hand fixed relative to body.')

    # Wait until the body arrives at the goal.
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)
    # input("Did spot turn back?")

def navigate_to_relative_pose(
    spot, 
    body_tform_goal: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0, 
    on_build_command=None,
    ) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.

    *** same as in navigation_utils but adds build_on_command
    """
    # Get the robot's current state.
    robot_state = spot.get_state()
    transforms = robot_state.kinematic_state.transforms_snapshot

    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(
            linear=Vec2(x=max_xytheta_vel[0], y=max_xytheta_vel[1]),
            angular=max_xytheta_vel[2],
        ),
        min_vel=SE2Velocity(
            linear=Vec2(x=min_xytheta_vel[0], y=min_xytheta_vel[1]),
            angular=min_xytheta_vel[2],
        ),
    )
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params,
        build_on_command=on_build_command
    )
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + timeout
    )
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = (
            feedback.feedback.synchronized_feedback.mobility_command_feedback
        )
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:  # pylint: disable=no-member,line-too-long
            spot.robot.logger.info("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (
            traj_feedback.status == traj_feedback.STATUS_AT_GOAL
            and traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED
        ):
            return
    if (time.perf_counter() - start_time) > timeout:
        spot.robot.logger.info("Timed out waiting for movement to execute!")

def mobility_params_for_slow_walk():
    """
    Limit the x speed of the robot during the mobility phases of this example.  This is purely
      for aesthetic purposes to observe the hand's behavior longer while walking.
    """
    speed_limit = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=0.2, y=2), angular=1),
        min_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=-0.2, y=-2), angular=-1))
    return spot_command_pb2.MobilityParams(vel_limit=speed_limit)

##### High level function to be called 
def drag_object(
    spot, 
    relative_pose, 
    image_source="frontleft_fisheye_image", 
    user_input=True,
    arm_on_side="right",
    pixel_xy=None,
    grasp_constraint=None,
    feedback=GraspFeedback(),
    debug=False,
    ):
    ## S0 - get relative pose of desired goal as a absolute pose in world frame
    transforms = spot.get_state().kinematic_state.transforms_snapshot
    body_0_tform_goal=math_helpers.SE2Pose(x=relative_pose.x, y=relative_pose.y, angle=relative_pose.angle)
    # print("Body 0 tform goal", body_0_tform_goal)
    out_tform_body_0 = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    # print("Out tform body 0", out_tform_body_0)
    out_tform_goal = out_tform_body_0 * body_0_tform_goal
    # print("Out tform goal", out_tform_goal)
    
    ## S1 - Get pixel coordinate 
    if user_input:  # Get image and ask user to select object in image
        # view = "frontleft_fisheye_image" 
        image, img = spot.get_image_RGB(image_source) # image - image request with info about camera; img - numpy array, actual image
        
        pixel_xy = get_user_grasp_input(spot, img) # returns pixel coordinates
    else:
        assert pixel_xy is not None, "Image pixel xy not provided"

        # Need image request details for transformations
        image, __ = spot.get_image_RGB(image_source)
    
    ## S2 - Grasp object based on pixel coordinates
    grasp_in_image(spot, image, pixel_xy, grasp_constraint)

    # Feedback Check #1 - after grasping
    robot_state = spot.get_state()
    feedback.initial_gripper_open_percentage = spot.get_state().manipulator_state.gripper_open_percentage
   
    if feedback.initial_gripper_open_percentage > 2.0: #2%
        feedback.initial_grasp = True

    ## S3 - Go into drag mode (freeze, and move) 
    move_into_drag_mode(spot,arm_on_side)
    # input("Is the spot in 'drag mode' with its arm to the side?")
    
    ## S4 - Move to relative pose with frozen joint 
    # print("Body 0 tform goal", body_0_tform_goal)
    # print("Out tform body 0", out_tform_body_0)
    # print("Out tform goal", out_tform_goal)
    transforms = spot.get_state().kinematic_state.transforms_snapshot
    out_tform_hand = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, "hand")
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    # print("Out tform hand", out_tform_hand)

    # Hand has its own yaw but we just care about the hand's x,y relation in the world frame. Combine with robot yaw instaed
    out_tform_hand_with_robot_yaw = math_helpers.SE2Pose(x=out_tform_hand.x, y=out_tform_hand.y, angle=out_tform_body.angle)
    # print("Out tform hand with robot yaw", out_tform_hand_with_robot_yaw)

    goal_tform_hand_with_robot_yaw = out_tform_goal.inverse() * out_tform_hand_with_robot_yaw
    # print("Goal tform hand with robot yaw", goal_tform_hand_with_robot_yaw)
    
    hand_with_robot_yaw_tform_goal = goal_tform_hand_with_robot_yaw.inverse()
    # print("hand with robot yaw tform goal", hand_with_robot_yaw_tform_goal)
    # input("Does hand to goal transformation look correct?")

    # Freeze arm joints and navigate to relative pose
    joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()    
    navigate_to_relative_pose(spot, hand_with_robot_yaw_tform_goal, on_build_command=joint_freeze_command)
    # input("Did spot bring the object to the goal position?")
    
    # Feedback Check #2 - after dragging
    robot_state = spot.get_state()
    feedback.final_gripper_open_percentage = spot.get_state().manipulator_state.gripper_open_percentage
   
    if feedback.final_gripper_open_percentage > 2.0: #2%
        feedback.final_grasp = True

    ## S5 - Release grip and stow arm
    open_gripper(spot)
    stow_arm(spot)
    close_gripper(spot)

    ## S6 - Move back to start (body_0)
    transforms = spot.get_state().kinematic_state.transforms_snapshot
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    # print("Out tform body", out_tform_body)
    body_0_tform_body = out_tform_body_0.inverse() * out_tform_body
    # print("body_0 tform body", body_0_tform_body)

    # Calculate relative pose of goal (body_0) in current body frame (body)
    body_tform_body_0 = body_0_tform_body.inverse()

    navigate_to_relative_pose(spot, body_tform_body_0)
    # input("Did spot go back to the start pose?")

    if feedback.success:
        print("Grasp successful")
    else:
        print("Unsuccessful grasp")
    # print("Finished drag sequence")