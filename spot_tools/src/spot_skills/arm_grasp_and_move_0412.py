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
"""
from arm_impedance_control_helpers import (apply_force_at_current_position,
                                           get_impedance_mobility_params, get_root_T_ground_body)

from bosdyn.client.math_helpers import Quat, SE3Pose, Vec3

from bosdyn.util import seconds_to_duration

"""
From arm_freeze example
"""
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

###
g_image_click = None
g_image_display = None

# impedance 
ENABLE_STAND_HIP_ASSIST = True
ENABLE_STAND_YAW_ASSIST = False


def arm_grasp_and_move(config):

    """A simple example of using the Boston Dynamics API to command Spot's arm."""
    
    
    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspPickupClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Pickup or drag (drawn from config, parameter passed into script)
    if config.move_type == 'pickup':
        pickup = True # pick up
        robot.logger.info('Pick up Mode')
    else:
        pickup = False # drag
        impedance_success = False
        robot.logger.info('Drag Mode')

    # Relative x, y (in meters)
    rel_x = float(config.relative_x)
    rel_y = float(config.relative_y) 
    robot.logger.info('Requested relative (x,y) = (%s, %s)', rel_x, rel_y)
    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    
    ## Since each part of sequence is blocking, get robot state (i.e. for gripper feedback, etc) in a blocking way. 
    # but async option is available for other applications

    # def gripper_state_async_callback(robot_state):
    #     """ Asynchronous callback function (from get_robot_state_async.py example)
        
    #     robot state has several categories including manipular state
    #     manipulator state has additional information - example:
    #             gripper_open_percentage: 51.573276519775391
    #             estimated_end_effector_force_in_hand {
    #                 x: 9.6913156509399414
    #                 y: 0.077649176120758057
    #                 z: 17.730426788330078
    #             }
    #             stow_state: STOWSTATE_STOWED
    #             velocity_of_hand_in_vision {
    #             linear {
    #                 x: 0.0022174888290464878
    #                 y: -0.00051479635294526815
    #                 z: -0.0049765077419579029
    #             }
    #             angular {
    #                 x: -0.0091756163164973259
    #                 y: 0.0079088704660534859
    #                 z: -0.0097299795597791672
    #             }
    #             }
    #             velocity_of_hand_in_odom {
    #             linear {
    #                 x: 0.00046961734187789261
    #                 y: -0.0022274947259575129
    #                 z: -0.0049765119329094887
    #             }
    #             angular {
    #                 x: 0.0032951682806015015
    #                 y: 0.0116569297388196
    #                 z: -0.0097299795597791672
    #             }
    #             } 
    #     """
    #     print('async_callback() called.')
    #     # nonlocal callback_is_done
    #     nonlocal gripper_open_percent
    #     gripper_open_percent = robot_state.result().manipulator_state.gripper_open_percentage
    #     print("Gripper Open Percentage: ", gripper_open_percent)
    #     # manipulator state also has other  
    #     # callback_is_done = True

    #     # nonlocal print_results
    #     # if print_results:
    #     #     print(results.result())


    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Take a picture with a camera
        robot.logger.info('Getting an image from: %s', config.image_source)
        image_responses = image_client.get_image_from_sources([config.image_source])

        if len(image_responses) != 1:
            print(f'Got invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        robot.logger.info(
            f'Picking object at image location ({g_image_click[0]}, {g_image_click[1]})')
        robot.logger.info('Picking object at image location (%s, %s)', g_image_click[0],
                          g_image_click[1])

        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # 4/12 - Get starting position before moving
        robot_state = robot_state_client.get_robot_state()
        # start_vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        start_odom_T_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
        # start_tf = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, "vision", "body")
        start_pos = start_odom_T_body.position  # x, y, z
        start_yaw = start_odom_T_body.rot.to_yaw()
        print("yawwww")
        # r, p, start_yaw, = quaternion_to_euler(start_rot) 
        print(f"Spot's starting position: {start_pos.x:.2f} m, {start_pos.y:.2f} m, {start_pos.z:.2f} m, {start_yaw:.2f} rad")
        
        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        add_grasp_constraint(config, grasp, robot_state_client)

        # Create request based on the "grasp" proto above 
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request for the robot to pick up the object
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        ###########
        # Get feedback from the robot (if grasp succeeded or failed)
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request for the response
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
            )

            ###########
            ## If successful, execute rest of sequence, lifting and moving to location
            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                
                if pickup:
                    ## Lift up if picking up (not dragging) (referenced from arm impedance example)
                    
                    # Pick a task frame that is beneath the robot body center, on the ground.
                    odom_T_task = get_root_T_ground_body(robot_state=robot_state_client.get_robot_state(),
                                                        root_frame_name=ODOM_FRAME_NAME)

                    # Set our tool frame to be the tip of the robot's bottom jaw. Flip the orientation so that
                    # when the hand is pointed downwards, the tool's z-axis is pointed upward.
                    wr1_T_tool = SE3Pose(0.23589, 0, -0.03943, Quat.from_pitch(-math.pi / 2))

                    # Now, do a Cartesian move to get the hand pointed downward 20cm above and 60cm in front of
                    # the task frame.
                    task_T_tool_desired = SE3Pose(0.6, 0, 0.2, Quat(1, 0, 0, 0))

                    # Pass in frames to function which creates and sends command to the robot
                    move_to_cartesian_pose_rt_task(robot, command_client, task_T_tool_desired, odom_T_task,
                                                wr1_T_tool) 
                    robot.logger.info('Lifting object')

                    # 4/12 - Check 1: Before bringing to desired location, check if still successfully holding object 
                    # robot.ensure_client(RobotStateClient.default_service_name)
                    if robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage < 1.0:
                        robot.logger.info('Gripper is closed, object not successfully grasped.')
                        
                        # walk back to start if fail to hold on?
                        walk_to_target(robot, robot_state_client, command_client, start_pos.x, start_pos.y, start_yaw)
                        break

                else: # drag - move arm to right
                    
                    ## 1. Keep arm in place relative to world frame (odom)
                    odom_T_arm = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "hand")
                    arm_command = RobotCommandBuilder.arm_pose_command_from_pose(odom_T_arm.to_proto(),
                                                                                ODOM_FRAME_NAME, seconds=2)
                    # Construct a RobotCommand from the arm command
                    command = RobotCommandBuilder.build_synchro_command(arm_command)
                    
                    # Send the request to the robot.
                    cmd_id = command_client.robot_command(command)
                    robot.logger.info('Keeping end-effector at location in odom front of and to the side of the robot.')

                    # Wait until the arm arrives at the goal.
                    block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)

                    ## 2. Turn to the left 90 degrees (feel free to adjust the angle or rel x, y)
                    mobility_params = mobility_params_for_slow_walk()
                    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    0.25, 0.25, 1.57,
                    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                    params=mobility_params)

                    # Send the request
                    end_time = 3
                    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
                    robot.logger.info('Walking with hand fixed relative to world.')

                    # Wait until the body arrives at the goal.
                    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)


                    # Need to adjust the input rel x, rel y to account for the arm being to off to the side
                    # Obtain transform from hand (arm) to body
                    body_T_hand = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, "body", "hand")
                    x_offset = body_T_hand.position.x
                    y_offset = body_T_hand.position.y
                    print(f"Hand to body offset: {x_offset:.2f} m, {y_offset:.2f} m")
                    rel_x = rel_x - x_offset # might need to add half of the body length?
                    rel_y = rel_y - y_offset 
                    print(f"Updated relative (x,y) = ({rel_x:.2f}, {rel_y:.2f})")

                    ##3. Now fix arm relative to body (should be off to the side)
                    body_T_arm = get_a_tform_b(robot_state_client.get_robot_state().kinematic_state.transforms_snapshot, "body", "hand")
                    arm_command = RobotCommandBuilder.arm_pose_command_from_pose(body_T_arm.to_proto(),
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
                    0.25, 0.25, -1.57,
                    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                    params=mobility_params)

                    # Send the request
                    end_time = 3
                    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
                    robot.logger.info('Walking with hand fixed relative to world.')

                    # Wait until the body arrives at the goal.
                    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)
                
                #     ## Lift up but less #TODO 

                    
                #     # Pick a task frame that is beneath the robot body center, on the ground.
                #     odom_T_task = get_root_T_ground_body(robot_state=robot_state_client.get_robot_state(),
                #                                         root_frame_name=ODOM_FRAME_NAME)

                #     # Set our tool frame to be the tip of the robot's bottom jaw. Flip the orientation so that
                #     # when the hand is pointed downwards, the tool's z-axis is pointed upward.
                #     wr1_T_tool = SE3Pose(0.23589, 0, -0.03943, Quat.from_pitch(-math.pi / 2))

                #     # Now, do a Cartesian move to get the hand pointed downward 20cm above and 60cm in front of
                #     # the task frame.
                #     task_T_tool_desired = SE3Pose(0.6, 0, 0.01, Quat(1, 0, 0, 0))

                #     # Pass in frames to function which creates and sends command to the robot
                #     move_to_cartesian_pose_rt_task(robot, command_client, task_T_tool_desired, odom_T_task,
                #                                 wr1_T_tool) 
                #     robot.logger.info('Lifting object')

                ############

                ## Walk to desired location with arm frozen 
                
                # Create a joint freeze command
                joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()

                # Create a synchronized command with the joint_freeze_command and a walking command (mobility)
                # WALK_DIST = 1.0  # meters # previously 0.75 * WALK_DIST
                # 4/12 - relative x and y should be relative to robot's original position
                
                # compute new relative x and y
                intermediate_robot_state = robot_state_client.get_robot_state()
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
                robot.logger.info('Walking with joint move to freeze.')

                # Wait until the body arrives at the goal.
                block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)
                
                ############
                
                # 4/12 - Check 2: After bringing to desired location, check if still successfully holding object 
                # robot.ensure_client(RobotStateClient.default_service_name)
                if robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage < 1.0:
                    robot.logger.info('Gripper is closed, object not successfully grasped.')

                    # walk back to start if fail to hold on?
                    walk_to_target(robot, robot_state_client, command_client, start_pos.x, start_pos.y, start_yaw)
                    break

                ####
                if pickup:
                    ## Bring object down using impedance command

                    # (if arm needs to be in a different pose while pushing down, insert here)

                    # Hold current pose in all other axes while pushing downward against the ground.
                    # Request a force be generated at the current position, in the negative-Z direction in the
                    # task frame.
                    force_dir_rt_task = Vec3(0, 0, -1) # adjust downward force here 
                    robot_cmd = apply_force_at_current_position(
                        force_dir_rt_task_in=force_dir_rt_task, force_magnitude=8,
                        robot_state=robot_state_client.get_robot_state(), root_frame_name=ODOM_FRAME_NAME,
                        root_T_task=odom_T_task, wr1_T_tool_nom=wr1_T_tool)

                    # Execute the impedance command
                    cmd_id = command_client.robot_command(robot_cmd)
                    robot.logger.info('Impedance command issued')

                # This might report STATUS_TRAJECTORY_COMPLETE or STATUS_TRAJECTORY_STALLED, depending on
                # the floor. STATUS_TRAJECTORY_STALLED is reported if the arm has stopped making progress to
                # the goal and the measured tool frame is far from the `desired_tool` along directions where
                # we expect good tracking. Since the robot can't push past the floor, the trajectory might
                # stop making progress, even though we will still be pushing against the floor. Unless the
                # floor has high friction (like carpet) we'd expect to have good tracking in all directions
                # except z. Because we have requested a feedforward force in z, we don't expect good
                # tracking in that direction. So we would expect the robot to report
                # STATUS_TRAJECTORY_COMPLETE in this case once arm motion settles.
                    impedance_success = block_until_arm_arrives(command_client, cmd_id, 10.0)
                
                
                ## if impedance command is successful (complete ?), then release item, lift, and walk back
                if impedance_success or not pickup:
                    robot.logger.info('Impedance move succeeded.')

                    ############
                    ## Open gripper to release object
                    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
                    robot.logger.info('Requesting open grip.')
                    open_gripper_command_id = command_client.robot_command(gripper_command)

                    block_until_arm_arrives(command_client, open_gripper_command_id, 2.0)

                    ## Lift arm up
                    odom_T_task = get_root_T_ground_body(robot_state=robot_state_client.get_robot_state(),
                                                    root_frame_name=ODOM_FRAME_NAME)

                    # Set our tool frame to be the tip of the robot's bottom jaw. Flip the orientation so that
                    # when the hand is pointed downwards, the tool's z-axis is pointed upward.
                    wr1_T_tool = SE3Pose(0.23589, 0, -0.03943, Quat.from_pitch(-math.pi / 2))

                    # Now, do a Cartesian move to get the hand pointed downward 20cm above and 60cm in front of
                    # the task frame.
                    task_T_tool_desired = SE3Pose(0.6, 0, 0.2, Quat(1, 0, 0, 0))
                    move_to_cartesian_pose_rt_task(robot, command_client, task_T_tool_desired, odom_T_task,
                                                wr1_T_tool)
                    robot.logger.info('Final lift')

                    ## Close gripper
                    gripper_command = RobotCommandBuilder.claw_gripper_close_command()
                    robot.logger.info('Requesting open grip.')
                    close_gripper_command_id = command_client.robot_command(gripper_command)

                    block_until_arm_arrives(command_client, close_gripper_command_id, 2.0)
                    
                    ## Stow arm 
                    stow_arm(robot, command_client)

                    ## Walk back (4/12 - want to move back to original position)
                    # Spot brought object to desired "rel x, rel y" location
                    moved_object_robot_state = robot_state_client.get_robot_state()
                    # end_tf = get_vision_tform_body(end_robot_state.kinematic_state.transforms_snapshot)
                    moved_object_odom_T_body = get_a_tform_b(moved_object_robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
                    moved_object_pos = moved_object_odom_T_body.position
                    moved_object_yaw = moved_object_odom_T_body.rot.to_yaw()
                    print(f"Spot's position after moving object: {moved_object_pos.x:.2f} m, {moved_object_pos.y:.2f} m, {moved_object_pos.z:.2f} m, {moved_object_yaw:.2f} rad")
                    
                    # Compute relative movement to bring spot back to starting position 
                    dx = moved_object_pos.x - start_pos.x
                    dy = moved_object_pos.y - start_pos.y
                    dtheta = moved_object_yaw - start_yaw
                    print(f"Spot moves dx={dx:.2f} m, dy={dy:.2f} m, dtheta={dtheta:.2f} rad to reach the starting pose.")

                    walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    -dx, -dy, -dtheta,
                    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                    params=mobility_params)

                    # Send the request
                    cmd_id = command_client.robot_command(walk_command, end_time_secs=time.time() + end_time)
                    robot.logger.info('Walking back.')

                    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)
                    
                    # final pose
                    end_robot_state = robot_state_client.get_robot_state()
                    # end_tf = get_vision_tform_body(end_robot_state.kinematic_state.transforms_snapshot)
                    end_odom_T_body = get_a_tform_b(end_robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
                    end_pos = end_odom_T_body.position
                    end_yaw = end_odom_T_body.rot.to_yaw()
                    print(f"Spot's ending position: {end_pos.x:.2f} m, {end_pos.y:.2f} m, {end_pos.z:.2f} m, {end_yaw:.2f} rad")
                    break # break once successful impedance
                else:
                    robot.logger.info('Impedance move didn\'t complete because it stalled or timed out.')
                
                ## Stow the arm
                stow_arm(robot, command_client)
                        
            ## Grasp failed -> break out of while loop
            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                
                # walk back to start if fail to grasp
                walk_to_target(robot, robot_state_client, command_client, start_pos.x, start_pos.y, start_yaw)
                break

            time.sleep(0.25)

        robot.logger.info('Finished sequence.')
        time.sleep(4.0)

        robot.logger.info('Sitting down and turning off.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)
    
def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

def unstow_arm(robot, command_client):
    stand_command = RobotCommandBuilder.synchro_stand_command(
        params=get_impedance_mobility_params())
    unstow = RobotCommandBuilder.arm_ready_command(build_on_command=stand_command)
    unstow_command_id = command_client.robot_command(unstow)
    robot.logger.info('Unstow command issued.')
    block_until_arm_arrives(command_client, unstow_command_id, 3.0)

def stow_arm(robot, command_client):
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow_cmd)
    robot.logger.info('Stow command issued.')
    block_until_arm_arrives(command_client, stow_command_id, 3.0)

def move_to_cartesian_pose_rt_task(robot, command_client, task_T_desired, root_T_task, wr1_T_tool):
    """
    Moves robot arm via cartesian trajectory (from arm impedance example)
    """
    robot_cmd = RobotCommandBuilder.synchro_stand_command(params=get_impedance_mobility_params())
    arm_cart_cmd = robot_cmd.synchronized_command.arm_command.arm_cartesian_command

    # Set up our root frame, task frame, and tool frame.
    arm_cart_cmd.root_frame_name = ODOM_FRAME_NAME
    arm_cart_cmd.root_tform_task.CopyFrom(root_T_task.to_proto())
    arm_cart_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())

    # Do a single point goto to a desired pose in the task frame.
    cartesian_traj = arm_cart_cmd.pose_trajectory_in_task
    traj_pt = cartesian_traj.points.add()
    traj_pt.time_since_reference.CopyFrom(seconds_to_duration(2.0))
    traj_pt.pose.CopyFrom(task_T_desired.to_proto())

    # Execute the Cartesian command.
    cmd_id = command_client.robot_command(robot_cmd)
    robot.logger.info('Arm cartesian command issued.')
    return block_until_arm_arrives(command_client, cmd_id, 3.0)


# from arm_freeze_hand.py for slow walk mobility params
def mobility_params_for_slow_walk():
    """
    Limit the x speed of the robot during the mobility phases of this example.  This is purely
      for aesthetic purposes to observe the hand's behavior longer while walking.
    """
    speed_limit = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=0.2, y=2), angular=1),
        min_vel=geometry_pb2.SE2Velocity(linear=geometry_pb2.Vec2(x=-0.2, y=-2), angular=-1))
    return spot_command_pb2.MobilityParams(vel_limit=speed_limit)

def walk_to_target(robot, robot_state_client, command_client, target_x, target_y, target_yaw):
    ## Walk back (4/12 - want to move back to original position)
    # Spot brought object to desired "rel x, rel y" location
    current_robot_state = robot_state_client.get_robot_state()
    # end_tf = get_vision_tform_body(end_robot_state.kinematic_state.transforms_snapshot)
    current_odom_T_body = get_a_tform_b(current_robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
    current_pos = current_odom_T_body.position
    current_yaw = current_odom_T_body.rot.to_yaw()
    print(f"Spot's current pose: {current_pos.x:.2f} m, {current_pos.y:.2f} m, {current_pos.z:.2f} m, {current_yaw:.2f} rad")
    
    # Compute relative movement to bring spot back to starting position 
    dx = current_pos.x - target_x
    dy = current_pos.y - target_y
    dtheta = current_yaw - target_yaw
    print(f"Spot moves dx={dx:.2f} m, dy={dy:.2f} m, dtheta={dtheta:.2f} rad to reach the starting pose.")

    mobility_params = mobility_params_for_slow_walk()
    joint_freeze_command = RobotCommandBuilder.arm_joint_freeze_command()
    joint_freeze_and_walk_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
    -dx, -dy, -dtheta,
    robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
    params=mobility_params, build_on_command=joint_freeze_command)
    
    # Send the request
    end_time = 10
    cmd_id = command_client.robot_command(joint_freeze_and_walk_command, end_time_secs=time.time() + end_time)
    robot.logger.info('Walking back.')
    block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=10)

    current_robot_state = robot_state_client.get_robot_state()
    current_odom_T_body = get_a_tform_b(current_robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, "body")
    current_pos = current_odom_T_body.position
    current_yaw = current_odom_T_body.rot.to_yaw()
    print(f"Spot's current pose: {current_pos.x:.2f} m, {current_pos.y:.2f} m, {current_pos.z:.2f} m, {current_yaw:.2f} rad")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')
    
    parser.add_argument('-m', '--move-type', help='Choose pick_up or drag', default='pickup')
    parser.add_argument('-x', '--relative-x', help='Enter x', default=0.5)
    parser.add_argument('-y', '--relative-y', help='Enter y', default=-0.5)

    options = parser.parse_args()

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print('Error: cannot force more than one type of grasp.  Choose only one.')
        sys.exit(1)

    try:
        arm_grasp_and_move(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
