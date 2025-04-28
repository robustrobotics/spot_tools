"""
Grasping an object with the arm (likely a backpack)
"""

import argparse
import sys
import time
from copy import copy

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import (
    geometry_pb2,
    manipulation_api_pb2,
    robot_state_pb2,
)
from bosdyn.client.frame_helpers import (
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives

from spot_skills.arm_impedance_control_helpers import (
    apply_force_at_current_position,
    get_root_T_ground_body,
)
from spot_skills.arm_utils import (
    arm_to_carry,
    arm_to_drop,
    close_gripper,
    open_gripper,
    stow_arm,
)

g_image_click = None
g_image_display = None


def wait_until_grasp_state_updates(grasp_override_command, robot_state_client):
    updated = False
    has_grasp_override = grasp_override_command.HasField("api_grasp_override")
    has_carry_state_override = grasp_override_command.HasField("carry_state_override")

    while not updated:
        robot_state = robot_state_client.get_robot_state()

        grasp_state_updated = (
            robot_state.manipulator_state.is_gripper_holding_item
            and (
                grasp_override_command.api_grasp_override.override_request
                == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING
            )
        ) or (
            not robot_state.manipulator_state.is_gripper_holding_item
            and grasp_override_command.api_grasp_override.override_request
            == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_NOT_HOLDING
        )

        carry_state_updated = has_carry_state_override and (
            robot_state.manipulator_state.carry_state
            == grasp_override_command.carry_state_override.override_request
        )
        updated = (not has_grasp_override or grasp_state_updated) and (
            not has_carry_state_override or carry_state_updated
        )
        time.sleep(0.1)


def force_stow_arm(manipulation_client, state_client, command_client):
    # grasp_override = manipulation_api_pb2.ApiGraspOverride()
    # grasp_override.override = manipulation_api_pb2.ApiGraspOverride.CARRIABLE_AND_STOWABLE

    ## Build the stow command with the grasp override
    # stow_command = RobotCommandBuilder.arm_stow_command()
    # stow_command.synchronized_command.arm_command.arm_cartesian_command.root_frame_name = 'body'
    # stow_command.synchronized_command.arm_command.arm_cartesian_command.wrist_tform_tool.rotation.w = 1
    # stow_command.synchronized_command.gripper_command.grasp_override.CopyFrom(grasp_override)

    ## Send the command
    # command_id = client.robot_command(stow_command)

    ## Optionally, you can block until the command is complete
    ##client.robot_command_feedback(command_id)

    ## Wait until the arm arrives at the goal.
    # block_until_arm_arrives(client, command_id, 2.0)

    carriable_and_stowable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
        override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE
    )

    grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
        override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING
    )

    override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
        api_grasp_override=grasp_holding_override,
        carry_state_override=carriable_and_stowable_override,
    )
    manipulation_client.grasp_override_command(override_request)
    # Wait for the override to take effect before trying to move the arm.
    wait_until_grasp_state_updates(override_request, state_client)

    robot_cmd = RobotCommandBuilder.arm_stow_command()
    cmd_id = command_client.robot_command(robot_cmd)
    block_until_arm_arrives(command_client, cmd_id)


def object_place(spot, semantic_class="bag", position=None):
    """Drop a grasped object."""

    if spot.is_fake:
        return True

    time.sleep(0.25)
    arm_to_carry(spot)
    stow_arm(spot)
    arm_to_drop(spot)

    odom_T_task = get_root_T_ground_body(
        robot_state=spot.get_state(), root_frame_name=ODOM_FRAME_NAME
    )
    wr1_T_tool = math_helpers.SE3Pose(
        0.23589, 0, -0.03943, math_helpers.Quat.from_pitch(-np.pi / 2)
    )
    force_dir_rt_task = math_helpers.Vec3(0, 0, -1)  # adjust downward force here
    robot_cmd = apply_force_at_current_position(
        force_dir_rt_task_in=force_dir_rt_task,
        force_magnitude=8,
        robot_state=spot.get_state(),
        root_frame_name=ODOM_FRAME_NAME,
        root_T_task=odom_T_task,
        wr1_T_tool_nom=wr1_T_tool,
    )

    # Execute the impedance command
    cmd_id = spot.command_client.robot_command(robot_cmd)
    spot.robot.logger.info("Impedance command issued")
    block_until_arm_arrives(spot.command_client, cmd_id, 10.0)
    input("Did impedance work")

    open_gripper(spot)
    # drop_object(spot)
    stow_arm(spot)
    close_gripper(spot)
    time.sleep(0.25)
    return True


def object_grasp(
    spot,
    image_source="hand_color_image",
    user_input=False,
    semantic_class="bag",
    grasp_constraint=None,
    labelspace_map=None,
    debug=False,
):
    debug_images = []
    if spot.is_fake:
        if debug:
            return True, None
        return True

    """Using the Boston Dynamics API to command Spot's arm."""

    if spot.semantic_name_to_id is None:
        raise Exception(
            "Must set spot.semantic_name_to_id in order to grasp object based on semantic class"
        )

    print(f'Grasping object of class "{semantic_class}"')
    if labelspace_map is not None:
        semantic_ids_to_grab = (
            labelspace_map[semantic_class]
            + labelspace_map["clothes"]
            + labelspace_map["bag"]
        )
    else:
        semantic_ids_to_grab = [
            spot.semantic_name_to_id[semantic_class],
            spot.semantic_name_to_id["bag"],
            spot.semantic_name_to_id["clothes"],
        ]

    open_gripper(spot)
    robot = spot.robot

    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client

    attempts = 0
    success = False
    img_source = image_source

    while attempts < 2 and not success:
        attempts += 1

        if not user_input:
            image, img = spot.get_image_alt(view=img_source)
            semantic_image = spot.segment_image(img)
            # Convert to grayscale if needed
            if len(semantic_image.shape) == 3:
                gray = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = semantic_image

            # Create a binary mask where the class of interest is white and the rest is black
            mask = np.zeros_like(gray)
            # mask[gray == class_index] = 255
            mask[np.isin(gray, semantic_ids_to_grab)] = 255

            debug_images.append(semantic_image)
            debug_images.append(mask)

            xy = get_class_centroid(semantic_image, semantic_ids_to_grab, img)
            print("Found object centroid:", xy)
            if xy is None:
                print("Object not found in image.")
                xy, image, img, image_source = look_for_object(
                    spot, semantic_ids_to_grab
                )
                if xy is None:
                    print("Object not found near robot.")
                    continue
        else:
            image, img = spot.get_image_alt(view=image_source)
            xy = get_user_grasp_input(spot, img)
            print("Found object centroid:", xy)
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

    close_cmd = RobotCommandBuilder.claw_gripper_close_command(
        build_on_command=None,
        max_acc=None,
        max_vel=None,
        disable_force_on_contact=False,
        max_torque=20,
    )
    spot.command_client.robot_command(close_cmd)
    time.sleep(0.25)

    # Move the arm to a carry position.
    print("")
    print("Grasp finished, search for a person...")
    carry_cmd = RobotCommandBuilder.arm_carry_command()
    spot.command_client.robot_command(carry_cmd)

    # Wait for the carry command to finish
    time.sleep(0.75)

    print("Force stowing arm!")
    # stow_arm(spot)
    force_stow_arm(manipulation_api_client, robot_state_client, spot.command_client)
    time.sleep(1)

    robot.logger.info("Finished grasp.")

    if debug:
        return success, debug_images
    return success


def object_grasp_YOLO(
    spot,
    image_source="hand_color_image",
    user_input=False,
    semantic_class="bag",
    grasp_constraint=None,
    labelspace_map=None,
    debug=False,
):
    debug_images = []
    if spot.is_fake:
        if debug:
            return True, None
        return True

    """Using the Boston Dynamics API to command Spot's arm."""
    if spot.yolo_model is None:
        raise Exception(
            "Must have initialized a YOLOWorld model to use object_grasp_YOLO."
        )

    print(f'Grasping object of class "{semantic_class}"')
    open_gripper(spot)
    robot = spot.robot

    robot_state_client = spot.state_client
    manipulation_api_client = spot.manipulation_api_client

    attempts = 0
    success = False
    img_source = image_source

    while attempts < 2 and not success:
        attempts += 1

        if not user_input:
            image, img = spot.get_image_RGB(view=img_source)
            xy = get_centroid_from_YOLO(
                spot, img, semantic_class, rotate=0, debug=debug
            )

            if xy is None:
                print("Object not found in image.")
                xy, image, img, image_source = look_for_object_YOLO(
                    spot, semantic_class, debug=debug
                )

                if xy is None:
                    print("Object not found near robot.")
                    continue
        else:
            image, img = spot.get_image_RGB(view=image_source)
            xy = get_user_grasp_input(spot, img)
            print("Found object centroid:", xy)
        pick_vec = geometry_pb2.Vec2(x=xy[0], y=xy[1])

        stow_arm(spot)

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

    close_cmd = RobotCommandBuilder.claw_gripper_close_command(
        build_on_command=None,
        max_acc=None,
        max_vel=None,
        disable_force_on_contact=False,
        max_torque=20,
    )
    spot.command_client.robot_command(close_cmd)
    time.sleep(0.25)

    # Move the arm to a carry position.
    print("")
    print("Grasp finished, search for a person...")
    carry_cmd = RobotCommandBuilder.arm_carry_command()
    spot.command_client.robot_command(carry_cmd)

    # Wait for the carry command to finish
    time.sleep(0.75)

    print("Force stowing arm!")
    # stow_arm(spot)
    force_stow_arm(manipulation_api_client, robot_state_client, spot.command_client)
    time.sleep(1)

    robot.logger.info("Finished grasp.")

    if debug:
        return success, debug_images
    return success


def get_centroid_from_YOLO(spot, img, semantic_class, rotate=0, debug=False):
    if rotate == 0:
        model_input = copy(img)
    elif rotate == 1:
        model_input = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 2:
        model_input = cv2.rotate(img, cv2.ROTATE_180)

    results = spot.yolo_model(model_input)

    best_box = None
    best_confidence = -1.0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            confidence = float(box.conf[0])

            # Check if the class name matches the semantic class we're looking for and if box is not too big
            box_height = box.xyxy[0][3] - box.xyxy[0][1]  # height of the bounding box
            if box_height > 0.5 * img.shape[0]:
                continue
            box_width = box.xyxy[0][2] - box.xyxy[0][0]  # width of the bounding box
            if (
                box_width > 0.5 * img.shape[1]
            ):  # If the box is more than half the width of the image, skip it
                continue

            if class_name == semantic_class and confidence > best_confidence:
                best_confidence = confidence
                best_box = box

    if best_box:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        class_id = int(best_box.cls[0])

        mid_x = x1 + (x2 - x1) // 2
        mid_y = y1 + (y2 - y1) // 2

        if rotate == 0:
            centroid_x = mid_x
            centroid_y = mid_y

        elif rotate == 1:
            centroid_x = mid_y
            centroid_y = img.shape[0] - mid_x

        elif rotate == 2:
            centroid_x = img.shape[1] - mid_x
            centroid_y = img.shape[0] - mid_y

        print("The centroid of the bounding box is at:", centroid_x, centroid_y)

        # # We need to rotate the bounding box coordinates if the image was rotated
        # if rotate == 1 or rotate == 2:
        #     # Rotate coordinates for 90 degrees clockwise
        #     x1, y1 = y1, img.shape[1] - x2
        #     x2, y2 = y2, img.shape[1] - x1
        # if rotate == 2:
        #     x1, y1 = img.shape[1] - x2, img.shape[0] - y2
        #     x2, y2 = img.shape[1] - x1, img.shape[0] - y1

        if debug:
            annotated_img = copy(img)

            # Draw bounding box and label
            # cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r.names[class_id]} {best_confidence:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (centroid_x, centroid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Label the centroid
            cv2.circle(annotated_img, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            cv2.putText(
                annotated_img,
                "Centroid",
                (centroid_x + 10, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            # Display or save the annotated image
            cv2.imshow("Most Confident Output", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return centroid_x, centroid_y  # Return the centroid of the bounding box

    else:
        return None


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
        image_title = "Click to grasp"
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def get_user_grasp_input(spot, img):
    """Show the image to the user and wait for them to click on a pixel"""

    spot.robot.logger.info("Click on an object to start grasping...")
    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    global g_image_click, g_image_display
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            # Quit
            print('"q" pressed, exiting.')
            exit(0)
    spot.robot.logger.info(
        f"Picking object at image location ({g_image_click[0]}, {g_image_click[1]})"
    )
    spot.robot.logger.info(
        "Picking object at image location (%s, %s)", g_image_click[0], g_image_click[1]
    )
    return g_image_click


def look_for_object(spot, semantic_ids):
    """Look for an object in the image sources. Return the centroid of the object, and the image source."""

    sources = spot.image_client.list_image_sources()

    for source in sources:
        if "depth" in source.name:
            continue
        image_source = source.name
        image, img = spot.get_image_alt(view=image_source)

        rotate = 0
        if "front" in image_source or "hand_image" in image_source:
            rotate = 1
        elif "right_fisheye_image" in image_source:
            rotate = 2

        semantic_image = spot.segment_image(img, rotate=rotate, show=False)
        xy = get_class_centroid(semantic_image, semantic_ids, img)
        print("Found object centroid:", xy)
        if xy is None:
            print(f"Object not found in {image_source}.")
            continue
        else:
            return xy, image, img, image_source

    return None, None, None, None


def look_for_object_YOLO(spot, semantic_class, debug=False):
    """Look for an object in the image sources using YOLO. Return the centroid of the object, and the image source."""

    sources = spot.image_client.list_image_sources()

    for source in sources:
        if (
            "depth" in source.name or source.name == "hand_image"
        ):  # "hand_image" is only in greyscale, "hand_color_image" is RGB
            continue

        image_source = source.name
        print("Getting image from source:", image_source)
        image, img = spot.get_image_RGB(view=image_source)

        rotate = 0

        if (
            "frontleft_fisheye_image" in image_source
            or "frontright_fisheye_image" in image_source
        ):
            rotate = 1  # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        elif "right_fisheye_image" in image_source:
            rotate = 2  # cv2.rotate(img, cv2.ROTATE_180)

        xy = get_centroid_from_YOLO(
            spot, img, semantic_class, rotate=rotate, debug=debug
        )
        print("Found object centroid:", xy)
        if xy is None:
            print(f"Object not found in {image_source}.")
            continue
        else:
            return xy, image, img, image_source

    return None, None, None, None


def get_class_centroid(segmented_image, class_indices, image):
    """Get the centroid of a class in a segmented image."""

    # Convert to grayscale if needed
    if len(segmented_image.shape) == 3:
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented_image

    # Create a binary mask where the class of interest is white and the rest is black
    mask = np.zeros_like(gray)
    # mask[gray == class_index] = 255
    mask[np.isin(gray, class_indices)] = 255

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # No contour found for the class

    # Assuming we take the largest contour if multiple are found
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate moments for the largest contour
    M = cv2.moments(largest_contour)

    # Calculate centroid using moments
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        seg_image_size = segmented_image.shape
        image_size = image.shape
        x_scale = image_size[1] / seg_image_size[1]
        y_scale = image_size[0] / seg_image_size[0]

        return (cX * x_scale, cY * y_scale)
    else:
        return None  # Centroid calculation failed


def add_grasp_constraint(grasp_constraint, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = False
    if (
        grasp_constraint == "force_top_down_grasp"
        or grasp_constraint == "force_horizontal_grasp"
    ):
        use_vector_constraint = True

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if grasp_constraint == "force_top_down_grasp":
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        elif grasp_constraint == "force_horizontal_grasp":
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        else:
            assert False, "Invalid grasp constraint"

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo
        )

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif grasp_constraint == "force_45_angle_grasp":
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
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif grasp_constraint == "force_squeeze_grasp":
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

    elif grasp_constraint is not None:
        assert False, "Invalid grasp constraint"


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-i",
        "--image-source",
        help="Get image from source",
        default="frontleft_fisheye_image",
    )
    parser.add_argument(
        "-t",
        "--force-top-down-grasp",
        help="Force the robot to use a top-down grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force-horizontal-grasp",
        help="Force the robot to use a horizontal grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--force-45-angle-grasp",
        help="Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--force-squeeze-grasp",
        help="Force the robot to use a squeeze grasp",
        action="store_true",
    )
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
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)

    try:
        # Assuming object_grasp is the correct function to call
        object_grasp(options)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
