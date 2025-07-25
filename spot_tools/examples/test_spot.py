import argparse
import base64
import io
import os  # Import the os package
import time
from copy import copy

import cv2
import numpy as np
from bosdyn.api.spot import door_pb2
from bosdyn.client import math_helpers
from openai import OpenAI  # Import the OpenAI package
from PIL import Image
from pydantic import BaseModel

from spot_executor.executor_feedback_pyplot import FeedbackCollector
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot
from spot_skills.arm_utils import (
    close_gripper,
    gaze_at_relative_pose,
    move_hand_to_relative_pose,
    open_gripper,
    stow_arm,
)
from spot_skills.detection_utils import YOLODetector
from spot_skills.door_utils import execute_open_door
from spot_skills.grasp_utils import object_grasp, object_place
from spot_skills.navigation_utils import (
    navigate_to_relative_pose,
)
from spot_skills.skills_definitions import OpenDoorFeedback, OpenDoorParams


def _run_walking_test(spot) -> None:
    # Put inside a function to avoid variable scoping issues.

    relative_poses = [
        (math_helpers.SE2Pose(x=0, y=0, angle=0), "Standing still"),
        (math_helpers.SE2Pose(x=0, y=0.5, angle=0), "Moving dy"),
        (math_helpers.SE2Pose(x=0, y=-0.5, angle=0), "Moving -dy"),
        (math_helpers.SE2Pose(x=0.5, y=0, angle=0), "Moving dx"),
        (math_helpers.SE2Pose(x=-0.5, y=0, angle=0), "Moving -dx"),
        (math_helpers.SE2Pose(x=0, y=0, angle=np.pi / 2), "Moving yaw"),
        (math_helpers.SE2Pose(x=0, y=0, angle=-np.pi / 2), "Moving -yaw"),
        (math_helpers.SE2Pose(x=1, y=0.5, angle=0), "Moving"),
        (math_helpers.SE2Pose(x=-1, y=-0.5, angle=0), "Moving back"),
        (math_helpers.SE2Pose(x=1, y=0.5, angle=np.pi), "Moving"),
        (math_helpers.SE2Pose(x=1, y=0.5, angle=-np.pi), "Moving back"),
    ]
    for relative_pose, msg in relative_poses:
        print(msg)
        navigate_to_relative_pose(spot, relative_pose)
        time.sleep(0.1)


def _run_hand_test(spot) -> None:
    # Put inside a function to avoid variable scoping issues.

    resting_pose = math_helpers.SE3Pose(x=0.80, y=0, z=0.45, rot=math_helpers.Quat())
    relative_down_pose = math_helpers.SE3Pose(
        x=0.0, y=0, z=0.0, rot=math_helpers.Quat.from_pitch(np.pi / 4)
    )
    resting_down_pose = resting_pose * relative_down_pose
    looking_down_and_rotated_right_pose = math_helpers.SE3Pose(
        x=0.9,
        y=0,
        z=0.0,
        rot=math_helpers.Quat.from_pitch(np.pi / 2)
        * math_helpers.Quat.from_roll(np.pi / 2),
    )
    print("Moving to a pose that looks down and rotates the gripper to the " + "right.")
    move_hand_to_relative_pose(spot, looking_down_and_rotated_right_pose)
    input("Press enter when ready to move on")

    print("Moving to a resting pose in front of the robot.")
    move_hand_to_relative_pose(spot, resting_pose)
    input("Press enter when ready to move on")

    print("Opening the gripper.")
    open_gripper(spot)
    input("Press enter when ready to move on")

    print("Moving to the same pose (should have no change).")
    move_hand_to_relative_pose(spot, resting_pose)
    input("Press enter when ready to move on")

    print("Closing the gripper.")
    move_hand_to_relative_pose(spot, resting_pose)
    close_gripper(spot)
    input("Press enter when ready to move on")

    print("Looking down and opening the gripper.")
    move_hand_to_relative_pose(spot, resting_down_pose)
    open_gripper(spot)
    input("Press enter when ready to move on")

    print("Closing the gripper, moving to resting pose")
    move_hand_to_relative_pose(spot, resting_pose)
    close_gripper(spot)


def _run_gaze_test(spot) -> None:
    resting_point = math_helpers.Vec3(x=0.80, y=0, z=0.45)
    relative_up = math_helpers.Vec3(x=0, y=0, z=15)
    print(resting_point)
    print(resting_point + relative_up)
    relative_poses = [
        # (resting_point, "Looking nowhere"),
        # (relative_up, "Looking up"),
        # (relative_up, "Looking up"),
        (math_helpers.Vec3(x=1.76, y=1, z=0.05), "Looking where Travis tells me to"),
        (math_helpers.Vec3(x=1.29, y=-0.34, z=-0.54), "Looking right"),
        # (math_helpers.Vec3(x=0, y=-0.5, ), "Looking -dy"),
        # (math_helpers.Vec3(x=0.5, y=0, z=0), "Looking dx"),
        # (math_helpers.Vec3(x=-0.5, y=0, ), "Looking -dx"),
        # (math_helpers.Vec3(x=0, y=0, ), "Looking yaw"),
        # (math_helpers.Vec3(x=0, y=0, ), "Looking -yaw"),
    ]
    for relative_pose, msg in relative_poses:
        print(msg)
        gaze_at_relative_pose(spot, relative_pose)
        input("Press enter when ready to move on")
        stow_arm(spot)
        input("Press enter when ready to move on")
        # time.sleep(0.5)


def _run_grasp_test(spot) -> None:
    spot.stand()
    stow_arm(spot)
    open_gripper(spot)
    relative_pose = math_helpers.Vec3(x=1, y=0, z=0)
    gaze_at_relative_pose(spot, relative_pose)
    time.sleep(0.2)

    detector = YOLODetector(
        spot,
        yolo_world_path="/home/rrg/data/models/yolov8s-world.pt",
    )
    object_grasp(
        spot,
        detector,
        image_source="hand_color_image",
        user_input=False,
        semantic_class="wood block",
        grasp_constraint=None,
        debug=True,
        feedback=FeedbackCollector(),
    )

    stow_arm(spot)


def _run_place_test(spot):
    object_place(spot)


def _run_segment_test(spot):
    open_gripper(spot)
    relative_pose = math_helpers.Vec3(x=1, y=0, z=0)
    gaze_at_relative_pose(spot, relative_pose)
    time.sleep(0.2)

    image, img = spot.get_image_alt(view="hand_color_image", show=True)
    segmented_image = spot.segment_image(img, show=True)
    return image, img, segmented_image


def _run_open_door_test(spot, model_path, max_tries=2) -> None:
    print("Opening the door...")

    trial_idx = 0
    parameters = OpenDoorParams()

    parameters.hinge_side = door_pb2.DoorCommand.HingeSide.HINGE_SIDE_RIGHT
    parameters.swing_direction = (
        door_pb2.DoorCommand.SwingDirection.SWING_DIRECTION_UNKNOWN
    )
    parameters.handle_type = door_pb2.DoorCommand.HandleType.HANDLE_TYPE_UNKNOWN

    feedback = OpenDoorFeedback()

    initial_pose = spot.get_pose()

    while trial_idx < max_tries:
        print(parameters)
        input("Does this seem right?")
        execute_open_door(spot, model_path, parameters, feedback, initial_pose)

        success = feedback.success()

        if success:
            print("The robot was able to open the door. SUCCESS!")
            break
        else:
            input("The robot was unable to open the door.")
            print("The robot was unable to open the door. FAILURE!")
            VLM_parameters = query_VLM(parameters, feedback)
            print(VLM_parameters)
            input("Does this seem reasonable?")
            parameters.hinge_side = VLM_parameters.hinge_side
            parameters.swing_direction = VLM_parameters.swing_direction
            parameters.handle_type = VLM_parameters.handle_type

        trial_idx += 1
        # Check that the feedback implies that the robot is successful
        # If not, query the VLM to edit the parameters and try again

    # # print(feedback.success)
    # # cv2.imshow('Detected Door', feedback.handle_detection)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # cv2.imshow('Ego View', feedback.ego_view)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()


def _run_YOLOWorld_test(spot):
    open_gripper(spot)
    hand_image_request, hand_image = spot.get_image_RGB()
    results = spot.yolo_model(hand_image)
    object_to_pick_up = "wood block"
    # Process results

    annotated_hand_image = copy(hand_image)
    # Show all of the detections
    for r in results:
        boxes = r.boxes  # Bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Draw bounding box and label
            cv2.rectangle(annotated_hand_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r.names[class_id]} {confidence:.2f}"
            cv2.putText(
                annotated_hand_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    # Display or save the annotated image
    cv2.imshow("YOLO Output", annotated_hand_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    most_confident = copy(hand_image)
    best_box = None
    best_confidence = -1.0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            confidence = float(box.conf[0])

            if class_name == object_to_pick_up and confidence > best_confidence:
                best_confidence = confidence
                best_box = box

    if best_box:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        class_id = int(best_box.cls[0])

        # Draw bounding box and label
        cv2.rectangle(most_confident, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{r.names[class_id]} {best_confidence:.2f}"
        cv2.putText(
            most_confident,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Display or save the annotated image
        cv2.imshow("Most Confident Output", most_confident)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        raise Exception("No objects of semantic class found")


def encode_numpy_image_to_base64(image_np: np.ndarray, format: str = "PNG"):
    pil_image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    b64_bytes = base64.b64encode(buffer.read())
    b64_string = b64_bytes.decode("utf-8")
    return b64_string


def query_VLM(parameters, feedback):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    class OpenDoorParams(BaseModel):
        hinge_side: int
        swing_direction: int
        handle_type: int
        CoT_reasoning: str

    side_by_side = feedback.ego_view

    # Encode the image to base64
    encoded_side_by_side = encode_numpy_image_to_base64(side_by_side)

    prompt = (
        "You are assisting a Boston Dynamics Spot robot that is trying to open a door. "
        "The robot has tried to open the door using its door opening skill, but has failed. "
        "The door opening skill requires the following parameters: hinge_side, swing_direction, handle_type. "
        "The hinge_side can be set to left (1) or right (2). "
        "The swing_direction can be set to unknown (0), pull (1), or push (2). "
        "The handle_type can be set to unknown (0), lever (1), or knob (2). "
        "In the robot's previous attempt to open the door, it used the following parameters: "
        "hinge_side: {}, swing_direction: {}, handle_type: {}. "
        "Given the robot's ego view (the perspective from the robot's vantage point) and the previous parameters, please update any parameters that need changing so the robot can try again. "
        "Please provide the updated parameters in a JSON format and include your reasoning. ".format(
            parameters.hinge_side, parameters.swing_direction, parameters.handle_type
        )
    )

    print(prompt)
    input("Does this seem right?")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "Assign the parameters correctly given the image.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_side_by_side}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        response_format=OpenDoorParams,
    )

    VLM_parameters = completion.choices[0].message.parsed
    return VLM_parameters


def save_images_constant_rate(spot, object_name, folder_name, rate=1.0):
    """
    Save images from the robot at a constant rate.
    :param spot: Spot object
    :param rate: Rate in Hz
    """
    # Set the rate
    rate = 1.0 / rate

    # Make directory called object_name under folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(os.path.join(folder_name, object_name)):
        os.makedirs(os.path.join(folder_name, object_name))
    folder_name = os.path.join(folder_name, object_name)

    idx = 0

    # Start saving images
    while True:
        print("Saving images at index {}".format(idx))
        # Get the image
        fl_img_response, fl_img = spot.get_image_RGB(view="frontleft_fisheye_image")
        fr_img_response, fr_img = spot.get_image_RGB(view="frontright_fisheye_image")
        # RGB to BGR
        stitched_image = spot.get_stitched_image_RGB(
            fl_img_response, fr_img_response, crop_image=True
        )
        stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
        # Save the image
        cv2.imwrite(
            f"{folder_name}/fl_img_{idx}.jpg",
            cv2.rotate(fl_img, cv2.ROTATE_90_CLOCKWISE),
        )
        cv2.imwrite(
            f"{folder_name}/fr_img_{idx}.jpg",
            cv2.rotate(fr_img, cv2.ROTATE_90_CLOCKWISE),
        )
        cv2.imwrite(f"{folder_name}/stitched_img_{idx}.jpg", stitched_image)

        # Wait for the next frame
        time.sleep(rate)

        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake", type=bool, default=False, help="Use a fake Spot robot for testing"
    )
    parser.add_argument("--ip", type=str, default="192.168.80.3")
    parser.add_argument("--username", type=str, default="user")
    parser.add_argument("--password", type=str, default="password")
    parser.add_argument(
        "-t", "--timeout", default=5, type=float, help="Timeout in seconds"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debug-level messages"
    )

    args = parser.parse_args()

    if args.fake:
        spot_init_pose2d = np.array([0, 0, 0, 0])
        spot = FakeSpot(
            username=args.username, password=args.password, init_pose=spot_init_pose2d
        )
    else:
        spot = Spot(ip=args.ip, username=args.username, password=args.password)

    print(f"Spot ID: {spot.id}")

    # spot.set_estop()
    # spot.take_lease()
    spot.robot.power_on(timeout_sec=20)
    spot.robot.time_sync.wait_for_sync()
    # spot.stand()
    # stow_arm(spot)

    # _run_open_door_test(spot, yoloworld_model_path)
    # _run_walking_test(spot)
    # _run_gaze_test(spot)
    # _run_place_test(spot)
    # _run_traj_test(spot)
    _run_grasp_test(spot)
    # _run_segment_test(spot)
    # _run_YOLOWorld_test(spot)

    spot.stand()
    spot.sit()
    # spot.safe_power_off()
