import argparse
import numpy as np 
import time 
from ultralytics import YOLOWorld
import cv2 

from bosdyn.client import math_helpers
from bosdyn.api.spot import door_pb2
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME
)


from spot_executor.spot import Spot
from spot_skills.arm_utils import (
    move_hand_to_relative_pose, 
    close_gripper,
    open_gripper,
    stow_arm, 
    gaze_at_relative_pose
    )
from spot_skills.navigation_utils import (
    navigate_to_relative_pose,
    follow_trajectory,
)
from spot_skills.grasp_utils import (
    object_grasp
)

from spot_skills.door_utils import (
    execute_open_door
)

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
        (relative_up, "Looking up"),
        (math_helpers.Vec3(x=0, y=5, z=0), "Looking left"),
        (math_helpers.Vec3(x=0, y=-5, z=0), "Looking right"),
        # (math_helpers.Vec3(x=0, y=-0.5, ), "Looking -dy"),
        # (math_helpers.Vec3(x=0.5, y=0, ), "Looking dx"),
        # (math_helpers.Vec3(x=-0.5, y=0, ), "Looking -dx"),
        # (math_helpers.Vec3(x=0, y=0, ), "Looking yaw"),
        # (math_helpers.Vec3(x=0, y=0, ), "Looking -yaw"),
    ]
    for relative_pose, msg in relative_poses:
        print(msg)
        spot.gaze_at_relative_pose(relative_pose)
        input("Press enter when ready to move on")
        # time.sleep(0.5)


def _run_traj_test(spot, frame=VISION_FRAME_NAME, stairs=False) -> None:
    relative_poses = [
        math_helpers.SE2Pose(x=0, y=0, angle=0),
        math_helpers.SE2Pose(x=1.0, y=-0.1, angle=0),
        math_helpers.SE2Pose(x=2.0, y=0.0, angle=0),
        math_helpers.SE2Pose(x=2.0, y=1.0, angle=0),
        math_helpers.SE2Pose(x=5.0, y=1.3, angle=0),
        math_helpers.SE2Pose(x=7.0, y=1.0, angle=180),
        # math_helpers.SE2Pose(x=-1, y=-0.5, angle=0),
        # math_helpers.SE2Pose(x=1, y=0.5, angle=np.pi),
        # math_helpers.SE2Pose(x=1, y=0.5, angle=-np.pi)
    ]
    waypoints_list = []
    current_pose = spot.get_pose()
    print(current_pose)
    for relative_pose in relative_poses:
        pose = current_pose * relative_pose
        print(pose)
        waypoint = [pose.x, pose.y, pose.angle]
        waypoints_list.append(waypoint)
    print(waypoints_list)
    from spot_skills.bezier_path import plot_curves, smooth_path

    path = smooth_path(waypoints_list, heading_mode="average", n_points=10)
    plot_curves(path, np.array(waypoints_list))

    try:
        return follow_trajectory(spot, waypoints_list, frame_name=frame, stairs=stairs)
    finally:
        # Send a Stop at the end,
        # Send a Stop at the end, regardless of what happened.
        # robot_command_client.robot_command(RobotCommandBuilder.stop_command())
        pass


def _run_grasp_test(spot) -> None:
    open_gripper(spot)
    relative_pose = math_helpers.Vec3(x=1, y=0, z=0)
    gaze_at_relative_pose(spot, relative_pose)
    time.sleep(0.2)

    object_grasp(
        spot,
        image_source="hand_color_image",
        user_input=False,
        #  semantic_model_path='data/models/efficientvit_seg_l2.onnx',
        semantic_class="bag",
        grasp_constraint=None,
    )

    open_gripper(spot)
    stow_arm(spot)
    close_gripper(spot)

    pass


def _run_segment_test(spot) -> None:
    open_gripper(spot)
    relative_pose = math_helpers.Vec3(x=1, y=0, z=0)
    gaze_at_relative_pose(spot, relative_pose)
    time.sleep(0.2)

    image, img = spot.get_image_alt(view="hand_color_image", show=True)
    segmented_image = spot.segment_image(img, show=True)


def _run_open_door_test(spot, model_path, max_tries) -> None:
    print("Opening the door...")

    trial_idx = 0 
    parameters = None

    while trial_idx < max_tries: 
        feedback_status = execute_open_door(spot, model_path, parameters)
        if feedback_status == door_pb2.DoorCommand.Feedback.STATUS_COMPLETED: 
            # The robot was successful in opening the door.
            break 
        else: 
            # Ask a VLM for new parameters and try again
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.80.3")
    parser.add_argument("--username", type=str, default="user")
    parser.add_argument("--password", type=str, default="password")
    parser.add_argument("-t", "--timeout", default=5, type=float, help="Timeout in seconds")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debug-level messages"
    )

    args = parser.parse_args()

    spot = Spot(username=args.username, password=args.password)
    print(spot.id)
    # assert False
    # spot.set_estop()
    # spot.take_lease()
    spot.robot.power_on(timeout_sec=20)
    spot.robot.time_sync.wait_for_sync()

    yoloworld_model_path = "/home/aaron/spot_tools/data/models/yolov8x-worldv2-door.pt"
    max_tries = 3
    _run_open_door_test(spot, yoloworld_model_path, max_tries)
    # _run_walking_test(spot)
    # _run_gaze_test(spot)
    # _run_traj_test(spot)
    # _run_grasp_test(spot)
    # _run_segment_test(spot)
    # spot.pitch_up()
    # print(look_for_object(spot, 'bag'))

    time.sleep(1)

    spot.stand()
    # spot.sit()
    # spot.sit()
    # spot.safe_power_off()
