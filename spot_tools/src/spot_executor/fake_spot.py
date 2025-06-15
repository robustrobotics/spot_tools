import threading
import time

import cv2
import numpy as np
from bosdyn.api import robot_state_pb2
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot, SE3Pose
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
)

from spot_executor.bad_proto_mock import FakeFeedbackWrapper


class FakeImageClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot

    class ImageSource:
        def __init__(self, name):
            self.name = name

    def list_image_sources(self):
        print("Pretending to list image sources.")
        return [
            self.ImageSource(name="back_depth"),
            self.ImageSource(name="back_depth_in_visual_frame"),
            self.ImageSource(name="back_fisheye_image"),
            self.ImageSource(name="frontleft_depth"),
            self.ImageSource(name="frontleft_depth_in_visual_frame"),
            self.ImageSource(name="frontleft_fisheye_image"),
            self.ImageSource(name="frontright_depth"),
            self.ImageSource(name="frontright_depth_in_visual_frame"),
            self.ImageSource(name="frontright_fisheye_image"),
            self.ImageSource(name="hand_color_image"),
            self.ImageSource(name="hand_color_in_hand_depth_frame"),
            self.ImageSource(name="hand_depth"),
            self.ImageSource(name="hand_depth_in_hand_color_frame"),
            self.ImageSource(name="hand_image"),
            self.ImageSource(name="left_depth"),
            self.ImageSource(name="left_depth_in_visual_frame"),
            self.ImageSource(name="left_fisheye_image"),
            self.ImageSource(name="right_depth"),
            self.ImageSource(name="right_depth_in_visual_frame"),
            self.ImageSource(name="right_fisheye_image"),
        ]


class FakeManipulationAPIClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot


class FakeStateClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot

    def get_robot_state(self, **kwargs):
        identity_p = SE3Pose()
        p = SE3Pose(position={"x": 1})

        # Set up the frame tree snapshot so that odom is the root and body is a child of odom, translated by 1m in x
        edge_odom = FrameTreeSnapshot.ParentEdge(
            parent_frame_name="", parent_tform_child=identity_p
        )
        edge_body = FrameTreeSnapshot.ParentEdge(
            parent_frame_name=ODOM_FRAME_NAME, parent_tform_child=p
        )

        snapshot = FrameTreeSnapshot(
            child_to_parent_edge_map={
                ODOM_FRAME_NAME: edge_odom,
                BODY_FRAME_NAME: edge_body,
            }
        )

        ks = robot_state_pb2.KinematicState(transforms_snapshot=snapshot)
        return robot_state_pb2.RobotState(kinematic_state=ks)


class FakeCommandClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot

    def robot_command(
        self, command, end_time_secs=None, timesync_endpoint=None, lease=None, **kwargs
    ):
        # lease=None, command=None, end_time_secs=None):
        print("Spot would execute command with params:")
        print(f"\tlease: {lease}")
        print(f"\tcommand: {command}")
        print(f"\tend_time_secs: {end_time_secs}")

        move_cmd = command.synchronized_command.HasField("mobility_command")
        if move_cmd:
            x = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.position.x
            y = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.position.y
            angle = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.angle

            z = self.fake_spot.get_pose()[2]
            self.fake_spot.set_pose((x, y, z, angle))
            self.fake_spot.moving = True
            self.fake_spot.last_move_command = time.time()

        time.sleep(0.5)

    def robot_command_feedback(self, cmd_id):
        print("Spot would return command feedback for cmd_id ", cmd_id)
        return FakeFeedbackWrapper()


class FakeTimeSync:
    def wait_for_sync(self):
        return True


class FakeRobot:
    def __init__(self, fake_spot):
        self.time_sync = FakeTimeSync()
        self.fake_spot = fake_spot

    def ensure_client(self, service_name):
        print(f"Pretending that service {service_name} exists.")
        if service_name == "robot-state":
            return FakeStateClient(self.fake_spot)
        elif service_name == "robot-command":
            return FakeCommandClient(self.fake_spot)
        else:
            raise ValueError(f"Unknown service name: {service_name}")

    def power_on(self, timeout_sec=None):
        print("Pretending to power on the fake robot.")


class FakeSpot:
    def __init__(
        self,
        username="",
        password="",
        init_pose=None,
        semantic_model_path=None,
    ):
        print("Initialized Fake Spot!")
        self.is_fake = True
        self.pose_lock = threading.Lock()
        self.vel_lock = threading.Lock()
        self.robot = FakeRobot(self)

        assert len(init_pose) == 4, "Spot pose is 4D: x,y,z,yaw"
        self.pose = init_pose

        self.state_client = FakeStateClient(self)
        self.manipulation_api_client = FakeManipulationAPIClient(self)
        self.image_client = FakeImageClient(self)

        self.moving = False
        self.last_move_command = time.time()

        self.cmd_vel_linear = np.zeros(3)
        self.cmd_vel_angular = np.zeros(3)

        self.id = "fake_spot_id"

    def step(self, dt):
        self.update_velocity_control(dt)

    def aquire_lease(self):
        print("Acquiring Lease")

    def take_lease(self):
        print("Taking Lease")

    def set_estop(self, name="my_estop", timeout=9.0):
        print(f"Setting estop {name} for {timeout} s")

    def update_velocity_control(self, dt):
        with self.vel_lock:
            vx, vy, vz = self.cmd_vel_linear
            _, _, dtheta = self.cmd_vel_angular

        cur_pose = self.get_pose()
        theta = cur_pose[3]
        dp = np.zeros(4)

        dp[0] = vx * np.cos(theta) + vy * np.sin(theta)
        dp[1] = vx * np.sin(theta) - vy * np.cos(theta)
        dp[2] = vz
        dp[3] = dtheta
        self.set_pose(cur_pose + dp * dt)

    def set_vel(self, v_linear, v_angular):
        with self.vel_lock:
            self.cmd_vel_linear = v_linear
            self.cmd_vel_angular = v_angular

    def get_pose(self):
        with self.pose_lock:
            return self.pose

    def set_pose(self, pose):
        with self.pose_lock:
            self.pose = pose

    def get_state(self):
        raise NotImplementedError(
            "get_state not implemented for FakeSpot (what is it supposed to return?)"
        )

    def get_image_RGB(self, view="hand_color_image", show=False):
        return self.get_image(view=view, show=show)

    def get_image_alt(self, view="hand_depth_image", show=False):
        return self.get_image(view=view, show=show)

    def get_image(self, view="hand_color_image", show=False):
        img = cv2.imread("/home/rrg/data/images/bag_image.jpg")

        return None, img

    def segment_image(
        self, image, model_path=None, rotate=0, class_name="bag", show=False
    ):
        raise NotImplementedError("segment_image not implemented for FakeSpot")

    def get_joint_states(self):
        joint_to_state = {}

        omg_hip = 2
        center_h2 = 1
        amp_h2 = 0.2
        center_k = -1.9
        amp_k = 0.3

        vl = np.linalg.norm(self.cmd_vel_linear)
        va = np.linalg.norm(self.cmd_vel_angular)

        if time.time() - self.last_move_command > 1:
            self.moving = False
        moving = 1 if vl > 0 or va > 0 or self.moving else 0

        t = time.time()

        joint_to_state["front_left_hip_x"] = 0
        joint_to_state["front_left_hip_y"] = center_h2 + moving * amp_h2 * np.cos(
            omg_hip * t
        )
        joint_to_state["front_left_knee"] = center_k + moving * amp_k * np.cos(
            omg_hip * t + np.pi / 2
        )

        joint_to_state["rear_left_hip_x"] = 0
        joint_to_state["rear_left_hip_y"] = center_h2 + moving * amp_h2 * np.cos(
            omg_hip * t + np.pi
        )
        joint_to_state["rear_left_knee"] = center_k + moving * amp_k * np.cos(
            omg_hip * t + 3 * np.pi / 2
        )

        joint_to_state["front_right_hip_x"] = 0
        joint_to_state["front_right_hip_y"] = center_h2 + moving * amp_h2 * np.cos(
            omg_hip * t + np.pi
        )
        joint_to_state["front_right_knee"] = center_k + moving * amp_k * np.cos(
            omg_hip * t + 3 * np.pi / 2
        )

        joint_to_state["rear_right_hip_x"] = 0
        joint_to_state["rear_right_hip_y"] = center_h2 + moving * amp_h2 * np.cos(
            omg_hip * t
        )
        joint_to_state["rear_right_knee"] = center_k + moving * amp_k * np.cos(
            omg_hip * t + np.pi / 2
        )

        return joint_to_state

    def stand(self):
        print("Simulating Spot standing up.")

    def sit(self):
        print("Simulating Spot sitting down.")

    def pitch_up(self):
        print("Simulating Spot pitching up.")
