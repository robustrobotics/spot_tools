#!/usr/bin/env python3
"""ROS Python Package for Hydra-ROS."""

import queue
import re
import threading

import bosdyn.client
import bosdyn.client.util
import geometry_msgs.msg
import rclpy
import std_msgs.msg
import tf2_ros
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_state import RobotStateClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState

SpotImage = image_pb2.Image
SpotPixelFormat = image_pb2.Image.PixelFormat


# borrowed from: https://github.com/heuristicus/spot_ros
JOINT_NAMES = {
    "fl.hx": "front_left_hip_x",
    "fl.hy": "front_left_hip_y",
    "fl.kn": "front_left_knee",
    "fr.hx": "front_right_hip_x",
    "fr.hy": "front_right_hip_y",
    "fr.kn": "front_right_knee",
    "hl.hx": "rear_left_hip_x",
    "hl.hy": "rear_left_hip_y",
    "hl.kn": "rear_left_knee",
    "hr.hx": "rear_right_hip_x",
    "hr.hy": "rear_right_hip_y",
    "hr.kn": "rear_right_knee",
    "arm0.sh0": "arm_joint1",
    "arm0.sh1": "arm_joint2",
    "arm0.el0": "arm_joint3",
    "arm0.el1": "arm_joint4",
    "arm0.wr0": "arm_joint5",
    "arm0.wr1": "arm_joint6",
    "arm0.f1x": "arm_gripper",
}


STATIC_IDS = [
    "body",
    "frontright.*",
    "frontleft.*",
    "left.*",
    "right.*",
    "rear.*",
]


def _compile_regex(filters):
    if len(filters) == 0:
        # see https://stackoverflow.com/a/942122
        return re.compile("(?!)")
    else:
        return re.compile("|".join(filters))


def _prefix_frame(tf_prefix, frame_id):
    return frame_id if tf_prefix == "" else f"{tf_prefix}/{frame_id}"


def _get_param(node, name, default):
    node.declare_parameter(name, default)
    value = node.get_parameter(name).get_parameter_value()
    return value


def _get_local_time(robot, robot_stamp):
    local_s = robot_stamp.seconds - robot.time_sync.endpoint.clock_skew.seconds
    local_ns = robot_stamp.nanos - robot.time_sync.endpoint.clock_skew.nanos
    if local_ns < 0:
        local_s -= 1.0
        local_ns += int(1e9)

    if local_ns < 0:
        raise ValueError(f"Invalid stamp {local_s} [s] {local_ns} [ns]")

    return rclpy.time.Time(seconds=int(local_s), nanoseconds=local_ns)


def _build_header(stamp: rclpy.time.Time, frame_id: str) -> std_msgs.msg.Header:
    header = std_msgs.msg.Header()
    header.stamp = stamp.to_msg()
    header.frame_id = frame_id
    return header


def _build_image_msg(header, shot):
    compressed = shot.image.format == SpotImage.FORMAT_JPEG
    msg = CompressedImage() if compressed else Image()
    msg.header = header
    if compressed:
        msg.format = "rgb8; jpeg compressed bgr8"
        msg.data = shot.image.data
        return msg

    msg.height = shot.image.rows
    msg.width = shot.image.cols
    if shot.image.pixel_format == SpotPixelFormat.PIXEL_FORMAT_DEPTH_U16:
        msg.encoding = "16UC1"
        msg.step = shot.image.cols * 2
    elif shot.image.pixel_format == SpotPixelFormat.PIXEL_FORMAT_RGB_U8:
        msg.encoding = "rgb8"
        msg.step = 3 * shot.image.cols
    else:
        return None

    msg.data = shot.image.data
    return msg


def _build_info_msg(header, response):
    intrinsics = response.source.pinhole.intrinsics

    msg = CameraInfo()
    msg.header = header
    msg.height = response.shot.image.rows
    msg.width = response.shot.image.cols
    msg.distortion_model = "plumb_bob"
    msg.k[0] = intrinsics.focal_length.x
    msg.k[2] = intrinsics.principal_point.x
    msg.k[4] = intrinsics.focal_length.y
    msg.k[5] = intrinsics.principal_point.y
    msg.p[0] = intrinsics.focal_length.x
    msg.p[2] = intrinsics.principal_point.x
    msg.p[5] = intrinsics.focal_length.y
    msg.p[6] = intrinsics.principal_point.y
    return msg


def _build_transform_msg(stamp, child_frame, parent_frame, pose):
    msg = geometry_msgs.msg.TransformStamped()
    msg.header.stamp = stamp.to_msg()
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = pose.position.x
    msg.transform.translation.y = pose.position.y
    msg.transform.translation.z = pose.position.z
    msg.transform.rotation.x = pose.rotation.x
    msg.transform.rotation.y = pose.rotation.y
    msg.transform.rotation.z = pose.rotation.z
    msg.transform.rotation.w = pose.rotation.w
    return msg


class CameraPublisher:
    """Publisher for a single camera."""

    def __init__(self, node, name, tf_prefix):
        self.name = name
        self.tf_prefix = tf_prefix
        self.color_suffix = _get_param(
            node, f"{name}.color_suffix", "fisheye_image"
        ).string_value
        self.depth_suffix = _get_param(
            node, f"{name}.depth_suffix", "depth_in_visual_frame"
        ).string_value

        param = node.declare_parameter(f"{name}.quality", 100.0)
        self.quality = param.get_parameter_value().double_value

        param = node.declare_parameter(f"{name}.use_compressed", True)
        self.use_compressed = param.get_parameter_value().bool_value

        color_topic = f"{name}/color/image_raw"
        depth_topic = f"{name}/depth/image_rect"
        color_info_topic = f"{name}/color/camera_info"
        depth_info_topic = f"{name}/depth/camera_info"

        if self.use_compressed:
            color_topic = f"{color_topic}/compressed"
            self._rgb_pub = node.create_publisher(CompressedImage, color_topic, 10)
        else:
            self._rgb_pub = node.create_publisher(Image, color_topic, 10)

        self._depth_pub = node.create_publisher(Image, depth_topic, 10)
        self._rgb_info_pub = node.create_publisher(CameraInfo, color_info_topic, 10)
        self._depth_info_pub = node.create_publisher(CameraInfo, depth_info_topic, 10)
        self._last_time = None

    @property
    def requests(self):
        rgb_kwargs = {"pixel_format": SpotPixelFormat.PIXEL_FORMAT_RGB_U8}
        rgb_kwargs["image_format"] = (
            SpotImage.FORMAT_JPEG if self.use_compressed else SpotImage.FORMAT_RAW
        )
        if self.quality:
            rgb_kwargs["quality_percent"] = self.quality

        depth_kwargs = {
            "image_format": SpotImage.FORMAT_RAW,
            "pixel_format": SpotPixelFormat.PIXEL_FORMAT_DEPTH_U16,
        }

        return [
            build_image_request(f"{self.name}_{self.color_suffix}", **rgb_kwargs),
            build_image_request(f"{self.name}_{self.depth_suffix}", **depth_kwargs),
        ]

    def _has_work(self):
        return (
            self._rgb_pub.get_subscription_count() > 0
            or self._rgb_info_pub.get_subscription_count() > 0
            or self._depth_pub.get_subscription_count() > 0
            or self._depth_info_pub.get_subscription_count() > 0
        )

    def publish(self, logger, robot, color, depth):
        if not self._has_work():
            return

        color_stamp = _get_local_time(robot, color.shot.acquisition_time)
        if self._last_time is not None and self._last_time == color_stamp:
            return  # skip previously published images

        self._last_time = color_stamp
        frame_name = _prefix_frame(self.tf_prefix, self.name)
        header = _build_header(color_stamp, frame_name)

        rgb_msg = _build_image_msg(header, color.shot)
        if rgb_msg is not None:
            self._rgb_pub.publish(rgb_msg)
            self._rgb_info_pub.publish(_build_info_msg(header, color))
        else:
            logger.logerr(f"Invalid pixel format: {color.shot.image.pixel_format}")

        depth_msg = _build_image_msg(header, depth.shot)
        if depth_msg is not None:
            self._depth_pub.publish(depth_msg)
            self._depth_info_pub.publish(_build_info_msg(header, depth))
        else:
            logger.logerr(f"Invalid pixel format: {depth.shot.image.pixel_format}")


def _check_seen(seen_dict, parent, child):
    if parent not in seen_dict:
        seen_dict[parent] = set([child])
        return True

    if child not in seen_dict[parent]:
        seen_dict[parent].add(child)
        return True

    return False


class SpotClientNode(Node):
    """Spot client for requesting (rectified) RGBD data."""

    def __init__(self):
        """Make a camera client."""
        super().__init__("spot_client_node")
        names = self._get_param(
            "cameras", ["frontleft", "frontright"]
        ).string_array_value

        self._robot = self._connect()
        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
        self._state_client = self._robot.ensure_client(
            RobotStateClient.default_service_name
        )

        self._tf_prefix = self._get_param("tf_prefix", "<ns>").string_value
        if self._tf_prefix == "<ns>":
            robot_ns = self.get_namespace()
            if robot_ns[0] == "/":
                robot_ns = robot_ns[1:]

            self._tf_prefix = robot_ns

        if len(self._tf_prefix) > 0 and self._tf_prefix[-1] == "/":
            self._tf_prefix = self._tf_prefix[:-2]

        self._cameras = {}
        for camera in names:
            self._cameras[camera] = CameraPublisher(self, camera, self._tf_prefix)

        allowed_parents = ["vision", "odom", "body"]
        self._parent_frame = self._get_param("parent_frame", "vision").string_value
        if self._parent_frame not in allowed_parents:
            err = f"Invalid parent '{self._parent_frame}'. Must be in {allowed_parents}"
            self.get_logger().error(err)

        excluded_frames = self._get_param("excluded_frames", []).string_array_value
        self._excluded_matcher = _compile_regex(excluded_frames)

        static_frames = self._get_param("static_frames", STATIC_IDS).string_array_value
        self._static_matcher = _compile_regex(static_frames)

        queue_size = self._get_param("max_tf_queue_size", 10).integer_value
        self._tf_queue = queue.Queue(queue_size)
        self._dynamic_pub = tf2_ros.TransformBroadcaster(self)
        self._static_pub = tf2_ros.StaticTransformBroadcaster(self)
        self._joint_pub = self.create_publisher(JointState, "joint_states", 10)

        self._tf_thread = threading.Thread(target=self._publish_transforms, daemon=True)
        self._tf_thread.start()

        cam_poll_period_s = self._get_param("camera_poll_period_s", 0.05).double_value
        self._camera_timer = self.create_timer(cam_poll_period_s, self._camera_callback)

        state_group = MutuallyExclusiveCallbackGroup()
        state_poll_period_s = self._get_param("state_poll_period_s", 0.01).double_value
        self._state_timer = self.create_timer(
            state_poll_period_s, self._state_callback, callback_group=state_group
        )

    def _get_param(self, name, default):
        return _get_param(self, name, default)

    def _connect(self):
        setup_logging = self._get_param("robot.setup_logging", True).bool_value
        should_retry = self._get_param("robot.should_retry", False).bool_value

        # uses node name to seed spot client
        name = self.get_name().replace("/", "_")
        if name[0] == "_":
            name = name[1:]

        robot_ip = self._get_param("robot.ip", "").string_value
        if robot_ip == "":
            raise ValueError("IP address required")

        username = self._get_param("robot.username", "").string_value
        if username == "":
            raise ValueError("Robot username required!")

        password = self._get_param("robot.password", "").string_value
        if password == "":
            raise ValueError("Robot password required!")

        sdk = bosdyn.client.create_standard_sdk(name)
        robot = sdk.create_robot(robot_ip)

        if setup_logging:
            bosdyn.client.util.setup_logging()

        def _get_login_info():
            return username, password

        while True:
            try:
                bosdyn.client.util.authenticate(robot, _get_login_info)
                break
            except Exception as e:
                self.get_logger().error(e)
                if not should_retry:
                    raise e

        robot.time_sync.wait_for_sync(10)
        return robot

    def _is_tf_static(self, child, parent):
        return self._static_matcher.match(child) and self._static_matcher.match(parent)

    def _publish_transforms(self):
        static_tfs = []
        static_seen = {}
        while rclpy.ok():
            new_static = False
            stamp, transforms, feet_pos = self._tf_queue.get()
            transform_map = transforms.child_to_parent_edge_map

            dynamic_tfs = []
            dynamic_seen = {}
            if feet_pos is not None:
                for name, pos in feet_pos.items():
                    msg = geometry_msgs.msg.TransformStamped()
                    msg.header.stamp = stamp.to_msg()
                    msg.header.frame_id = _prefix_frame(self._tf_prefix, "body")
                    msg.child_frame_id = _prefix_frame(self._tf_prefix, f"foot_{name}")
                    msg.transform.translation.x = pos.x
                    msg.transform.translation.y = pos.y
                    msg.transform.translation.z = pos.z
                    dynamic_tfs.append(msg)

            for frame in transform_map:
                if self._excluded_matcher.match(frame):
                    continue

                transform = transform_map.get(frame)
                parent_frame = transform.parent_frame_name
                if frame == "" or parent_frame == "":
                    continue

                pose = transform.parent_tform_child
                if frame == self._parent_frame:
                    pose = SE3Pose.from_proto(pose).inverse().to_proto()
                    parent_frame, frame = frame, parent_frame

                is_static = self._is_tf_static(frame, parent_frame)
                # check if repeated static frame
                if is_static and not _check_seen(static_seen, parent_frame, frame):
                    continue

                # check if repeated dynamic frame
                if not is_static and not _check_seen(dynamic_seen, parent_frame, frame):
                    continue

                frame = _prefix_frame(self._tf_prefix, frame)
                parent_frame = _prefix_frame(self._tf_prefix, parent_frame)
                msg = _build_transform_msg(stamp, frame, parent_frame, pose)

                if is_static:
                    new_static = True
                    self.get_logger().info(
                        f"New static TF {parent_frame}_T_{frame} @ {stamp.nanoseconds} [ns]"
                    )
                    static_tfs.append(msg)
                else:
                    dynamic_tfs.append(msg)

            if new_static:
                self._static_pub.sendTransform(static_tfs)

            self._dynamic_pub.sendTransform(dynamic_tfs)

    def _camera_callback(self):
        """Poll Spot for new image messages and publish."""
        names = []
        requests = []
        for name, camera in self._cameras.items():
            requests += camera.requests
            names.append(name)

        responses = self._image_client.get_image(requests)
        for resp in responses:
            stamp = _get_local_time(self._robot, resp.shot.acquisition_time)
            try:
                self._tf_queue.put_nowait((stamp, resp.shot.transforms_snapshot, None))
            except queue.Full:
                self.get_logger().warn(f"TF queue is full! Dropping TF @ {stamp.nanoseconds} [ns]")

        for idx, name in enumerate(names):
            cam = self._cameras[name]
            rgb = responses[2 * idx]
            depth = responses[2 * idx + 1]
            cam.publish(self.get_logger(), self._robot, rgb, depth)

    def _state_callback(self):
        """Poll Spot for new image messages and publish."""
        state = self._state_client.get_robot_state()
        pos_state = state.kinematic_state
        stamp = _get_local_time(self._robot, pos_state.acquisition_timestamp)

        feet_names = ["front_left", "front_right", "rear_left", "rear_right"]
        feet_pos = {
            feet_names[i]: foot.foot_position_rt_body
            for i, foot in enumerate(state.foot_state)
        }

        try:
            self._tf_queue.put_nowait((stamp, pos_state.transforms_snapshot, feet_pos))
        except queue.Full:
            self.get_logger().warn(f"TF queue is full! Dropping TF @ {stamp.nanoseconds} [ns]")


        msg = JointState()
        msg.header.stamp = stamp.to_msg()
        for joint in pos_state.joint_states:
            joint_name = JOINT_NAMES.get(joint.name)
            if joint_name is None:
                self.get_logger.debug(f"unknown joint '{joint.name}'")
                continue

            prefixed_joint_name = _prefix_frame(self._tf_prefix, joint_name)
            msg.name.append(prefixed_joint_name)
            msg.position.append(joint.position.value)
            msg.velocity.append(joint.velocity.value)
            msg.effort.append(joint.load.value)

        self._joint_pub.publish(msg)


def main(args=None):
    """Start client and node."""
    rclpy.init(args=args)
    try:
        node = SpotClientNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
