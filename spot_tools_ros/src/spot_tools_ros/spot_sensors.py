#!/usr/bin/env python3
"""ROS Python Package for Hydra-ROS."""

import geometry_msgs.msg
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import std_msgs.msg
import tf2_ros

from rclpy.node import Node
import rclpy
import os

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request


SpotImage = image_pb2.Image
SpotPixelFormat = image_pb2.Image.PixelFormat


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

    return rclpy.Time(int(local_s), local_ns)


def _build_header(stamp, frame_id):
    header = std_msgs.msg.Header()
    header.stamp = stamp
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
    msg.K[0] = intrinsics.focal_length.x
    msg.K[2] = intrinsics.principal_point.x
    msg.K[4] = intrinsics.focal_length.y
    msg.K[5] = intrinsics.principal_point.y
    msg.P[0] = intrinsics.focal_length.x
    msg.P[2] = intrinsics.principal_point.x
    msg.P[5] = intrinsics.focal_length.y
    msg.P[6] = intrinsics.principal_point.y
    return msg


def _build_transform_msg(robot, shot, child_frame, transform):
    pose = transform.parent_tform_child

    msg = geometry_msgs.msg.TransformStamped()
    msg.header.stamp = _get_local_time(robot, shot.acquisition_time)
    msg.header.frame_id = transform.parent_frame_name
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

    def __init__(self, node, name):
        self.name = name
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

        color_topic = f"spot/{name}/color/image_raw"
        depth_topic = f"spot/{name}/depth/image_rect"
        color_info_topic = f"spot/{name}/color/camera_info"
        depth_info_topic = f"spot/{name}/depth/camera_info"

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

    def _publish(self, logger, robot, color, depth):
        if not self._has_work():
            return

        color_stamp = _get_local_time(robot, color.shot.acquisition_time)
        if self._last_time is not None and self._last_time == color_stamp:
            return  # skip previously published images

        self._last_time = color_stamp
        header = _build_header(color_stamp, self._name)

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


class SpotClientNode(Node):
    """Spot client for requesting (rectified) RGBD data."""

    def __init__(self):
        """Make a camera client."""
        super().__init__("spot_client_node")
        robot_ip = self._get_param("robot.ip", "192.168.80.3").string_value
        setup_logging = self._get_param("robot.setup_logging", True).bool_value
        should_retry = self._get_param("robot.should_retry", False).bool_value
        should_connect = self._get_param("connect_to_robot", False).bool_value
        poll_period_s = self._get_param("poll_period_s", 0.05).double_value
        names = self._get_param("cameras", ["frontleft", "frontright"]).string_array_value

        self._cameras = {}
        for camera in names:
            self._cameras[camera] = CameraPublisher(self, camera)

        if should_connect:
            self._robot = self._connect(robot_ip, setup_logging, should_retry)
            self._client = self._robot.ensure_client(ImageClient.default_service_name)
        else:
            self._robot = None
            self._client = None

        # TODO(nathan) configurable exluded frames
        self._excluded_frames = ["vision", "odom", "body"]
        self._tfs = []
        self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._timer = self.create_timer(poll_period_s, self._callback)

    def _get_param(self, name, default):
        return _get_param(self, name, default)

    def _connect(self, robot_ip, setup_logging, should_retry):
        # uses node name to seed spot client
        name = self.get_name().replace("/", "_")
        if name[0] == "_":
            name = name[1:]

        sdk = bosdyn.client.create_standard_sdk(name)
        robot = sdk.create_robot(robot_ip)

        if setup_logging:
            bosdyn.client.util.setup_logging()

        env_prefix = self._get_param("env_prefix", "ADT4_BOSDYN").string_value
        def _get_login_info():
            pass_env = f"{env_prefix}_PASSWORD"
            user_env = f"{env_prefix}_USERNAME"
            if user_env not in os.environ or pass_env not in os.environ:
                return None, None

            return os.environ[user_env], os.environ[pass_env]

        while True:
            try:
                bosdyn.client.util.authenticate(robot, _get_login_info)
                return
            except Exception as e:
                self.get_logger().error(e)
                if not should_retry:
                    raise e

        robot.time_sync.wait_for_sync(10)
        return robot

    def _publish_transforms(self, shot):
        transform_map = shot.transforms_snapshot.child_to_parent_edge_map
        for frame in transform_map:
            if frame in self._excluded_frames:
                continue

            transform = transform_map.get(frame)
            parent_frame = transform.parent_frame_name
            existing = [(t.header.frame_id, t.child_frame_id) for t in self._tfs]
            if (parent_frame, frame) in existing:
                continue

            self._tfs.append(_build_transform_msg(self._robot, shot, frame, transform))
            self._tf_broadcaster.sendTransform(self._tfs)

    def _callback(self):
        """Poll Spot for new image messages and publish."""
        names = []
        requests = []
        for name, camera in self._cameras.items():
            requests += camera.requests
            names.append(name)

        if self._client is not None:
            responses = self._client.get_image(requests)
        else:
            print(requests)
            responses = []

        # shots = [resp.shot for resp in responses]
        for idx, name in enumerate(names):
            cam = self._cameras[name]
            rgb = responses[2 * idx]
            depth = responses[2 * idx + 1]
            cam.publish(self.get_logger(), self._robot, rgb, depth)


def main(args=None):
    """Start client and node."""
    rclpy.init(args=args)
    try:
        node = SpotClientNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
