import time

import numpy as np
import rclpy
import rclpy.time
import spot_executor as se
import tf2_ros
import tf_transformations
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D

# from cv_bridge import CvBridge
from nav_msgs.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import from_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from ros_system_monitor_msgs.msg import NodeInfoMsg
from sensor_msgs.msg import Image
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot
from visualization_msgs.msg import Marker, MarkerArray

from spot_tools_ros.fake_spot_ros import FakeSpotRos
from spot_tools_ros.utils import waypoints_to_path


def get_robot_pose(tf_buffer, parent_frame: str, child_frame: str):
    """
    Looks up the transform from parent_frame to child_frame and returns [x, y, z, yaw].

    """
    # TODO: use Time(0) instead of now?
    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(
            parent_frame,
            child_frame,
            now,
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        transform = tf_buffer.lookup_transform(parent_frame, child_frame, now)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to Euler angles
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        return np.array([translation.x, translation.y, translation.z]), rotation

    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise


def load_inverse_semantic_id_map_from_label_space(fn):
    with open(fn, "r") as fo:
        labelspace_yaml = yaml.safe_load(fo)
    label_to_id = {e["name"]: e["label"] for e in labelspace_yaml["label_names"]}
    return label_to_id


def pt_to_marker(pt, ns, mid, color):
    m = Marker()
    m.header.frame_id = "vision"
    m.header.stamp = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()
    m.ns = ns
    m.id = mid
    m.type = m.SPHERE
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x = pt.x
    m.pose.position.y = pt.y
    m.scale.x = 0.3
    m.scale.y = 0.3
    m.scale.z = 0.3
    m.color.a = 1.0
    m.color.r = float(color[0])
    m.color.g = float(color[1])
    m.color.b = float(color[2])

    return m


def build_progress_markers(current_point, target_point):
    ma = MarkerArray()
    m1 = pt_to_marker(current_point, "path_progress", 0, [0, 1, 1])
    ma.markers.append(m1)
    m2 = pt_to_marker(target_point, "path_progress", 1, [1, 0, 1])
    ma.markers.append(m2)
    return ma


class RosFeedbackCollector:
    def pick_image_feedback(self, semantic_image, mask_image):
        bridge = CvBridge()
        semantic_hand_msg = bridge.cv2_to_imgmsg(semantic_image, encoding="passthrough")
        mask_img_msg = bridge.cv2_to_imgmsg(mask_image, encoding="passthrough")
        self.semantic_hand_pub.publish(semantic_hand_msg)
        self.semantic_mask_pub.publish(mask_img_msg)

    def follow_path_feedback(self, path):
        path_debug_viz = waypoints_to_path("vision", path)
        self.smooth_path_publisher.publish(path_debug_viz)

    def path_following_progress_feedback(self, progress_point, target_point):
        self.progress_point_pub.publish(
            build_progress_markers(progress_point, target_point)
        )

    def gaze_feedback(self, pose, gaze_point):
        pass

    def print(self, level, string):
        match level:
            case "DEBUG":
                log_fn = self.logger.debug
            case "INFO":
                log_fn = self.logger.info
            case "WARNING":
                log_fn = self.logger.warning
            case "ERROR":
                log_fn = self.logger.error
            case _:
                raise ValueError(f"Invalid log level {level}")
        log_fn(str(string))

    def feedback_viz_2(self, y):
        pass

    def register_publishers(self, node):
        self.logger = node.get_logger()

        # ROS 2 transient local QoS for "latching" behavior
        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Keeping the "~/" prefix notation for private topics in ROS 2
        self.smooth_path_publisher = node.create_publisher(
            Path, "~/smooth_path_publisher", qos_profile=latching_qos
        )

        self.progress_visualizer_pub = node.create_publisher(
            MarkerArray, "~/path_progress_visualizer", qos_profile=latching_qos
        )

        self.semantic_hand_pub = node.create_publisher(
            Image, "~/semantic_hand_image", qos_profile=latching_qos
        )

        self.semantic_mask_pub = node.create_publisher(
            Image, "~/semantic_mask_image", qos_profile=latching_qos
        )

        self.progress_point_pub = node.create_publisher(
            MarkerArray, "~/progress_point_visualizer", qos_profile=latching_qos
        )


class SpotExecutorRos(Node):
    def __init__(self):
        super().__init__("spot_executor_ros")
        self.debug = False

        self.feedback_collector = RosFeedbackCollector()
        self.feedback_collector.register_publishers(self)

        # Connectivity parameters
        self.declare_parameter("spot_ip", "")
        self.declare_parameter("bosdyn_client_username", "")
        self.declare_parameter("bosdyn_client_password", "")
        spot_ip = self.get_parameter("spot_ip").value
        assert spot_ip != ""
        bdai_username = self.get_parameter("bosdyn_client_username").value
        assert bdai_username != ""
        bdai_password = self.get_parameter("bosdyn_client_password").value
        assert bdai_password != ""

        # Follow Skill
        self.declare_parameter("follower_lookahead", 0.0)
        follower_lookahead = self.get_parameter("follower_lookahead").value
        assert follower_lookahead > 0

        self.declare_parameter("goal_tolerance", 0.0)
        goal_tolerance = self.get_parameter("goal_tolerance").value
        assert goal_tolerance > 0

        # Pick/Inspect Skill
        self.declare_parameter("semantic_model_path", "")
        self.declare_parameter("labelspace_path", "")
        self.declare_parameter("labelspace_grouping_path", "")
        # semantic_model_path = self.get_parameter("semantic_model_path").value
        # labelspace_path = self.get_parameter("labelspace_path").value
        # semantic_name_to_id = load_inverse_semantic_id_map_from_label_space(
        #    labelspace_path
        # )
        # labelspace_grouping_path = self.get_parameter("labelspace_grouping_path").value
        # with open(labelspace_grouping_path, "r") as f:
        #    grouping_info = yaml.safe_load(f)
        # turn list of dictionaries into single dictionary
        self.labelspace_map = {}
        # offset = int(grouping_info["offset"])
        # for group in grouping_info["groups"]:
        #    self.labelspace_map[group["name"]] = [g + offset for g in group["labels"]]

        # Robot Initialization
        self.declare_parameter("use_fake_spot_interface", False)
        use_fake_spot_interface = self.get_parameter("use_fake_spot_interface").value

        if use_fake_spot_interface:
            self.declare_parameter("fake_spot_external_pose", False)
            external_pose = self.get_parameter("fake_spot_external_pose").value

            self.declare_parameter("use_fake_spot_pose", False)
            if self.get_parameter("use_fake_spot_pose").value:
                self.declare_parameter("fake_spot_x", np.inf)
                self.declare_parameter("fake_spot_y", np.inf)
                self.declare_parameter("fake_spot_z", np.inf)
                self.declare_parameter("fake_spot_yaw", np.inf)
                spot_x = self.get_parameter("fake_spot_x").value
                spot_y = self.get_parameter("fake_spot_y").value
                spot_z = self.get_parameter("fake_spot_z").value
                spot_yaw = self.get_parameter("fake_spot_yaw").value

                spot_init_pose2d = np.array([spot_x, spot_y, spot_z, spot_yaw])
                assert not any(np.isinf(spot_init_pose2d)), (
                    "Must set fake_spot_x, fake_spot_y, fake_spot_z, fake_spot_yaw"
                )
            else:
                spot_init_pose2d = np.array([0, 0, 0, 0])

            self.get_logger().info(str(spot_init_pose2d))
            self.get_logger().info("About to initialize fake spot")
            self.spot_interface = FakeSpot(
                username=bdai_username,
                password=bdai_password,
                init_pose=spot_init_pose2d,
                semantic_model_path=None,
            )

            self.declare_parameter("odom_frame", "")
            odom_frame = self.get_parameter("odom_frame").value
            assert odom_frame != ""

            self.declare_parameter("body_frame", "")
            body_frame = self.get_parameter("body_frame").value
            assert body_frame != ""

            self.spot_ros_interface = FakeSpotRos(
                self,
                self.spot_interface,
                odom_frame,
                body_frame,
                external_pose=external_pose,
            )

        else:
            self.get_logger().info("About to initialize Spot")
            self.get_logger().info(f"{bdai_username=}, {bdai_password=}, {spot_ip=}")
            self.spot_interface = Spot(
                username=bdai_username,
                password=bdai_password,
                ip=spot_ip,
            )

        self.get_logger().info("Initialized!")
        self.status_str = "Idle"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # <spot_vision_frame> must get mapped to the TF frame corresponding
        # to Spot's vision odom estimate.
        special_tf_remaps = {"<spot_vision_frame>": odom_frame}

        def tf_lookup_fn(parent, child):
            if parent in special_tf_remaps:
                parent = special_tf_remaps[parent]
            if child in special_tf_remaps:
                child = special_tf_remaps[child]
            try:
                return get_robot_pose(self.tf_buffer, parent, child)
            except tf2_ros.TransformException as e:
                self.get_logger.warn(f"Failed to get transform: {e}")

        self.spot_executor = se.SpotExecutor(
            self.spot_interface, tf_lookup_fn, follower_lookahead, goal_tolerance
        )

        self.action_sequence_sub = self.create_subscription(
            ActionSequenceMsg,
            "~/action_sequence_subscriber",
            self.process_action_sequence,
            10,
        )

        if not use_fake_spot_interface:
            self.pose_pub = self.create_publisher(Pose2D, "~/spot_pose", 1)
            timer_period = 1.0  # seconds
            self.pose_timer = self.create_timer(timer_period, self.publish_pose)

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )

    def publish_pose(self):
        if self.spot_interface is None:
            self.get_logger().warn(
                "Spot interface not initialized, cannot publish pose."
            )
            return
        else:
            pose = self.spot_interface.get_pose()
            if pose is None:
                self.get_logger().warn("Spot interface returned None for pose.")
                return
        # msg = Pose2D(x=pose.x, y=pose.y, theta=pose.angle)
        msg = Pose2D(x=pose[0], y=pose[1], theta=pose[2])

        self.pose_pub.publish(msg)
        self.status_str = f"Publishing pose: {pose}"
        self.pose_pub.publish(msg)
        self.get_logger().info(f"Publishing: {msg}")

    def hb_callback(self):
        msg = NodeInfoMsg()
        msg.nickname = "spot_executor"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = NodeInfoMsg.NOMINAL
        msg.notes = self.status_str
        self.heartbeat_pub.publish(msg)

    def process_action_sequence(self, msg):
        self.status_str = "Processing action sequence"
        self.get_logger().info("Starting action sequence")
        sequence = from_msg(msg)
        self.spot_executor.process_action_sequence(sequence, self.feedback_collector)
        self.get_logger().info("Finished execution action sequence.")
        self.status_str = "Idle"


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SpotExecutorRos()

        ros_executor = MultiThreadedExecutor()
        ros_executor.add_node(node)

        try:
            ros_executor.spin()
        finally:
            ros_executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
