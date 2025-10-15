import threading
import time
import os

import cv2
import numpy as np

import rclpy
import rclpy.time
import spot_executor as se
import tf2_ros
import tf_transformations
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface.mid_level_planner import MidLevelPlanner, IdentityPlanner, OccupancyMap

from robot_executor_interface_ros.action_descriptions_ros import from_msg
from robot_executor_msgs.msg import ActionSequenceMsg, ActionMsg
from ros_system_monitor_msgs.msg import NodeInfoMsg
from sensor_msgs.msg import Image
from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot
from spot_skills.detection_utils import YOLODetector
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


from spot_tools_ros.fake_spot_ros import FakeSpotRos
from spot_tools_ros.occupancy_grid_ros_updater import OccupancyGridROSUpdater
from spot_tools_ros.utils import waypoints_to_path, pose_to_homo, get_tf_pose


def load_inverse_semantic_id_map_from_label_space(fn):
    with open(fn, "r") as fo:
        labelspace_yaml = yaml.safe_load(fo)
    label_to_id = {e["name"]: e["label"] for e in labelspace_yaml["label_names"]}
    return label_to_id


def pt_to_marker(pt, ns, mid, color, fid="vision"):
    m = Marker()
    m.header.frame_id = fid
    m.header.stamp = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()
    m.ns = ns
    m.id = mid
    m.type = m.SPHERE
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x = pt.x
    m.pose.position.y = pt.y
    m.scale.x = 0.6
    m.scale.y = 0.6
    m.scale.z = 0.6
    m.color.a = 1.0
    m.color.r = float(color[0])
    m.color.g = float(color[1])
    m.color.b = float(color[2])

    return m


def build_progress_markers(current_point, target_point, frame):
    ma = MarkerArray()
    m1 = pt_to_marker(current_point, "path_progress", 0, [0, 1, 1], fid=frame)
    ma.markers.append(m1)
    m2 = pt_to_marker(target_point, "path_progress", 1, [1, 0, 1], fid=frame)
    ma.markers.append(m2)
    return ma

def build_marker(pt, ns, frame, mid=0, color=[1,0,0]):
    ma = MarkerArray()
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()
    m.ns = ns
    m.id = mid
    m.type = m.SPHERE
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.pose.position.x = pt[0]
    m.pose.position.y = pt[1]
    m.scale.x = 0.3
    m.scale.y = 0.3
    m.scale.z = 0.3
    m.color.a = 1.0
    m.color.r = float(color[0])
    m.color.g = float(color[1])
    m.color.b = float(color[2])
    ma.markers.append(m)
    return ma


class RosFeedbackCollector:
    def __init__(self, odom_frame):
        self.pick_confirmation_event = threading.Event()
        self.pick_confirmation_response = False

        self.break_out_of_waiting_loop = False
        self.odom_frame = odom_frame

    def bounding_box_detection_feedback(
        self, annotated_img, centroid_x, centroid_y, semantic_class
    ):
        bridge = CvBridge()

        if centroid_x is not None and centroid_y is not None:
            label = f"{semantic_class}"
            cv2.putText(
                annotated_img,
                label,
                (centroid_x - 100, centroid_y - 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 0, 225),
                20,
            )

            # Label the centroid
            cv2.drawMarker(
                annotated_img,
                (centroid_x, centroid_y),
                (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=200,
                thickness=30,
            )

        annotated_img_msg = bridge.cv2_to_imgmsg(annotated_img, encoding="passthrough")
        self.annotated_img_pub.publish(annotated_img_msg)

        self.pick_confirmation_event.clear()

        # Wait until input is received and self.pick_confirmation_response is set
        while (
            not self.break_out_of_waiting_loop
            and not self.pick_confirmation_event.is_set()
        ):
            self.logger.info("Waiting for user to confirm pick action...")
            self.pick_confirmation_event.wait(timeout=5)

        if self.break_out_of_waiting_loop:
            self.logger.info("ROBOT WAS PREEMPTED")
            self.pick_confirmation_response = False

        # This boolean determines whether the executor keeps going
        return self.pick_confirmation_response

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
            build_progress_markers(progress_point, target_point, self.odom_frame)
        )
    
    def path_follow_MLP_feedback(self, path, target_point_metric):        
        self.mlp_path_publisher.publish(waypoints_to_path(self.odom_frame, path)) # TODO: parameterize frame name
        target_point_metric_flattened = [p[0] for p in target_point_metric]
        self.mlp_target_publisher.publish(
            build_marker(target_point_metric_flattened, "projected target point", self.odom_frame)
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

        self.mlp_path_publisher = node.create_publisher(
            Path, "~/mlp_path_publisher", qos_profile=latching_qos
        )

        self.mlp_target_publisher = node.create_publisher(
            MarkerArray, "~/mlp_target_publisher", qos_profile=latching_qos
        )


        self.annotated_img_pub = node.create_publisher(
            Image, "~/annotated_image", qos_profile=latching_qos
        )

        node.create_subscription(
            Bool,
            "~/pick_confirmation",
            self.pick_confirmation_callback,
            10,
        )

    def pick_confirmation_callback(self, msg):
        if msg.data:
            self.logger.info("Detection is valid. Continuing pick action!")
            self.pick_confirmation_response = True
        else:
            self.logger.warn("Detection is invalid. Discontinuing pick action.")
            self.pick_confirmation_response = False

        self.pick_confirmation_event.set()


class SpotExecutorRos(Node):
    def __init__(self):
        super().__init__("spot_executor_ros")
        self.debug = False
        self.background_thread = None

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
        
        self.declare_parameter("occupancy_frame", "")
        self.occupancy_frame = self.get_parameter("occupancy_frame").value
        self.get_logger().info(f"{self.occupancy_frame=}")

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
        self.declare_parameter("detector_model_path", "")
        detector_model_path = self.get_parameter("detector_model_path").value
        # semantic_model_path = self.get_parameter("semantic_model_path").value
        # labelspace_path = self.get_parameter("labelspace_path").value
        # semantic_name_to_id = load_inverse_semantic_id_map_from_label_space(
        #    labelspace_path
        # )
        # labelspace_grouping_path = self.get_parameter("labelspace_grouping_path").value
        # with open(labelspace_grouping_path, "r") as f:
        #    grouping_info = yaml.safe_load(f)
        # turn list of dictionaries into single dictionary
        # self.labelspace_map = {}
        # offset = int(grouping_info["offset"])
        # for group in grouping_info["groups"]:
        #    self.labelspace_map[group["name"]] = [g + offset for g in group["labels"]]

        self.declare_parameter("odom_frame", "")
        odom_frame = self.get_parameter("odom_frame").value
        assert odom_frame != ""
        self.odom_frame = odom_frame
        
        self.declare_parameter("body_frame", "")
        body_frame = self.get_parameter("body_frame").value
        assert body_frame != ""
        self.body_frame = body_frame

        self.feedback_collector = RosFeedbackCollector(self.odom_frame)
        self.feedback_collector.register_publishers(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Robot Initialization
        self.declare_parameter("use_fake_spot_interface", False)
        use_fake_spot_interface = self.get_parameter("use_fake_spot_interface").value
        self.declare_parameter("use_mid_level_planner", False)
        use_mid_level_planner = self.get_parameter("use_mid_level_planner").value
        self.get_logger().info(f"{use_mid_level_planner=}")
        
        # mid-level planner parameters
        self.declare_parameter("lookahead_distance", 50)
        lookahead_distance = self.get_parameter("lookahead_distance").value
        assert lookahead_distance > 0
        self.declare_parameter("occupancy_inflation_radius", 0.5)
        occupancy_inflation_radius = self.get_parameter("occupancy_inflation_radius").value
        assert occupancy_inflation_radius > 0
        
        self.declare_parameter("use_fake_occupancy_map", False)
        use_fake_occupancy_map = self.get_parameter("use_fake_occupancy_map").value
        self.declare_parameter("use_fake_path_plan", False)
        use_fake_path_plan = self.get_parameter("use_fake_path_plan").value
        self.get_logger().info(f"{use_fake_occupancy_map=}, {use_fake_path_plan=}")

        # mid-level planner initialization
        mid_level_planner_type = "astar"
        # mid_level_planner_type = "identity"
        match mid_level_planner_type:
            case "astar":
                self.occupancy_map = OccupancyMap(self.feedback_collector, inflate_radius_meters=occupancy_inflation_radius)
                self.occupancy_map_updater = OccupancyGridROSUpdater(self, self.body_frame, self.odom_frame, self.occupancy_map, self.feedback_collector, self.tf_buffer)
                self.mid_level_planner = MidLevelPlanner(self.occupancy_map, self.feedback_collector, lookahead_distance_grid=lookahead_distance)
                self.get_logger().info("Using A* mid-level planner")
            case "identity":
                self.mid_level_planner = IdentityPlanner(self.feedback_collector)
            case _:
                raise ValueError(f"Invalid mid-level planner type {mid_level_planner_type}")

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

        # <spot_vision_frame> must get mapped to the TF frame corresponding
        # to Spot's vision odom estimate.
        special_tf_remaps = {"<spot_vision_frame>": odom_frame}

        def tf_lookup_fn(parent, child):
            if parent in special_tf_remaps:
                parent = special_tf_remaps[parent]
            if child in special_tf_remaps:
                child = special_tf_remaps[child]
            try:
                return get_tf_pose(self.tf_buffer, parent, child)
            except tf2_ros.TransformException as e:
                self.get_logger.warn(f"Failed to get transform: {e}")
        self.tf_lookup_fn = tf_lookup_fn # TODO: use this to test transformation
        
        detector = YOLODetector(
            self.spot_interface,
            yolo_world_path=detector_model_path,
        )

        self.spot_executor = se.SpotExecutor(
            self.spot_interface,
            detector,
            tf_lookup_fn,
            self.mid_level_planner,
            follower_lookahead,
            goal_tolerance,
            self.feedback_collector, 
            use_fake_path_plan,
        )

        self.action_sequence_sub = self.create_subscription(
            ActionSequenceMsg,
            "~/action_sequence_subscriber",
            self.process_action_sequence,
            10,
        )

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )
            
    def hb_callback(self):
        msg = NodeInfoMsg()
        msg.nickname = "spot_executor"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = NodeInfoMsg.NOMINAL
        msg.notes = self.status_str
        self.heartbeat_pub.publish(msg)
            
    def process_action_sequence(self, msg):
        def process_sequence():
            self.status_str = "Processing action sequence"
            self.get_logger().info("Starting action sequence")
            sequence = from_msg(msg)
            self.spot_executor.process_action_sequence(
                sequence, self.feedback_collector
            )
            # TODO: pointer to the occupancy grid inside spot_executor
            self.get_logger().info("Finished execution action sequence.")
            self.status_str = "Idle"

        if self.background_thread is not None and self.background_thread.is_alive():
            self.spot_executor.terminate_sequence(self.feedback_collector)

        self.feedback_collector.break_out_of_waiting_loop = False
        self.background_thread = threading.Thread(target=process_sequence, daemon=False)
        self.background_thread.start()

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
