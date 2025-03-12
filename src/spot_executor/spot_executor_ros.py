import numpy as np
import tf2_ros
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import yaml
from nav_msgs.msg import Path
from phoenix_tamp_planner.action_descriptions import (
    ActionSequence,
)
from phoenix_tamp_planner.msg import ActionSequenceMsg
from sensor_msgs.msg import Image
from spot_tools.mapping.outdoor_dsg_utils import (
    load_inverse_semantic_id_map_from_label_space,
)
from visualization_msgs.msg import MarkerArray

from spot_executor.fake_spot import FakeSpot
from spot_executor.spot import Spot

class FeedbackCollector:
    def feedback_viz_1(self, x):
        pass
    def feedback_viz_2(self, y):
        pass

    def register_publishers(self, node):

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        self.smooth_path_publisher = node.create_publisher(
            "~/smooth_path_publisher", Path, 1, qos_profile=latching_qos
        )

        self.progress_visualizer_pub = node.create_publisher(
            "~/path_progress_visualizer", MarkerArray, 1, qos_profile=latching_qos)

        self.semantic_hand_pub = node.create_publisher(
            "~/semantic_hand_image", Image, 1, qos_profile=latching_qos)

        self.semantic_mask_pub = node.create_publisher(
            "~semantic_mask_image", Image, 1, qos_profile=latching_qos)


class SpotExecutorRos(Node):
    def __init__(self):
        self.debug = False

        self.feedback_collector = FeedbackCollector()

        #self.subscriber = self.create_subscription(
        #    NodeInfoMsg, "~/node_diagnostic_collector", self.callback, 10
        #)

        self.declare_param("my_param", 1)
        myparam = self.get_param("my_param").value

        self.fixed_frame = rospy.get_param("~fixed_frame")

        # Connectivity
        spot_ip = rospy.get_param("~spot_ip")
        bdai_username = rospy.get_param("~bosdyn_client_username")
        bdai_password = rospy.get_param("~bosdyn_client_password")

        # Follow Skill
        self.follower_lookahead = rospy.get_param("~follower_lookahead")

        # Pick/Inspect Skill
        semantic_model_path = rospy.get_param("~semantic_model_path")
        labelspace_path = rospy.get_param("~labelspace_path")
        semantic_name_to_id = load_inverse_semantic_id_map_from_label_space(
            labelspace_path
        )
        labelspace_grouping_path = rospy.get_param("~labelspace_grouping_path")
        with open(labelspace_grouping_path, "r") as f:
            grouping_info = yaml.safe_load(f)
        # turn list of dictionaries into single dictionary
        self.labelspace_map = {}
        offset = int(grouping_info["offset"])
        for group in grouping_info["groups"]:
            self.labelspace_map[group["name"]] = [g + offset for g in group["labels"]]


        # Robot Initialization
        use_fake_spot_interface = rospy.get_param("~use_fake_spot_interface")
        if use_fake_spot_interface:
            external_pose = rospy.get_param("~fake_spot_external_pose")
            static_pose = rospy.get_param("~fake_spot_static_pose")

            if rospy.get_param("~use_fake_spot_pose"):
                spot_x = rospy.get_param("~fake_spot_x")
                spot_y = rospy.get_param("~fake_spot_y")
                spot_z = rospy.get_param("~fake_spot_z")
                spot_yaw = rospy.get_param("~fake_spot_yaw")
                spot_init_pose2d = np.array([spot_x, spot_y, spot_z, spot_yaw])
            else:
                spot_init_pose2d = None

            print("About to initialize fake spot")
            self.spot_interface = FakeSpot(
                username=bdai_username,
                password=bdai_password,
                external_pose=external_pose,
                static_pose=static_pose,
                init_pose=spot_init_pose2d,
                semantic_model_path=semantic_model_path,
                semantic_name_to_id=semantic_name_to_id,
            )
        else:
            print("About to initialize Spot")
            self.spot_interface = Spot(
                username=bdai_username,
                password=bdai_password,
                ip=spot_ip,
                semantic_model_path=semantic_model_path,
                semantic_name_to_id=semantic_name_to_id,
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.action_sequence_sub = rclpy.create_subscription(
            "~action_sequence_subscriber",
            ActionSequenceMsg,
            self.process_action_sequence,
        )


    def process_action_sequence(self, msg):
        sequence = ActionSequence.from_msg(msg)
        self.spot_executor.process_action_sequence(sequence, self.feedback_collector)

def main(args=None):
    rclpy.init(args=args)
    node = SpotExecutorRos()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
