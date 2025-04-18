import numpy as np
import rclpy
from rclpy.node import Node
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from visualization_msgs.msg import MarkerArray


class Tester(Node):
    def __init__(self):
        super().__init__("Tester")
        # ROS 2 transient local QoS for "latching" behavior
        # latching_qos = QoSProfile(
        #    depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        # )

        # Keeping the "~/" prefix notation for private topics in ROS 2
        publisher = self.create_publisher(
            ActionSequenceMsg, "/spot_executor_node/action_sequence_subscriber", 1
        )

        viz_publisher = self.create_publisher(MarkerArray, "/planner/visualization", 1)

        path = np.array(
            [
                [0.0, 0],
                [1.0, 0],
                [3.0, 5],
                [5.0, 5],
            ]
        )

        follow_cmd = Follow("vision", path)

        gaze_cmd = Gaze(
            "vision", np.array([5.0, 5, 0]), np.array([7.0, 7, 0]), stow_after=True
        )

        seq = ActionSequence("id0", "spot", [follow_cmd, gaze_cmd])

        publisher.publish(to_msg(seq))
        viz_publisher.publish(to_viz_msg(seq, "planner_ns"))


def main(args=None):
    rclpy.init(args=args)
    node = Tester()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
