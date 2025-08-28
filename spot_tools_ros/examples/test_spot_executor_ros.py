import numpy as np
import rclpy
from rclpy.node import Node
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from visualization_msgs.msg import MarkerArray

from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
)


class Tester(Node):
    def __init__(self):
        super().__init__("Tester")
        # ROS 2 transient local QoS for "latching" behavior
        # latching_qos = QoSProfile(
        #    depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        # )

        # Keeping the "~/" prefix notation for private topics in ROS 2
        publisher = self.create_publisher(
            ActionSequenceMsg, "/hamilton/omniplanner_node/compiled_plan_out", 1
        )

        viz_publisher = self.create_publisher(MarkerArray, "/planner/visualization", 1)

        path = np.array(
            [
                [0.0, 0],
                [10.8, 0],
                # [3.0, 5],
                # [5.0, 5],
            ]
        )

        follow_cmd = Follow("hamilton/odom", path)

        # gaze_cmd = Gaze(
        #    "hamilton/odom",
        #    np.array([5.0, 5, 0]),
        #    np.array([7.0, 7, 0]),
        #    stow_after=True,
        # )

        # pick_cmd = Pick(
        #    "hamilton/odom", "bag", np.array([5.0, 5, 0]), np.array([7.0, 7, 0])
        # )

        seq = ActionSequence("id0", "spot", [follow_cmd])

        publisher.publish(to_msg(seq))
        viz_publisher.publish(to_viz_msg(seq, "planner_ns"))


def main(args=None):
    rclpy.init(args=args)
    node = Tester()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
