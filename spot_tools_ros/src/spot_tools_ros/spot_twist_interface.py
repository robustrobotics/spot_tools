#!/usr/bin/env python3
"""Minimal ROS 2 node: subscribes to /cmd_vel and commands Spot's body velocity."""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from spot_executor.spot import Spot


class SpotTwistBridge(Node):
    def __init__(self):
        super().__init__("spot_twist_bridge")

        ###
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

        # Twist receiver params
        # self.declare_parameter("follower_lookahead", 0.0)
        # follower_lookahead = self.get_parameter("follower_lookahead").value
        # assert follower_lookahead > 0

        self.get_logger().info("About to initialize Spot")
        self.get_logger().info(f"{bdai_username=}, {bdai_password=}, {spot_ip=}")
        self.spot_interface = Spot(
            username=bdai_username,
            password=bdai_password,
            ip=spot_ip,
        )
        self.spot_interface.robot.time_sync.wait_for_sync()
        self.spot_interface.take_lease()

        self.sub = self.create_subscription(Twist, "~/cmd_vel", self.twist_cb, 10)

    def twist_cb(self, msg: Twist):
        """Convert a Twist message into a Spot velocity command."""
        # msg.linear.x  -> forward/back  (m/s)
        # msg.linear.y  -> left/right    (m/s)
        # msg.angular.z -> rotation      (rad/s)
        self.spot_interface.set_twist(msg.linear.x, msg.linear.y, msg.angular.z)


def main(args=None):
    rclpy.init(args=args)
    node = SpotTwistBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
