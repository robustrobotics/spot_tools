#!/usr/bin/env python3
"""Minimal ROS 2 node: subscribes to /cmd_vel and commands Spot's body velocity."""

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from spot_executor.spot import Spot


def clip(x, lower, upper):
    if x < lower:
        return lower
    if x > upper:
        return upper
    return x


class SpotTwistBridge(Node):
    def __init__(self):
        super().__init__("spot_twist_bridge")

        self.declare_parameter("spot_ip", "")
        self.declare_parameter("bosdyn_client_username", "")
        self.declare_parameter("bosdyn_client_password", "")
        self.declare_parameter("body_frame", "")
        spot_ip = self.get_parameter("spot_ip").value
        assert spot_ip != ""
        bdai_username = self.get_parameter("bosdyn_client_username").value
        assert bdai_username != ""
        bdai_password = self.get_parameter("bosdyn_client_password").value
        assert bdai_password != ""
        self.body_frame = self.get_parameter("body_frame").value
        assert self.body_frame != ""

        self.get_logger().info("About to initialize Spot")
        self.get_logger().info(f"{bdai_username=}, {bdai_password=}, {spot_ip=}")
        self.spot_interface = Spot(
            username=bdai_username,
            password=bdai_password,
            ip=spot_ip,
        )
        self.spot_interface.robot.time_sync.wait_for_sync()
        self.spot_interface.take_lease()

        self.sub = self.create_subscription(
            TwistStamped, "~/cmd_vel", self.twist_cb, 10
        )

    def twist_cb(self, msg: TwistStamped):
        """Convert a Twist message into a Spot velocity command."""
        # msg.twist.linear.x  -> forward/back  (m/s)
        # msg.twist.linear.y  -> left/right    (m/s)
        # msg.twist.angular.z -> rotation      (rad/s)
        self.get_logger().info(
            f"Setting twist for spot: {[msg.linear.x, msg.linear.y, msg.angular.z]}"
        )
        assert msg.header.frame_id == self.body_frame, (
            "Currently Spot only accepts messages in its body frame"
        )
        vx = clip(msg.twist.linear.x, -0.5, 0.5)
        vy = clip(msg.twist.linear.x, -0.5, 0.5)
        omega_z = clip(msg.twist.linear.x, -0.3, 0.3)
        self.get_logger().info(f"Clipped values: {[vx, vy, omega_z]}")
        self.spot_interface.set_twist(vx, vy, omega_z)


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
