#!/usr/bin/env python3

import argparse
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from visualization_msgs.msg import MarkerArray

from robot_executor_interface.action_descriptions import ActionSequence, Follow
from spot_tools_ros.utils import get_tf_pose


# get time message
def gtm():
    return rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()


class FakePathPublisher(Node):
    def __init__(
        self,
        goal_x,
        goal_y,
        map_frame,
        robot_name,
        follow_robot=False,
        publish_rate=1.0,
    ):
        super().__init__("fake_path_publisher")

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.map_frame = map_frame
        self.robot_name = robot_name
        self.robot_frame = self.robot_name + "/body"
        self.follow_robot = follow_robot

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(
            ActionSequenceMsg,
            f"/{self.robot_name}/omniplanner_node/compiled_plan_out",
            10,
        )
        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.viz_publisher = self.create_publisher(
            MarkerArray,
            f"/{self.robot_name}/omniplanner_node/compiled_plan_viz_out",
            qos_profile=latching_qos,
        )

        if follow_robot:
            # Publish periodically if following robot
            self.timer = self.create_timer(1.0 / publish_rate, self.publish_path)
            self.get_logger().info(
                f"Publishing path few meters ahead of the robot at {publish_rate} Hz"
            )
        else:
            # Publish once
            self.timer = self.create_timer(1.0, self.publish_once)
            self.get_logger().info(f"Publishing path to goal ({goal_x}, {goal_y}) once")

    # def to_viz_msg_follow(self, action, marker_ns):
    #     points = []
    #     for p in action.path.poses:
    #         pt = Point()
    #         pt.x = p.pose.position.x
    #         pt.y = p.pose.position.y
    #         pt.z = p.pose.position.z
    #         points.append(pt)
    #     m = Marker()
    #     m.header.frame_id = self.map_frame
    #     m.header.stamp = gtm()
    #     m.ns = marker_ns
    #     m.id = 0
    #     m.type = m.LINE_STRIP
    #     m.action = m.ADD
    #     m.pose.orientation.w = 1.0
    #     m.scale.x = 0.2
    #     m.scale.y = 0.2
    #     m.color.a = 1.0
    #     m.color.r = 0.0
    #     m.color.g = 1.0
    #     m.color.b = 0.0
    #     m.points = points

    #     start = Marker()
    #     start.header.frame_id = self.map_frame
    #     start.header.stamp = gtm()
    #     start.ns = marker_ns
    #     start.id = 1
    #     start.type = m.SPHERE
    #     start.action = m.ADD
    #     start.pose.orientation.w = 1.0
    #     start.scale.x = 0.4
    #     start.scale.y = 0.4
    #     start.scale.z = 0.4
    #     start.color.a = 0.5
    #     start.color.r = 1.0
    #     start.color.g = 0.0
    #     start.color.b = 0.0
    #     start.pose.position.x = points[0].x
    #     start.pose.position.y = points[0].y
    #     start.pose.position.z = points[0].z

    #     end = Marker()
    #     end.header.frame_id = self.map_frame
    #     end.header.stamp = gtm()
    #     end.ns = marker_ns
    #     end.id = 2
    #     end.type = m.SPHERE
    #     end.action = m.ADD
    #     end.pose.orientation.w = 1.0
    #     end.scale.x = 0.4
    #     end.scale.y = 0.4
    #     end.scale.z = 0.4
    #     end.color.a = 0.5
    #     end.color.r = 0.0
    #     end.color.g = 0.0
    #     end.color.b = 1.0
    #     end.pose.position.x = points[-1].x
    #     end.pose.position.y = points[-1].y
    #     end.pose.position.z = points[-1].z

    #     return [m, start, end]

    # def to_viz_msg(self, action, marker_ns):
    #     ma = MarkerArray()
    #     for ix, a in enumerate(action.actions):
    #         ma.markers += self.to_viz_msg_follow(a, marker_ns + f"/{ix}")
    #     return ma

    def generate_waypoints(self, start_x, start_y, start_yaw, num_waypoints=10):
        """Generate linear waypoints from start to goal"""
        waypoints = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            x = start_x + alpha * (self.goal_x - start_x)
            y = start_y + alpha * (self.goal_y - start_y)
            # Interpolate yaw towards goal direction
            goal_yaw = np.arctan2(self.goal_y - start_y, self.goal_x - start_x)
            yaw = start_yaw + alpha * (goal_yaw - start_yaw)
            waypoints.append([x, y, yaw])
        return np.array(waypoints)

    def publish_path(self):
        """Publish path from current robot pose to goal"""
        try:
            # Get current robot pose
            robot_pose, (roll, pitch, robot_yaw) = get_tf_pose(
                self.tf_buffer, self.map_frame, self.robot_frame, get_euler=True
            )

            # Generate waypoints
            if self.follow_robot:
                waypoints = np.array(
                    [
                        [robot_pose[0], robot_pose[1], robot_yaw],
                        [
                            robot_pose[0] + 20 * np.cos(robot_yaw),
                            robot_pose[1] + 20 * np.sin(robot_yaw),
                            robot_yaw,
                        ],
                    ]
                )
            else:
                waypoints = self.generate_waypoints(
                    robot_pose[0], robot_pose[1], robot_yaw
                )

            # Create ActionSequenceMsg
            # msg = ActionSequenceMsg()
            # msg.header.stamp = self.get_clock().now().to_msg()
            # msg.header.frame_id = self.map_frame
            # msg.plan_id = "fake_path_plan"
            # msg.robot_name = self.robot_name
            # msg.actions = []

            # # Create ActionMsg with path
            # action_msg = ActionMsg()
            # action_msg.action_type = "FOLLOW"
            # path = Path()
            # path.header.frame_id = self.map_frame
            # path.header.stamp = self.get_clock().now().to_msg()
            # path.poses = []

            # # Add waypoints to path
            # for wp in waypoints:
            #     pose_stamped = PoseStamped()
            #     pose_stamped.header.frame_id = self.map_frame
            #     pose_stamped.header.stamp = self.get_clock().now().to_msg()
            #     pose_stamped.pose.position.x = wp[0]
            #     pose_stamped.pose.position.y = wp[1]
            #     pose_stamped.pose.position.z = 0.0
            #     quat = tf_transformations.quaternion_from_euler(0, 0, wp[2])
            #     pose_stamped.pose.orientation.x = quat[0]
            #     pose_stamped.pose.orientation.y = quat[1]
            #     pose_stamped.pose.orientation.z = quat[2]
            #     pose_stamped.pose.orientation.w = quat[3]
            #     path.poses.append(pose_stamped)

            # action_msg.path = path
            # msg.actions.append(action_msg)

            actions = []
            actions.append(Follow(frame=self.map_frame, path2d=waypoints.T))
            action_sequence = ActionSequence(
                plan_id="fake_path_plan",
                robot_name=self.robot_name,
                actions=actions,
            )
            msg = to_msg(action_sequence)

            self.publisher.publish(msg)
            self.viz_publisher.publish(
                to_viz_msg(action_sequence, self.robot_name)
            )  # somehow throw an error about the msg type, it has a '_' before the actual msg name
            # self.viz_publisher.publish(self.to_viz_msg(msg, self.robot_name))
            self.get_logger().info(f"Published path with {len(waypoints)} waypoints")

        except Exception as e:
            self.get_logger().error(f"Failed to publish path: {e}")

    def publish_once(self):
        """Publish path once and shutdown"""
        self.publish_path()
        self.timer.cancel()
        raise KeyboardInterrupt  # To exit the spin loop


def main():
    parser = argparse.ArgumentParser(description="Fake path publisher")
    parser.add_argument("goal_x", type=float, help="Goal x position")
    parser.add_argument("goal_y", type=float, help="Goal y position")
    parser.add_argument(
        "--follow_robot",
        action="store_true",
        help="Continuously update path based on robot pose",
    )
    parser.add_argument("--map_frame", type=str, default="map", help="Map frame")
    parser.add_argument("--robot_name", type=str, default="hamilton", help="Robot name")
    parser.add_argument("--rate", type=float, default=1.0, help="Publishing rate (Hz)")

    args = parser.parse_args()

    rclpy.init()

    node = FakePathPublisher(
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        map_frame=args.map_frame,
        robot_name=args.robot_name,
        follow_robot=args.follow_robot,
        publish_rate=args.rate,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
