#!/usr/bin/env python3

import os
import numpy as np
import argparse
import rclpy
import tf2_ros
import tf_transformations
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


def get_robot_pose(tf_buffer, parent_frame: str, child_frame: str):
    """Looks up the transform from parent_frame to child_frame"""
    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(parent_frame, child_frame, now, timeout=rclpy.duration.Duration(seconds=1.0))
        transform = tf_buffer.lookup_transform(parent_frame, child_frame, now)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        return np.array([translation.x, translation.y, translation.z]), rotation
    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise


class FakeOccupancyPublisher(Node):
    def __init__(self, occupancy_grid, resolution, robot_name, publish_rate=10.0, use_sim_time=False):
        super().__init__("fake_occupancy_publisher")

        # Enable simulation time for bag playback
        # self.declare_parameter('use_sim_time', use_sim_time)

        self.occupancy_grid = occupancy_grid.astype(np.int8)
        self.resolution = resolution
        self.map_frame = robot_name + "/map"
        self.robot_frame = robot_name + "/body"
        self.map_height, self.map_width = occupancy_grid.shape

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(OccupancyGrid, f"/{robot_name}/hydra/tsdf/occupancy", 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_map)

        self.get_logger().info(f"Publishing {self.map_width}x{self.map_height} map at {publish_rate} Hz")
        
        self.map_origin = None

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = Pose()

        # Get robot pose and center map on it
        try:
            if self.map_origin is None:
                robot_pose = get_robot_pose(self.tf_buffer, self.map_frame, self.robot_frame)
                self.map_origin = (robot_pose[0][0], robot_pose[0][1], robot_pose[0][2])
            msg.info.origin.position.x = self.map_origin[0]
            msg.info.origin.position.y = self.map_origin[1]
            msg.info.origin.position.z = self.map_origin[2]
        except:
            msg.info.origin.position.x = -16.0
            msg.info.origin.position.y = 0.0
            msg.info.origin.position.z = 0.0
        
        msg.info.origin.position.x = -16.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0
        msg.data = self.occupancy_grid.flatten().tolist()

        self.publisher.publish(msg)


def create_test_map(width=200, height=200):
    """Create a test map with obstacles"""
    grid = np.zeros((height, width), dtype=np.int8)
    grid[50:150, 50:150] = 100  # Vertical wall
    # grid[height//2-5:height//2+5, :] = 100  # Horizontal wall
    # center_i, center_j = height//4, width//2
    # y, x = np.ogrid[:height, :width]
    # mask = (x - center_j)**2 + (y - center_i)**2 <= 100
    # grid[mask] = 100  # Circular obstacle
    return grid


def main():
    parser = argparse.ArgumentParser(description='Fake occupancy publisher')
    parser.add_argument('--robot_name', type=str, default='hamilton', help='Robot name')
    args = parser.parse_args()
    rclpy.init()

    # Create or load occupancy map
    occupancy_map = create_test_map(width=200, height=200)
    # Or: occupancy_map = np.load("path/to/map.npy")

    node = FakeOccupancyPublisher(
        occupancy_grid=occupancy_map,
        resolution=0.12,
        robot_name=args.robot_name,
        publish_rate=10.0,
        use_sim_time=False  # Set to True when playing from bag
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
