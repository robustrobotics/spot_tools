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
from functools import partial
from scipy.spatial.transform import Rotation

def pose_to_homo(pose, quat):
    '''
    Input:
        - pose: list [x, y, z]
        - quat: ros2 geometry_msgs.msg.Quaternion
    '''
    # Convert pose and quaternion to a 4x4 homogeneous transformation matrix
    trans = np.array(pose)
    rot_mat = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    homo_mat = np.eye(4)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat
        
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
    def __init__(self, occupancy_grid, resolution, robot_name, crop_distance=-1, publish_rate=10.0):
        super().__init__("fake_occupancy_publisher")

        self.occupancy_grid = occupancy_grid.astype(np.int8)
        self.resolution = resolution
        self.map_frame = robot_name + "/map"
        self.robot_frame = robot_name + "/body"
        self.odom_frame = robot_name + "/odom"
        self.map_height, self.map_width = occupancy_grid.shape


        self.crop_distance = crop_distance  # meters
        self.crop = self.crop_distance > 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        special_tf_remaps = {"<spot_vision_frame>": self.odom_frame}

        def tf_lookup_fn(parent, child):
            if parent in special_tf_remaps:
                parent = special_tf_remaps[parent]
            if child in special_tf_remaps:
                child = special_tf_remaps[child]
            try:
                return get_robot_pose(self.tf_buffer, parent, child)
            except tf2_ros.TransformException as e:
                self.get_logger.warn(f"Failed to get transform: {e}")
        self.tf_lookup_fn = tf_lookup_fn # TODO: use this to test transformation


        self.publisher = self.create_publisher(OccupancyGrid, f"/{robot_name}/hydra/tsdf/occupancy", 10)
        self.unmodified_publisher = self.create_publisher(OccupancyGrid, f"/{robot_name}/hydra/tsdf/occupancy/unmodified", 10)

        self.timer = self.create_timer(
            1.0 / publish_rate,
            partial(self.publish_map, self.publisher, crop_to_robot=self.crop),
        )

        self.unmodified_timer = self.create_timer(
            1.0 / publish_rate,
            partial(self.publish_map, self.unmodified_publisher, crop_to_robot=False),
        )


        self.get_logger().info(f"Publishing {self.map_width}x{self.map_height} map at {publish_rate} Hz")
        
        self.map_origin = None


    def global_pose_to_grid_cell(self, pose, map_origin):
        '''
        Input:
            - pose: (4,1) numpy array in map (global) frame
        Output:
            - (x, y): tuple in grid frame
        
        indexing: (i, j) = (row, col) = (y, x)
        '''
        pose_in_grid_frame = np.linalg.inv(map_origin) @ pose # (4,1)
        
        # Convert the pose to grid coordinates
        grid_j = int(pose_in_grid_frame[0, 0] / self.resolution)
        grid_i = int(pose_in_grid_frame[1, 0] / self.resolution)
                
        return (grid_i, grid_j)

    def crop_around_robot(self, robot_pose_odom_frame, msg_info_origin):
        '''
        Input:
            - robot_pose_odom_frame: tuple of (translation, rotation) from tf_lookup_fn
            - msg_info_origin: Pose message containing map origin
            - crop_size: float, size of the crop in meters
        Output:
            - cropped_grid: flattened numpy array
        '''
        
        # Get transformation from odom to map frame
        odom_to_map = self.tf_lookup_fn(self.odom_frame, self.map_frame)
        
        # Convert to homogeneous transformation matrices
        robot_pose_homo = pose_to_homo(robot_pose_odom_frame[0], robot_pose_odom_frame[1])
        odom_to_map_homo = pose_to_homo(odom_to_map[0], odom_to_map[1])
        map_origin_homo = pose_to_homo([msg_info_origin.position.x, msg_info_origin.position.y, msg_info_origin.position.z], msg_info_origin.orientation)

        # Transform map origin to odom frame (following spot_executor_ros pattern)
        map_origin_odom_frame = odom_to_map_homo @ map_origin_homo
        
        # Create robot position vector in odom frame
        robot_pose_vector = robot_pose_homo[:, 3].reshape(4, 1)  # Extract position column

        robot_grid_cell = self.global_pose_to_grid_cell(robot_pose_vector, map_origin_odom_frame)
        rows, cols = np.ogrid[:self.occupancy_grid.shape[0], :self.occupancy_grid.shape[1]]
        dist2 = (rows - robot_grid_cell[0])**2 + (cols - robot_grid_cell[1])**2

        # Convert crop_size from meters to grid cells
        crop_size_cells = self.crop_distance / self.resolution

        # Mask: True for points *outside* the circle
        mask_outside = dist2 > crop_size_cells**2

        cropped_grid = np.copy(self.occupancy_grid)

        cropped_grid[mask_outside] = int(-1)  # Unknown

        return cropped_grid.flatten().tolist()


    def publish_map(self, publisher, crop_to_robot=False):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin = Pose()
        
        msg.info.origin.position.x = -16.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0
        if crop_to_robot:
            robot_pose_odom_frame = self.tf_lookup_fn(self.odom_frame, self.robot_frame)
            msg.data = self.crop_around_robot(robot_pose_odom_frame, msg.info.origin)
        else:
            msg.data = self.occupancy_grid.flatten().tolist()

        publisher.publish(msg)  # For visualization in RViz


def create_test_map(width=200, height=200):
    """Create a test map with obstacles"""
    grid = np.zeros((height, width), dtype=np.int8)
    grid[50:150, 50:150] = 100  # Vertical wall

    grid[:10, 0:20] = 100  # Horizontal wall
    return grid




def main():
    parser = argparse.ArgumentParser(description='Fake occupancy publisher')
    parser.add_argument('--robot_name', type=str, default='hamilton', help='Robot name')
    parser.add_argument('--resolution', type=float, default=0.12, help='Map resolution in meters/cell')
    parser.add_argument('--crop_distance', type=float, default=5.0, help='Crop distance in meters (set to -1 to disable cropping)')
    parser.add_argument('--publish_rate', type=float, default=10.0, help='Publish rate in Hz')
    args = parser.parse_args()
    rclpy.init()

    # Create or load occupancy map
    occupancy_map = create_test_map(width=200, height=200)
    # Or: occupancy_map = np.load("path/to/map.npy")

    node = FakeOccupancyPublisher(
        occupancy_grid=occupancy_map,
        resolution=args.resolution,
        robot_name=args.robot_name,
        publish_rate=args.publish_rate,
        crop_distance=args.crop_distance
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
