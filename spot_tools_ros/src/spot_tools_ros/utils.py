import time
from itertools import zip_longest

import numpy as np
import rclpy.time
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path
from tf_transformations import euler_from_quaternion, quaternion_from_euler


def waypoints_to_path(fixed_frame, waypoints):
    now = rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()


    frame_name = fixed_frame
    path_viz = Path()
    path_viz.header.stamp = now
    path_viz.header.frame_id = frame_name

    q = [0, 0, 0, 1]
    for w, wnext in zip_longest(waypoints, waypoints[1:]):
        p = PoseStamped()
        p.header.frame_id = frame_name
        p.header.stamp = now
        p.pose.position.x = w[0]
        p.pose.position.y = w[1]
        p.pose.position.z = 0.0

        if wnext is not None:
            d_next = wnext[:2] - w[:2]
            theta = np.arctan2(d_next[1], d_next[0])
            q = quaternion_from_euler(0, 0, theta)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]

        path_viz.poses.append(p)
    return path_viz


def path_to_waypoints(path):
    # NOTE: Currently ignores the frame that the path is in
    waypoints = []
    for p in path.poses:
        quat = p.pose.orientation
        q = [quat.x, quat.y, quat.z, quat.w]
        _, _, psi = euler_from_quaternion(q)
        waypoints.append([p.pose.position.x, p.pose.position.y, psi])

    return np.array(waypoints)


