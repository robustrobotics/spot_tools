import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class FakeSpotRos:
    def __init__(
        self,
        host_node,
        spot,
        odom_frame,
        body_frame,
        external_pose=False,
        semantic_model_path=None,
        semantic_name_to_id=None,
    ):
        self.host_node = host_node
        self.robot = spot

        self.odom_frame_name = odom_frame
        self.body_frame_name = body_frame
        self.tf_prefix = "/".join(body_frame.split("/")[:-1])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.host_node)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.host_node)
        self.semantic_name_to_id = semantic_name_to_id

        self.joint_state_publisher = host_node.create_publisher(
            JointState, "~/joint_states", 10
        )

        if not external_pose:
            timer_group = MutuallyExclusiveCallbackGroup()
            self.timer = host_node.create_timer(
                0.1, self.update_pose_tf, callback_group=timer_group
            )

            dt = 0.05
            self.timer = host_node.create_timer(
                dt, lambda: self.robot.step(dt), callback_group=timer_group
            )

            self.cmd_vel_linear = np.zeros(3)
            self.cmd_vel_angular = np.zeros(3)
            host_node.create_subscription(Twist, "~/cmd_vel", self.twist_command_cb, 10)

    def twist_command_cb(self, msg):
        vl = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        va = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        self.robot.set_vel(vl, va)

    def get_pose(self):
        try:
            trans, rot = self.tf_buffer.lookup_transform(
                self.odom_frame_name, self.body_frame_name, rclpy.time.Time()
            )
        except tf2_ros.TransformException as e:
            self.host_node.get_logger().warn(f"Could not get transform: {e}")
        _, _, yaw = euler_from_quaternion(rot)
        return np.array([trans[0], trans[1], trans[2]])

    def update_pose_tf(self):
        pose = self.robot.get_pose()
        if pose is None:
            self.host_node.get_logger().warn("Spot pose not set, cannot update!")
            return

        trans = (pose[0], pose[1], pose[2])
        yaw = pose[3]

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = self.host_node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = self.odom_frame_name
        transform_stamped.child_frame_id = self.body_frame_name
        transform_stamped.transform.translation.x = trans[0]
        transform_stamped.transform.translation.y = trans[1]
        transform_stamped.transform.translation.z = trans[2]
        q = quaternion_from_euler(0, 0, yaw)
        transform_stamped.transform.rotation.x = q[0]
        transform_stamped.transform.rotation.y = q[1]
        transform_stamped.transform.rotation.z = q[2]
        transform_stamped.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(transform_stamped)
        self.update_joint_states()

    def update_joint_states(self):
        jsm = JointState()
        jsm.header.stamp = rclpy.time.Time(seconds=time.time()).to_msg()

        joint_state_map = self.robot.get_joint_states()
        values = list(map(float, joint_state_map.values()))  # I hate ROS2 so much
        jsm.name = [f"{self.tf_prefix}/{k}" for k in joint_state_map.keys()]
        jsm.position = values

        self.joint_state_publisher.publish(jsm)
