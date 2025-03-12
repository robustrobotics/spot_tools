
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

from tf_transformations import euler_from_quaternion


class FakeSpotRos:
    def __init__(
        self,
        spot,
        external_pose=False,
        semantic_model_path=None,
        semantic_name_to_id=None,
    ):

        self.robot = spot

        self.tflistener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.semantic_name_to_id = semantic_name_to_id

        self.joint_state_publisher = rospy.Publisher(
            "~joint_states", JointState, queue_size=1
        )

        if not external_pose:
            rospy.Timer(rospy.Duration(0.1), self.update_pose_tf)
            rospy.Timer(rospy.Duration(0.05), lambda x: self.spot.step(.05))

            self.cmd_vel_linear = np.zeros(3)
            self.cmd_vel_angular = np.zeros(3)
            rospy.Subscriber("~cmd_vel", Twist, self.twist_command_cb)


    def twist_command_cb(self, msg):
        vl = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        va = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        self.robot.set_vel(vl, va)

    def get_pose(self):
        self.tflistener.waitForTransform(
            "vision", "body", rospy.Time(0), rospy.Duration(1.0)
        )
        trans, rot = self.tflistener.lookupTransform("vision", "body", rospy.Time(0))
        _, _, yaw = euler_from_quaternion(rot)
        return np.array([trans[0], trans[1], trans[2]])

    def update_pose_tf(self, _):
        with self.pose_lock:
            if self.pose is None:
                rospy.logwarn("Spot pose not set, cannot update!")
                return
            trans = (self.pose[0], self.pose[1], self.pose[2])
            yaw = self.pose[3]
        self.tf_broadcaster.sendTransform(
            trans,
            tf.transformations.quaternion_from_euler(0, 0, yaw),
            rospy.Time.now(),
            "body",
            "vision",
        )
        self.update_joint_states()

    def update_joint_states(self):
        jsm = JointState()
        jsm.header.stamp = rospy.Time.now()

        joint_state_map = self.robot.get_joint_states()
        jsm.name = list(joint_state_map.keys())
        jsm.position = list(joint_state_map.values())

        self.joint_state_publisher.publish(jsm)
