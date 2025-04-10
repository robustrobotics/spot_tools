import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np
import rospy
import tf
import tf2_ros
from bosdyn.api import arm_command_pb2
from geometry_msgs.msg import TransformStamped, Twist
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion


def set_robot_pose2d(parent_frame, child_frame, pose2d):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform.translation.x = pose2d[0]
    static_transformStamped.transform.translation.y = pose2d[1]
    static_transformStamped.transform.translation.z = 0

    quat = tf.transformations.quaternion_from_euler(0, 0, pose2d[3])
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    broadcaster.sendTransform(static_transformStamped)


class FakeProtoInterface:
    def HasField(self, field):
        for f in dataclasses.fields(self):
            if f.name == field:
                return True
        return False


@dataclass
class FakePose:
    x: float
    y: float
    z: float

    def __sub__(self, other):
        try:
            return FakePose(self.x - other.x, self.y - other.y, self.z - other.z)
        except:
            raise TypeError(
                "Subtraction only supported between FakePose and Pose instances"
            )


@dataclass
class FakeMobilityFeedback:
    status: str = "ThisIsFakeMobilityFeedback"


@dataclass
class FakeArmGazeFeedback:
    status: int = arm_command_pb2.GazeCommand.Feedback.STATUS_TRAJECTORY_COMPLETE


@dataclass
class FakeArmFeedback(FakeProtoInterface):
    status: str = "ThisIsFakeArmFeedback"
    arm_gaze_feedback: FakeArmGazeFeedback = FakeArmGazeFeedback()


@dataclass
class FakeSynchronizedFeedback:
    mobility_command_feedback: FakeMobilityFeedback = FakeMobilityFeedback()
    arm_command_feedback: FakeArmFeedback = FakeArmFeedback()


@dataclass
class FakeFeedback:
    synchronized_feedback: FakeSynchronizedFeedback = FakeSynchronizedFeedback()


@dataclass
class FakeFeedbackWrapper:
    feedback: FakeFeedback = FakeFeedback()


class FakeStateClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot

    def get_robot_state(self, **kwargs):
        """Obtain current state of the robot.

        Returns:
            RobotState: The current robot state.

        Raises:
            RpcError: Problem communicating with the robot.
        """
        return self.fake_spot.get_pose()


class FakeCommandClient:
    def __init__(self, fake_spot):
        self.fake_spot = fake_spot

    def robot_command(
        self, command, end_time_secs=None, timesync_endpoint=None, lease=None, **kwargs
    ):
        # lease=None, command=None, end_time_secs=None):
        print("Spot would execute command with params:")
        print(f"\tlease: {lease}")
        print(f"\tcommand: {command}")
        print(f"\tend_time_secs: {end_time_secs}")

        move_cmd = command.synchronized_command.HasField("mobility_command")
        if move_cmd:
            x = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.position.x
            y = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.position.y
            angle = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points[
                0
            ].pose.angle

            z = self.fake_spot.get_pose().z
            self.fake_spot.set_pose((x, y, z, angle))
            self.fake_spot.moving = True
            self.fake_spot.last_move_command = rospy.Time.now()
        time.sleep(0.5)

    def robot_command_feedback(self, cmd_id):
        print("Spot would return command feedback for cmd_id ", cmd_id)
        return FakeFeedbackWrapper()


class FakeTimeSync:
    def wait_for_sync(self):
        return True


class FakeRobot:
    def __init__(self, fake_spot):
        self.time_sync = FakeTimeSync()
        self.fake_spot = fake_spot

    def ensure_client(self, service_name):
        print(f"Pretending that service {service_name} exists.")
        return FakeCommandClient(self.fake_spot)


class FakeSpot:
    def __init__(
        self,
        username="",
        password="",
        external_pose=False,
        static_pose=False,
        init_pose=None,
        semantic_model_path=None,
        semantic_name_to_id=None,
    ):
        print("Initialized Fake Spot!")
        self.is_fake = True
        self.pose_lock = threading.Lock()
        self.robot = FakeRobot(self)
        self.pose = init_pose
        self.tflistener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.semantic_name_to_id = semantic_name_to_id

        self.state_client = FakeStateClient(self)
        self.image_client = None
        self.manipulation_api_client = None
        self.joint_state_publisher = rospy.Publisher(
            "~joint_states", JointState, queue_size=1
        )

        self.moving = False
        self.last_move_command = rospy.Time.now()

        # init_pose is set to None when we want to run the planner while playing back the bag.
        # If we do this, we lose the ability to have spot pretend to execute the plan.
        if not external_pose and not static_pose:
            rospy.Timer(rospy.Duration(0.1), self.update_pose_tf)

            self.cmd_vel_linear = np.zeros(3)
            self.cmd_vel_angular = np.zeros(3)
            rospy.Subscriber("~cmd_vel", Twist, self.twist_command_cb)
            rospy.Timer(rospy.Duration(0.05), self.update_teleop)

        if not external_pose and static_pose:
            set_robot_pose2d("vision", "body", self.pose)

    def aquire_lease(self):
        pass

    def take_lease(self):
        pass

    def update_teleop(self, _):
        vx, vy, vz = self.cmd_vel_linear
        _, _, dtheta = self.cmd_vel_angular
        theta = self.pose[3]

        dp = np.zeros(4)

        dp[0] = vx * np.cos(theta) + vy * np.sin(theta)
        dp[1] = vx * np.sin(theta) - vy * np.cos(theta)
        dp[2] = vz
        dp[3] = dtheta
        self.set_pose(self.pose + dp * 0.05)

    def twist_command_cb(self, msg):
        self.cmd_vel_linear = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        self.cmd_vel_angular = np.array([msg.angular.x, msg.angular.y, msg.angular.z])

    def get_pose(self):
        self.tflistener.waitForTransform(
            "vision", "body", rospy.Time(0), rospy.Duration(1.0)
        )
        trans, rot = self.tflistener.lookupTransform("vision", "body", rospy.Time(0))
        # _, _, yaw = euler_from_quaternion(rot)
        # return np.array([trans[0], trans[1], yaw])
        return FakePose(trans[0], trans[1], trans[2])

    def set_pose(self, pose):
        with self.pose_lock:
            self.pose = pose

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

        t = rospy.Time.now().to_sec()
        jsm.header.stamp = rospy.Time.now()

        omg_hip = 2
        center_h2 = 1
        amp_h2 = 0.2
        center_k = -1.9
        amp_k = 0.3

        vl = np.linalg.norm(self.cmd_vel_linear)
        va = np.linalg.norm(self.cmd_vel_angular)

        if rospy.Time.now() - self.last_move_command > rospy.Duration(1):
            self.moving = False
        moving = 1 if vl > 0 or va > 0 or self.moving else 0

        jsm.name.append("front_left_hip_x")
        jsm.position.append(0)
        jsm.name.append("front_left_hip_y")
        jsm.position.append(center_h2 + moving * amp_h2 * np.cos(omg_hip * t))
        jsm.name.append("front_left_knee")
        jsm.position.append(center_k + moving * amp_k * np.cos(omg_hip * t + np.pi / 2))

        jsm.name.append("rear_left_hip_x")
        jsm.position.append(0)
        jsm.name.append("rear_left_hip_y")
        jsm.position.append(center_h2 + moving * amp_h2 * np.cos(omg_hip * t + np.pi))
        jsm.name.append("rear_left_knee")
        jsm.position.append(
            center_k + moving * amp_k * np.cos(omg_hip * t + 3 * np.pi / 2)
        )

        jsm.name.append("front_right_hip_x")
        jsm.position.append(0)
        jsm.name.append("front_right_hip_y")
        jsm.position.append(center_h2 + moving * amp_h2 * np.cos(omg_hip * t + np.pi))
        jsm.name.append("front_right_knee")
        jsm.position.append(
            center_k + moving * amp_k * np.cos(omg_hip * t + 3 * np.pi / 2)
        )

        jsm.name.append("rear_right_hip_x")
        jsm.position.append(0)
        jsm.name.append("rear_right_hip_y")
        jsm.position.append(center_h2 + moving * amp_h2 * np.cos(omg_hip * t))
        jsm.name.append("rear_right_knee")
        jsm.position.append(center_k + moving * amp_k * np.cos(omg_hip * t + np.pi / 2))

        self.joint_state_publisher.publish(jsm)
