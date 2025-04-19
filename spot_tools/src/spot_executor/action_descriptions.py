from dataclasses import dataclass

import numpy as np
import rospy
from geometry_msgs.msg import Point
from phoenix_tamp_planner.msg import ActionMsg, ActionSequenceMsg
from phoenix_tamp_planner.utils import path_to_waypoints, waypoints_to_path
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class ActionSequence:
    plan_id: str
    robot_name: str
    actions: list

    @classmethod
    def from_msg(cls, msg):
        actions = []
        for a in msg.actions:
            if a.action_type == a.FOLLOW:
                actions.append(Follow.from_msg(a))
            elif a.action_type == a.GAZE:
                actions.append(Gaze.from_msg(a))
            elif a.action_type == a.PICK:
                actions.append(Pick.from_msg(a))
            elif a.action_type == a.PLACE:
                actions.append(Place.from_msg(a))
            else:
                raise Exception(f"Received invalid action type {a.action_type}")

        return cls(plan_id=msg.plan_id, robot_name=msg.robot_name, actions=actions)

    def to_msg(self):
        msg = ActionSequenceMsg()
        msg.plan_id = self.plan_id
        msg.robot_name = self.robot_name
        msg.header.stamp = rospy.Time.now()
        msg.actions = [a.to_msg() for a in self.actions]
        return msg

    def to_viz_msg(self, marker_ns_prefix):
        ma = MarkerArray()
        for ix, a in enumerate(self.actions):
            ma.markers += a.to_viz_msg(marker_ns_prefix + f"/{ix}")
        return ma


@dataclass
class Follow:
    frame: str
    path2d: np.ndarray

    @classmethod
    def from_msg(cls, msg):
        return cls(frame=msg.path.header.frame_id, path2d=path_to_waypoints(msg.path))

    def to_msg(self):
        msg = ActionMsg()
        msg.action_type = msg.FOLLOW
        msg.path = waypoints_to_path(self.frame, self.path2d)
        return msg

    def to_viz_msg(self, marker_ns):
        points = []
        for p in self.path2d:
            pt = Point()
            pt.x = p[0]
            pt.y = p[1]
            pt.z = 0
            points.append(pt)
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = rospy.Time.now()
        m.ns = marker_ns
        m.id = 0
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.pose.orientation.w = 1
        m.scale.x = 0.1
        m.color.a = 1
        m.color.r = 0
        m.color.g = 1
        m.color.b = 0
        m.points = points
        return [m]


@dataclass
class Gaze:
    frame: str
    robot_point: np.ndarray
    gaze_point: np.ndarray
    stow_after: bool = False

    @classmethod
    def from_msg(cls, msg):
        robot_point = np.array(
            [msg.robot_point.x, msg.robot_point.y, msg.robot_point.z]
        )
        gaze_point = np.array([msg.gaze_point.x, msg.gaze_point.y, msg.gaze_point.z])
        stow_after = msg.place_frame == "STOW"

        return cls(
            frame=msg.gaze_frame,
            robot_point=robot_point,
            gaze_point=gaze_point,
            stow_after=stow_after,
        )

    def to_msg(self):
        msg = ActionMsg()
        msg.action_type = msg.GAZE

        msg.robot_point.x = self.robot_point[0]
        msg.robot_point.y = self.robot_point[1]
        msg.robot_point.z = self.robot_point[2]

        msg.gaze_point.x = self.gaze_point[0]
        msg.gaze_point.y = self.gaze_point[1]
        msg.gaze_point.z = self.gaze_point[2]

        msg.place_frame = "STOW" if self.stow_after else "NO_STOW"

        return msg

    def to_viz_msg(self, marker_ns):
        pt1 = Point()
        pt1.x = self.robot_point[0]
        pt1.y = self.robot_point[1]
        pt1.z = self.robot_point[2]
        pt2 = Point()
        pt2.x = self.gaze_point[0]
        pt2.y = self.gaze_point[1]
        pt2.z = self.gaze_point[2]

        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = rospy.Time.now()
        m.ns = marker_ns
        m.id = 0
        m.type = m.ARROW
        m.action = m.ADD
        m.pose.orientation.w = 1
        m.scale.x = 0.1  # shaft diameter
        m.scale.y = 0.2  # head diameter
        m.color.a = 1
        m.color.r = 1
        m.color.g = 0
        m.color.b = 0
        m.points = [pt1, pt2]

        return [m]


@dataclass
class Pick:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray

    @classmethod
    def from_msg(cls, msg):
        robot_point = np.array(
            [msg.robot_point.x, msg.robot_point.y, msg.robot_point.z]
        )
        object_point = np.array(
            [msg.object_point.x, msg.object_point.y, msg.object_point.z]
        )

        return cls(
            frame=msg.pick_frame,
            object_class=msg.object_class,
            robot_point=robot_point,
            object_point=object_point,
        )

    def to_msg(self):
        msg = ActionMsg()
        msg.action_type = msg.PICK

        msg.robot_point.x = self.robot_point[0]
        msg.robot_point.y = self.robot_point[1]
        msg.robot_point.z = self.robot_point[2]

        msg.object_point.x = self.object_point[0]
        msg.object_point.y = self.object_point[1]
        msg.object_point.z = self.object_point[2]

        msg.object_class = self.object_class

        return msg

    def to_viz_msg(self, marker_ns):
        pt1 = Point()
        pt1.x = self.robot_point[0]
        pt1.y = self.robot_point[1]
        pt1.z = self.robot_point[2]
        pt2 = Point()
        pt2.x = self.object_point[0]
        pt2.y = self.object_point[1]
        pt2.z = self.object_point[2]

        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = rospy.Time.now()
        m.ns = marker_ns
        m.id = 0
        m.type = m.ARROW
        m.action = m.ADD
        m.pose.orientation.w = 1
        m.scale.x = 0.1  # shaft diameter
        m.scale.y = 0.2  # head diameter
        m.color.a = 1
        m.color.r = 0
        m.color.g = 1
        m.color.b = 0
        m.points = [pt1, pt2]

        return [m]


@dataclass
class Place:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray

    @classmethod
    def from_msg(cls, msg):
        robot_point = np.array(
            [msg.robot_point.x, msg.robot_point.y, msg.robot_point.z]
        )
        object_point = np.array(
            [msg.object_point.x, msg.object_point.y, msg.object_point.z]
        )

        return cls(
            frame=msg.place_frame,
            object_class=msg.object_class,
            robot_point=robot_point,
            object_point=object_point,
        )

    def to_msg(self):
        msg = ActionMsg()
        msg.action_type = msg.PLACE

        msg.robot_point.x = self.robot_point[0]
        msg.robot_point.y = self.robot_point[1]
        msg.robot_point.z = self.robot_point[2]

        msg.object_point.x = self.object_point[0]
        msg.object_point.y = self.object_point[1]
        msg.object_point.z = self.object_point[2]

        return msg

    def to_viz_msg(self, marker_ns):
        pt1 = Point()
        pt1.x = self.robot_point[0]
        pt1.y = self.robot_point[1]
        pt1.z = self.robot_point[2]
        pt2 = Point()
        pt2.x = self.object_point[0]
        pt2.y = self.object_point[1]
        pt2.z = self.object_point[2]

        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = rospy.Time.now()
        m.ns = marker_ns
        m.id = 0
        m.type = m.ARROW
        m.action = m.ADD
        m.pose.orientation.w = 1
        m.scale.x = 0.1  # shaft diameter
        m.scale.y = 0.2  # head diameter
        m.color.a = 1
        m.color.r = 0
        m.color.g = 0
        m.color.b = 1
        m.points = [pt1, pt2]

        return [m]
