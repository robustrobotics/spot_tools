import hashlib
import time
from functools import singledispatch

import numpy as np
import rclpy.time
from geometry_msgs.msg import Point
from robot_executor_msgs.msg import ActionMsg, ActionSequenceMsg
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)
from spot_tools_ros.utils import path_to_waypoints, waypoints_to_path

# Visually distinct colors for differentiating robot plans in RViz.
# Avoids pure green (blends with occupancy grids).
_ROBOT_COLORS = [
    ColorRGBA(r=0.0, g=0.6, b=1.0, a=1.0),   # sky blue
    ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0),   # orange
    ColorRGBA(r=0.8, g=0.0, b=0.8, a=1.0),   # magenta
    ColorRGBA(r=0.0, g=0.9, b=0.9, a=1.0),   # cyan
    ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0),  # gold
    ColorRGBA(r=1.0, g=0.0, b=0.3, a=1.0),   # rose
    ColorRGBA(r=0.4, g=0.2, b=1.0, a=1.0),   # purple
    ColorRGBA(r=0.0, g=0.8, b=0.4, a=1.0),   # teal green
]


def robot_color(marker_ns: str) -> ColorRGBA:
    """Return a deterministic color for a robot based on its namespace/name."""
    # Strip sub-action suffixes (e.g. "spot1/0") so all actions for one robot match.
    robot_name = marker_ns.split("/")[0]
    idx = int(hashlib.md5(robot_name.encode()).hexdigest(), 16) % len(_ROBOT_COLORS)
    c = _ROBOT_COLORS[idx]
    return ColorRGBA(r=c.r, g=c.g, b=c.b, a=c.a)


def robot_label_marker(marker_ns: str, frame_id: str, x: float, y: float) -> Marker:
    """Create a TEXT_VIEW_FACING marker showing the robot name at (x, y)."""
    robot_name = marker_ns.split("/")[0]
    label = Marker()
    label.header.frame_id = frame_id
    label.header.stamp = gtm()
    label.ns = marker_ns
    label.id = 999
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.position.x = float(x)
    label.pose.position.y = float(y)
    label.pose.position.z = 0.5
    label.pose.orientation.w = 1.0
    label.scale.z = 0.4
    label.color = robot_color(marker_ns)
    label.text = robot_name
    return label


# get time message
def gtm():
    return rclpy.time.Time(nanoseconds=time.time() * 1e9).to_msg()


@singledispatch
def to_msg(action):
    raise Exception(f"Unknown action type {type(action)}")


@singledispatch
def to_viz_msg(action, marker_ns):
    raise Exception(f"Unknown action type {type(action)}")


@to_msg.register
def _(action: ActionSequence):
    msg = ActionSequenceMsg()
    msg.plan_id = action.plan_id
    msg.robot_name = action.robot_name
    msg.header.stamp = gtm()

    print([to_msg(a) for a in action.actions])
    msg.actions = [to_msg(a) for a in action.actions]
    return msg


@to_viz_msg.register
def _(action: ActionSequence, marker_ns):
    ma = MarkerArray()
    for ix, a in enumerate(action.actions):
        ma.markers += to_viz_msg(a, marker_ns + f"/{ix}")

    # Add a text label at the start of the first action
    if action.actions:
        first = action.actions[0]
        if isinstance(first, Follow) and len(first.path2d) > 0:
            label = robot_label_marker(
                marker_ns, first.frame, first.path2d[0][0], first.path2d[0][1]
            )
            ma.markers.append(label)

    return ma


def from_msg(msg):
    actions = []
    for a in msg.actions:
        match a.action_type:
            case a.FOLLOW:
                actions.append(follow_from_msg(a))
            case a.PICK:
                actions.append(pick_from_msg(a))
            case a.PLACE:
                actions.append(place_from_msg(a))
            case a.GAZE:
                actions.append(gaze_from_msg(a))
            case _:
                raise Exception(f"Received invalid action type {a.action_type}")
    return ActionSequence(
        plan_id=msg.plan_id, robot_name=msg.robot_name, actions=actions
    )


def follow_from_msg(msg):
    return Follow(frame=msg.path.header.frame_id, path2d=path_to_waypoints(msg.path))


@to_msg.register
def _(action: Follow):
    msg = ActionMsg()
    msg.action_type = msg.FOLLOW
    msg.path = waypoints_to_path(action.frame, action.path2d)
    return msg


@to_viz_msg.register
def _(action: Follow, marker_ns):
    points = []
    for p in action.path2d:
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        pt.z = 0.0
        points.append(pt)
    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.LINE_STRIP
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    c = robot_color(marker_ns)
    m.scale.x = 0.2
    m.scale.y = 0.2
    m.color = c
    m.points = points

    start = Marker()
    start.header.frame_id = action.frame
    start.header.stamp = gtm()
    start.ns = marker_ns
    start.id = 1
    start.type = m.SPHERE
    start.action = m.ADD
    start.pose.orientation.w = 1.0
    start.scale.x = 0.4
    start.scale.y = 0.4
    start.scale.z = 0.4
    start.color = ColorRGBA(r=c.r * 0.6, g=c.g * 0.6, b=c.b * 0.6, a=0.5)
    start.pose.position.x = points[0].x
    start.pose.position.y = points[0].y
    start.pose.position.z = points[0].z

    end = Marker()
    end.header.frame_id = action.frame
    end.header.stamp = gtm()
    end.ns = marker_ns
    end.id = 2
    end.type = m.SPHERE
    end.action = m.ADD
    end.pose.orientation.w = 1.0
    end.scale.x = 0.4
    end.scale.y = 0.4
    end.scale.z = 0.4
    end.color = ColorRGBA(r=min(c.r + 0.3, 1.0), g=min(c.g + 0.3, 1.0), b=min(c.b + 0.3, 1.0), a=0.5)
    end.pose.position.x = points[-1].x
    end.pose.position.y = points[-1].y
    end.pose.position.z = points[-1].z

    return [m, start, end]


def gaze_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    gaze_point = np.array([msg.gaze_point.x, msg.gaze_point.y, msg.gaze_point.z])
    stow_after = msg.place_frame == "STOW"
    object_id = msg.object_id

    return Gaze(
        frame=msg.gaze_frame,
        robot_point=robot_point,
        gaze_point=gaze_point,
        stow_after=stow_after,
        object_id=object_id,
    )


@to_msg.register
def _(action: Gaze):
    msg = ActionMsg()
    msg.action_type = msg.GAZE

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.gaze_point.x = action.gaze_point[0]
    msg.gaze_point.y = action.gaze_point[1]
    msg.gaze_point.z = action.gaze_point[2]

    msg.place_frame = "STOW" if action.stow_after else "NO_STOW"

    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Gaze, marker_ns):
    pt1 = Point()
    pt1.x = action.robot_point[0]
    pt1.y = action.robot_point[1]
    pt1.z = action.robot_point[2]
    pt2 = Point()
    pt2.x = action.gaze_point[0]
    pt2.y = action.gaze_point[1]
    pt2.z = action.gaze_point[2]

    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.ARROW
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.1  # shaft diameter
    m.scale.y = 0.2  # head diameter
    m.color = robot_color(marker_ns)
    m.points = [pt1, pt2]

    return [m]


def pick_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    object_point = np.array(
        [msg.object_point.x, msg.object_point.y, msg.object_point.z]
    )

    return Pick(
        frame=msg.pick_frame,
        object_class=msg.object_class,
        robot_point=robot_point,
        object_point=object_point,
        object_id=msg.object_id,
    )


@to_msg.register
def _(action: Pick):
    msg = ActionMsg()
    msg.action_type = msg.PICK

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.object_point.x = action.object_point[0]
    msg.object_point.y = action.object_point[1]
    msg.object_point.z = action.object_point[2]

    msg.object_class = action.object_class
    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Pick, marker_ns):
    pt1 = Point()
    pt1.x = action.robot_point[0]
    pt1.y = action.robot_point[1]
    pt1.z = action.robot_point[2]
    pt2 = Point()
    pt2.x = action.object_point[0]
    pt2.y = action.object_point[1]
    pt2.z = action.object_point[2]

    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.ARROW
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.1  # shaft diameter
    m.scale.y = 0.2  # head diameter
    m.color = robot_color(marker_ns)
    m.points = [pt1, pt2]

    return [m]


def place_from_msg(msg):
    robot_point = np.array([msg.robot_point.x, msg.robot_point.y, msg.robot_point.z])
    object_point = np.array(
        [msg.object_point.x, msg.object_point.y, msg.object_point.z]
    )

    return Place(
        frame=msg.place_frame,
        object_class=msg.object_class,
        robot_point=robot_point,
        object_point=object_point,
        object_id=msg.object_id,
    )


@to_msg.register
def _(action: Place):
    msg = ActionMsg()
    msg.action_type = msg.PLACE

    msg.robot_point.x = action.robot_point[0]
    msg.robot_point.y = action.robot_point[1]
    msg.robot_point.z = action.robot_point[2]

    msg.object_point.x = action.object_point[0]
    msg.object_point.y = action.object_point[1]
    msg.object_point.z = action.object_point[2]

    msg.object_id = action.object_id

    return msg


@to_viz_msg.register
def _(action: Place, marker_ns):
    pt1 = Point()
    pt1.x = action.robot_point[0]
    pt1.y = action.robot_point[1]
    pt1.z = action.robot_point[2]
    pt2 = Point()
    pt2.x = action.object_point[0]
    pt2.y = action.object_point[1]
    pt2.z = action.object_point[2]

    m = Marker()
    m.header.frame_id = action.frame
    m.header.stamp = gtm()
    m.ns = marker_ns
    m.id = 0
    m.type = m.ARROW
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.1  # shaft diameter
    m.scale.y = 0.2  # head diameter
    m.color = robot_color(marker_ns)
    m.points = [pt1, pt2]

    return [m]
