import argparse
import io
import math
import sys
import time
from typing import Tuple

import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import basic_command_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.manipulation_api_pb2 import (
    ManipulationApiFeedbackRequest,
    ManipulationApiRequest,
    WalkToObjectInImage,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
    get_se2_a_tform_b,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from PIL import Image

from spot_skills.arm_utils import (
    change_gripper,
    close_gripper,
    gaze_at_relative_pose,
    move_hand_to_relative_pose,
    open_gripper,
)
from spot_skills.navigation_utils import (
    navigate_to_absolute_pose,
    navigate_to_relative_pose,
)


def move(spot, start_pose, end_pose, trajectory):
    return traj
