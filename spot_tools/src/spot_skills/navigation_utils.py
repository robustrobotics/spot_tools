"""Interface for moving the spot frame."""

import math
import time
from typing import Tuple, Union

import numpy as np
import rospy
import shapely
from bosdyn.api import geometry_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.sdk import Robot
from numpy.typing import ArrayLike
from visualization_msgs.msg import Marker, MarkerArray

# Global constants to define spot motion speed
MAX_LINEAR_VEL = 0.75
MAX_ROTATION_VEL = 0.65


def navigate_to_relative_pose(
    spot,
    body_tform_goal: math_helpers.SE2Pose,
    max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
    min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0),
    timeout: float = 20.0,
) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.
    """
    # Get the robot's current state.
    robot_state = spot.get_state()
    transforms = robot_state.kinematic_state.transforms_snapshot

    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(
            linear=Vec2(x=max_xytheta_vel[0], y=max_xytheta_vel[1]),
            angular=max_xytheta_vel[2],
        ),
        min_vel=SE2Velocity(
            linear=Vec2(x=min_xytheta_vel[0], y=min_xytheta_vel[1]),
            angular=min_xytheta_vel[2],
        ),
    )
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params,
    )
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + timeout
    )
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = (
            feedback.feedback.synchronized_feedback.mobility_command_feedback
        )
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:  # pylint: disable=no-member,line-too-long
            spot.robot.logger.info("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (
            traj_feedback.status == traj_feedback.STATUS_AT_GOAL
            and traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED
        ):
            return
    if (time.perf_counter() - start_time) > timeout:
        spot.robot.logger.info("Timed out waiting for movement to execute!")


def navigate_to_absolute_pose(
    spot, waypoint, frame_name=VISION_FRAME_NAME, stairs=False
):
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )

    params = RobotCommandBuilder.mobility_params(
        stair_hint=stairs, locomotion_hint=robot_command_pb2.LocomotionHint.HINT_AUTO
    )

    max_vel_linear = geometry_pb2.Vec2(x=MAX_LINEAR_VEL, y=MAX_LINEAR_VEL)
    max_vel_se2 = geometry_pb2.SE2Velocity(
        linear=max_vel_linear, angular=MAX_ROTATION_VEL
    )
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params.vel_limit.CopyFrom(vel_limit)
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=waypoint.x,
        goal_y=waypoint.y,
        goal_heading=waypoint.angle,
        frame_name=frame_name,
        params=params,
    )
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(
        lease=None, command=robot_cmd, end_time_secs=time.time() + end_time
    )
    return cmd_id


def follow_trajectory(
    spot,
    waypoints_list: ArrayLike,
    frame_name=VISION_FRAME_NAME,
    # trajectory_publisher: Union[
    #     spot_ros_utils.TrajectoryPublisher, None
    # ] = None,
    stairs=False,
) -> bool:
    """
    Follows a trajectory by commanding the robot to move to each waypoint in the specified frame.
    Args:
        waypoints_list (ArrayLike): List of list of positions in the format [x, y].
        frame_name (str): Desired frame for the trajectory.
        robot_command_client (RobotCommandClient): Client for sending robot commands.
        robot_state_client (RobotStateClient): Client for receiving robot state.
        occupancy_grid_subscriber (Union[spot_ros_utils.OccupancyGridSubscriber, None], optional): Subscriber for occupancy grid updates. Defaults to None.
        stairs (bool, optional): Flag indicating whether the robot is navigating stairs. Defaults to False.
    Returns:
        bool: True if the trajectory is successfully followed, False otherwise.
    """
    robot_command_client = spot.robot.ensure_client(
        RobotCommandClient.default_service_name
    )

    tform_body_in_vision = spot.get_pose()

    current_path = waypoints_list

    i = 0
    follower = Follower(waypoints_list, heading_mode="leading")
    # Follow each waypoint.
    timeout = 0
    trigger_replan = False
    inner_timed_out = 0

    while not follower.at_end(tform_body_in_vision.x, tform_body_in_vision.y):
        timeout += 1
        if timeout > 500:
            print("Exiting via timeout")
            return False

        current_path = follower.select_furthest(
            tform_body_in_vision.x, tform_body_in_vision.y, current_path=current_path
        )

        current_waypoint = math_helpers.SE2Pose(
            x=current_path[0, 0], y=current_path[0, 1], angle=current_path[0, 2]
        )

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.

        cmd_id = navigate_to_absolute_pose(
            spot, current_waypoint, frame_name, stairs=stairs
        )

        # Wait until the robot has reached the goal.
        inner_timeout = time.time()
        while True:
            curr_time = time.time()
            if curr_time - inner_timeout > 2:
                print("timeout trying to reach waypoint, try to go to next waypoint")
                current_path = current_path[1:]
                inner_timed_out += 1
                if inner_timed_out > 1:
                    trigger_replan = True
                break
            # Check progress towards waypoint.
            tform_body_in_vision = spot.get_pose()
            dist_to_intermediate_waypoint = (
                (tform_body_in_vision.x - current_waypoint.x) ** 2
                + (tform_body_in_vision.y - current_waypoint.y) ** 2
            ) ** 0.5

            # Check robot feedback.
            feedback = robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = (
                feedback.feedback.synchronized_feedback.mobility_command_feedback
            )
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print(mobility_feedback.status)
                print(
                    "Failed to reach intermediate goal: [{}, {}, {}]".format(
                        current_waypoint.x, current_waypoint.y, current_waypoint.angle
                    )
                )
                # current_path = current_path[1:]
                trigger_replan = True
                break
            if (dist_to_intermediate_waypoint < 0.1 and len(current_path) > 1) or (
                dist_to_intermediate_waypoint < 0.1 and len(current_path) == 1
            ):  # and
                # traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print(
                    "Reached intermediate goal: [{}, {}, {}]".format(
                        current_waypoint.x, current_waypoint.y, current_waypoint.angle
                    )
                )
                current_path = current_path[1:]
                i += 1
                break
        if len(current_path) == 0:
            tform_body_in_vision = spot.get_pose()
            dist_to_intermediate_waypoint = (
                (tform_body_in_vision.x - current_waypoint.x) ** 2
                + (tform_body_in_vision.y - current_waypoint.y) ** 2
            ) ** 0.5
            print(
                f"Exiting unable to fully reach goal safely, distance: {dist_to_intermediate_waypoint}"
            )
            return True
    print("Exiting via success")
    return True


def pt_to_marker(pt, ns, mid, color):
    m = Marker()
    m.header.frame_id = "vision"
    m.header.stamp = rospy.Time.now()
    m.ns = ns
    m.id = mid
    m.type = m.SPHERE
    m.action = m.ADD
    m.pose.orientation.w = 1
    m.pose.position.x = pt.x
    m.pose.position.y = pt.y
    m.scale.x = 0.3
    m.scale.y = 0.3
    m.scale.z = 0.3
    m.color.a = 1
    m.color.r = color[0]
    m.color.g = color[1]
    m.color.b = color[2]

    return m


def build_progress_markers(current_point, target_point):
    ma = MarkerArray()
    m1 = pt_to_marker(current_point, "path_progress", 0, [0, 1, 1])
    ma.markers.append(m1)
    m2 = pt_to_marker(target_point, "path_progress", 1, [1, 0, 1])
    ma.markers.append(m2)
    return ma


def follow_trajectory_continuous(
    spot,
    waypoints_list: ArrayLike,
    lookahead_distance: float,
    goal_tolerance: float,
    timeout: float,
    frame_name=VISION_FRAME_NAME,
    stairs=False,
    progress_point_publisher=None,
) -> bool:
    """
    Follows a trajectory by commanding the robot to move to each waypoint in the specified frame.
    Args:
        waypoints_list (ArrayLike): List of list of positions in the format [x, y].
        frame_name (str): Desired frame for the trajectory.
        robot_command_client (RobotCommandClient): Client for sending robot commands.
        robot_state_client (RobotStateClient): Client for receiving robot state.
        occupancy_grid_subscriber (Union[spot_ros_utils.OccupancyGridSubscriber, None], optional): Subscriber for occupancy grid updates. Defaults to None.
        stairs (bool, optional): Flag indicating whether the robot is navigating stairs. Defaults to False.
    Returns:
        bool: True if the trajectory is successfully followed, False otherwise.
    """

    spot.robot.ensure_client(RobotCommandClient.default_service_name)

    path = shapely.LineString(waypoints_list[:, :2])
    end_pt = waypoints_list[-1, :2]
    t0 = rospy.Time.now()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if (rospy.Time.now() - t0).to_sec() > timeout:
            # TODO: I think we need to tell Spot to stop?
            # TODO: Also, we should probably have a finer-grained
            # check about making progress
            return False
        tform_body_in_vision = spot.get_pose()
        distance_from_end = np.linalg.norm(
            end_pt - np.array([tform_body_in_vision.x, tform_body_in_vision.y])
        )
        if distance_from_end < goal_tolerance:
            print("Spot reached end of path")
            break

        # 1. project to current path distance
        current_point = shapely.Point(tform_body_in_vision.x, tform_body_in_vision.y)
        progress_distance = shapely.line_locate_point(path, current_point)
        progress_point = shapely.line_interpolate_point(path, progress_distance)
        # 2. get line point at lookahead
        target_distance = progress_distance + lookahead_distance
        target_point = shapely.line_interpolate_point(path, target_distance)

        yaw_angle = np.arctan2(
            target_point.y - current_point.y, target_point.x - current_point.x
        )

        if progress_point_publisher is not None:
            progress_point_publisher.publish(
                build_progress_markers(progress_point, target_point)
            )
        # 3. send command
        current_waypoint = math_helpers.SE2Pose(
            x=target_point.x, y=target_point.y, angle=yaw_angle
        )

        cmd_id = navigate_to_absolute_pose(
            spot, current_waypoint, frame_name, stairs=stairs
        )
        rate.sleep()

    return True


def turn_to_point(spot, current_position, target_position):
    d = [
        target_position[0] - current_position.x,
        target_position[1] - current_position.y,
    ]
    angle = np.arctan2(d[1], d[0])
    waypoint = math_helpers.SE2Pose(
        x=current_position.x, y=current_position.y, angle=angle
    )
    navigate_to_absolute_pose(spot, waypoint, "vision", stairs=False)


class Follower:
    def __init__(
        self, waypoints: ArrayLike, dr: float = 1.0, heading_mode: str = "trailing"
    ):
        """
        Initialize a PathFollower.
        This takes a global plan of waypoints, assuming you primarily just want to get to the final point.
        Throughout the trajectory, this will pick what should be the next waypoint for the agent to traverse to.
        The heading_mode can be either "leading" or "trailing".
        This determines whether the agent should arrive at the waypoint lined up to the next one or aligned with the previous waypoint.
        """
        self.path = np.array(waypoints)
        self.ind = 0
        # path_xy_interpolated = waypoints
        # self.path = np.concatenate((path_xy_interpolated, yaw.reshape(-1, 1)), axis=1)

    def select_next(
        self, x: float, y: float, distance_threshold: float = 3
    ) -> ArrayLike:
        """Given the x,y coordinates of the agent this returns the next waypoint"""

        self.ind += 1
        return self.path[self.ind :, :]

    def select_furthest(
        self, x: float, y: float, distance_threshold: float = 2.0, current_path=None
    ) -> ArrayLike:
        """Given the x,y coordinates of the agent this returns the intermediate waypoint
        furthest along the list that is less than 'distance_threshold' from the agent.
        If not waypoints are within that distance, then the nearest waypoint is selected.
        """
        if current_path is not None:
            self.path = current_path
        distances = np.linalg.norm(self.path[:, :2] - [x, y], axis=1)
        less_than_threshold = np.where(distances < distance_threshold)[0]
        if less_than_threshold.size > 0:
            ind = less_than_threshold[-1]  # get the last index
        else:
            ind = np.argmin(distances)  # get the index of the minimum value

        # Print out indicies that were skipped
        print("Waypoints that were skipped:", np.arange(self.ind + 1, ind))

        self.ind = ind

        return self.path[ind:, :]

    def at_end(self, x: float, y: float, threshold: float = 0.5):
        """
        Returns True if the agent is within threshold distance of the final goal.
        """
        return (
            np.linalg.norm(
                [
                    x - self.path[-1, 0],
                    y - self.path[-1, 1],
                ]
            )
            <= threshold
        )


# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     bosdyn.client.util.add_base_arguments(parser)
#     parser.add_argument('--frame', choices=[VISION_FRAME_NAME, ODOM_FRAME_NAME],
#                         default=VISION_FRAME_NAME, help='Send the command in this frame.')
#     parser.add_argument('--stairs', action='store_true', help='Move the robot in stairs mode.')
#     options = parser.parse_args()
#     bosdyn.client.util.setup_logging(options.verbose)

#     # Specify desired trajectory.
#     # Default frame is VISION.
#     waypoints_list = [[0., 0.],
#                       [2., 0.],
#                       [2., 5.],
#                       [-2., 5.],
#                       [0., 0.]]

#     # Create robot object.
#     sdk = bosdyn.client.create_standard_sdk('RobotCommandMaster')
#     robot = sdk.create_robot(options.hostname)
#     bosdyn.client.util.authenticate(robot)

#     # Check that an estop is connected with the robot so that the robot commands can be executed.
#     assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
#                                     'such as the estop SDK example, to configure E-Stop.'

#     # Create the lease client.
#     lease_client = robot.ensure_client(LeaseClient.default_service_name)

#     # Setup clients for the robot state and robot command services.
#     robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
#     robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

#     with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
#         # Power on the robot and stand it up.
#         robot.time_sync.wait_for_sync()
#         robot.power_on()
#         blocking_stand(robot_command_client)

#         try:
#             return follow_trajectory(waypoints_list, options.frame,
#                                      robot_command_client, robot_state_client, stairs=options.stairs)
#         finally:
#             # Send a Stop at the end,
#             # Send a Stop at the end, regardless of what happened.
#             # robot_command_client.robot_command(RobotCommandBuilder.stop_command())
#             pass


# if __name__ == '__main__':
#     if not main():
#         sys.exit(1)
