import numpy as np
import rospy
import skimage as ski
import tf2_ros
from cv_bridge import CvBridge
from phoenix_tamp_planner.action_descriptions import (
    Follow,
    Gaze,
    Pick,
    Place,
)
from phoenix_tamp_planner.utils import transform_command_frame, waypoints_to_path

from spot_skills.arm_utils import gaze_at_vision_pose
from spot_skills.bezier_path import smooth_path
from spot_skills.grasp_utils import object_grasp, object_place
from spot_skills.navigation_utils import (
    follow_trajectory,
    follow_trajectory_continuous,
    turn_to_point,
)


class SpotExecutor:
    def __init__(self, spot_interface):
        self.debug = False
        self.spot_interface = spot_interface

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def process_action_sequence(self, sequence):
        print("Would like to execute: ")
        for command in sequence.actions:
            print(command)
        resp = input("Proceed? [Y/n]")
        if str.lower(resp) not in ["y", "yes", ""]:
            print("Aborting")
            return

        self.spot_interface.robot.time_sync.wait_for_sync()
        self.spot_interface.take_lease()

        for ix, command in enumerate(sequence.actions):
            pick_next = False
            if ix < len(sequence.actions) - 1:
                pick_next = type(sequence.actions[ix + 1]) is Pick
            print("Spot executor executing command: ")
            print(command)
            if type(command) is Follow:
                self.execute_follow(command)

            elif type(command) is Gaze:
                self.execute_gaze(command, pick_next=pick_next)

            elif type(command) is Pick:
                self.execute_pick(command)

            elif type(command) is Place:
                self.execute_place(command)

            else:
                raise Exception(
                    f"SpotExecutor received unknown command type {type(command)}"
                )

    def execute_gaze(self, command, pick_next=False):
        print("Executing `gaze` command")
        current_position = self.spot_interface.get_pose()
        turn_to_point(self.spot_interface, current_position, command.gaze_point)
        # stow_after = command.stow_after
        stow_after = not pick_next
        success = gaze_at_vision_pose(
            self.spot_interface, command.gaze_point, stow_after=stow_after
        )
        print("Finished `gaze` command")
        return success

    def execute_pick(self, command):
        print("Executing `pick` command")
        success = object_grasp(
            self.spot_interface,
            semantic_class=command.object_class,
            labelspace_map=self.labelspace_map,
            debug=self.debug,
        )

        if self.debug:
            success, debug_images = success
            bridge = CvBridge()
            sem_img = ski.util.img_as_ubyte(debug_images[0])
            print("looking for classes: ", self.labelspace_map[command.object_class])
            print("unique semantic labels: ", np.unique(sem_img))
            outline_img = ski.util.img_as_ubyte(debug_images[1])
            semantic_hand_msg = bridge.cv2_to_imgmsg(sem_img, encoding="passthrough")
            mask_img_msg = bridge.cv2_to_imgmsg(outline_img, encoding="passthrough")
            self.semantic_hand_pub.publish(semantic_hand_msg)
            self.semantic_mask_pub.publish(mask_img_msg)

        print("Finished `pick` command")
        return success

    def execute_place(self, command):
        print("Executing `place` command")
        success = object_place(self.spot_interface, semantic_class=command.object_class)
        print("Finished `place` command")
        return success

    def execute_follow(self, command):
        print("Executing `follow` command")
        if self.fixed_frame == "vision":
            command_to_send = command.path2d
        else:
            command_to_send = transform_command_frame(
                self.tf_buffer,
                self.fixed_frame,
                "vision",
                command.path2d,
            )

        use_continuous_follower = True
        if use_continuous_follower:
            goal_tolerance = 1
            path_distance = np.sum(
                np.linalg.norm(np.diff(command_to_send[:, :2], axis=0), axis=1)
            )
            timeout = path_distance * 6
            print(
                f"Using continous follower with params:\n\tlookahead: {self.follower_lookahead}\n\tgoal tolerance: {goal_tolerance}\n\ttimeout: {timeout}"
            )
            path_debug_viz = waypoints_to_path("vision", command_to_send)
            self.smooth_path_publisher.publish(path_debug_viz)
            ret = follow_trajectory_continuous(
                self.spot_interface,
                command_to_send,
                self.follower_lookahead,
                goal_tolerance,
                timeout,
                progress_point_publisher=self.progress_visualizer_pub,
            )
        else:
            smooth_trajectory = smooth_path(
                command_to_send, heading_mode="average", n_points=10
            )
            path = waypoints_to_path(self.fixed_frame, smooth_trajectory)
            path.header.stamp = rospy.Time.now()
            self.smooth_path_publisher.publish(path)
            ret = follow_trajectory(self.spot_interface, smooth_trajectory)
        return ret
