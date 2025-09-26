import time

import numpy as np
import skimage as ski
from robot_executor_interface.action_descriptions import (
    Follow,
    Gaze,
    Pick,
    Place,
)
from scipy.spatial.transform import Rotation

from spot_skills.arm_utils import gaze_at_vision_pose
from spot_skills.grasp_utils import object_grasp, object_place
from spot_skills.navigation_utils import (
    follow_trajectory_continuous,
    turn_to_point,
)
from robot_executor_interface.mid_level_planner import MidLevelPlanner


def transform_command_frame(tf_trans, tf_q, command, feedback=None):
    # command is Nx3 numpy array

    R = Rotation.from_quat([tf_q.x, tf_q.y, tf_q.z, tf_q.w])
    _, _, yaw = R.as_euler("xyz", degrees=False)

    for ix in range(len(command)):
        c = command[ix]
        x, y = c[0:2]

        command[ix, 0] = np.cos(yaw) * x - np.sin(yaw) * y + tf_trans[0]
        command[ix, 1] = np.sin(yaw) * x + np.cos(yaw) * y + tf_trans[1]
        command[ix, 2] += yaw
    return command


class SpotExecutor:
    def __init__(
        self,
        spot_interface,
        detector,
        transform_lookup,
        follower_lookahead=2,
        goal_tolerance=2.8,
    ):
        self.debug = False
        self.spot_interface = spot_interface
        self.transform_lookup = transform_lookup
        self.follower_lookahead = follower_lookahead
        self.goal_tolerance = goal_tolerance
        self.detector = detector
        self.keep_going = True
        self.processing_action_sequence = False
        self.mid_level_planner = MidLevelPlanner()

    def terminate_sequence(self, feedback):
        # Tell the actions sequence to break
        self.keep_going = False

        # Blocking the thread so that it terminates cleanly by
        # terminating the pick action and waiting for processing to end
        feedback.break_out_of_waiting_loop = True

        # Block until action sequence is done executing
        while self.processing_action_sequence:
            feedback.print("INFO", "Waiting for previous action sequence to terminate.")
            time.sleep(1)

    def process_action_sequence(self, sequence, feedback):
        self.processing_action_sequence = True
        self.keep_going = True
        try:
            feedback.print("INFO", "Would like to execute: ")
            for command in sequence.actions:
                feedback.print("INFO", command)

            self.spot_interface.robot.time_sync.wait_for_sync()
            self.spot_interface.take_lease()

            for ix, command in enumerate(sequence.actions):
                if not self.keep_going:
                    feedback.print("INFO", "Action sequence was pre-empted.")
                    break
                pick_next = False
                if ix < len(sequence.actions) - 1:
                    pick_next = type(sequence.actions[ix + 1]) is Pick
                feedback.print("INFO", "Spot executor executing command: ")
                feedback.print("INFO", command)
                if type(command) is Follow:
                    self.execute_follow(command, feedback)

                elif type(command) is Gaze:
                    self.execute_gaze(command, feedback, pick_next=pick_next)

                elif type(command) is Pick:
                    self.execute_pick(command, feedback)

                elif type(command) is Place:
                    self.execute_place(command, feedback)

                else:
                    raise Exception(
                        f"SpotExecutor received unknown command type {type(command)}"
                    )
        except Exception as ex:
            self.processing_action_sequence = False
            raise ex

        self.processing_action_sequence = False

    def execute_gaze(self, command, feedback, pick_next=False):
        # TODO: need to transform command to robot odom frame
        feedback.print("INFO", "Executing `gaze` command")
        current_pose = self.spot_interface.get_pose()
        turn_to_point(self.spot_interface, current_pose, command.gaze_point)
        # stow_after = command.stow_after
        stow_after = not pick_next
        success = gaze_at_vision_pose(
            self.spot_interface, command.gaze_point, stow_after=stow_after
        )
        feedback.gaze_feedback(current_pose, command.gaze_point)
        feedback.print("INFO", "Finished `gaze` command")
        return success

    def execute_pick(self, command, feedback):
        feedback.print("INFO", "Executing `pick` command")

        success = object_grasp(
            self.spot_interface,
            self.detector,
            image_source="hand_color_image",
            user_input=False,
            semantic_class=command.object_class,
            feedback=feedback,
        )

        if self.debug:
            success, debug_images = success
            sem_img = ski.util.img_as_ubyte(debug_images[0])
            feedback.print(
                "INFO",
                "looking for classes: ",
                self.spot_interface.labelspace_map[command.object_class],
            )
            feedback.print("INFO", "unique semantic labels: ", np.unique(sem_img))
            outline_img = ski.util.img_as_ubyte(debug_images[1])

            feedback.pick_image_feedback(sem_img, outline_img)

        feedback.print("INFO", "Finished `pick` command")
        return success

    def execute_place(self, command, feedback):
        feedback.print("INFO", "Executing `place` command")
        success = object_place(self.spot_interface, semantic_class=command.object_class)
        feedback.print("INFO", "Finished `place` command")
        return success

    def execute_follow(self, command, feedback):
        feedback.print("INFO", "Executing `follow` command")
        feedback.print(
            "INFO", f"transforming path from {command.frame} to <spot_vision_frame>"
        )

        # <spot_vision_frame> gets remapped to the actual robot odom frame name
        # by the transform_lookup function.
        t, r = self.transform_lookup("<spot_vision_frame>", command.frame)
        command_to_send = transform_command_frame(
            t, r, command.path2d, feedback=feedback
        ) # the path2d in is our odom frame
        # TODO: Have a class that keep track of the occupancy grid
        # /home/multyxu/dcist_ws/src/awesome_dcist_t4/spot_tools/robot_executor_interface/robot_executor_interface/src/robot_executor_interface 
        path_distance = np.sum(
            np.linalg.norm(np.diff(command_to_send[:, :2], axis=0), axis=1)
        )
        timeout = path_distance * 6
        feedback.print(
            "INFO",
            f"Using continous follower with params:\n\tlookahead: {self.follower_lookahead}\n\tgoal tolerance: {self.goal_tolerance}\n\ttimeout: {timeout}",
        )

        feedback.follow_path_feedback(command_to_send)
        
        # MLP
        # blabala  -> fail
        
        # fall back
        ret = follow_trajectory_continuous(
            self.spot_interface,
            command_to_send,
            self.follower_lookahead,
            self.goal_tolerance,
            timeout,
            feedback=feedback,
        ) # TODO: pass in the self.mid_level_planner
        return ret
