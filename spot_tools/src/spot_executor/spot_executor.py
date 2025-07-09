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


def transform_command_frame(tf_trans, tf_q, command, feedback=None):
    # command is Nx3 numpy array
    print("command in: ", command)

    R = Rotation.from_quat([tf_q.x, tf_q.y, tf_q.z, tf_q.w])
    _, _, yaw = R.as_euler("xyz", degrees=False)

    if feedback is not None:
        feedback.print("INFO", "rotating path")
        feedback.print("INFO", f"translation: {str(tf_trans)}")
        feedback.print("INFO", f"rotration: {yaw}")

    feedback.print("INFO", "path before:")
    feedback.print("INFO", str(command))
    for ix in range(len(command)):
        c = command[ix]
        x, y = c[0:2]

        # command[ix, 0] = np.cos(yaw) * c[0] - np.sin(yaw) * c[1] + tf_trans[0]
        # command[ix, 1] = np.sin(yaw) * c[0] + np.cos(yaw) * c[1] + tf_trans[1]

        command[ix, 0] = np.cos(yaw) * x - np.sin(yaw) * y + tf_trans[0]
        command[ix, 1] = np.sin(yaw) * x + np.cos(yaw) * y + tf_trans[1]

        # TODO: check if this actually transforms yaw correctly
        command[ix, 2] += yaw

    feedback.print("INFO", "path after:")
    feedback.print("INFO", str(command))

    return command


class SpotExecutor:
    def __init__(self, spot_interface, transform_lookup):
        self.debug = False
        self.spot_interface = spot_interface
        self.transform_lookup = transform_lookup
        self.fixed_frame = "map"
        self.follower_lookahead = 2

    def process_action_sequence(self, sequence, feedback):
        feedback.print("INFO", "Would like to execute: ")
        for command in sequence.actions:
            feedback.print("INFO", command)

        self.spot_interface.robot.time_sync.wait_for_sync()
        self.spot_interface.take_lease()

        for ix, command in enumerate(sequence.actions):
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

    def execute_gaze(self, command, feedback, pick_next=False):
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
            semantic_class=command.object_class,
            labelspace_map=self.labelspace_map,
            debug=self.debug,
        )

        if self.debug:
            success, debug_images = success
            sem_img = ski.util.img_as_ubyte(debug_images[0])
            feedback.print(
                "INFO",
                "looking for classes: ",
                self.labelspace_map[command.object_class],
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
        feedback.print("INFO", f"self.fixed_frame: {self.fixed_frame}")
        if self.fixed_frame == "vision":
            feedback.print("INFO", "passing through command")
            command_to_send = command.path2d
        else:
            feedback.print(
                "INFO", f"transforming path from {self.fixed_frame} to hamilton/odom"
            )
            # t, r = self.transform_lookup(self.fixed_frame, "hamilton/odom")
            t, r = self.transform_lookup("hamilton/odom", self.fixed_frame)
            command_to_send = transform_command_frame(
                t, r, command.path2d, feedback=feedback
            )

        goal_tolerance = 2.8
        path_distance = np.sum(
            np.linalg.norm(np.diff(command_to_send[:, :2], axis=0), axis=1)
        )
        timeout = path_distance * 6
        feedback.print(
            "INFO",
            f"Using continous follower with params:\n\tlookahead: {self.follower_lookahead}\n\tgoal tolerance: {goal_tolerance}\n\ttimeout: {timeout}",
        )

        feedback.follow_path_feedback(command_to_send)
        ret = follow_trajectory_continuous(
            self.spot_interface,
            command_to_send,
            self.follower_lookahead,
            goal_tolerance,
            timeout,
            feedback=feedback,
        )
        return ret
