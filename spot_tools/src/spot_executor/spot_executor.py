import threading
import time

import numpy as np
import skimage as ski
from bosdyn.api.robot_state_pb2 import BehaviorFault
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import BehaviorFaultError
from robot_executor_interface.action_descriptions import (
    Follow,
    Gaze,
    Pick,
    Place,
)
from scipy.spatial.transform import Rotation

from spot_skills.arm_utils import gaze_at_vision_pose
from spot_skills.grasp_utils import object_grasp, object_place, stow_arm
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


# The lease manager should run in a separate thread to handle the exchange
# of the lease e.g., when the tablet takes control of the robot.
class LeaseManager:
    def __init__(self, spot_interface, feedback=None):
        self.spot_interface = spot_interface
        self.monitoring_thread = None
        self.feedback = feedback

        self.initialize_thread()
        self.taking_back_lease = False

        leases = self.spot_interface.lease_client.list_leases()
        self.owner = leases[0].lease_owner
        self.owner_name = self.owner.client_name

    def initialize_thread(self):
        def monitor_lease():
            while True:
                leases = self.spot_interface.lease_client.list_leases()

                # owner of the full lease
                self.owner = leases[0].lease_owner
                self.owner_name = self.owner.client_name

                # If nobody owns the lease, then the owner string is empty.
                # We should try to take the lease back in that case.
                if self.owner_name == "":
                    # We should set the feedback's break_out_of_waiting_loop to True
                    # so that the pick skill gets immediately cancelled if it is running.
                    if self.feedback is not None:
                        self.feedback.break_out_of_waiting_loop = True

                    self.taking_back_lease = True
                    if self.feedback is not None:
                        self.feedback.print(
                            "INFO",
                            "LEASE MANAGER THREAD: Trying to take lease back, since nobody owns it.",
                        )
                    self.spot_interface.take_lease()
                    try:
                        stow_arm(self.spot_interface)
                        self.spot_interface.stand()
                    except BehaviorFaultError:
                        fault_ids = []
                        for fault in (
                            self.spot_interface.get_state().behavior_fault_state.faults
                        ):
                            if fault.cause == BehaviorFault.CAUSE_LEASE_TIMEOUT:
                                fault_ids.append(fault.behavior_fault_id)
                        for fault_id in fault_ids:
                            if self.feedback is not None:
                                self.feedback.print(
                                    "INFO",
                                    f"LEASE MANAGER THREAD: Clearing behavior fault {fault_id}",
                                )
                            self.spot_interface.command_client.clear_behavior_fault(
                                fault_id
                            )

                        if (
                            len(
                                self.spot_interface.get_state().behavior_fault_state.faults
                            )
                            == 0
                        ):
                            self.spot_interface.stand()
                        else:
                            if self.feedback is not None:
                                self.feedback.print(
                                    "WARN",
                                    "LEASE MANAGER THREAD: Could not clear all behavior faults, cannot stand.",
                                )
                    time.sleep(1)
                    if self.feedback is not None:
                        self.feedback.break_out_of_waiting_loop = False
                    self.taking_back_lease = False
                time.sleep(0.5)

        self.monitoring_thread = threading.Thread(target=monitor_lease, daemon=False)
        self.monitoring_thread.start()


class SpotExecutor:
    def __init__(
        self,
        spot_interface,
        detector,
        transform_lookup,
        planner,
        follower_lookahead=2,
        goal_tolerance=2.8,
        feedback=None,
        use_fake_path_planner=False,
    ):
        self.debug = False
        self.spot_interface = spot_interface
        self.transform_lookup = transform_lookup
        self.follower_lookahead = follower_lookahead
        self.goal_tolerance = goal_tolerance
        self.detector = detector
        self.keep_going = True
        self.processing_action_sequence = False
        self.mid_level_planner = planner
        self.use_fake_path_planner = use_fake_path_planner

        self.lease_manager = None

    def initialize_lease_manager(self, feedback):
        self.lease_manager = LeaseManager(self.spot_interface, feedback)

    def terminate_sequence(self, feedback):
        # Tell the actions sequence to break
        self.keep_going = False

        # Blocking the thread so that it terminates cleanly by
        # terminating the pick action and waiting for processing to end
        feedback.break_out_of_waiting_loop = True

        # Block until action sequence is done executing
        while self.processing_action_sequence:
            feedback.print(
                "INFO",
                "Waiting for previous action sequence to terminate. You must release the lease!",
            )
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

            ix = 0
            inner_loop_attempts = 0
            while ix < len(sequence.actions):
                # If the lease manager is actively taking back the lease and getting the
                # robot to stand back up, we don't want to send it any commands. It will break.
                if (
                    self.lease_manager is not None
                    and self.lease_manager.taking_back_lease
                ):
                    feedback.print(
                        "INFO",
                        "Waiting for lease manager to finish taking back lease...",
                    )
                    time.sleep(1)
                    continue

                # If we don't own the lease, we don't try to take any actions
                if (
                    self.lease_manager is not None
                    and not self.lease_manager.owner_name.startswith("understanding")
                ):
                    time.sleep(0.5)
                    continue

                command = sequence.actions[ix]

                if not self.keep_going:
                    feedback.print("INFO", "Action sequence was pre-empted.")
                    break
                pick_next = False
                if ix < len(sequence.actions) - 1:
                    pick_next = type(sequence.actions[ix + 1]) is Pick
                feedback.print("INFO", "\n")
                feedback.print("INFO", "Spot executor executing command: ")
                feedback.print("INFO", command)

                success = False
                try:
                    if type(command) is Follow:
                        success = self.execute_follow(command, feedback)
                        feedback.print("INFO", f"Finished `follow` command with return {success}")

                    elif type(command) is Gaze:
                        success = self.execute_gaze(
                            command, feedback, pick_next=pick_next
                        )

                    elif type(command) is Pick:
                        success = self.execute_pick(command, feedback)

                    elif type(command) is Place:
                        success = self.execute_place(command, feedback)

                    else:
                        raise Exception(
                            f"SpotExecutor received unknown command type {type(command)}"
                        )
                    if success or inner_loop_attempts > 1:
                        ix += 1
                        inner_loop_attempts = 0
                    else:
                        inner_loop_attempts += 1
                        time.sleep(1)

                except LeaseUseError:
                    # feedback.print("INFO", "Lost lease, stopping action sequence.")
                    # Wait until the lease manager has taken the lease back
                    time.sleep(2)

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
        feedback.print("INFO", f"Pick skill success: {success}")
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
        )

        path_distance = np.sum(
            np.linalg.norm(np.diff(command_to_send[:, :2], axis=0), axis=1)
        )
        timeout = path_distance * 6
        feedback.print(
            "INFO",
            f"Using continous follower with params:\n\tlookahead: {self.follower_lookahead}\n\tgoal tolerance: {self.goal_tolerance}\n\ttimeout: {timeout}",
        )

        feedback.follow_path_feedback(command_to_send)
        
        if self.mid_level_planner is not None and self.use_fake_path_planner:
            # this only publish the path but does not actually command the spot to follow it
            # TODO: need to refactor this part
            ret = False
            mlp_success, planning_output = self.mid_level_planner.plan_path(command_to_send[:, :2])
            path = planning_output['path_shapely']
            path_wp = planning_output['path_waypoints_metric']
            target_point_metric = planning_output['target_point_metric']
            if not mlp_success:
                feedback.print("INFO", "Mid-level planner failed to find a path")
            if target_point_metric is not None:
                feedback.path_follow_MLP_feedback(path_wp, target_point_metric)
        else:
            ret = follow_trajectory_continuous(
                self.spot_interface,
                command_to_send,
                self.follower_lookahead,
                self.goal_tolerance,
                timeout,
                self.mid_level_planner,
                feedback=feedback,
            )
        return ret
