# from std_msgs.msg import Bool
from cv_bridge import CvBridge
import numpy as np
import rospy
from spot_tools.mapping.outdoor_dsg_utils import \
    load_inverse_semantic_id_map_from_label_space
from spot_skills.arm_utils import gaze_at_vision_pose
from spot_skills.bezier_path import smooth_path
from spot_executor.fake_spot import FakeSpot
from spot_skills.grasp_utils import object_grasp, object_place
from spot_skills.navigation_utils import (follow_trajectory,
                                                  follow_trajectory_continuous,
                                                  turn_to_point)
from spot_executor.spot import Spot
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image

from phoenix_tamp_planner.action_descriptions import (ActionSequence, Follow,
                                                      Gaze, Pick, Place)
from phoenix_tamp_planner.msg import ActionSequenceMsg
from phoenix_tamp_planner.utils import (transform_command_frame,
                                        waypoints_to_path)
import tf2_ros

import yaml
import skimage as ski

class SpotExecutor:
    def __init__(self):
        self.debug = False
        self.fixed_frame = rospy.get_param("~fixed_frame")
        spot_ip = rospy.get_param("~spot_ip")
        bdai_username = rospy.get_param("~bosdyn_client_username")
        bdai_password = rospy.get_param("~bosdyn_client_password")

        self.follower_lookahead = rospy.get_param("~follower_lookahead")

        semantic_model_path = rospy.get_param("~semantic_model_path")

        labelspace_path = rospy.get_param("~labelspace_path")
        semantic_name_to_id = load_inverse_semantic_id_map_from_label_space(
            labelspace_path
        )
        labelspace_grouping_path= rospy.get_param("~labelspace_grouping_path")
        with open(labelspace_grouping_path, "r") as f:
            grouping_info = yaml.safe_load(f)
        
        # turn list of dictionaries into single dictionary
        self.labelspace_map = {}
        offset = int(grouping_info['offset'])
        for group in grouping_info['groups']:
            self.labelspace_map[group['name']] = [g + offset for g in group['labels']]

        use_fake_spot_interface = rospy.get_param("~use_fake_spot_interface")
        if use_fake_spot_interface:
            external_pose = rospy.get_param("~fake_spot_external_pose")
            static_pose = rospy.get_param("~fake_spot_static_pose")

            if rospy.get_param("~use_fake_spot_pose"):
                spot_x = rospy.get_param("~fake_spot_x")
                spot_y = rospy.get_param("~fake_spot_y")
                spot_z = rospy.get_param("~fake_spot_z")
                spot_yaw = rospy.get_param("~fake_spot_yaw")
                spot_init_pose2d = np.array([spot_x, spot_y, spot_z, spot_yaw])
            else:
                spot_init_pose2d = None

            print("About to initialize fake spot")
            self.spot_interface = FakeSpot(
                username=bdai_username,
                password=bdai_password,
                external_pose=external_pose,
                static_pose=static_pose,
                init_pose=spot_init_pose2d,
                semantic_model_path=semantic_model_path,
                semantic_name_to_id=semantic_name_to_id,
            )
        else:
            print("About to initialize Spot")
            self.spot_interface = Spot(
                username=bdai_username,
                password=bdai_password,
                ip=spot_ip,
                semantic_model_path=semantic_model_path,
                semantic_name_to_id=semantic_name_to_id,
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.smooth_path_publisher = rospy.Publisher(
            "~smooth_path_publisher", Path, queue_size=1, latch=True
        )

        self.progress_visualizer_pub = rospy.Publisher(
            "~path_progress_visualizer", MarkerArray, queue_size=1
        )

        self.semantic_hand_pub = rospy.Publisher(
            "~semantic_hand_image", Image, queue_size=1, latch=True
        )

        self.semantic_mask_pub = rospy.Publisher(
            "~semantic_mask_image", Image, queue_size=1, latch=True
        )

        self.action_sequence_sub = rospy.Subscriber(
            "~action_sequence_subscriber",
            ActionSequenceMsg,
            self.process_action_sequence,
        )

        # TODO: Subscribe to "soft estop" ?
        # self.soft_estop_sub = rospy.Subscriber("~soft_estop", Bool, self.soft_estop_cb)

    def process_action_sequence(self, msg):
        sequence = ActionSequence.from_msg(msg)
        print("Would like to execute: ")
        for command in sequence.actions:
            print(command)
        resp = input("Proceed? [Y/n]")
        if str.lower(resp) not in ['y', 'yes', '']:
            print("Aborting")
            return
        self.spot_interface.robot.time_sync.wait_for_sync()
        self.spot_interface.take_lease()

        for ix, command in enumerate(sequence.actions):
            pick_next = False
            if ix < len(sequence.actions) - 1:
                pick_next = type(sequence.actions[ix + 1]) == Pick
            print("Spot executor executing command: ")
            print(command)
            if type(command) == Follow:
                self.execute_follow(command)

            elif type(command) == Gaze:
                self.execute_gaze(command, pick_next=pick_next)

            elif type(command) == Pick:
                self.execute_pick(command)

            elif type(command) == Place:
                self.execute_place(command)

            else:
                raise Exception(
                    f"SpotExecutor received unknown command type {type(command)}"
                )

    def execute_gaze(self, command, pick_next=False):
        print("Executing `gaze` command")
        current_position = self.spot_interface.get_pose()
        # current_position = np.array(
        #    [current_position.x, current_position.y, current_position.z]
        # )
        turn_to_point(self.spot_interface, current_position, command.gaze_point)
        #stow_after = command.stow_after
        stow_after = not pick_next
        success = gaze_at_vision_pose(self.spot_interface, command.gaze_point, stow_after=stow_after)
        print("Finished `gaze` command")
        return success

    def execute_pick(self, command):
        print("Executing `pick` command")
        success = object_grasp(self.spot_interface, semantic_class=command.object_class, labelspace_map=self.labelspace_map, debug=self.debug)

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
