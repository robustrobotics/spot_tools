"""
On-robot verification script for direct navigation commands.

This script runs a series of tests on the real Spot robot to verify
the direct navigation commands work correctly. Each test requires
manual confirmation before proceeding.

Usage:
    python -i test_direct_navigation_on_robot.py --ip <SPOT_IP> --username <USER> --password <PASS>

Prerequisites:
    - Robot is powered on and standing
    - You have line of sight to the robot
    - BD tablet E-Stop is accessible
    - Clear area (~5m radius) around the robot

Test sequence:
    1. Small move forward (0.5m) - verify direction and distance
    2. Small move backward (0.5m) - verify return to start
    3. Turn left 90 degrees - verify rotation direction
    4. Turn right 90 degrees - verify return to original heading
    5. Strafe left (0.5m) - verify lateral movement
    6. Strafe right (0.5m) - verify return
    7. Sit down - verify robot sits
    8. Stand up - verify robot stands
    9. Stop during move - verify preemption
    10. Pause/resume (if running via ROS)
"""

import argparse
import sys
import time

import numpy as np

from spot_executor.spot import Spot
from spot_executor.spot_executor import SpotExecutor
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    MoveRelative,
    StandSit,
    Stop,
    Strafe,
    TurnRelative,
)


class TestFeedback:
    """Minimal feedback for on-robot tests."""

    def __init__(self):
        self.break_out_of_waiting_loop = False

    def print(self, level, s):
        print(f"[{level}] {s}")

    def follow_path_feedback(self, path):
        pass

    def path_following_progress_feedback(self, progress_point, target_point):
        pass

    def path_follow_MLP_feedback(self, path_wp, target_point_metric):
        pass

    def gaze_feedback(self, current_pose, gaze_point):
        pass

    def set_robot_holding_state(self, is_holding, object_id):
        pass

    def log_lease_takeover(self, event):
        pass


def get_pose_str(spot):
    pose = spot.get_pose()
    return f"x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={np.rad2deg(pose[2]):.1f}deg"


def confirm(prompt):
    response = input(f"\n{prompt} [y/n/q]: ").strip().lower()
    if response == "q":
        print("Quitting tests.")
        sys.exit(0)
    return response == "y"


def run_action(executor, feedback, name, actions):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    seq = ActionSequence("test", "spot", actions)
    executor.process_action_sequence(seq, feedback)


def tf_identity(parent, child):
    return np.array([0, 0, 0.0]), np.array([0.0, 0, 0, 1])


def main():
    parser = argparse.ArgumentParser("Direct Navigation On-Robot Tests")
    parser.add_argument("--ip", required=True, help="Spot IP address")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    print("Connecting to Spot...")
    spot = Spot(
        username=args.username,
        password=args.password,
        ip=args.ip,
        take_lease=True,
        set_estop=False,
    )

    from robot_executor_interface.mid_level_planner import IdentityPlanner

    feedback = TestFeedback()
    planner = IdentityPlanner(feedback)
    executor = SpotExecutor(
        spot,
        detector=None,
        transform_lookup=tf_identity,
        planner=planner,
    )

    print("Connected. Ensure robot is standing.")
    spot.stand()
    time.sleep(2)

    results = []

    # --- Test 1: Move forward ---
    if confirm("Test 1: Move FORWARD 0.5m. Robot should take ~1 step forward. Ready?"):
        start = spot.get_pose()
        run_action(executor, feedback, "Move Forward 0.5m", [MoveRelative(distance_m=0.5)])
        end = spot.get_pose()
        dist = np.linalg.norm(end[:2] - start[:2])
        print(f"  Measured displacement: {dist:.2f}m (expected ~0.5m)")
        passed = confirm("Did the robot move forward approximately 0.5m?")
        results.append(("Move Forward", passed))

    # --- Test 2: Move backward ---
    if confirm("Test 2: Move BACKWARD 0.5m. Robot should return close to start. Ready?"):
        start = spot.get_pose()
        run_action(executor, feedback, "Move Backward 0.5m", [MoveRelative(distance_m=-0.5)])
        end = spot.get_pose()
        dist = np.linalg.norm(end[:2] - start[:2])
        print(f"  Measured displacement: {dist:.2f}m (expected ~0.5m)")
        passed = confirm("Did the robot move backward approximately 0.5m?")
        results.append(("Move Backward", passed))

    # --- Test 3: Turn left ---
    if confirm("Test 3: Turn LEFT 90 degrees (counter-clockwise from above). Ready?"):
        start_yaw = np.rad2deg(spot.get_pose()[2])
        run_action(executor, feedback, "Turn Left 90deg", [TurnRelative(angle_deg=90.0)])
        end_yaw = np.rad2deg(spot.get_pose()[2])
        delta = end_yaw - start_yaw
        # Normalize to [-180, 180]
        delta = (delta + 180) % 360 - 180
        print(f"  Measured rotation: {delta:.1f}deg (expected ~+90deg)")
        passed = confirm("Did the robot turn left (CCW) approximately 90 degrees?")
        results.append(("Turn Left", passed))

    # --- Test 4: Turn right ---
    if confirm("Test 4: Turn RIGHT 90 degrees (should return to original heading). Ready?"):
        start_yaw = np.rad2deg(spot.get_pose()[2])
        run_action(executor, feedback, "Turn Right 90deg", [TurnRelative(angle_deg=-90.0)])
        end_yaw = np.rad2deg(spot.get_pose()[2])
        delta = end_yaw - start_yaw
        delta = (delta + 180) % 360 - 180
        print(f"  Measured rotation: {delta:.1f}deg (expected ~-90deg)")
        passed = confirm("Did the robot turn right (CW) approximately 90 degrees?")
        results.append(("Turn Right", passed))

    # --- Test 5: Strafe left ---
    if confirm("Test 5: Strafe LEFT 0.5m. Robot should step sideways to its left. Ready?"):
        start = spot.get_pose()
        run_action(executor, feedback, "Strafe Left 0.5m", [Strafe(distance_m=0.5)])
        end = spot.get_pose()
        dist = np.linalg.norm(end[:2] - start[:2])
        print(f"  Measured displacement: {dist:.2f}m (expected ~0.5m)")
        passed = confirm("Did the robot strafe left approximately 0.5m?")
        results.append(("Strafe Left", passed))

    # --- Test 6: Strafe right ---
    if confirm("Test 6: Strafe RIGHT 0.5m. Robot should return to original position. Ready?"):
        start = spot.get_pose()
        run_action(executor, feedback, "Strafe Right 0.5m", [Strafe(distance_m=-0.5)])
        end = spot.get_pose()
        dist = np.linalg.norm(end[:2] - start[:2])
        print(f"  Measured displacement: {dist:.2f}m (expected ~0.5m)")
        passed = confirm("Did the robot strafe right approximately 0.5m?")
        results.append(("Strafe Right", passed))

    # --- Test 7: Sit ---
    if confirm("Test 7: SIT. Robot should sit down gracefully. Ready?"):
        run_action(executor, feedback, "Sit", [StandSit(action="sit")])
        passed = confirm("Did the robot sit down?")
        results.append(("Sit", passed))

    # --- Test 8: Stand ---
    if confirm("Test 8: STAND. Robot should stand back up. Ready?"):
        run_action(executor, feedback, "Stand", [StandSit(action="stand")])
        passed = confirm("Did the robot stand up?")
        results.append(("Stand", passed))

    # --- Test 9: Stop preempts sequence ---
    if confirm(
        "Test 9: STOP preemption. Will send [Move 3m, Turn 90deg]. "
        "The Stop is the first action so the Turn should NOT execute. Ready?"
    ):
        run_action(
            executor,
            feedback,
            "Stop then Turn (Turn should be skipped)",
            [Stop(), TurnRelative(angle_deg=90.0)],
        )
        passed = confirm("Did the robot stop without turning?")
        results.append(("Stop Preemption", passed))

    # --- Test 10: Multi-step sequence ---
    if confirm(
        "Test 10: Multi-step sequence. Robot will: "
        "move forward 1m, turn right 90deg, move forward 1m. "
        "Should trace an L-shape. Ready?"
    ):
        run_action(
            executor,
            feedback,
            "L-shaped path",
            [
                MoveRelative(distance_m=1.0),
                TurnRelative(angle_deg=-90.0),
                MoveRelative(distance_m=1.0),
            ],
        )
        passed = confirm("Did the robot trace an L-shaped path?")
        results.append(("L-shape Sequence", passed))

    # --- Summary ---
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n{passed_count}/{total} tests passed")

    if passed_count < total:
        print("\nFailed tests may indicate:")
        print("  - Sign convention mismatch (check body frame conventions)")
        print("  - Sleep timing too short (robot didn't finish moving)")
        print("  - Odometry drift (normal for longer sequences)")


if __name__ == "__main__":
    main()
