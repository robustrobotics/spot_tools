"""
Unit tests for direct navigation commands (MoveRelative, TurnRelative, Strafe, Stop, StandSit).

These tests use FakeSpot and mock navigate_to_relative_pose (which requires get_state()
that FakeSpot doesn't implement) to verify executor dispatch and logic.

Run with: python -m pytest spot_tools/tests/test_direct_navigation.py -v
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    MoveRelative,
    StandSit,
    Stop,
    Strafe,
    TurnRelative,
)

from spot_executor.fake_spot import FakeSpot
from spot_executor.spot_executor import SpotExecutor


# --- Test Fixtures ---


class MinimalFeedback:
    """Minimal feedback collector for tests (no matplotlib dependency)."""

    def __init__(self):
        self.break_out_of_waiting_loop = False
        self.logs = []

    def print(self, level, s):
        self.logs.append((level, str(s)))

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


def tf_identity(parent, child):
    return np.array([0, 0, 0.0]), np.array([0.0, 0, 0, 1])


@pytest.fixture
def fake_spot():
    return FakeSpot(init_pose=np.array([0.0, 0.0, 0.0, 0.0]))


@pytest.fixture
def feedback():
    return MinimalFeedback()


@pytest.fixture
def executor(fake_spot):
    from robot_executor_interface.mid_level_planner import IdentityPlanner

    planner = IdentityPlanner(MinimalFeedback())
    return SpotExecutor(
        fake_spot,
        detector=None,
        transform_lookup=tf_identity,
        planner=planner,
    )


# --- Action Dataclass Tests ---


class TestActionDataclasses:
    def test_move_relative_creation(self):
        cmd = MoveRelative(distance_m=2.5)
        assert cmd.distance_m == 2.5

    def test_move_relative_negative(self):
        cmd = MoveRelative(distance_m=-1.0)
        assert cmd.distance_m == -1.0

    def test_turn_relative_creation(self):
        cmd = TurnRelative(angle_deg=90.0)
        assert cmd.angle_deg == 90.0

    def test_turn_relative_negative(self):
        cmd = TurnRelative(angle_deg=-45.0)
        assert cmd.angle_deg == -45.0

    def test_strafe_creation(self):
        cmd = Strafe(distance_m=1.5)
        assert cmd.distance_m == 1.5

    def test_stop_creation(self):
        cmd = Stop()
        assert isinstance(cmd, Stop)

    def test_stand_sit_stand(self):
        cmd = StandSit(action="stand")
        assert cmd.action == "stand"

    def test_stand_sit_sit(self):
        cmd = StandSit(action="sit")
        assert cmd.action == "sit"


# --- Executor Dispatch Tests ---


class TestExecutorDispatch:
    """Test that SpotExecutor correctly dispatches to the right execute_* method."""

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_move_relative_dispatches(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [MoveRelative(distance_m=2.0)])
        executor.process_action_sequence(seq, feedback)
        mock_nav.assert_called_once()
        args = mock_nav.call_args
        assert args[0][1].x == 2.0
        assert args[0][1].y == 0.0
        assert args[0][1].angle == 0.0

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_move_relative_backward(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [MoveRelative(distance_m=-3.0)])
        executor.process_action_sequence(seq, feedback)
        args = mock_nav.call_args
        assert args[0][1].x == -3.0

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_turn_relative_dispatches(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [TurnRelative(angle_deg=90.0)])
        executor.process_action_sequence(seq, feedback)
        mock_nav.assert_called_once()
        args = mock_nav.call_args
        assert args[0][1].x == 0.0
        assert args[0][1].y == 0.0
        assert abs(args[0][1].angle - np.deg2rad(90.0)) < 1e-6

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_turn_relative_negative(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [TurnRelative(angle_deg=-45.0)])
        executor.process_action_sequence(seq, feedback)
        args = mock_nav.call_args
        assert abs(args[0][1].angle - np.deg2rad(-45.0)) < 1e-6

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_strafe_dispatches(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [Strafe(distance_m=1.5)])
        executor.process_action_sequence(seq, feedback)
        mock_nav.assert_called_once()
        args = mock_nav.call_args
        assert args[0][1].x == 0.0
        assert args[0][1].y == 1.5
        assert args[0][1].angle == 0.0

    @patch("spot_executor.spot_executor.navigate_to_absolute_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_stop_dispatches(self, mock_sleep, mock_nav, executor, fake_spot, feedback):
        fake_spot.set_pose(np.array([5.0, 3.0, 0.0, 1.2]))
        seq = ActionSequence("test", "spot", [Stop()])
        executor.process_action_sequence(seq, feedback)
        mock_nav.assert_called_once()
        # Verify it commands hold at current pose
        args = mock_nav.call_args
        waypoint = args[0][1]
        assert abs(waypoint.x - 5.0) < 1e-6
        assert abs(waypoint.y - 3.0) < 1e-6
        assert abs(waypoint.angle - 1.2) < 1e-6

    def test_stand_sit_stand(self, executor, fake_spot, feedback):
        fake_spot.stand = MagicMock()
        seq = ActionSequence("test", "spot", [StandSit(action="stand")])
        executor.process_action_sequence(seq, feedback)
        fake_spot.stand.assert_called_once()

    def test_stand_sit_sit(self, executor, fake_spot, feedback):
        fake_spot.sit = MagicMock()
        seq = ActionSequence("test", "spot", [StandSit(action="sit")])
        executor.process_action_sequence(seq, feedback)
        fake_spot.sit.assert_called_once()

    def test_stand_sit_invalid_raises(self, executor, feedback):
        seq = ActionSequence("test", "spot", [StandSit(action="jump")])
        with pytest.raises(ValueError, match="Unknown StandSit action"):
            executor.process_action_sequence(seq, feedback)


# --- Stop Preemption Tests ---


class TestStopPreemption:
    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.navigate_to_absolute_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_stop_cancels_remaining_actions(
        self, mock_sleep, mock_abs_nav, mock_rel_nav, executor, fake_spot, feedback
    ):
        """Stop should prevent subsequent actions from executing."""
        fake_spot.set_pose(np.array([0.0, 0.0, 0.0, 0.0]))
        seq = ActionSequence(
            "test",
            "spot",
            [Stop(), MoveRelative(distance_m=5.0)],
        )
        executor.process_action_sequence(seq, feedback)
        # navigate_to_absolute_pose called for Stop's hold-position
        mock_abs_nav.assert_called_once()
        # navigate_to_relative_pose should NOT be called (MoveRelative was skipped)
        mock_rel_nav.assert_not_called()

    @patch("spot_executor.spot_executor.navigate_to_absolute_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_stop_sets_break_flag(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence("test", "spot", [Stop()])
        executor.process_action_sequence(seq, feedback)
        assert feedback.break_out_of_waiting_loop is True


# --- Sequence Tests ---


class TestActionSequences:
    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_multiple_moves_in_sequence(self, mock_sleep, mock_nav, executor, feedback):
        seq = ActionSequence(
            "test",
            "spot",
            [
                MoveRelative(distance_m=1.0),
                TurnRelative(angle_deg=90.0),
                MoveRelative(distance_m=2.0),
            ],
        )
        executor.process_action_sequence(seq, feedback)
        assert mock_nav.call_count == 3

    @patch("spot_executor.spot_executor.navigate_to_relative_pose")
    @patch("spot_executor.spot_executor.time.sleep")
    def test_move_then_sit(self, mock_sleep, mock_nav, executor, fake_spot, feedback):
        fake_spot.sit = MagicMock()
        seq = ActionSequence(
            "test",
            "spot",
            [MoveRelative(distance_m=1.0), StandSit(action="sit")],
        )
        executor.process_action_sequence(seq, feedback)
        mock_nav.assert_called_once()
        fake_spot.sit.assert_called_once()


# --- ROS Message Serialization Tests ---


class TestMessageSerialization:
    def test_move_relative_roundtrip(self):
        from robot_executor_interface.action_descriptions import ActionSequence
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence(
            "test", "spot", [MoveRelative(distance_m=2.5)]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert len(result.actions) == 1
        assert isinstance(result.actions[0], MoveRelative)
        assert result.actions[0].distance_m == 2.5

    def test_turn_relative_roundtrip(self):
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence(
            "test", "spot", [TurnRelative(angle_deg=-45.0)]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], TurnRelative)
        assert result.actions[0].angle_deg == -45.0

    def test_strafe_roundtrip(self):
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence("test", "spot", [Strafe(distance_m=1.0)])
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], Strafe)
        assert result.actions[0].distance_m == 1.0

    def test_stop_roundtrip(self):
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence("test", "spot", [Stop()])
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], Stop)

    def test_stand_sit_roundtrip(self):
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence(
            "test", "spot", [StandSit(action="sit")]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], StandSit)
        assert result.actions[0].action == "sit"

    def test_mixed_sequence_roundtrip(self):
        from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg

        original = ActionSequence(
            "test",
            "spot",
            [
                MoveRelative(distance_m=1.0),
                TurnRelative(angle_deg=90.0),
                Strafe(distance_m=-0.5),
                StandSit(action="stand"),
            ],
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert len(result.actions) == 4
        assert isinstance(result.actions[0], MoveRelative)
        assert isinstance(result.actions[1], TurnRelative)
        assert isinstance(result.actions[2], Strafe)
        assert isinstance(result.actions[3], StandSit)
