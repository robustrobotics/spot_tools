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
