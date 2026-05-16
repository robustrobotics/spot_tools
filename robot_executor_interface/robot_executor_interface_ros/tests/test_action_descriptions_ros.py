"""
Tests for ROS message serialization of direct navigation actions.

Tests conversion between action dataclasses and ROS ActionMsg messages,
verifying roundtrip serialization/deserialization.
"""

import pytest
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    MoveRelative,
    StandSit,
    Stop,
    Strafe,
    TurnRelative,
)
from robot_executor_interface_ros.action_descriptions_ros import from_msg, to_msg


class TestDirectNavigationSerialization:
    """Test ROS message serialization for direct navigation commands."""

    def test_move_relative_roundtrip(self):
        original = ActionSequence(
            "test", "spot", [MoveRelative(distance_m=2.5)]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert len(result.actions) == 1
        assert isinstance(result.actions[0], MoveRelative)
        assert result.actions[0].distance_m == 2.5

    def test_move_relative_negative(self):
        original = ActionSequence(
            "test", "spot", [MoveRelative(distance_m=-3.0)]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert result.actions[0].distance_m == -3.0

    def test_turn_relative_roundtrip(self):
        original = ActionSequence(
            "test", "spot", [TurnRelative(angle_deg=-45.0)]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], TurnRelative)
        assert result.actions[0].angle_deg == -45.0

    def test_strafe_roundtrip(self):
        original = ActionSequence("test", "spot", [Strafe(distance_m=1.0)])
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], Strafe)
        assert result.actions[0].distance_m == 1.0

    def test_stop_roundtrip(self):
        original = ActionSequence("test", "spot", [Stop()])
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], Stop)

    def test_stand_sit_stand(self):
        original = ActionSequence(
            "test", "spot", [StandSit(action="stand")]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], StandSit)
        assert result.actions[0].action == "stand"

    def test_stand_sit_sit(self):
        original = ActionSequence(
            "test", "spot", [StandSit(action="sit")]
        )
        msg = to_msg(original)
        result = from_msg(msg)
        assert isinstance(result.actions[0], StandSit)
        assert result.actions[0].action == "sit"

    def test_mixed_sequence_roundtrip(self):
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
