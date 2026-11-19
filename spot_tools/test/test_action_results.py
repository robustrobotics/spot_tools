"""Exceptional action failures reach the per-action result channel exactly once."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Place,
)

import spot_executor.spot_executor as se


@pytest.mark.parametrize("setup_fails", [False, True])
def test_exception_reports_failure_with_action_identity(setup_fails):
    def boom(*_):
        raise RuntimeError("skill failed")

    spot = NS(
        robot=NS(time_sync=NS(wait_for_sync=boom if setup_fails else lambda: None)),
        take_lease=lambda: None,
    )
    ex = se.SpotExecutor(spot, None, None, None)
    ex.execute_follow = boom
    results = []
    feedback = NS(
        print=lambda *_: None, action_result=lambda *args: results.append(args)
    )
    action = Follow("map", np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="skill failed"):
        ex.process_action_sequence(ActionSequence("plan", "robot", [action]), feedback)
    assert results == [(action, 0, "FAILED", "RuntimeError: skill failed")]
    assert not ex.processing_action_sequence


def test_broken_feedback_does_not_mask_skill_exception():
    def boom(*_):
        raise RuntimeError("skill failed")

    ex = se.SpotExecutor(
        NS(robot=NS(time_sync=NS(wait_for_sync=lambda: None)), take_lease=lambda: None),
        None,
        None,
        None,
    )
    ex.execute_follow = boom
    feedback = NS(print=lambda *_: None, action_result=boom)
    with pytest.raises(RuntimeError, match="skill failed"):
        ex.process_action_sequence(
            ActionSequence("p", "r", [Follow("map", np.zeros((2, 3)))]), feedback
        )
    assert not ex.processing_action_sequence


def _run(actions, outcomes, monkeypatch):
    """Run a sequence whose attempts end as outcomes says, in order; returns
    (attempts made, action results reported)."""
    monkeypatch.setattr(se.time, "sleep", lambda *_: None)
    ex = se.SpotExecutor(
        NS(robot=NS(time_sync=NS(wait_for_sync=lambda: None)), take_lease=lambda: None),
        None,
        None,
        None,
    )
    left = list(outcomes)
    ex.execute_follow = ex.execute_place = lambda *_: left.pop(0)
    results = []
    feedback = NS(
        print=lambda *_: None, action_result=lambda *args: results.append(args[1:])
    )
    ex.process_action_sequence(ActionSequence("plan", "robot", actions), feedback)
    return len(outcomes) - len(left), results


def test_a_follow_interrupted_many_times_is_still_finished(monkeypatch):
    """2026-11-18: the operator took hamilton's lease three times during one
    long follow. The follow was given up, and the pick after it started far
    from the cone."""
    follow = Follow("map", np.zeros((2, 3)))
    tries, results = _run([follow, follow], [False] * 10 + [True, True], monkeypatch)
    assert tries == 12
    assert results == [(0, "SUCCESS", ""), (1, "SUCCESS", "")]


def test_a_follow_is_given_up_after_a_hundred_attempts(monkeypatch):
    follow = Follow("map", np.zeros((2, 3)))
    tries, results = _run([follow], [False] * 150, monkeypatch)
    assert tries == se.MAX_FOLLOW_ATTEMPTS == 100
    assert results == [(0, "FAILED", "gave up after 100 attempts")]


def test_other_actions_are_still_given_up_after_three_attempts(monkeypatch):
    place = Place("", "", np.zeros(3), np.zeros(3), "o1")
    tries, results = _run([place], [False] * 10, monkeypatch)
    assert tries == 3
    assert results == [(0, "FAILED", "gave up after 3 attempts")]
