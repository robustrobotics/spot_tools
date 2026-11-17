"""After a pick the robot believes worked, the operator says whether it did.

"picked" continues, "retry" lets go and picks again (as often as the operator
asks), "give up" ends the action sequence so nothing runs as if the robot held
the object.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import ActionSequence, Pick, Place

import spot_executor.spot_executor as se

PICK = Pick("map", "bag", np.zeros(3), np.zeros(3), "o7")
PLACE = Place("map", "bag", np.zeros(3), np.zeros(3), "o7")


class Feedback:
    def __init__(self, answers=None, can_ask=True):
        self.answers = list(answers or [])
        self.asked = []
        self.results = []
        self.done = []
        self.holding = []
        self.break_out_of_waiting_loop = False
        if can_ask:
            self.confirm_pick_result = self._confirm

    def print(self, *_):
        pass

    def _confirm(self, **question):
        self.asked.append(question)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def action_result(self, command, index, status, detail):
        self.results.append((type(command).__name__, index, status, detail))

    def report_action_done(self, sequence, command, success, note=""):
        self.done.append((type(command).__name__, success) + ((note,) if note else ()))

    def set_robot_holding_state(self, is_holding, object_id, position=None):
        self.holding.append((is_holding, object_id))
        return True


@pytest.fixture()
def rig(monkeypatch):
    calls = NS(grasp=[], release=0, place=0, grasp_results=[])

    def grasp(*_, **__):
        calls.grasp.append(1)
        return calls.grasp_results.pop(0) if calls.grasp_results else True

    def release(_spot):
        calls.release += 1

    def place(*_, **__):
        calls.place += 1
        return True

    monkeypatch.setattr(se, "object_grasp", grasp)
    monkeypatch.setattr(se, "release_failed_grasp", release)
    monkeypatch.setattr(se, "object_place", place)
    monkeypatch.setattr(se, "capture_pick_result_images", lambda *_, **__: [])
    monkeypatch.setattr(se.time, "sleep", lambda _s: None)
    state = NS(manipulator_state=NS(is_gripper_holding_item=True))
    spot = NS(
        robot=NS(time_sync=NS(wait_for_sync=lambda: None)),
        take_lease=lambda: None,
        get_state=lambda: state,
        get_pose=lambda: np.zeros(3),
    )

    def run(feedback, actions=(PICK, PLACE), **kwargs):
        ex = se.SpotExecutor(spot, None, None, None, **kwargs)
        ex._check_place_distance = lambda *_: None
        ex.process_action_sequence(ActionSequence("p", "r", list(actions)), feedback)
        return ex

    return NS(run=run, calls=calls)


def test_picked_continues_with_the_plan(rig):
    fb = Feedback(["picked"])
    rig.run(fb)
    assert len(rig.calls.grasp) == 1 and rig.calls.release == 0
    assert fb.asked == [
        {
            "object_id": "o7",
            "object_class": "bag",
            "attempt": 1,
            "images": [],
            "gripper_holding": True,
        }
    ]
    assert fb.holding[0] == (True, "O7")
    assert rig.calls.place == 1
    assert [r[:3] for r in fb.results] == [
        ("Pick", 0, "SUCCESS"),
        ("Place", 1, "SUCCESS"),
    ]


def test_retry_picks_again_as_often_as_the_operator_asks(rig):
    """More retries than the executor's own three attempts allow."""
    fb = Feedback(["retry"] * 5 + ["picked"])
    rig.run(fb)
    assert len(rig.calls.grasp) == 6
    assert rig.calls.release == 5
    assert [q["attempt"] for q in fb.asked] == [1, 2, 3, 4, 5, 6]
    # holding is reported once, for the pick that worked
    assert [h for h in fb.holding if h[0]] == [(True, "O7")]
    assert [r[:3] for r in fb.results] == [
        ("Pick", 0, "SUCCESS"),
        ("Place", 1, "SUCCESS"),
    ]
    assert fb.done == [("Pick", True), ("Place", True)]


def test_give_up_ends_the_sequence_before_the_place(rig):
    fb = Feedback(["retry", "give_up"])
    ex = rig.run(fb)
    assert len(rig.calls.grasp) == 2 and rig.calls.release == 2
    assert rig.calls.place == 0
    assert fb.holding == []
    assert fb.results == [
        ("Pick", 0, "FAILED", "operator reported the pick failed and gave up")
    ]
    # the note tells the planner not to replan this pick
    assert fb.done == [("Pick", False, "operator_gave_up")]
    assert not ex.processing_action_sequence


def test_a_pick_the_robot_knows_failed_is_not_asked_about(rig):
    """The robot's own failures keep the three-attempt limit."""
    rig.calls.grasp_results = [False, False, False]
    fb = Feedback([])
    rig.run(fb, actions=(PICK,))
    assert len(rig.calls.grasp) == 3
    assert fb.asked == [] and rig.calls.release == 0
    assert fb.results == [("Pick", 0, "FAILED", "gave up after 3 attempts")]


def test_the_question_can_be_turned_off(rig):
    fb = Feedback([])
    rig.run(fb, confirm_pick_result=False)
    assert fb.asked == [] and fb.holding[0] == (True, "O7") and rig.calls.place == 1


def test_a_feedback_that_cannot_ask_keeps_the_robots_belief(rig):
    fb = Feedback([], can_ask=False)
    rig.run(fb)
    assert fb.holding[0] == (True, "O7") and rig.calls.place == 1


@pytest.mark.parametrize("answer", [RuntimeError("no publisher"), "maybe", None])
def test_a_broken_or_unknown_answer_keeps_the_robots_belief(rig, answer):
    fb = Feedback([answer])
    rig.run(fb)
    assert rig.calls.release == 0
    assert fb.holding[0] == (True, "O7") and rig.calls.place == 1


def test_a_stop_while_waiting_keeps_the_robots_belief(rig):
    """The answer that arrives with the stop is not acted on: nothing is released."""
    fb = Feedback([])
    ex_box = {}

    def confirm(**_):
        ex_box["ex"].keep_going = False
        return "retry"

    fb.confirm_pick_result = confirm
    ex = se.SpotExecutor(
        NS(
            robot=NS(time_sync=NS(wait_for_sync=lambda: None)),
            take_lease=lambda: None,
            get_state=lambda: None,
        ),
        None,
        None,
        None,
    )
    ex_box["ex"] = ex
    ex.process_action_sequence(ActionSequence("p", "r", [PICK, PLACE]), fb)
    assert rig.calls.release == 0 and len(rig.calls.grasp) == 1
    assert fb.holding[0] == (True, "O7")
    assert rig.calls.place == 0


def test_result_images_are_upright_and_small_enough_for_the_radio():
    from spot_skills.grasp_utils import PICK_RESULT_MAX_SIDE, capture_pick_result_images

    frames = {
        "hand_color_image": np.zeros((480, 640, 3), np.uint8),
        "frontleft_fisheye_image": np.zeros((1200, 1600, 3), np.uint8),
    }

    def get_image_RGB(view):
        if view not in frames:
            raise RuntimeError("camera offline")
        return None, frames[view]

    logged = []
    images = capture_pick_result_images(NS(get_image_RGB=get_image_RGB), logged.append)
    assert [name for name, _ in images] == [
        "hand_color_image",
        "frontleft_fisheye_image",
    ]
    # the gripper camera is passed through; the sideways front camera is turned
    # upright (landscape -> portrait) and scaled down
    assert images[0][1].shape == (480, 640, 3)
    assert images[1][1].shape == (PICK_RESULT_MAX_SIDE, 480, 3)
    assert len(logged) == 1 and "frontright_fisheye_image" in logged[0]
