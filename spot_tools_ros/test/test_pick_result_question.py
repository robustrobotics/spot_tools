"""The after-pick question on the wire: one JSON question, one matching answer."""

import json
import threading
import time
from base64 import b64decode
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from spot_tools_ros.spot_executor_ros import RosFeedbackCollector  # noqa: E402


class Pub:
    def __init__(self):
        self.sent = []
        self.event = threading.Event()

    def publish(self, msg):
        self.sent.append(json.loads(msg.data))
        self.event.set()


@pytest.fixture()
def collector(tmp_path):
    fc = RosFeedbackCollector("odom", str(tmp_path), "")
    fc.logger = SimpleNamespace(info=lambda *_: None, warn=lambda *_: None)
    fc.pick_result_request_pub = Pub()
    return fc


def ask(fc, **kwargs):
    box = {}
    args = dict(
        object_id="o7",
        object_class="bag",
        attempt=2,
        images=[("hand_color_image", np.full((8, 12, 3), 200, np.uint8))],
        gripper_holding=True,
    )
    args.update(kwargs)
    t = threading.Thread(
        target=lambda: box.update(result=fc.confirm_pick_result(**args)), daemon=True
    )
    t.start()
    assert fc.pick_result_request_pub.event.wait(5)
    return t, box


def answer(fc, **fields):
    fc.pick_result_callback(SimpleNamespace(data=json.dumps(fields)))


@pytest.mark.parametrize("result", ["picked", "retry", "give_up"])
def test_question_is_answered_by_the_matching_request_id(collector, result):
    t, box = ask(collector)
    question = collector.pick_result_request_pub.sent[0]
    assert question["object_id"] == "o7" and question["object_class"] == "bag"
    assert question["attempt"] == 2 and question["gripper_holding"] is True
    (image,) = question["images"]
    assert image["name"] == "hand_color_image"
    jpeg = np.frombuffer(b64decode(image["jpeg_b64"]), np.uint8)
    assert cv2.imdecode(jpeg, cv2.IMREAD_COLOR).shape == (8, 12, 3)

    answer(collector, request_id=question["request_id"], result=result)
    t.join(5)
    assert box == {"result": result}
    # every UI is told the question is closed, and how
    assert collector.pick_result_request_pub.sent[-1] == {
        "request_id": question["request_id"],
        "closed": result,
    }


def test_answers_for_another_question_or_malformed_are_ignored(collector):
    t, box = ask(collector)
    request_id = collector.pick_result_request_pub.sent[0]["request_id"]
    answer(collector, request_id="someone-else", result="give_up")
    answer(collector, request_id=request_id, result="explode")
    answer(collector, request_id=request_id)
    collector.pick_result_callback(SimpleNamespace(data="not json"))
    time.sleep(0.2)
    assert t.is_alive() and box == {}
    answer(collector, request_id=request_id, result="retry")
    t.join(5)
    assert box == {"result": "retry"}


def test_an_answer_with_no_question_pending_is_ignored(collector):
    answer(collector, request_id="old", result="give_up")
    t, box = ask(collector)
    time.sleep(0.2)
    assert t.is_alive()
    collector.break_out_of_waiting_loop = True
    t.join(5)


def test_a_stop_while_waiting_returns_picked_and_closes_the_question(collector):
    t, box = ask(collector)
    request_id = collector.pick_result_request_pub.sent[0]["request_id"]
    collector.break_out_of_waiting_loop = True
    t.join(5)
    assert box == {"result": "picked"}
    assert collector.pick_result_request_pub.sent[-1] == {
        "request_id": request_id,
        "closed": "stopped",
    }


def test_an_unanswered_question_is_repeated(collector):
    collector.pick_result_repeat_s = 1.0
    t, box = ask(collector)
    deadline = time.time() + 5
    while len(collector.pick_result_request_pub.sent) < 2 and time.time() < deadline:
        time.sleep(0.1)
    first, second = collector.pick_result_request_pub.sent[:2]
    assert first == second
    answer(collector, request_id=first["request_id"], result="picked")
    t.join(5)


def test_a_given_up_pick_is_marked_in_the_action_done_report(collector):
    """The planner replans a failed pick, unless the operator gave it up."""
    from rclpy.clock import Clock

    from robot_executor_interface.action_descriptions import Pick

    sent = []
    collector._clock = Clock()
    collector.action_done_pub = SimpleNamespace(publish=sent.append)
    sequence = SimpleNamespace(plan_id="plan-1", robot_name="euclid")
    pick = Pick("map", "bag", np.zeros(3), np.zeros(3), "o7")
    collector.report_action_done(sequence, pick, False)
    collector.report_action_done(sequence, pick, False, note="operator_gave_up")
    assert [(m.action_type, m.success, m.object_id) for m in sent] == [
        ("PICK", False, "o7")
    ] * 2
    assert [m.header.frame_id for m in sent] == ["", "operator_gave_up"]
