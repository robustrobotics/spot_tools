from dataclasses import dataclass, field, fields

from bosdyn.api import arm_command_pb2


class FakeProtoInterface:
    def HasField(self, field):
        for f in fields(self):
            if f.name == field:
                return True
        return False


@dataclass
class FakePose:
    x: float
    y: float
    z: float

    def __sub__(self, other):
        try:
            return FakePose(self.x - other.x, self.y - other.y, self.z - other.z)
        except Exception as _:
            raise TypeError(
                "Subtraction only supported between FakePose and Pose instances"
            )


@dataclass
class FakeMobilityFeedback:
    status: str = "ThisIsFakeMobilityFeedback"


@dataclass
class FakeArmGazeFeedback:
    status: int = arm_command_pb2.GazeCommand.Feedback.STATUS_TRAJECTORY_COMPLETE


@dataclass
class FakeArmFeedback(FakeProtoInterface):
    status: str = "ThisIsFakeArmFeedback"
    arm_gaze_feedback: FakeArmGazeFeedback = field(default_factory=FakeArmGazeFeedback)


@dataclass
class FakeSynchronizedFeedback:
    mobility_command_feedback: FakeMobilityFeedback = field(
        default_factory=FakeMobilityFeedback
    )
    arm_command_feedback: FakeArmFeedback = field(default_factory=FakeArmFeedback)


@dataclass
class FakeFeedback:
    synchronized_feedback: FakeSynchronizedFeedback = field(
        default_factory=FakeSynchronizedFeedback
    )


@dataclass
class FakeFeedbackWrapper:
    feedback: FakeFeedback = field(default_factory=FakeFeedback)
