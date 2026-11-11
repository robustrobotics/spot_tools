from dataclasses import dataclass

import numpy as np


@dataclass
class ActionSequence:
    plan_id: str
    robot_name: str
    actions: list


@dataclass
class Follow:
    frame: str
    path2d: np.ndarray


@dataclass
class Gaze:
    frame: str
    robot_point: np.ndarray
    gaze_point: np.ndarray
    object_id: str
    stow_after: bool = False


@dataclass
class Pick:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str


@dataclass
class Place:
    frame: str
    object_class: str
    robot_point: np.ndarray
    object_point: np.ndarray
    object_id: str


@dataclass
class MoveRelative:
    distance_m: float  # positive=forward, negative=backward


@dataclass
class TurnRelative:
    angle_deg: float  # positive=left(CCW), negative=right(CW)


@dataclass
class Strafe:
    distance_m: float  # positive=left, negative=right


@dataclass
class Stop:
    pass


@dataclass
class StandSit:
    action: str  # "stand" or "sit"
