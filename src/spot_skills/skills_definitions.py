from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class OpenDoorSkillParams:
    hinge_side: int # e.g., door_pb2.DoorCommand.HingeSide
    swing_direction: int # e.g., door_pb2.DoorCommand.SwingDirection
    handle_type: int # e.g., door_pb2.DoorCommand.HandleType

@dataclass
class OpenDoorSkillFeedback:
    walked_to_door: bool
    detected_door: bool
    opened_door: bool 
    error_message: Optional[str] = None
    ego_view: Optional[np.ndarray] = None
    handle_detection: Optional[np.ndarray] = None



