from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

# ------------------------------ Open Door Skill ------------------------------ #
@dataclass
class OpenDoorParams:
    hinge_side: int = 0         # i.e., DoorCommand.HingeSide.HINGE_SIDE_UNKNOWN
    swing_direction: int = 0    # i.e., DoorCommand.SwingDirection.SWING_DIRECTION_UNKNOWN
    handle_type: int = 0        # i.e., DoorCommand.HandleType.HANDLE_TYPE_UNKNOWN

@dataclass
class OpenDoorFeedback:
    walked_to_door: bool = False
    detected_door: bool = False
    opened_door: bool = False 
    error_message: Optional[str] = None
    ego_view: Optional[np.ndarray] = None
    handle_detection: Optional[np.ndarray] = None



