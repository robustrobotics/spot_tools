from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

# ------------------------------ Open Door Skill ------------------------------ #
@dataclass
class OpenDoorParams:
    hinge_side: int = 1         # i.e., DoorCommand.HingeSide.HINGE_SIDE_LEFT
    swing_direction: int = 1    # i.e., DoorCommand.SwingDirection.SWING_DIRECTION_UNKNOWN
    handle_type: int = 0        # i.e., DoorCommand.HandleType.HANDLE_TYPE_UNKNOWN

@dataclass
class OpenDoorFeedback:
    detected_door: bool = False
    walked_to_door: bool = False
    opened_door: bool = False 
    error_message: Optional[str] = None
    ego_view: Optional[np.ndarray] = None
    handle_detection: Optional[np.ndarray] = None

    def success(self):
        return self.walked_to_door and self.detected_door and self.opened_door


    def get_status(self): 
        status = ""
        if self.detected_door: 
            status += "The robot detected the door.\n"
        else: 
            status += "The robot was unable to detect the door."
            return status
        if self.walked_to_door: 
            status += "The robot walked to the door.\n"
        else: 
            status += "The robot could not walk to the door."
            return status 
        if self.opened_door: 
            status += "The robot was able to open the door. SUCCESS!"
            return status 
        else: 
            status += "The robot was unable to open the door. FAILURE!"
            return status 

        


