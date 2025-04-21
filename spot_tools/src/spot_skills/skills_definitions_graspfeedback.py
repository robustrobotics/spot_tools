from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

# ------------------------------ Drag Skill ------------------------------ #
# @dataclass
# class OpenDoorParams:
#     hinge_side: int = 1         # i.e., DoorCommand.HingeSide.HINGE_SIDE_LEFT
#     swing_direction: int = 0    # i.e., DoorCommand.SwingDirection.SWING_DIRECTION_UNKNOWN
#     handle_type: int = 0        # i.e., DoorCommand.HandleType.HANDLE_TYPE_UNKNOWN

@dataclass
class GraspFeedback:
    initial_gripper_open_percentage: float = 0.0
    final_gripper_open_percentage: float = 0.0
    initial_grasp: bool = False
    final_grasp: bool = False

    # detected_door: bool = False
    # walked_to_door: bool = False
    # opened_door: bool = False 
    # error_message: Optional[str] = None
    # ego_view: Optional[np.ndarray] = None
    # handle_detection: Optional[np.ndarray] = None

    def success(self):
        return self.initial_grasp and self.final_grasp


    def get_status(self): 
        status = ""
        if self.initial_grasp: 
            status += "The initial grasp is successful.\n"
        else: 
            status += "The initial grasp is unsuccessful."
            return status
        if self.final_grasp: 
            status += "The final grasp is successful. \n"
        else: 
            status += "The final grasp is unsuccessful."
            return status 

        


