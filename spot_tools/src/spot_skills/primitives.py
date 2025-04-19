from spot_skills.arm_utils import (
    close_gripper,
    open_gripper,
    stow_arm,
)
from spot_skills.navigation_utils import (
    navigate_to_absolute_pose,
    navigate_to_relative_pose,
)

def execute_recovery_action(
    spot, recover_arm=True, absolute_pose=None, relative_pose=None
):
    if recover_arm:
        open_gripper(spot)
        stow_arm(spot)
        close_gripper(spot)

    if absolute_pose:
        navigate_to_absolute_pose(spot, absolute_pose)
        return

    elif relative_pose:
        navigate_to_relative_pose(spot, relative_pose)
        return

    else:
        return
