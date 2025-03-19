import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze

import spot_executor as se


def tf_lookup(parent, child):
    print(f"Faking lookup for {parent}->{child}")
    return np.array([0, 0, 0.0]), np.array([0.0, 0, 0, 1])


spot = se.FakeSpot(init_pose=np.array([0, 1, 0, 0]))
executor = se.SpotExecutor(spot, tf_lookup)

collector = se.FeedbackCollector()


path = np.array(
    [
        [0.0, 0],
        [1.0, 0],
        [3.0, 5],
        [5.0, 5],
    ]
)

follow_cmd = Follow("map", path)

gaze_cmd = Gaze("map", np.array([5, 5, 0]), np.array([7, 7, 0]), stow_after=True)

seq = ActionSequence("id0", "spot", [follow_cmd, gaze_cmd])

executor.process_action_sequence(seq, collector)
