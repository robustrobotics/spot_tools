# Spot Tools
A place for spot tools

```bash
python3 -m venv spot_tools_env
pip install -e .
```
To run the linter on everything:
```
pre-commit run --all-files
```


# Direct Navigation Commands

In addition to high-level PDDL-based planning, the spot executor supports
direct navigation commands for teleoperation-style control through the chat
interface. These commands bypass the PDDL planning pipeline and are sent
directly to the executor.

## Available Commands

| Command | Action Type | Parameters | Description |
|---------|-------------|------------|-------------|
| Move | `MOVE_RELATIVE` | `distance_m` (float) | Move forward (positive) or backward (negative) by a distance in meters |
| Turn | `TURN_RELATIVE` | `angle_deg` (float) | Turn left/CCW (positive) or right/CW (negative) by an angle in degrees |
| Strafe | `STRAFE` | `distance_m` (float) | Move left (positive) or right (negative) sideways by a distance in meters |
| Stop | `STOP` | none | Halt all motion and cancel any in-progress action sequence |
| Stand/Sit | `STAND_SIT` | `action` ("stand" or "sit") | Stand up or sit down |

## Architecture

These commands are defined as action dataclasses in
`robot_executor_interface/action_descriptions.py` alongside the existing
`Follow`, `Gaze`, `Pick`, and `Place` actions. They are dispatched by the
`SpotExecutor` in `spot_executor.py`, which uses `navigate_to_relative_pose()`
from `navigation_utils.py` to convert body-frame relative commands into
vision-frame absolute trajectories.

The corresponding LLM tools are defined in `heracles_agents` at
`src/heracles_agents/tools/navigation_tools.py`. When the LLM calls a
navigation tool, it publishes an `ActionSequenceMsg` via `ros2 topic pub`
directly to the executor's action sequence topic.

```
Chat input -> LLM tool call (e.g., move_relative) -> ros2 topic pub ActionSequenceMsg -> SpotExecutor
```

## Testing Direct Commands

You can manually publish direct commands for testing without the chat interface:

```bash
# Move forward 2 meters
ros2 topic pub /hamilton/spot_executor_node/action_sequence_subscriber \
  robot_executor_msgs/msg/ActionSequenceMsg \
  "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'MOVE_RELATIVE', scalar_value: 2.0}]}" -1

# Turn right 90 degrees
ros2 topic pub /hamilton/spot_executor_node/action_sequence_subscriber \
  robot_executor_msgs/msg/ActionSequenceMsg \
  "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'TURN_RELATIVE', scalar_value: -90.0}]}" -1

# Stop
ros2 topic pub /hamilton/spot_executor_node/action_sequence_subscriber \
  robot_executor_msgs/msg/ActionSequenceMsg \
  "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STOP'}]}" -1

# Sit down
ros2 topic pub /hamilton/spot_executor_node/action_sequence_subscriber \
  robot_executor_msgs/msg/ActionSequenceMsg \
  "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STAND_SIT', stand_sit_action: 'sit'}]}" -1
```

After modifying `ActionMsg.msg`, rebuild with:
```bash
colcon build --packages-select robot_executor_msgs
```

## Pause / Resume

The executor supports a lightweight pause/resume mechanism via a ROS topic.
When paused, the robot cancels any in-progress action sequence, holds its
current pose (stays standing), and rejects new action sequences until resumed.

```bash
# Pause — robot stops and holds position
ros2 topic pub /hamilton/spot_executor_node/pause std_msgs/msg/Bool "{data: true}" -1

# Resume — robot accepts commands again
ros2 topic pub /hamilton/spot_executor_node/pause std_msgs/msg/Bool "{data: false}" -1
```

This can be bound to a joystick button, keyboard shortcut, or RViz panel
button for one-press pause without going through the chat/LLM stack. The robot
stays powered on and standing — no lease transfer, no E-Stop, no recovery
sequence needed.

## Stopping Behavior

There are three levels of stopping:

- **Pause** (`~/pause` topic): Cancels the action sequence and holds position.
  The robot stays standing with motors on. Resume at any time by publishing
  `false`. No LLM or chat interface needed. Best for: operator wants to
  temporarily halt the robot.

- **Software stop** (`STOP` command via chat): Same effect as pause, but
  triggered through the chat interface / LLM tool call. Adds LLM latency.
  Best for: "I changed my mind" during a chat session.

- **Hardware E-Stop** (tablet): Cuts motor power immediately — the robot will
  collapse. This is always available and should be used for safety-critical
  situations. The robot requires a full recovery sequence (clear faults, power
  on, stand) before it can move again.


# Examples

You can find an example of the ROS-free spot executor in
`examples/test_spot_executor.py`. You should be able to run this and see
a little plot of the agent moving along a path. Run it with `python -i` so
that the plot stays at the end. If it crashes with an inscrutable error,
you may need to `pip install opencv-python-headless` because of a conflict
between opencv's QT version and matplotlib's QT version.


# Testing Mid-Level Planner in Simulation

You can find the implemented mid-level planner in `robot_executor_interface/src/robot_executor_interface/mid_level_planner.py`.

## Setup and Testing Instructions

The mid-level planner can be tested using the DCIST launch system with fake publishers to simulate robot behavior and occupancy data.

### 1. Launch the DCIST System

First, launch the complete DCIST system in simulation mode:

```bash
ADT4_SIM_TIME=false tmuxp load dcist_launch_system/tmux/autogenerated/spot_prior_dsg-spot_prior_dsg.yaml
```

This will start all the necessary nodes including the spot executor and planning systems.

### 2. Start the Fake Occupancy Publisher

In a separate terminal, launch the fake occupancy publisher to provide simulated occupancy grid data:

```bash
ros2 run spot_tools_ros fake_occupancy_publisher
```

This publishes a test occupancy grid that the mid-level planner uses for obstacle avoidance and path planning.

#### Occupancy Publisher Parameters

You can customize the occupancy grid simulation using the following parameters:

- `--num_obstacles <N>`: Number of simulated obstacles in the occupancy grid (default: 5)
- `--crop_distance <D>`: How far the robot can "see" in meters - areas beyond this distance are marked as unknown (default: 5.0, set to -1 to disable cropping)
- `--resolution <R>`: Map resolution in meters per cell (default: 0.12)
- `--robot_name <NAME>`: Robot name for topic namespacing (default: 'hamilton')
- `--publish_rate <RATE>`: Publishing frequency in Hz (default: 10.0)

**Example with custom parameters:**
```bash
ros2 run spot_tools_ros fake_occupancy_publisher --num_obstacles 10 --crop_distance 8.0
```

This creates a more challenging environment with 10 obstacles and allows the robot to "see" up to 8 meters away.

### 3. Send Path Commands

In another terminal, use the fake path publisher to send waypoint commands to the planner:

```bash
ros2 run spot_tools_ros fake_path_publisher <x> <y>
```

Where `<x>` and `<y>` are the desired target coordinates. For example:
```bash
ros2 run spot_tools_ros fake_path_publisher -6 1
```

### 4. Visualization in RViz

Launch RViz to visualize the planning behavior and monitor the following key topics:

1. **High-level Path**: `/hamilton/omniplanner_node/compiled_plan_viz_out`
   - Shows the simulated high-level path from the omniplanner

2. **Mid-Level Planner Output**: `/hamilton/spot_executor_node/mlp_path_publisher`
   - Displays the locally planned path from the mid-level planner
   - This path incorporates obstacle avoidance and local optimization

3. **Occupancy Grid**: `/hamilton/hydra/tsdf/occupancy`
   - Shows the TSDF occupancy map used for planning
   - Visualizes obstacles and free space

### Expected Behavior

The mid-level planner should:
- Receive high-level waypoints from the fake path publisher
- Process the occupancy grid from the fake occupancy publisher
- Generate locally optimal paths that avoid obstacles
- Publish the resulting path for execution by the robot controller

This testing setup allows you to validate the mid-level planner's obstacle avoidance, path smoothing, and local optimization capabilities in a controlled simulation environment. 