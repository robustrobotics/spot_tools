#!/usr/bin/env bash
# Quick navigation test commands for Spot direct control.
# Usage: source nav_test.sh
#
# Then call: forward 0.5 | backward 1.0 | turn_left 90 | turn_right 45
#            strafe_left 0.5 | strafe_right 0.5 | stand | sit | stop
#            pause | resume
#
# The EXEC_TOPIC must match the remapped action_sequence_subscriber topic for
# your deployment. In the hamilton launch config this is remapped to:
#   omniplanner_node/compiled_plan_out
# Adjust EXEC_TOPIC below if deploying on a different robot or launch config.

# Source ROS and workspace if not already set up
if [ -z "$RMW_IMPLEMENTATION" ]; then
    source /opt/ros/jazzy/setup.zsh 2>/dev/null || source /opt/ros/jazzy/setup.bash 2>/dev/null
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    export ZENOH_ROUTER_CONFIG_URI=/home/swarm/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5
fi

ADT4_SETUP=/home/swarm/dcist_ws/install/setup.zsh
[ -f "$ADT4_SETUP" ] && source "$ADT4_SETUP"

EXEC_TOPIC="/hamilton/omniplanner_node/compiled_plan_out"
PAUSE_TOPIC="/hamilton/spot_executor_node/pause"
MSG_TYPE="robot_executor_msgs/msg/ActionSequenceMsg"

_pub() { ros2 topic pub "$EXEC_TOPIC" "$MSG_TYPE" "$1" -1; }

forward()      { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'MOVE_RELATIVE', scalar_value: ${1:-1.0}}]}"; }
backward()     { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'MOVE_RELATIVE', scalar_value: -${1:-1.0}}]}"; }
turn_left()    { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'TURN_RELATIVE', scalar_value: ${1:-90.0}}]}"; }
turn_right()   { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'TURN_RELATIVE', scalar_value: -${1:-90.0}}]}"; }
strafe_left()  { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STRAFE', scalar_value: ${1:-0.5}}]}"; }
strafe_right() { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STRAFE', scalar_value: -${1:-0.5}}]}"; }
stand()        { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STAND_SIT', stand_sit_action: 'stand'}]}"; }
sit()          { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STAND_SIT', stand_sit_action: 'sit'}]}"; }
stop()         { _pub "{plan_id: 'test', robot_name: 'spot', actions: [{action_type: 'STOP'}]}"; }
pause()        { ros2 topic pub "$PAUSE_TOPIC" std_msgs/msg/Bool "{data: true}" -1; }
resume()       { ros2 topic pub "$PAUSE_TOPIC" std_msgs/msg/Bool "{data: false}" -1; }

echo "Nav test commands loaded:"
echo "  forward [m]      backward [m]"
echo "  turn_left [deg]  turn_right [deg]"
echo "  strafe_left [m]  strafe_right [m]"
echo "  stand            sit"
echo "  stop             pause    resume"
echo "Defaults: distance=1.0m, angle=90deg, strafe=0.5m"
