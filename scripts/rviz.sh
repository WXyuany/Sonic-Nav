#!/bin/bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
rviz2
