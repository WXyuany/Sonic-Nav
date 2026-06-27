#!/bin/bash
# Usage: bash scripts/switch_scene.sh <scene_name>
# Scenes: default, dynamic, stairs, uneven

SCENE=${1:-default}
YAML="$HOME/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
SCENE_DIR="$HOME/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1"

case $SCENE in
  default)  FILE="scene_43dof.xml" ;;
  dynamic)  FILE="scene_dynamic.xml" ;;
  stairs)   FILE="scene_stairs.xml" ;;
  uneven)   FILE="scene_uneven.xml" ;;
  *) echo "Usage: $0 {default|dynamic|stairs|uneven}"; exit 1 ;;
esac

sed -i "s|scene_[a-z]*.xml|$FILE|" "$YAML"
echo "Switched to: $SCENE ($FILE) — restart sim to apply"
