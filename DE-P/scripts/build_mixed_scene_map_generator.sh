#!/usr/bin/env bash
set -euo pipefail

root=/home/zjh/YOPO/DE-P
sim=/home/zjh/YOPO/Simulator/src
build="$root/artifacts/mixed_scene_map_generator/build"
binary="$root/artifacts/mixed_scene_map_generator/yopo_mixed_scene_map_generator"

source /opt/ros/noetic/setup.bash
rm -f "$build/CMakeCache.txt"
cmake -S "$root/tools/mixed_scene_map_generator" -B "$build" \
  -DYOPO_SIMULATOR_SOURCE="$sim" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --parallel 8
install -m 0755 "$build/yopo_mixed_scene_map_generator" "$binary"
printf '%s\n' "$binary"
