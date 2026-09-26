# ba_netvlad_ros

Live wrapper for the offline `ba_netvlad` library (blur gate + NetVLAD + Super
Dictionary + temporal voting) as a ROS 2 node, for testing BA-NetVLAD loop
closure on a TurtleBot4.

> **Not tested against real ROS 2 or hardware.** This package was written in a
> dev environment with no ROS 2 / rclpy / colcon installed — it could not be
> built or run here. It targets ROS 2 Humble APIs and should be correct, but
> verify topic names, QoS, message field access, and every numeric threshold
> against your actual robot before trusting it near the robot. Report back
> whatever colcon/runtime errors you hit and they'll get fixed.

## What this is (and is not)

This node implements **Step 1–4 of BA-NetVLAD only**: blur scoring, a global
NetVLAD descriptor per frame, cosine matching against a growable Super
Dictionary, and `tau`/`delta`/temporal-voting confirmation. On an accepted
match it publishes a hypothesis: *"the robot is probably at the pose stored for
keyframe X."*

It is **not**:
- A SLAM backend. It does not build a map, does not run scan matching, and
  does not maintain a pose graph.
- A drop-in loop-closure plugin for slam_toolbox or Cartographer. Neither
  exposes a simple "accept this external loop-closure hypothesis" topic/service
  out of the box — you would need to either (a) feed the published pose into
  your EKF/localization fusion as an observation, or (b) patch/extend the SLAM
  node. **RTAB-Map is the practical exception**: it has a real API for
  external loop-closure links (see "Integrating with a SLAM backend" below).
- Running ORB anywhere. The reference `ba_netvlad` library has no ORB/tracking
  implementation (see the repo's `main.tex`, section on future work) — the
  "sharp branch runs ORB" claim in the method write-up is architecture, not
  code that exists yet. This node runs NetVLAD on every frame regardless of
  the blur-gate reading; `beta`/`is_sharp` are published as diagnostics only.

## Package layout

```
ba_netvlad_ros/
  lcd_node.py             the node (see its module docstring for the pipeline)
  dynamic_dictionary.py   growable version of ba_netvlad.dictionary.CosineDict
config/params.yaml        every parameter, documented, with placeholder paths
launch/lcd.launch.py      ros2 launch entry point
```

## Install

```bash
# 1. Python deps for ba_netvlad itself, into the SAME Python environment ROS 2 uses
#    (a venv layered on top of the system/ROS Python, or --user installs --
#    torch has no rosdep key, so it will not come from `rosdep install`).
pip install torch torchvision opencv-python numpy onnxruntime

# 2. Clone/copy the ba_netvlad repo somewhere on the robot or workstation, e.g.:
export BA_NETVLAD_REPO_ROOT=/home/ubuntu/ba_netvlad
# add that line to your shell rc file so it's set for every future ros2 launch

# 3. Build this package
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ba_netvlad_ros
source install/setup.bash
```

## Before the first run

1. **Recalibrate the blur gate for your camera.** The offline `epsilon` values
   in `main.tex` (1521.5 for KITTI, ~1200/1000 for KITTI05/06) are meaningless
   for an OAK-D Lite. Record a short clip, extract frames, run
   `scripts/01_calibrate_blur_gate.py --frames <dir>`, and put the suggested
   value into `config/params.yaml`.
2. **Retrain or fine-tune for your environment.** A KITTI-only checkpoint is
   an outdoor-driving domain; the GPW/City Centre zero-shot results in
   `main.tex` show this does not transfer to a visually different domain
   (indoor hallway vs. outdoor driving is at least as large a gap). Follow
   Stage A in the offline-first plan: record a teleop run, extract
   frames + `/odom` poses into a `poses.csv`, and fine-tune the same way
   `scripts/09_finetune_gpw.py` / `scripts/10_finetune_citycentre.py` do
   (place identity from real (x, y) here, not from a synchronized-camera
   convention).
3. **Re-tune `tau`/`delta` for that checkpoint** from a recorded run's PR
   curve (`scripts/04_replay_evaluate.py` + a sweep), the same way every
   dataset in `main.tex` needed its own operating point.

## Field-testing procedure

1. **Dry run offline first.** Record a bag of a small loop (rectangle/circle
   in a hallway) with at least one deliberate fast rotation. Extract frames +
   odom into the offline `SequenceDataset` layout and run
   `scripts/04_replay_evaluate.py` against them *before* touching the live
   node — this validates the model/thresholds on real TurtleBot4 imagery with
   zero ROS risk, and is exactly how the KITTI/GPW/City Centre numbers in
   `main.tex` were produced.
2. **Small-loop live test.** `ros2 launch ba_netvlad_ros lcd.launch.py` with
   `mapping_mode: true`, teleop the same rectangle/circle, and watch
   `~/loop_closure_debug` and the node's log for an accepted closure when the
   robot returns to the start.
3. **Call `~/save_dictionary`** (`std_srvs/Trigger`) to persist what was
   mapped, then optionally set `mapping_mode` to `false` via
   `~/set_mapping_mode_off` and re-run the same loop to test pure localization
   against the frozen dictionary.
4. **Adverse-condition tests**, per the original plan: dim the lights (tests
   robustness independent of blur), and deliberately spin fast at a few points
   in the loop (tests the actual BA-NetVLAD claim — recall should barely drop,
   per the KITTI motion-blur benchmark in `main.tex`).

## Integrating with a SLAM backend

- **RTAB-Map**: has a real external-loop-closure mechanism
  (`rtabmap_ros`'s user-data / global-pose APIs). A small bridge node
  subscribing to `~/loop_closure_pose` and calling into that API is the
  practical path — not included here; write it once you've confirmed the
  detector itself works on your robot, since RTAB-Map's exact API surface
  varies by version.
- **slam_toolbox / Cartographer**: no such entry point exists out of the box.
  The realistic use of `~/loop_closure_pose` here is as an extra observation
  into your own EKF/localization fusion (e.g. `robot_localization`), or purely
  as an offline diagnostic (log accepted closures, verify them by eye against
  `~/loop_closure_debug`, and manually judge whether the map drifted less on
  the runs where a closure fired) rather than a plug-and-play map correction.

## Parameters

See `config/params.yaml` — every field has a unit/range comment inline. The
only ones without a safe default are `model_path` (required) and `epsilon`
(defaults to 0.0, which makes the blur gate report every frame as blurry — a
harmless-but-useless placeholder until you calibrate it).
