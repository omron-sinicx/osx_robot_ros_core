# osx_splines

Robot-side environment and evaluation scripts for **casp** (B-spline FCT policies,
`dependencies/casp`). Temporary package: a copy of `osx_ur5e` taken from the
`spline-vel-engine` branch at commit `38c00cd`, kept separate so casp can run while
`osx_ur5e` is being changed for the RA-L work. The two are to be unified later; the
common ancestor of both env copies is `osx_ur5e` at `b5cfc4f`.

## What differs from `osx_ur5e` on `spline-vel-engine`

- Python module is `osx_splines` (`from osx_splines.fdcc_env import FDCCEnv`), so both
  packages can live in the same catkin workspace.
- Imports `casp` instead of `comet` everywhere; the Hydra `config_path` of the scripts
  points at `dependencies/casp/configs`.
- `controller.target_wrench_frame` defaults to `base` (was `ee`). ur_control's
  `CompliantController` calls `set_hand_frame_control(False)` at construction, so the
  FZI controller reads target wrenches in the robot base frame; rotating into the
  end-effector frame was wrong. Re-run the sign check from
  `dependencies/casp/docs/bspline_fct.md` (section 8.7) before trusting
  `feedforward.sign`.
- Only the casp entry points are copied: `evaluate_policy.py`, `replay_episode.py`,
  `test_replay_episode.py`, `test_replay_episode_bspline.py`, `test_waypoint_stride.py`,
  `test_waypoint_sync.py`. Data collection, bag conversion, GoPro and teleop tooling
  stay in `osx_ur5e`.
- No launch files: robot and camera bring-up still come from `osx_ur5e`
  (`connect_real_robot.launch`, `camera_bringup.launch`).

## Requirements on the robot PC

- `casp` installed in the container's Python (`pip install -e dependencies/casp`), next to
  or instead of `comet`; the two packages do not collide.
- Datasets under `dependencies/casp/data` for the replay scripts, checkpoints under
  `dependencies/casp/rollouts` for `evaluate_policy.py` (their saved `config.yaml` /
  `eval_config.yaml` must say `casp.*`, `${casp_root:}` and `trainer: casp`).

## Running

```
roslaunch osx_ur5e connect_real_robot.launch
roslaunch osx_ur5e camera_bringup.launch
rosrun osx_splines evaluate_policy.py --config-name edge_sink_wipe_dm model=casp eval.base.ckpt_name=<run>
rosrun osx_splines test_replay_episode_bspline.py --config-name edge_sink_wipe_dm
```

ROS-free unit test, from the workspace root with the pixi python:

```
PYTHONPATH=dependencies/casp pixi run --frozen python -m pytest ws/src/osx/osx_core/osx_splines/tests -q
```
