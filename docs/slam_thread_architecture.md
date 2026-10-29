# ROS process architecture

The three standard launches start `tracker_node` and `pose_graph_node` in separate
processes, as in commit `3791b363`. Visualization and streamers keep their existing
processes. The Python GIL is separate in the tracking and backend processes.
The optional `gelslam_node` entry point remains an experimental single-process
adapter; none of the standard launches starts it.

## Data flow and ownership

- Tracking owns reconstruction, the frontend prior, initialization and IMU buffers.
  Image callbacks and the IMU-ready guard share one mutually exclusive group.
  When IMU is enabled, tracking waits until the latest accepted IMU timestamp
  reaches the image's selected IMU time. One pending image is retained; a newer
  image replaces it. Accepted IMU callbacks trigger the guard in the image group,
  without tracking or blocking in the IMU group. The ROS executor can continue
  receiving IMU while an image waits or a ready image is being tracked.
  The experimental single-process adapter enqueues guard wakeups to its tracking
  worker, which retains sole ownership of image and tracker mutations.
- Reliable `keyframe` messages carry immutable surface observations, a full-precision
  navigation seed, the complete previous-keyframe IMU interval, tracking session ID,
  original trial ID and the anchor basis used to initialize the seed.
- Backend ROS callbacks enqueue keyframes. One backend worker owns decoding, feature
  search, mutable databases, loop PGO, groups and coverage.
  Queue draining preserves accepted keyframes when an optimization takes a long time.
- The backend returns only poses, velocities and biases on `backend_feedback`, with
  session, epoch, solution revision and per-keyframe revisions. Surface maps are not
  sent back. Other-session, duplicate and older results cannot replace a newer result.
- `frame_pose` carries lightweight frame metadata for final backend trajectory
  evaluation. Existing `frame` images remain available to visualizers. The backend
  does not subscribe to every full frame image just to record its pose.

SharedMap is local to the backend process. It retains immutable observations and
atomic solution versions used by the LocalMapping service. It is not shared memory
between ROS processes. Snapshot and feedback callbacks do not run an optimizer.

All standard launches accept `backend_feedback:=false`. It disables backend
feedback publication and the tracker feedback subscription. Backend
loop PGO, map/mesh output and final backend evaluation continue. The backend
records the effective flag and the number of published feedback messages in
its process diagnostics. The bag launch defaults this flag to `backend`; both
live launches default it to true.

## Correction at image F_j

Let K_k be a corrected keyframe in the active original tracking trial, F_i the last
accepted image and F_j the image now being processed.

A pure gravity-preserving frame correction transports F_i's pose, world velocity and full marginal
prior together. F_i keeps its timestamp and sensor-frame biases. The saved last
tactile pose follows the same correction. This special case needs no older
K_k-to-F_j IMU history. Global 15D corrections that change velocity, either bias,
or gravity attitude use the fixed-keyframe path below.

Navigation feedback uses this fixed-keyframe path:

1. At the start of F_j, tracking reads one immutable feedback snapshot.
2. A correction requires the complete retained raw IMU interval t_k to t_j.
   Missing endpoints or a gap above the configured bound defer the correction.
3. The accepted F_i pose follows the corrected reference K_k while preserving
   its previously accepted K_k-to-F_i relative pose. F_i's world velocity follows
   the same correction rotation. This update does not propagate F_i with IMU.
   F_i retains its timestamp and sensor-frame biases.
4. The corrected K_k pose, velocity and biases initialize the IMU prediction for F_j.
   A fixed-anchor solve uses direct tactile rows, IMU and bias factors. It omits the
   stale marginal prior from F_i.
5. Only an accepted F_j posterior replaces the frontend prior. A rejected target
   retains its correction anchor. Following accepted images resume the usual prior path.
6. Each incoming keyframe seed is converted from its transmitted tracking anchor
   basis into the corresponding stored backend anchor basis. This also handles a seed
   captured before a backend correction and multiple queued keyframes in one batch.
   This changes initialization and adds no measurement factor.

Feedback states use the original tracking trial coordinates. Coordinate conversion
rotates velocity but leaves sensor-frame biases unchanged. Session IDs prevent feedback
from a previous tracker instance from being applied after a restart. The feedback age
counter measures reception-to-application time in the tracking process.

## Preserved estimator behavior

- Each backend keyframe retains pose, velocity and both biases: 15 dimensions.
  These states are the tracking keyframe seeds. The backend has no local VI
  window, no direct normal/height keyframe factor and no marginalization.
- Keyframe batches without an accepted loop run no optimizer. Their seeds are
  published with the last global pose, velocity and bias corrections.
  Accepted loop events run the separate global 15D PGO.
- Native midpoint IMU integration, fixed-anchor frontend feedback and bounded raw IMU
  retention remain enabled according to the existing configuration.
- `backend:=false` in the bag launch disables optimization and frontend feedback and
  retains raw keyframe output. `debug_visualization_enabled` still controls construction
  and publication of the three frontend debug outputs, independently of rendering.

## Separate loop correction

Global PGO has an IMU Pose3 (6), world velocity (3), accelerometer bias (3) and
gyroscope bias (3) at every keyframe. It uses the existing split-pose retraction
and sparse QR/LM solver. Roll and pitch are variables alongside yaw and XYZ.
The pose of each remaining component root is fixed exactly for the gauge.
Root velocity and bias priors remain soft observations, including at merged roots.

An accepted loop triggers a global 15D solve over all retained keyframes.
Temporal edges contain native 9D IMU preintegration plus 6D bias random walk.
They use raw keyframe IMU intervals, configured integration/noise and map gravity.
Missing endpoints or gaps above the configured bound reject the solve.

Contact-valid temporal targets also retain a 6D relative-pose odometry observation
from the frontend keyframe seeds. This is a fused pose observation,
not an additional raw tactile factor. Incoming IMU-only recovery pairs omit
pose odometry and keep only IMU/bias factors. Global PGO never re-evaluates normal
or height maps. Registered loops are fixed 6D GelSight relative poses, with
rotation sigma 0.01 rad and translation sigma 0.0001 m. The calibrated mounting
maps their pose Jacobians to the IMU states. Loop search still uses SIFT and
NormalFlow SCR/CCS, supplies target_T_reference, and PGO stores its inverse.

The global pose correction is C_k = P_k inverse(L_k). Map publication uses C_k L_k.
Velocity additionally receives the solved offset from R(C_k) v_local, and sensor
biases receive their solved additive offsets. The map-state API and ROS feedback
publish those optimized 15D values. A newly appended temporal keyframe inherits
all three corrections until the next loop solve. A new trial starts with its
recorded gravity aligned to the first trial's map gravity. Disconnected component
alignment may use full rotation and translation; the global IMU factors constrain
the subsequent navigation solution.
Conversion back to a tracking trial removes only its gravity-preserving frame
gauge. Estimated roll/pitch changes remain in the returned navigation state;
they are not canceled by using the optimized root pose as a frame transform.

Global optimization does not change the stored keyframe seeds. Trial merging
removes redundant global pose gauges while retaining initial velocity/bias
observations. Snapshot format v4 stores the pose, velocity
and bias offsets plus navigation anchor priors. Existing v3 pose-correction
snapshots retain their poses, initialize navigation offsets to zero, and run a
full 15D solve when the next loop requests it. Old v2 snapshots with direct loops
still require replay because their registered relative measurements were not saved.

## Shutdown and diagnostics

Each completed global solve logs 15D per keyframe, the optimized state dimension,
and the initial/final cost. backend_process_summary.json records both
loop_pgo_degrees_of_freedom and loop_pgo_optimized_state_dimension. The latter
is zero if no global solve ran during that session.

After the final image callback, tracking waits for keyframe/frame-pose DDS delivery
and publishes a session end marker containing its last keyframe ID. On shutdown,
backend reception continues briefly until that keyframe has reached its queue; then
its worker drains the queue and joins before final evaluation, mesh and optional state
export. Backend launch termination timeouts allow long Global PGO calls to finish.
Worker exceptions reach the backend coordinator and cause a failed process exit.

`tracker_process_config_supplied.yaml` and `backend_process_config_supplied.yaml`
record actual configurations and process IDs. Their corresponding
`*_process_summary.json` files record session/counter and shutdown information.
ROS message interfaces changed; rebuild both `gelslam_msgs` and `gelslam` before use.
