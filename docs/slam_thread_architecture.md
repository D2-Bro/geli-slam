# ROS process architecture

The three standard launches start `tracker_node` and `pose_graph_node` in separate
processes, as in commit `3791b363`. Visualization and streamers keep their existing
processes. The Python GIL is separate in the tracking and backend processes.
The optional `gelslam_node` entry point remains an experimental single-process
adapter; none of the standard launches starts it.

## Data flow and ownership

- Tracking owns reconstruction, the frontend prior, initialization and IMU buffers.
  Image callbacks are mutually exclusive. The ROS executor can ingest IMU separately.
- Reliable `keyframe` messages carry immutable surface observations, a full-precision
  navigation seed, the complete previous-keyframe IMU interval, tracking session ID,
  original trial ID and the anchor basis used to initialize the seed.
- Backend ROS callbacks enqueue keyframes. One backend worker owns decoding, feature
  search, mutable databases, local VI, separate loop PGO, groups and coverage.
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
feedback publication and the tracker feedback subscription. Backend local VI,
loop PGO, map/mesh output and final backend evaluation continue. The backend
records the effective flag and the number of published feedback messages in
its process diagnostics. The bag launch defaults this flag to `backend`; both
live launches default it to true.

## Correction at image F_j

Let K_k be a corrected keyframe in the active original tracking trial, F_i the last
accepted image and F_j the image now being processed.

1. At the start of F_j, tracking reads one immutable feedback snapshot.
2. A correction requires the complete retained raw IMU interval t_k to t_j.
   Missing endpoints or a gap above the configured bound defer the correction.
3. The accepted F_i pose follows the corrected reference K_k while preserving
   its previously accepted K_k-to-F_i relative pose. This update does not
   propagate F_i with IMU. F_i retains its timestamp, velocity and biases.
4. The corrected K_k pose, velocity and biases initialize the IMU prediction for F_j.
   A fixed-anchor solve uses direct tactile rows, IMU and bias factors. It omits the
   stale marginal prior from F_i.
5. Only an accepted F_j posterior replaces the frontend prior. A rejected target
   retains its correction anchor. Following accepted images resume the usual prior path.
6. Each incoming keyframe seed is converted from its transmitted tracking anchor
   basis into the corresponding local VI anchor basis. This also handles a seed
   captured before a backend correction and multiple queued keyframes in one batch.
   This changes initialization and adds no measurement factor.

Feedback states use the original tracking trial coordinates. Coordinate conversion
rotates velocity but leaves sensor-frame biases unchanged. Session IDs prevent feedback
from a previous tracker instance from being applied after a restart. The feedback age
counter measures reception-to-application time in the tracking process.

## Preserved estimator behavior

- Each backend keyframe retains pose, velocity and both biases: 15 dimensions.
- Each observed keyframe pair has one `gelsight` factor. It stacks the normal
  residual rows and one mean-height row, followed by separate native 9D IMU and
  6D bias random-walk factors and initialization priors. Only temporal neighbors
  have tactile factors. Relative-pose measurement factors remain absent from local VI.
- The local window, marginalization and iteration settings remain in YAML.
- Local VI runs only when the incoming batch adds at least one tactile factor.
  IMU recovery keyframes and bias random-walk factors are still appended. Their
  seeds are published without solving or advancing the marginalization window.
  The first valid contact after an invalid tactile endpoint also adds no tactile
  factor. A later tactile pair resumes local VI, including recovery states still
  inside the window. Initialization-only batches retain their supplied seeds.
  Accepted loop events run the separate 4DoF PGO even without a new tactile factor;
  loop-only events do not run local VI. Mixed batches run both when required.
- With `marginalization_enabled: false`, the temporal predecessor immediately
  before the window has its complete 15D navigation state fixed. This requires
  an actual IMU link into the window. Trial-start anchors inside the window and
  any other outside observation endpoints have only their poses fixed; their
  velocities and biases remain free. Fixed increment columns are removed from
  sparse QR, so fixed values stay exact.
  The crossing 9D IMU factor uses 0.01 times its native information matrix and
  block Huber loss with delta `sqrt(16.92)`, following ORB-SLAM3. It is cloned
  when constructing the local window. Stored factors, preintegration data,
  interior IMU factors and all bias random-walk factors keep their original
  information. Window advances do not accumulate the information scaling.
  Existing observations and initialization priors among included states stay
  in the graph when they constrain free variables. Outside selection does not
  recursively expand through older temporal links. A full window normally has
  `local_window_size` states plus its outside context. For example, K50..K59
  fixes all 15 components of temporal predecessor K49; an inside trial-start
  anchor K53 fixes only its pose. A loop event retains these boundary choices
  and runs separate 4DoF PGO.
- With `marginalization_enabled: true`, the existing Schur window prior remains
  available. It retains the information matrix and uses a zero-gradient anchor.
- Normal and height share one callback and projection. Their sigmas remain
  independent. Normal rows use mean squared error without Huber loss; no
  robust weight is applied to either residual block or the combined vector.
- Residual-only evaluation skips pose Jacobians.
  Both backend maps use origin's `wide_remap` /
  `cv2.remap(INTER_LINEAR)`, with float32 maps and pixel coordinates. Contact
  selection uses `INTER_NEAREST` and origin's image bounds. The normal/height
  maps are not converted to full-image float64. Target-map image gradients and
  neighboring-pixel float64 gathers are not needed for linearization.
- Backend normal geometry now uses the same reference-map IC basis and
  right-local chart mapping as `DenseNormalResidual.evaluate_ic()`. Its 5D
  cached reference basis is embedded in 6D with a zero reference-Z column.
  The shared mapping helper is used by the frontend and backend. Active rows
  and chart mapping update with the current endpoint poses.
- The backend mean-height residual is still resampled at each pose, but its
  linear model holds rotation, XY and correspondences fixed and uses only
  target GelSight body-local Z, exactly as the frontend's height row. With
  `T = inverse(target) * reference`, `J_target = [0,0,0,0,0,1]` and
  `J_reference = -J_target * Ad(T)` before sigma/extrinsic mapping. This
  reference transport preserves the common rigid-motion nullspace. It can
  contain reference rotation/XY entries; the IMU extrinsic also introduces
  lever-arm rotation entries. Z-only describes the relative target-GelSight
  model, not a world-Z constraint or zeros in every IMU Jacobian column.
  The row stays in the same GelSight factor, independently weighted from the
  normal rows. Splitting it out gives the same linear least-squares system.
- IC and conditional-Z rows are approximations to the full resampled cost.
  OpenCV sampling and independent sigmas remain unchanged. Backend normal
  rows use ordinary least squares in every iteration. The frontend retains its
  configured initial Huber phase. The backend still uses LM; it does not inherit
  the frontend's undamped GN update or iteration-dependent Huber schedule.
- Native midpoint IMU integration, fixed-anchor frontend feedback and bounded raw IMU
  retention remain enabled according to the existing configuration.
- `backend:=false` in the bag launch disables optimization and frontend feedback and
  retains raw keyframe output. `debug_visualization_enabled` still controls construction
  and publication of the three frontend debug outputs, independently of rendering.

## Separate loop correction

`LoopPoseGraph` contains world yaw and XYZ variables for each GelSight keyframe.
Local VI first solves its configured window only if the batch adds a tactile
factor. PGO then uses relative poses between
temporal local VI neighbors and fixed `reference_T_target` loop measurements.
Loop search supplies `target_T_reference`; PGO stores its inverse. SIFT inliers
and normalflow SCR/CCS still validate registration before insertion.

The 4DoF PGO follows ORB-SLAM3's world-frame yaw/XYZ updates, with the existing
temporal odometry/registered-loop edge structure. It uses
rotation sigma 0.01 rad and translation sigma 0.0001 m for both edge types.
Origin's translation sigma is 0.1 mm. This implementation uses metres throughout
and retains the backend's split relative-pose errors and sparse QR/LM optimizer.
All six measurement residuals remain in each edge; each endpoint Jacobian has
only four columns. A remaining trial root has all four increments fixed exactly.
The update is `R_map = Exp(up * yaw) R_base`, with independent world XYZ.
`up` comes from the recorded gravity, so this also works in tilted world axes.
Local gravity directions from disconnected trials are aligned to the first
trial's map gravity. Subsequent corrections, including component alignment at
loop insertion, contain only gravity-axis yaw and translation. Registration tilt
errors cannot change the local VI gravity attitude.

Let `L_k` be the local VI GelSight pose at keyframe K_k. Let `P_k` be the loop PGO
solution. The saved correction is `C_k = P_k inverse(L_k)`. Map publication uses
`C_k L_k`. The IMU pose and world velocity use the same correction and rotation;
sensor-frame biases keep their local VI values. Tracking-trial feedback removes
the original backend group's root map transform. A new temporal keyframe inherits
its predecessor's correction until the next PGO. A new disconnected trial starts
with identity map correction when its gravity axes already match the map. Other
gravity axes receive a fixed gravity alignment, followed by yaw/XYZ corrections.

PGO never changes local navigation states, gravity, initialization factors or the
local marginal prior. Trial merging removes only the redundant PGO gauge anchor;
each independent local VI trial retains its original pose/velocity/bias priors.
Local-only solves retain the last PGO corrections. They do not optimize loop edges.
Snapshot format v3 stores local states, fixed loop measurements and map corrections.
New loop snapshots also record four optimization degrees of freedom. Historical
6D corrections remain viewable; the next PGO projects their initialization into
the gravity-preserving subspace before solving the retained measurements.
Old v2 direct-loop snapshots require replay because their registered relative
measurements were not saved. Old v2 snapshots without loops remain loadable.

## Shutdown and diagnostics

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
