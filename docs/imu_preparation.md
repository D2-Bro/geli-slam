# IMU Integration Preparation

## Recorded source clock option

For paired `/gelsight/image_raw` and `/gelsight/timing` messages and paired
`/imu/data_raw` and `/imu/device_time_ms` messages, set this `imu_config` field
in the YAML passed as `config_path`:

```yaml
timestamp_source: recorded_source
```

Without `recorded_source_bag_path`, the tracker subscribes to
`camera_timing_topic` (default `/gelsight/timing`, `std_msgs/String`) and
`imu_device_time_topic` (default `/imu/device_time_ms`, `std_msgs/UInt32`).
It uses camera `source_pts_ns` and unwrapped IMU device milliseconds for
integration. Image and IMU subscriptions still use their configured input
topics. No bag database is opened in this mode.

Camera timing is matched by `host_wall_ns` to the original image header.
The IMU timing message has no header or sample identifier. Its messages are
paired with IMU measurements in FIFO order. Start the tracker before the two
sensor publishers or bag player, and deliver both IMU topics in the same order
without independent message loss. Both subscriptions use reliable queues of
depth 1000. A missing headerless partner cannot always be detected from the
remaining data. Camera drops do not shift timing onto the next image.

The two source clocks have different origins. During startup,
`recorded_source_warmup_s` seconds (default 1.0, at least three matched pairs
per sensor) determine separate median host offsets. Source clock rates remain
one second per SI second; offsets are frozen before any sample is released.
This avoids moving previously used navigation times when later receipt jitter
changes. Topic mode does not estimate long-term clock-rate drift. Pending
queues are bounded; invalid clock transitions or overflow stop source timing
with an error, without a header-time fallback.

For deterministic offline pairing and full-recording clock-rate estimation,
optionally set `recorded_source_bag_path: /absolute/path/to/bag_0.db3`.
The tracker then reads the paired timing topics from that database at startup,
matches replayed messages to their recorded partners, and fits separate robust
affine source-to-host maps. This explicit database mode does not subscribe to
the timing topics. An unmatched replayed message is dropped.

In both modes, `image_to_imu_time_offset_s` is added to camera integration
times. Published frame/keyframe headers retain the original ROS headers.
Host alignment does not measure camera exposure delay or the camera-to-IMU
latency difference. Calibrate the remaining offset before treating the
trajectory as a synchronized measurement result.

The recorded IMU millisecond counter can wrap at 60,000 ms or at the uint32
limit. The loader infers the period from a backward jump larger than half the
counter range, then retains that period for the recording. For example,
`59997, 2, 7` becomes `59997, 60002, 60007` ms. This also supports repeated
minute wraps. Duplicate device times, small backward jumps, and inconsistent
counter formats remain errors. This unwrapping applies to `recorded_source`
mode; `header` mode continues to use the incoming ROS header times.

The tracker node subscribes to `/imu/data` and buffers calibrated gyro/accel
measurements as timestamped SI-unit `ImuSample` records. With
`frontend_optimization` and a calibrated `gelsight_T_imu`, each contact
frame uses NormalFlow normal rows, a native GTSAM `ImuFactor`, and a separate
IMU bias random-walk factor.
GTSAM preintegrates the exact image-to-image IMU interval. The frontend
predicts the target IMU pose and IMU-origin world velocity from the last accepted
frontend state and its biases. That prediction initializes the dense
normal-flow and IMU solve, which updates pose, velocity, accel bias, and gyro
bias. Set `imu_config.frontend_initialize_from_imu_preintegration: false` to
initialize the current solve with a copy of the last accepted frontend state
at the current image timestamp. The IMU preintegration remains in the solve
as an `ImuFactor`. When the setting is true, the same preintegration
is used for prediction and optimization. Raw IMU measurements are integrated
at the IMU origin. No sensor-to-body offset correction is applied to them.
The calibrated transform is applied to tactile observations instead.

Set `imu_config.frontend_imu_prediction_only: true` to use the existing IMU
prediction as the contact target State without optimization. This optional
boolean defaults to false and requires the frontend to be enabled and
initialized. It overrides `frontend_initialize_from_imu_preintegration: false`
and both fixed/joint optimization modes. Pose and world velocity are predicted
from the last accepted navigation State through the target image time; both
biases are copied from that State. No NormalFlow factor is built, no optimizer
iteration runs, and carried/pending marginal priors are cleared. The existing
SCR/CCS gate and normal-correspondence projection are bypassed as well.
Each valid IMU prediction becomes the next navigation State, regardless of
visual contact overlap or similarity. Missing IMU coverage or an invalid
prediction still rejects the target without a GelSight-only registration.
GT input controls, when enabled, still run before prediction. Startup, no-contact propagation,
and backend behavior are unchanged. Statistics use `imu_prediction_only_calls`,
`imu_prediction_only_accepted` instead of counting these predictions as VI
optimization calls. The legacy `imu_prediction_only_quality_rejected` counter
remains zero because this mode no longer evaluates visual quality.

Bias estimation from motion requires joint state optimization
(`frontend_fix_previous_state: false` and `gtsam_vi_marginalization: true`).

`imu_config.integration_method` accepts only `midpoint` (default). It averages
the two observations bounding each sample interval, subtracts the anchor biases,
and composes rotation with `Exp(omega * dt)`. It rotates the averaged specific
force with the half-step attitude, including `Exp(omega * dt / 2)`. Bias
Jacobians and covariance follow this method. Image-time endpoint handling and
the GelSLAM solver remain as configured.

This shared path covers static startup propagation, frontend prediction/recovery,
calibrated VI no-contact propagation, and backend IMU factors. The legacy
`InertialPropagator` path with frontend VI disabled keeps its existing rule.
The native `setUseMidpointIntegration` API comes from the patches documented in
the README. Rebuild GTSAM after applying them.

The experimental `imu_config.coast_velocity_fit` option uses past accepted
contact positions to estimate the velocity at the last accepted contact time.
`inertial_quadratic` first subtracts the measured IMU motion over that history;
`quadratic` fits positions directly. The fit changes no-contact predictions
only. The accepted contact state, recovery state and correlated prior retain
their existing ownership. A recovery can therefore return to the original
VI prediction; this option does not solve that recovery error.

The code default is disabled. The supplied `bag_simulation_config.yaml` and
`figure8_imu_refined_experiment.yaml` now enable it. A valid fit needs at least five contact states over
0.3 s inside a 0.4 s window. Fit residual, velocity change and sample-gap gates
reject unsuitable history. See `docs/imu_coast_experiments_20261021.txt` for the
equations, frame labels, comparisons and limits. The separate
`figure8_imu_refined_experiment.yaml` profile is specific to its named bag.
Both simulation profiles enable the contact velocity fit and stationary gyro-bias
uncertainty calculation only. They retain normal-flow weight 1.0 and
`gtsam_integration_sigma_m: 0.01`. The historical report's final combined metrics
also included weight changes and do not describe these current profiles.

`static_initialization.gyro_bias_uncertainty_from_samples` is a separate opt-in
initialization option. It retains the stationary gyro mean and computes that
mean's endpoint bias uncertainty from the configured gyro white-noise density,
bias random walk and selected sample times. For N samples at times t_i, the
white variance is sigma_w^2 * sum(1/dt_i) / N^2. The first sample uses the first
observed period. Endpoint-minus-mean random-walk variance is
sigma_wb^2 * sum(dt_i * (i/N)^2), i=1..N-1. The resulting standard deviation
replaces the initial gyro-bias sigma at the last stationary sample. Startup
preintegration then propagates it to the first contact time. Configured fixed
biases remain constants. The code default is false; both simulation profiles
explicitly enable this option. Calibrated noise densities and
a stationary interval are required for this model to describe the sensor.

`gtsam_integration_sigma_m` is a legacy parameter name. GTSAM adds
`sigma**2 * dt` to position covariance. Its effective units are m/sqrt(s),
and it is separate from accelerometer and gyroscope measurement densities.
Changing it leaves the mean preintegration unchanged but changes its weight
in optimization. A smaller value alone can worsen tracking; the reported
experiments vary it together with the tactile observation weight.

Set `imu_config.frontend_fix_previous_state: true` to hold the previous
pose, world velocity, accelerometer bias, and gyro bias constant throughout
each frontend optimization. Only the target state is optimized. This flag
defaults to false and takes precedence over `gtsam_vi_marginalization`:
the fixed-state mode bypasses joint marginalization, clears any carried
prior, and does not create a new marginal prior. It fixes the previous
navigation state supplied to the solve, not the visual reference frame.
Existing GT input controls, if enabled, run before that state is fixed.
The motion factor uses the fixed previous bias. Target bias is connected
only by the bias random-walk factor, so its optimum equals the fixed
previous bias. This mode does not estimate new biases from tactile motion.
Prediction, the frame quality gate, and IMU recovery remain unchanged.

Set `imu_config.gtsam_vi_height_z_enabled: true` to add a conditional height
measurement to the contact frontend. It defaults to false; the bag simulation
config enables it. Each GN iteration computes the mean of target height minus
the transformed reference-point Z over valid shared contacts. Its linear model
retains only the target GelSight body-local Z translation derivative (+1), then
maps that row to IMU increments using the calibrated sensor transform. Rotation,
XY and correspondences are held fixed in this height row's derivative, but
the measurement is resampled at the updated pose. IMU/prior coupling can still
change other state components. This is a conditional approximation, not the full
height-map Jacobian. The normal-only Z override does not clamp this measurement.
The height row is included in cost diagnostics and the retained marginal prior.

`gtsam_vi_height_z_sigma_m` sets the standard deviation of this **mean** height
measurement in metres. Its initial default is `0.0001` (0.1 mm), not a calibrated
sensor uncertainty. Do not reduce it by the square root of the number of pixels:
height integration errors are correlated. Disabling the flag restores the
normal/IMU solve for a fresh run. Prediction-only and no-contact propagation do
not evaluate this factor. A running solver reads these options at construction;
restart it after changing the config. A prior produced with height enabled
already contains that measurement information.

The frontend graph state is `T_WI`, IMU-origin velocity in world axes, and
accelerometer/gyro biases in IMU axes. `T_GI` maps IMU coordinates into
GelSight coordinates. NormalFlow evaluates `T_WG = T_WI * inverse(T_GI)`.
Its pose Jacobian includes the full offset derivative: rotating the IMU also
moves the GelSight origin. The same conversion is used for quality checks,
full GN updates, prediction output, and the retained marginal prior's tactile rows.
Frame and keyframe messages, backend solutions, and RViz remain GelSight poses.

The backend has IMU Pose3 graph variables only. Tracker GelSight relative poses form odometry edges.
Accepted NormalFlow registrations form loop edges. The keyframe message does
not carry IMU preintegration. Missing frontend IMU coverage rejects the
target. The frontend does not fall back to visual-only NormalFlow registration.
Backend edge/prior errors remain in GelSight coordinates with the full
Jacobian to IMU variables. Their configured noise values retain their meaning.
Backend pose caches and saved/output solutions remain GelSight poses; saved
graphs include the extrinsic used to rebuild their IMU variables.
Old tracker snapshots with active GelSight navigation states must start a new
trial. They are rejected rather than read as IMU states with an incorrect prior.

The EBIMU publisher must use `frame_id:=ebimu_link`, which is the default in
`ebimu_pkg`. Its `linear_acceleration` currently contains gravity (`<soa1>`),
matching the specific-force convention used by `InertialPropagator`.

## Runtime connection

Start the IMU publisher with its Quaternion ROS profile, then start the
GelSLAM tracker using a config where `imu_config.enabled` is true:

```bash
ros2 run ebimu_pkg ebimu_publisher --ros-args \
  -p port:=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 \
  -p configure_receiver:=true -p profile:=ros_quaternion

ros2 run gelslam tracker_node --ros-args -p config_path:=/absolute/path/to/config.yaml
```

The first command changes the receiver from an intake-test profile to the ROS
profile. The tracker logs `Buffering IMU samples` only after it has subscribed
successfully; samples with an unset, mismatched, duplicate, or out-of-order
header timestamp are dropped. Leave `gelsight_T_imu: null` until its physical
mounting calibration is known: buffering remains active, but propagation stays
disabled and the original new-trial fallback is retained.

## Tracking lifecycle

The tracker exposes the following lifecycle states:

```text
CONTACT_TRACKING -> NO_CONTACT_COASTING -> REACQUIRE -> CONTACT_TRACKING
                  \-> IMU_COASTING -----------------------------/
```

While no tactile contact is available, there is no corrective tactile residual.
With a calibrated transform, the tracker holds the last reliable tactile
surface and publishes it at the propagated IMU pose. A tactile `LoseTrackError`
uses the same path instead of immediately opening a new trial. On contact,
normal-flow retries against the existing reference using the propagated pose as
its initialization. A new trial is opened only when no valid IMU coast is
available.

When frontend VI is disabled, a contact gap anchors IMU propagation once at the
last accepted tactile pose. The anchor takes the tactile rotation and position.
Its velocity is the position difference of the last two tactile poses divided
by their time difference, or zero when only one pose exists. The IMU then
propagates from this anchor.

With frontend VI enabled, a contact gap copies the last accepted frontend
state. IMU-only prediction uses the same GTSAM preintegration, fixed trial
coordinates, IMU-origin velocity, gravity vector, and
IMU-axis biases as the contact frontend. It does not estimate velocity again
from tactile pose differences or reset velocity after a reference change.
Prediction always integrates IMU measurements, including when
`frontend_initialize_from_imu_preintegration` is false for visual solves.
If frontend initialization is pending, coasting also waits for that state.

The contact-gap prediction is separate from the accepted optimization state.
One GTSAM preintegration accumulates received IMU intervals from the last
accepted state. Each prediction applies the full accumulated interval to that
fixed anchor; intermediate predictions do not become optimization variables.
On contact return, the visual-inertial solve reuses this preintegration and
the anchor's marginal prior. It does not integrate the gap again or treat a
prediction as an independent measurement.

Only received sample boundaries are retained in the accumulator. The final
interval to a contact image timestamp is completed on a copy. A rejected
contact, including a second registration attempt at the same timestamp, cannot
insert an interpolated endpoint into the persistent integration history.
The buffer still retains the full raw interval for diagnostics and a fresh
integration if the anchor, bias, gravity, or IMU noise configuration changes.
Accepted contact clears the accumulator and gap prediction. The next gap
starts from the newly accepted state. Publication alone converts the predicted
trial pose to the active reference keyframe coordinates; the message interface
is unchanged. Existing duration limits and the non-VI coasting path are unchanged.

`KeyFrameMsg.trial_t_curr` carries each keyframe pose in its trial's first
keyframe coordinates. The backend uses that pose directly as its state seed.
`FrameMsg.ref_t_curr` remains relative to the active reference keyframe.
The tracker computes a trial-frame pose for ordinary frames only inside its
visual-inertial state, where consecutive IMU residuals require one fixed
coordinate frame. It does not add that pose to ordinary frame messages.

`max_coast_duration_s` limits this open-loop bridge. When that time expires,
the tracker does not publish another IMU-only tactile pose. This prevents a
long unobserved interval from becoming a contact-reacquisition prior.

## Joint normal-flow + IMU optimization

Set these fields only after mounting and time calibration:

```yaml
imu_config:
  frontend_optimization: true
  gelsight_T_imu: [[...], [...], [...], [0, 0, 0, 1]]
  # Add to an image time to obtain its matching IMU time. Calibrate this
  # jointly with the physical mount; zero is valid only for synchronized clocks.
  image_to_imu_time_offset_s: 0.0
  max_coast_duration_s: 1.0
  # Optional gravity in the first GelSight trial frame.
  frontend_gravity_mps2: [0.0, 0.0, -9.81]
  accelerometer_noise_density: 0.02
  gyroscope_noise_density: 0.002
  accelerometer_bias_random_walk: 0.0002
  gyroscope_bias_random_walk: 0.00002
```

When `frontend_gravity_mps2` is set directly, express it in the trial's first
GelSight frame. The frontend uses that vector without rotating its pose or
velocity. An absent value with `fixed_world_z` uses -Z. Published tracker and
keyframe poses use the same trial frame. The
four noise fields are continuous-time SI noise values and should be replaced
by the calibrated sensor values.

`frontend_optimization` enables the tracker-local solve. Each iteration
selects the current shared contact pixels and calculates the original
NormalFlow normal residuals and Jacobians. GTSAM linearizes its native
`CombinedImuFactor` at the same state. A `GaussianFactorGraph` solves both
sets of rows with the previous navigation state fixed. A successful joint
solve keeps its optimized Z pose. The frontend no longer has a standalone
visual-only NormalFlow fallback.

After a successful joint update, the tracker keeps the solved velocity and
biases for the next frame. GTSAM computes the IMU residual, tangent Jacobian,
preintegration covariance, and bias evolution. The former custom calculation
uses its own SE(3) pose log residual, covariance propagation, and bias
Jacobians. Both have 15 IMU rows, but their numerical values need not match.
The configured IMU noise, gravity direction, and sensor transform must match
the native factor's conventions.

## Stationary IMU initialization before contact

To use the first few seconds of IMU measurements, set:

```yaml
imu_config:
  static_initialization:
    enabled: true
    mode: startup_duration
    duration_s: 5.0
```

`startup_duration` starts at the first accepted IMU measurement in the selected
IMU clock. `duration_s` is the length of that startup interval in seconds.
The node's startup time and the first processed image do not set this origin.
The sensor must be stationary during this interval. `real_far_config.yaml`
uses the first 5 seconds for `far-agc-off-pilot-20261015/trial_004`.
The tracker uses only IMU data up to the current image time to complete the
interval. It sets zero velocity at the interval end, then predicts from that
anchor to the first contact. It does not detect a later stationary interval
or reset the origin when the propagation deadline expires. Missing start
coverage requires replay from the beginning. Do not set `start_time_s` or
`end_time_s` in this mode. Existing `fixed_time` settings remain absolute.

To use the operator rest events recorded on `/geli/events`, set:

```yaml
imu_config:
  timestamp_source: header
  static_initialization:
    enabled: true
    mode: recorded_rest_event
    events_topic: /geli/events
    motion_check: false
```

`recorded_rest_event` averages the IMU from `recorded_rest_started` to
`recorded_rest_completed`. This mean sets the gravity direction and the gyro
bias. Zero velocity is set at `trial_ready` (GO), and the tracker predicts from
GO to the first contact. The interval comes from each bag's event times, not
from the bag start. The tracker uses the `host_wall_ns` field of each event as
the IMU time. Header stamps and `timestamp_source: recorded_source` times are
both on the host receipt axis; they differ from the event time by a few
milliseconds.

With `motion_check: true`, the tracker searches the IMU from GO to the first
contact. It finds the first sample whose acceleration differs from the rest
mean by more than `motion_accel_threshold_mps2` (default 0.15), or whose
angular rate differs by more than `motion_gyro_threshold_rps` (default 0.01).
The zero-velocity anchor then moves to `motion_pre_peak_margin_s` (default
0.1 s) before that sample. The attitude at the moved anchor comes from
integration from GO. Without such a sample, the anchor stays at GO.
`max_propagation_s` limits the time from the zero-velocity anchor to the first
contact.

Set `imu_config.static_initialization.enabled: true` to initialize from a
stationary, no-contact IMU interval. The hardware and real-bag configurations
default to `false`, except for the far preset above. The simulation
configuration enables the fixed interval below for the figure-eight bag.

For `fixture-rock-long-figure8-20261001-v1-gelslam-image/bag`, use:

```yaml
imu_config:
  static_initialization:
    enabled: true
    mode: fixed_time
    start_time_s: 1.0
    end_time_s: 2.935
    min_samples: 50
    max_sample_gap_s: 0.05
    max_propagation_s: 5.0
    velocity_sigma_mps: 0.01
```

These are absolute timestamps in the selected IMU clock, not wall time or
seconds since node startup. The bag starts at header time 1.000 s. Both
GelSight and IMU GT poses remain exactly unchanged through 2.935 s. The first
changed pose sample is at 2.940 s (200 Hz GT). Thus the configured interval
lasts 1.935 s after the start of this bag. GT was used offline to select these
two constants. No runtime GT pose, velocity, or stationary detector is used.

In `fixed_time` mode, all IMU samples in the closed interval
`[start_time_s, end_time_s]` supply the mean acceleration and gyro rate. Both
boundaries must have measurement coverage; non-sample boundaries are
interpolated. The state at exactly `end_time_s` has zero velocity. The tracker
then integrates from that fixed time to the first contact. The interval and
its means are never updated during the approach. Variance, gyro-rate,
gravity-norm and mean-change stationary thresholds are not evaluated.
Minimum sample count, raw sample gaps, nonzero specific force and propagation
duration checks still apply. A missing start boundary requires replay from
before `start_time_s`; a later interval is never substituted. The fixed window
is for startup only. It is not reused to initialize a later disconnected
trial or after its propagation deadline. Use different constants for a
different dataset. Set `enabled: false` to restore the existing GT/fixed-gravity
initialization options.

The previous automatic detector remains available as an explicit alternative:

```yaml
imu_config:
  enabled: true
  frontend_optimization: true
  gtsam_vi_marginalization: true
  # Supply the calibrated 4x4 gelsight_T_imu transform in this same section.
  static_initialization:
    enabled: true
    mode: stationary_detection
    window_s: 1.0
    min_samples: 50
    accel_std_threshold_mps2: 0.05
    gyro_std_threshold_rps: 0.005
    gyro_mean_threshold_rps: 0.05
    accel_mean_change_threshold_mps2: 0.1
    gravity_norm_tolerance_mps2: 0.3
    max_sample_gap_s: 0.05
    max_propagation_s: 5.0
    velocity_sigma_mps: 0.01
```

In `stationary_detection` mode, keep the sensor stationary and out of contact for at least the configured
window before moving toward the surface. Acceleration must include gravity.
The detector uses two adjacent half-windows. It checks acceleration and gyro
dispersion, mean gyro rate relative to the configured initial gyro bias, and
the corrected acceleration magnitude. It selects the older stationary
half-window. Detected motion freezes that anchor before the approach.
This independent implementation follows the two-window approach described
in the [OpenVINS static initializer](https://docs.openvins.com/classov__init_1_1StaticInitializer.html).

At the selected stationary IMU timestamp `t_s`, GelSight velocity is zero and
gyro bias is the mean measured gyro rate. The configured accelerometer bias
is the starting value. Only its component along the measured specific force
receives a correction from the known gravity magnitude. A single stationary
orientation does not identify all accelerometer bias components or absolute
yaw. Sample averages do not remove these ambiguities. The configured initial
pose/bias prior sigmas remain uncertainty bounds; the stationary velocity
sigma comes from `velocity_sigma_mps`.

At the first valid contact image time `t_c`, GTSAM integrates the full interval
`t_s -> t_c` at the IMU origin. It predicts IMU pose and IMU-origin velocity.
At `t_s`, the initializer sets gravity to `(0, 0, -gravity_mps2)` and computes
the initial IMU rotation from the stationary specific-force direction. It
retains the calibrated IMU offset from the GelSight origin. Startup prediction
therefore uses gravity-aligned world axes from the start. At `t_c`, only a
rotation about world Z and a translation set the first-contact heading and
origin. Gravity stays on -Z. The tracker uses this aligned state directly.
Its IMU and bias
factors propagate the complete 15D prior. This startup prediction runs even
if `frontend_initialize_from_imu_preintegration` is false for contact solves.
The calibrated VI frontend always aligns its trial world to gravity. The first
published GelSight contact position is `(0, 0, 0)`. Its rotation is the gravity
alignment rotation, not generally identity. The frontend estimates in these
aligned trial axes, where gravity is `(0, 0, -gravity_mps2)`. The origin is the
first contact GelSight origin, not the IMU origin. The calibrated sensor offset
is retained in `T_WI = T_WG * T_GI`. The initial IMU position is therefore
generally nonzero. Absolute yaw remains unobservable.

The same rigid coordinate change transforms pose, velocity and the full 15D
prior, including its cross terms. IMU biases remain in IMU sensor axes. The
pre-contact integrated position mean is discarded at this origin reset.
The propagated prior retains its position uncertainty and cross correlations;
the prior does not make the first state a hard constraint during a later
joint solve. The backend uses the published first rotation as its anchor and
keeps it during pose-graph construction, group recovery, and save/load.
Alignment applies to the calibrated VI frontend, including static, GT and
fixed-gravity initialization. It does not supply gravity to visual-only tracking
or change the standalone IMU-only diagnostic node. The static switch and its
absolute time bounds are independent of gravity alignment.

Velocity is not reset to zero. Contact acceptance then supplies the existing frontend
and subsequent no-contact coasting paths. The backend remains pose-only.

The startup interval has its own `max_propagation_s` limit. The existing
`max_coast_duration_s` still limits gaps after an accepted contact. Both interval
endpoints must have IMU coverage. Raw sample gaps, including gaps hidden by
endpoint interpolation, must not exceed `max_sample_gap_s`. The image-to-IMU
time offset and recorded-source clock selection apply before these checks.
Stationary detection never uses samples later than the current image time.

While enabled, this mode takes precedence over `frontend_gravity_initialization`,
`frontend_gravity_mps2`, `gravity_initialization`, and
`frontend_first_trial_velocity_from_gt`. The tracker logs superseded fields
and does not subscribe to GT for initialization. `gravity_mps2`, sensor biases,
noise, mounting calibration, timing and residual weights keep their stated
roles. Startup uncertainty propagation uses physical IMU noise without the
contact objective's relative `imu_residual_weight` scale. With the switch off,
the original gravity and GT initialization settings apply again.

Invalid prerequisites or option values produce a startup error. In particular,
the default hardware configuration still needs its calibrated `gelsight_T_imu`;
enabling this switch does not supply calibration. If no usable rest anchor or
IMU coverage exists, the tracker logs the reason and does not register a node.
It does not substitute GT, fixed -Z gravity, or zero contact velocity. Continued
contact cannot supply a missing no-contact rest window. In automatic mode, after a timed-out
approach, repeat the no-contact rest. A later disconnected tactile trial also
requires a new no-contact rest interval; a successful short-gap reacquisition
continues the accepted VI state without reinitialization.

## Simulation GT initialization

The diagnostic accepts independent controls under `imu_config`:

```yaml
frontend_gt_previous_state_test:
  enabled: true
  previous_pose_from_gt: true
  previous_velocity_from_gt: true
  reference_pose_from_gt: false
  reset_marginal_prior: false
  previous_state_time: accepted_state
```

This example changes only the previous IMU pose/velocity. Toggle each field
independently. For a reference-only test, disable both previous-state fields
and enable `reference_pose_from_gt`. For a prior-only test, disable all GT
fields and enable `reset_marginal_prior`. Pose-only and velocity-only tests are
also supported. `enabled: false` disables all interventions, regardless of
subfield values. Existing boolean `true` remains the original all-on diagnostic
with `previous_state_time: previous_image`; boolean `false` remains off.
The supplied preset's enabled value is preserved when expanding its controls.

Use `accepted_state` for independent ablations: it retains the previously
accepted navigation state's timestamp, so preintegration boundaries stay fixed
and the old prior refers to the same physical state. The shipped mapping configs
use this policy so changing only pose or velocity remains valid after an image
is rejected. `previous_image` selects
the immediately preceding input image, including rejected images. If that moves
the timestamp, both pose and velocity must come from GT. An old marginal prior
cannot constrain a different IMU time; reset it or choose `accepted_state`.
Reference-only and prior-only controls leave the previous state/time unchanged.

Enabled GT fields use full `/ground_truth/imu_pose` messages (or
`ground_truth_imu_pose_topic`). Positions use linear interpolation, rotations
SLERP, and velocity native central differences plus linear interpolation.
There is no smoothing or extrapolation. Missing required GT coverage stops the
test explicitly. Prior-only and all-false controls require no diagnostic GT.

One fixed transform from the trial's initial IMU state expresses GT in the
existing navigation world. This transform is retained until the next trial;
the initial anchor GT is not queried again after it leaves the bounded buffer.
The current target still uses the existing frontend
initializer and joint optimizer. Target GT is not an optimizer input. Biases
remain estimated. No-contact-only predictions keep their existing path.
Unavailable VI inputs or a failed solve reject the target without a visual-only
fallback. The previous GT state is a starting value, not a hard
constraint; joint optimization can change it.

With `reset_marginal_prior: true`, the old posterior is discarded and the solver
creates its usual initial prior. With it false, the same-time frozen prior's
anchor, matrix and residual gradient remain unchanged, even if its mean differs
from the GT starting value. It is linearized at that starting value. A missing
prior still uses the usual initial prior. If the accepted state jumped to a
different time after ordinary tracking/recovery, the invalid old-time prior
uses the normal initial-prior path, recorded as `initial_timestamp_change`.
This mandatory restart is separate from a requested per-frame reset. With the
diagnostic disabled, the original automatic prior selection remains unchanged.

`latest_gt_state_test` records controls, timestamps, corrected/predicted/optimized
states and `prior_source` (`initial`, `retained`, `initial_timestamp_change` or `none`). A prior toggle has no
effect when marginalization is disabled. Use a common time policy and compare
one field/group at a time, then combinations to check interactions. Good
prediction followed by larger optimization error narrows the investigation to
the optimization/measurement path; it does not alone prove an optimizer bug.

For simulation only, set `frontend_gravity_initialization` to
`ground_truth_initial_pose`. One `/ground_truth/gel_pose` orientation near the
start of each trial sets gravity for the frontend and for no-contact IMU
propagation. `gt_trial_attitude_max_age_s` limits the permitted time gap. A
trial waits for a nearby GT sample instead of reusing a prior trial's gravity.
The frontend uses this gravity vector directly in the trial coordinates. Its
VI pose, velocity, and no-contact gravity use fixed trial coordinates.
With frontend VI disabled, legacy no-contact propagation instead expresses
gravity in the active reference keyframe coordinates. To check the factor
graph before implementing VI initialization, set
`frontend_first_trial_velocity_from_gt: true`. The tracker computes central
differences of native GT IMU positions from `ground_truth_imu_pose_topic`
(default `/ground_truth/imu_pose`), including unequal sample spacing, then
linearly interpolates velocity to the initialization IMU timestamp. It rotates
that IMU-origin velocity from GT world axes into trial axes. This calculation
does not use measured gyro or a GelSight-to-IMU velocity correction.
`frontend_gt_velocity_window_s` limits each three-sample stencil's span;
`frontend_gt_velocity_max_age_s` limits support distance from the IMU time.
Both default to 0.05 s. Initialization waits for samples on both sides of the
IMU timestamp. A later GT callback completes the original timestamp and pose;
it does not move initialization to a later image or substitute zero velocity.
No extra smoothing or extrapolation is applied.
This is a simulation-only initialization check. The trial-start GT
attitudes are not tracking factors or pose corrections.

Once the first trial has a complete accepted state, the tracker spin owner
stops callback dispatch and drains all submitted executor work. It then deletes
only the initialization GT IMU subscription and clears its position buffer.
A new executor resumes the same node, state, IMU buffer and marginal prior.
Subscription deletion does not run in a timer or sensor callback. Continuous
GT diagnostics and the separate GT velocity-evaluation subscription remain active.
The experimental single-process adapter also places retirement after its queued
tracking inputs and waits for that worker before resuming ROS dispatch.

The backend does not require gravity. It uses relative pose edges with separate
odometry and loop noise settings. It saves edge data and solved poses so it
can rebuild the GTSAM graph after loading.

## Common keyframe interval

`gelslam_config.track_config.keyframe_max_interval_s` defaults to 0.5 seconds.
After this interval without a new keyframe, the next successfully tracked
contact frame becomes a keyframe. This condition applies to joint VI,
fixed-previous-State VI, GT-input tests, and IMU prediction-only mode.
It uses the selected camera acquisition time (`header` or `recorded_source`),
not callback wall time. All existing keyframe creation paths restart this
clock with the emitted keyframe's image timestamp. A new-trial or recovery
keyframe is not duplicated by the timer. No-contact IMU coast frames and
failed targets do not become timed keyframes. A contact gap can therefore
exceed this interval; the next valid contact target is eligible immediately.
Set the interval to 0 to disable only this additional condition.

The timed keyframe becomes the visual reference. Its tracker frame message
uses its new keyframe ID and identity relative pose. The world navigation
pose, velocity, biases, gravity, and retained marginal prior are preserved.

## Frontend failure counters

When the tracker node exits, it writes `Frontend tracking summary` before
destroying the node. It first waits for image callbacks to finish. Counts
cover the current run and persist across trial and reference-keyframe changes.
Loading a saved tracker starts a new set of counts. No configuration option
is required.

If an IMU-initialized contact solve returns `initial_shared_points_lt_10`, the
tracker retries the same reference/target pair once. It copies the previous
navigation State's pose, velocity, and both biases into the target's initial
guess, sets its timestamp to the target image's IMU time, and rebuilds the
initial visual correspondences. The reference, anchor-to-target IMU samples,
preintegration, and physical IMU prediction used by the consistency gate stay
the same. The retry uses the selected ordinary, fixed-keyframe, or recovery
reference VI solver. It must pass the existing SCR/CCS and IMU consistency
checks before the tracker accepts its State and commits its marginal prior.
An initializer already configured to use the previous State does not repeat
the same attempt. IMU prediction-only mode does not use this retry.

`previous_state_retry_calls` counts these additional solver calls.
`previous_state_retry_accepted` counts retries accepted by all quality checks.
The retry log records anchor and target timestamps and both initial point
counts. `vi_solve_calls` includes both the initial call and this retry.

`initial_shared_points_lt_10` counts frontend attempts that still fail the
initial fewer-than-10 shared-point check after any seed retry. Other reasons
are `vi_unavailable`, `imu_prediction_unavailable`, and `vi_refinement_failed_other`.
`frontend_failed_calls` counts these rejected frontend attempts. They raise
`LoseTrackError` and enter the existing tracking failure / IMU recovery path.
No GelSight-only registration runs, including when the frontend solver is
not enabled, calibration is missing, or IMU coverage is unavailable. If recovery
also has no usable IMU interval, retain the accepted navigation State and drop
the target. A valid IMU prediction may still recover a later input as a keyframe.
The removed `_run_normalflow_fallback` and `_track_normalflow_only` methods
are no longer available. `normalflow_only_calls`, `normalflow_only_accepted`,
`normalflow_only_quality_rejected`, and `normalflow_only_registration_failed`
remain as legacy summary keys; new execution paths never increment them.
`vi_solve_calls`, `vi_accepted`, and `vi_quality_rejected` report the VI path.
An accepted VI result means the outer SCR/CCS gate also passed.

These are call counts, not distinct image counts. Reacquisition can try a
keyframe and then the last tactile image for the same target image. Each
attempt counts separately. A VI result rejected by SCR/CCS is not counted
as `frontend_failed_calls`. The counters do not change pose estimation.

## Image/IMU input synchronization

With IMU enabled, `tracker_node` defers each image until the newest accepted
IMU timestamp reaches its selected camera acquisition time plus
`image_to_imu_time_offset_s`. This applies to header and recorded-source clocks,
contact tracking, and no-contact propagation. The pending slot holds only the
latest image; a newer image replaces an image still waiting for IMU.
No new configuration option is required. IMU-disabled input runs immediately.

Accepted IMU callbacks trigger a ROS guard in the same mutually exclusive
callback group as image input. That guard checks readiness and then runs image
tracking and publication. IMU reception remains in its separate callback group.
Waiting does not block an executor thread or add a polling timer. A pending
image cannot block shutdown when IMU stops or a bag ends before its image time.

Once ready, the existing buffer interpolates the image endpoint from received
measurements. Missing anchor coverage, excessive sample gaps, and invalid IMU
predictions still use the existing failure handling. The synchronization does
not change preintegration, visual factors, or quality checks. Direct calls to
`Tracker.track()` outside this ROS node retain their existing input contract.

At shutdown, `Image/IMU synchronization summary` reports `waited_images`,
`replaced_images`, and `pending_images`. The same values are saved under
`image_imu_sync` in `tracker_process_summary.json`. Waiting/replaced images do
not enter frontend tracking and do not count as failed frontend attempts.

## Timestamp contract

`FrameMsg` and `KeyFrameMsg` carry the source tactile image header. In the
default `header` mode, the tracker adds `image_to_imu_time_offset_s` to the
image header time. In `recorded_source` mode, it adds the offset to the
host-aligned camera source time and integrates host-aligned IMU device times.
The tracker then linearly interpolates a bracketing IMU sample when available. If DDS
dispatches an image before the matching IMU callback, the newest already
received sample is used; this avoids a spurious new trial but may lag the image
pose by one IMU period.

## Simulation bag replay

The supplied simulation bags use `/image` and `/imu/data_raw`. The ordinary
`config.yaml` is a hardware template (`/imu/data` topic and no mounting
extrinsic), so it intentionally cannot coast these bags. Use
`config/bag_simulation_config.yaml`, which carries the simulation-only
`T_SI` extrinsic, and play both observation topics:

```bash
ros2 launch gelslam bag_replay_launch.py \
  config_path:=/home/dongjae/gelslam/src/gelslam/config/bag_simulation_config.yaml

ros2 bag play /home/dongjae/dataset/GeLISLAM/fixture-rock-zigzag-gap-20260929-v1-gelslam-image/bag \
  --clock 100 --topics /image /imu/data_raw /ground_truth/gel_pose /ground_truth/imu_pose
```

The replay launch starts RViz by default.  For an RViz-only view (without the
legacy Open3D window), add `rendering:=false`.  Its fixed frame is
`gelslam_gravity_aligned`: gray is the pose-graph coverage mesh, red is the latest tactile
patch, and the TF display shows `gelsight_estimate` with the configured
`imu_estimate` child axis.  The IMU axis is the configured tactile-to-IMU
extrinsic, not an independently estimated IMU orientation.
RViz obtains the actual initialized frontend gravity. For an aligned frontend
world, the TF from `gelslam_gravity_aligned` to `gelslam_map` is identity; it does
not apply the alignment a second time. Legacy trial axes use the display-only
alignment. Each disconnected trial has its own initial position and axes;
loop closure can later place the trial in a merged backend map.

The simulation transform and noise are not a substitute for calibration of the
physical EBIMU/GelSight rig.

On the first contact image after a gap, the tracker first propagates to that
image timestamp and uses the resulting pose as the normal-flow initial guess.
It tries the active keyframe first, then retries against the last actual
tactile frame.  If neither tactile registration succeeds, it continues IMU
coasting only while there is no tactile contact.  During contact, tactile
recovery remains authoritative: a failed normal-flow update never publishes an
IMU-only pose.  `coast_during_contact_tracking_failure` is false by default;
set it true only for an explicit experimental comparison.

## Hardware-dependent follow-up

Before enabling pose propagation, measure or validate:

1. IMU-to-GelSight extrinsic transform and axis conventions.
2. Camera/IMU time offset.
3. Accelerometer and gyroscope noise density and bias random walk.
4. Stationary initialization for gravity, velocity, and bias.

The frontend IMU factor uses SI units. The backend Pose3 graph uses metres.
