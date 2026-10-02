# IMU Integration Preparation

The tracker node subscribes to `/imu/data` and buffers calibrated gyro/accel
measurements as timestamped SI-unit `ImuSample` records. It does not yet add
IMU factors to the pose graph. When a calibrated `gelsight_T_imu` is present,
it can dead-reckon the last tactile pose through a contact loss or a
`LoseTrackError`; this is propagation-only, not an optimizer residual.

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

No IMU factor is inserted into the pose graph in this step. On reacquisition,
a future backend can add a preintegrated IMU factor with the tactile residual.

## Timestamp contract

`FrameMsg` and `KeyFrameMsg` carry the source tactile image header. IMU samples
are linearly interpolated at a bracketing image timestamp when available. If
DDS dispatches an image before the same-time IMU callback, the newest already
received sample is used; this avoids a spurious new trial but may lag the image
pose by one IMU period.

## Simulation bag replay

The supplied simulation bags use `/image` and `/imu/data_raw`. Use
`config/bag_simulation_config.yaml`, which carries the simulation-only
`T_SI` extrinsic, and play both observation topics:

```bash
ros2 launch gelslam bag_replay_launch.py \
  config_path:=/home/dongjae/gelslam/src/gelslam/config/bag_simulation_config.yaml

ros2 bag play /home/dongjae/dataset/GeLISLAM/fixture-rock-zigzag-gap-20260929-v1-gelslam-image/bag \
  --clock 100 --topics /image /imu/data_raw /tf_static
```

The replay launch starts RViz by default.  For an RViz-only view (without the
legacy Open3D window), add `rendering:=false`.  Its fixed frame is
`gelslam_map`: gray is the pose-graph coverage mesh, red is the latest tactile
patch, and the TF display shows `gelsight_estimate` with the configured
`imu_estimate` child axis.  The IMU axis is the configured tactile-to-IMU
extrinsic, not an independently estimated IMU orientation.

The simulation transform and noise are not a substitute for calibration of the
physical EBIMU/GelSight rig.

## Hardware-dependent follow-up

Before enabling pose propagation, measure or validate:

1. IMU-to-GelSight extrinsic transform and axis conventions.
2. Camera/IMU time offset.
3. Accelerometer and gyroscope noise density and bias random walk.
4. Stationary initialization for gravity, velocity, and bias.

The graph integration phase will use SI units and add pose, velocity, and bias
states with a preintegrated IMU factor only after those values are available.
