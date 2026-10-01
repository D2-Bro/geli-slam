# IMU Integration Preparation

The tracker node subscribes to `/imu/data` and buffers calibrated gyro/accel
measurements as timestamped SI-unit `ImuSample` records. It does not yet add
IMU factors to the pose graph or alter tactile frame poses from IMU propagation.

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
header timestamp are dropped.

## Tracking lifecycle

The tracker exposes the following lifecycle states:

```text
CONTACT_TRACKING -> NO_CONTACT_COASTING -> REACQUIRE -> CONTACT_TRACKING
```

While no tactile contact is available, there is no corrective tactile residual.
The current implementation only buffers IMU data in this state. A later change
will propagate the last corrected navigation state; it must not run an optimizer
without an independent observation.

On reacquisition, the preintegrated interval from the last contact keyframe to
the new contact keyframe can be added to the backend together with the new
tactile residual. Until that feature exists, the established new-trial fallback
is unchanged.

## Timestamp contract

`FrameMsg` and `KeyFrameMsg` carry the source tactile image header. Live frames
use the node clock at image acquisition, and replay/offline frames use a stable
frame-index/FPS timeline. Future IMU integration must use these timestamps to
select samples in the interval between keyframes.

## Hardware-dependent follow-up

Before enabling pose propagation, measure or validate:

1. IMU-to-GelSight extrinsic transform and axis conventions.
2. Camera/IMU time offset.
3. Accelerometer and gyroscope noise density and bias random walk.
4. Stationary initialization for gravity, velocity, and bias.

The graph integration phase will use SI units and add pose, velocity, and bias
states with a preintegrated IMU factor only after those values are available.
