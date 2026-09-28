# IMU Integration Preparation

This repository currently remains tactile-only. The code in this change records
timestamps and provides hardware-independent IMU buffering and propagation
utilities; it does not subscribe to an IMU or add IMU factors to the pose graph.

## Tracking lifecycle

The tracker exposes the following lifecycle states:

```text
CONTACT_TRACKING -> NO_CONTACT_COASTING -> REACQUIRE -> CONTACT_TRACKING
```

While no tactile contact is available, there is no corrective tactile residual.
When an IMU is enabled in a future change, this state will only propagate the
last corrected navigation state. It must not run an optimizer without an
independent observation.

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

Before enabling `imu_config.enabled`, measure or validate:

1. IMU-to-GelSight extrinsic transform and axis conventions.
2. Camera/IMU time offset.
3. Accelerometer and gyroscope noise density and bias random walk.
4. Stationary initialization for gravity, velocity, and bias.

The graph integration phase will use SI units and add pose, velocity, and bias
states with a preintegrated IMU factor only after those values are available.
