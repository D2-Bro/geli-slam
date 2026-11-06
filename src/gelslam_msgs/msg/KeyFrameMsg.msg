# KeyFrameMsg.msg

# Timestamp and source frame of the tactile image used for this keyframe.
std_msgs/Header header

# Height map (H, W) as an Image
sensor_msgs/Image height_map
# Contact mask (H, W) as an Image
sensor_msgs/Image contact_mask
# Normal map (H, W, 3) as an Image
sensor_msgs/Image normal_map
# Laplacian map (H, W) as an Image
sensor_msgs/Image laplacian_map

# Keyframe's ID number
int32 kid
# Keyframe's Frame ID number
int32 fid
# New trial flag
bool new_trial_flag
# Pose of this keyframe in its trial's first-keyframe coordinates (4x4).
float32[] trial_t_curr

# IMU-origin navigation seed at this keyframe's synchronized IMU time.
# World axes are the same as trial_t_curr. These are initial values, not factors.
bool navigation_valid
float64 navigation_timestamp
float64[3] velocity
float64[3] accel_bias
float64[3] gyro_bias
float64[3] gravity
# False for an IMU-only recovery keyframe whose tactile match was rejected.
bool tactile_valid true
# Previous emitted keyframe to this keyframe, including both time endpoints.
# Rows: [timestamp_s, ax, ay, az, wx, wy, wz], in calibrated IMU units/axes.
float64[] imu_measurements

# Process-independent stream identity and seed basis. These add no factors.
string session_id
int32 trial_id -1
int32 anchor_kid -1
uint64 solution_revision
# Optional full-precision IMU pose, in the original tracking trial (16 entries).
float64[] navigation_pose
# Optional 26-double anchor row: timestamp, IMU pose, velocity, ba, bg.
float64[] anchor_state
# Optional frontend re-estimate of the previous keyframe (anchor_kid) after its
# publication, relative to anchor_state (empty: none). 27 doubles: kid, navigation
# timestamp, 4x4 IMU-pose change inv(anchor pose) * estimated pose (anchor body
# frame), velocity change in anchor body axes, accel bias change, gyro bias change.
# The backend applies it to that keyframe's stored seed.
float64[] keyframe_update
