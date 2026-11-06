# FrameMsg.msg

# Timestamp and source frame of the tactile image used for this frame.
std_msgs/Header header

# Height map (H, W) as an Image
sensor_msgs/Image height_map
# Contact mask (H, W) as an Image
sensor_msgs/Image contact_mask
# Normal map (H, W, 3) as an Image
sensor_msgs/Image normal_map
# Laplacian map (H, W) as an Image
sensor_msgs/Image laplacian_map

# Frame's ID number
int32 fid
# Reference keyframe's ID number
int32 ref_kid
# Transformation matrix (4, 4) as a flattened array
float32[] ref_t_curr

# Estimation mode and actual pose time in the image clock.
string tracking_status "contact_tracking"
bool pose_valid true
float64 pose_timestamp

# Frontend estimate of the reference keyframe (ref_kid) when this frame was
# published, relative to that keyframe's published pose: inv(published) * estimate
# (GelSight poses, 4x4 flattened). Empty: the estimate is the published pose.
# Once the keyframe's keyframe_update reaches the backend, the final evaluation
# resolves this frame on the keyframe's final pose through this estimate.
float64[] published_t_ref_estimate
