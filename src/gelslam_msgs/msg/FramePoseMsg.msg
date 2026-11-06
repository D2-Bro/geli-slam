# Lightweight frame record for final backend trajectory evaluation.
std_msgs/Header header
int32 fid
int32 ref_kid
float64[16] ref_t_curr

# Estimation mode and actual pose time in the image clock.
string tracking_status "contact_tracking"
bool pose_valid true
float64 pose_timestamp

# Frontend estimate of ref_kid at publication relative to its published pose,
# inv(published) * estimate, 4x4 flattened; empty: the published pose (FrameMsg).
float64[] published_t_ref_estimate
