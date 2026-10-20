# Optimized IMU-origin states expressed in each original tracking trial.
std_msgs/Header header
string session_id
uint64 epoch
uint64 revision
int32[] kids
int32[] trial_ids
uint64[] state_revisions
# One 26-double row per kid: timestamp, pose(16), velocity(3), ba(3), bg(3).
float64[] states
