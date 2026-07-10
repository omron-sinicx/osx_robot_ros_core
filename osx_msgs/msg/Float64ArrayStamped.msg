# Generic stamped array of float64 values.
# Used by data collection to record command-side values that are not
# otherwise published as stamped topics (e.g. compliance stiffness commands).
std_msgs/Header header
float64[] data
