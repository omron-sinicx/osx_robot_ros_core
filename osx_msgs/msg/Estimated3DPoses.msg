int64                     class_id     # part ID detected
float64                   width        # shorted side (m)
float64                   length       # longest side (m)
float64                   confidence   # confidence localization
geometry_msgs/PoseStamped pose         # 3D poses (might have multiple poses in belt detection)
string                    info         # Additional info about the object
