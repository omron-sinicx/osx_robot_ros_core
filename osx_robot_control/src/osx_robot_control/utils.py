from typing import Union
import rospy
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg


def transform_pose(buffer: tf2_ros.Buffer,
                   target_frame: str,
                   pose_stamped: geometry_msgs.msg.PoseStamped,
                   timeout: rospy.Duration = rospy.Duration(1.0)) -> Union[geometry_msgs.msg.PoseStamped, bool]:
    """
    Transform a pose to a target frame using tf2_ros.
    Similar interface to the old tf.TransformListener.transformPose()

    Args:
        buffer (tf2_ros.Buffer): The tf2 buffer to use for transforms
        target_frame (str): The frame to transform the pose to
        pose_stamped (geometry_msgs.msg.PoseStamped): The pose to transform
        timeout (rospy.Duration): How long to wait for the transform

    Returns:
        geometry_msgs.msg.PoseStamped: The transformed pose, or False if transform failed
    """
    try:
        transform = buffer.lookup_transform(
            target_frame,
            pose_stamped.header.frame_id,
            pose_stamped.header.stamp,
            timeout
        )
        return tf2_geometry_msgs.do_transform_pose(
            pose_stamped,
            transform
        )
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to transform pose: {str(e)}")
        return False
