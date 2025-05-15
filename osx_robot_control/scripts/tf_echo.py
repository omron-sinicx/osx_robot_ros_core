#!/usr/bin/python
"""
Basically same as `rosrun tf tf_echo` but prints more decimal points
"""
import rospy
import tf
import numpy as np

if __name__ == '__main__':
    rospy.init_node('tf_a')
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    counter = 0
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/a_bot_right_inner_knuckle', '/a_bot_right_inner_finger', rospy.Time(0))
            print (np.round(trans, 6), np.roll(np.round(rot, 6), 1))
        except (tf.LookupException, tf.ConnectivityException):
            continue

        rate.sleep()
        counter += 1
        if counter > 2:
            break
