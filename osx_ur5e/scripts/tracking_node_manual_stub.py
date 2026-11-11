#!/usr/bin/env python3
"""Fills in the one service the SAM 3 tracker does not provide.

evaluate_policy_points.py waits for three services from its tracker before it
starts: <ns>/reset_and_track, <ns>/reset_and_track_manual and <ns>/stop_track.
The SAM 3 RGB-D tracker (tracking_node_rgbd.py --sam_points, started with
~/sam3/docker/run_tracker.sh) provides the first and last, plus
~environment_state, ~visualization_image and ~num_points, but has no manual
point picking -- SAM 3 places every point itself. Without this stub the
evaluator times out after 15 s with "tracking_node services not available".

This node advertises only <ns>/reset_and_track_manual and always answers
success=False, so choosing 'm' at the evaluator's reset prompt just brings the
retry prompt back (Enter or 'r' re-runs SAM 3). It never touches the cameras or the tracker.

Usage (in the lab container, alongside run_tracker.sh on the host):
    rosrun osx_ur5e tracking_node_manual_stub.py
    rosrun osx_ur5e tracking_node_manual_stub.py --tracking_node_ns tracking_node
"""

import argparse
import sys

import rospy
from std_srvs.srv import Trigger, TriggerResponse

MESSAGE = ("Manual point picking is not available with the SAM 3 tracker; "
           "press Enter (or 'r') to re-run SAM 3 on the current frame.")


def _handle_reset_and_track_manual(req) -> TriggerResponse:
    rospy.logwarn("reset_and_track_manual called -- not supported by the SAM 3 tracker.")
    return TriggerResponse(success=False, message=MESSAGE)


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Stub <ns>/reset_and_track_manual for the SAM 3 tracker")
    parser.add_argument("--tracking_node_ns", type=str, default="tracking_node",
                        help="Same value as evaluate_policy_points.py's --tracking_node_ns.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    ns = "/" + args.tracking_node_ns.strip("/")
    rospy.init_node("tracking_node_manual_stub", anonymous=False)
    rospy.Service(f"{ns}/reset_and_track_manual", Trigger, _handle_reset_and_track_manual)
    rospy.loginfo(f"{ns}/reset_and_track_manual stub ready (always answers success=False).")
    rospy.spin()


if __name__ == "__main__":
    main()
