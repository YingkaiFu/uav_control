#! /usr/bin/env python3
import rospy

if __name__ == "__main__":
    rospy.init_node("HW")
    name = "qsl"
    rospy.loginfo("HW:%s",name)

