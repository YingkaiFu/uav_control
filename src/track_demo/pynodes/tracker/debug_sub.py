#!/usr/bin/env python3
import sys
import os
path = sys.path[0]
path = path + '/../../src/tracker_lib'
print(path)
sys.path.append(path)
sys.path.append("/../../../../devel/lib/python3/dist-packages")
sys.path.insert(0,'/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib/python3/dist-packages/')
import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from geometry_msgs.msg import Pose
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from loguru import logger
from threading import Lock
from prometheus_msgs.msg import DetectionInfo, MultiDetectionInfo,WindowPosition

from config.config import cfg, specify_task
from model import builder as model_builder
from pipeline import builder as pipeline_builder
import math
import time
from metavision_player.msg import SerializeImage

data_sub = rospy.get_param('~event_data', '/prophesee/event_data')
show_sub = rospy.get_param('~event_show', '/prophesee/event_frame')

def callback_show(data):
    global cv_image,getshow
    time1 = time.time()
    bridge = CvBridge()
    # rospy.loginfo("yes")
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    print(time.time()-time1)
    data = cv_image.copy()
    cv2.imshow("asd",data)
    cv2.waitKey(1)

def callback(data):
    global g_image
    # print(type(data.serialize_image))
    time1 = time.time()
    array = np.array(data.serialize_image, dtype='int8')
    # time2 = time.time()
    # print(data.height, data.width, data.channel)
    # print(time2-time1)
    event_data = array.reshape(data.height, data.width)
    event_data = (event_data-np.min(event_data))/(np.max(event_data)-np.min(event_data))*255
    print(time.time()-time1)
    # event_data = event_data.astype(float)
    # g_image = event_data
    cv2.imshow("asd1",event_data)
    cv2.waitKey(1)
    # bridge = CvBridge()
    # # rospy.loginfo("yes")
    # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    # data = cv_image.copy()

if __name__=="__main__":
    rospy.init_node("HW")
    global cv_image,getshow
    getshow=False
    rospy.loginfo("yes")
    rospy.Subscriber(data_sub, SerializeImage, callback, queue_size=1)
    # rospy.Subscriber(show_sub, Image, callback_show, queue_size=1)

    # rospy.Subscriber(show_sub, Image, callback_show, queue_size=1)
    rospy.spin()