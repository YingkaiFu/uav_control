#!/usr/bin/env python3
import sys
import os
path = sys.path[0]
path = path + '/../../src/pylibs'
print(path)
sys.path.append(path)
sys.path.append("/../../../../devel/lib/python3/dist-packages")
sys.path.insert(0,'/opt/ros/noetic/lib/python3/dist-packages/')
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
from event_stream.msg import SerializeImage

image_lock = Lock()
show_lock = Lock()
camera_matrix = np.zeros((3, 3), np.float32)
distortion_coefficients = np.zeros((5,), np.float32)
kcf_tracker_h = 1.0

rospy.init_node('siamrpn_tracker', anonymous=True)


def xywh2xyxy(rect):
    rect = np.array(rect, dtype=np.float32)
    return np.concatenate([
        rect[..., [0]], rect[..., [1]], rect[..., [2]] + rect[..., [0]] - 1,
        rect[..., [3]] + rect[..., [1]] - 1
    ],
                          axis=-1)

def draw_circle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, init, flag, g_image, start

    if 1:
        if event == cv2.EVENT_LBUTTONDOWN and flag == 1:
            drawing = True
            x1, y1 = x, y
            x2, y2 = -1, -1
            flag = 2
            
            init = False    
            
        x2, y2 = x, y
        if event == cv2.EVENT_LBUTTONUP and flag == 2:
            w = x2-x1
            h = y2 -y1
            if w>0 and w*h>50:
                init = True   
                start = False   
                flag = 1
                drawing = False
                # print(init)
                # print([x1,y1,x2,y2])
            else:
                x1, x2, y1, y2 = -1, -1, -1, -1
        if drawing is True:
            x2, y2 = x, y
            cv2.rectangle(g_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if event == cv2.EVENT_MBUTTONDOWN:
        flag = 1
        init = False
        x1, x2, y1, y2 = -1, -1, -1, -1

class MouseInfo:
    def __init__(self):
        self.flag = False
        self.down_xy = [0, 0]
        self.up_xy = [0, 0]
        self.cur_draw_xy = [0, 0]
        self.show_draw = False
        self.finish = False
        self.past_ts = time.time()
        self._r_double_button = False

    def __call__(self, event, x, y, flags, params):
        self.cur_draw_xy = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            if time.time() - self.past_ts < 0.5:
                self._r_double_button = True
                # print("Double Right Button")
            self.past_ts = time.time()

        if event == cv2.EVENT_LBUTTONDOWN and self.flag == False:
            self.down_xy = [x, y]
            self.up_xy = [0, 0]
            self.flag = not self.flag
            self.show_draw = True

        if event == cv2.EVENT_LBUTTONUP and self.flag == True:
            self.up_xy = [x, y]
            self.flag = not self.flag
            self.show_draw = False
            self.finish = True

    def r_double_event(self) -> bool:
        # 是否完成双击
        tmp = self._r_double_button
        self._r_double_button = False
        return tmp

    def finish_event(self) -> bool:
        # 是否完成框选
        tmp = self.finish
        self.finish = False
        return tmp

def callback(data):
    global  g_image, getim
    array = np.array(data.serialize_image, dtype='float32')
    g_image = array.reshape(data.height, data.width, data.channel)
    getim = True

def callback_show(data):
    global s_image,getshow
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    s_image =  cv_image
    getshow = True

def winpos_callback(data):
    global x1, y1, x2, y2, init, start
    x1=data.origin_x
    y1=data.origin_y
    x2=x1+data.width
    y2=y1+data.height
    if data.mode==1:
        init = True
        start = False
    else:
        init = False
        start = False
    print(data)
    
def showImage(data_sub,show_sub, camera_matrix, kcf_tracker_h, uav_id,conf):
    global g_image, getim,s_image,getshow

    start = False
    getim = False
    getshow = False
    flag_lose = False
    count_lose = 0


    root_cfg = cfg
    root_cfg.merge_from_file(conf)
    logger.info("Load experiment configuration at: %s" % conf)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    window_name = task_cfg.exp_name
    # build model
    model = model_builder.build(task, task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
    dev = torch.device("cuda:0")
    pipeline.set_device(dev)

    rospy.Subscriber(data_sub, SerializeImage, callback, queue_size=1)
    rospy.Subscriber(show_sub, Image, callback_show, queue_size=1)
    rospy.Subscriber("/detection/bbox_draw",WindowPosition,winpos_callback)
    pub = rospy.Publisher("/uav" + str(uav_id) + '/prometheus/object_detection/siamrpn_tracker', DetectionInfo, queue_size=1)

    cv2.namedWindow('image')
    draw_bbox = MouseInfo()
    cv2.setMouseCallback('image', draw_bbox)
    # rate = rospy.Rate(100)
    inde = 1
    begin = False
    while not rospy.is_shutdown():
        if not begin:
            time.sleep(1)
            begin = True
        # if getim and getshow:
        getim = False
        getshow = False
        ## ! 
        d_info = DetectionInfo()
        d_info.frame = 0
        ## ! 
        
        image = g_image.copy()
        # image = np.concatenate((image, image, image), axis=-1).astype(float)
        image_show = s_image.copy()

        if start is False and draw_bbox.finish_event():
            mouse_bbox = [
                min(draw_bbox.down_xy[0], draw_bbox.up_xy[0]),
                min(draw_bbox.down_xy[1], draw_bbox.up_xy[1]),
                max(draw_bbox.down_xy[0], draw_bbox.up_xy[0]),
                max(draw_bbox.down_xy[1], draw_bbox.up_xy[1]),
            ]
            target_pos = np.array([(mouse_bbox[0] + mouse_bbox[2]) / 2, (mouse_bbox[1] + mouse_bbox[3]) / 2])
            target_sz = np.array([mouse_bbox[2] - mouse_bbox[0], mouse_bbox[3] - mouse_bbox[1]])
            if (target_sz[0]**2 + target_sz[1]**2) < 100:
                continue
            boxes = np.array([mouse_bbox[0], mouse_bbox[1],mouse_bbox[2]-mouse_bbox[0],mouse_bbox[3]-mouse_bbox[1]]).astype(int)
            pipeline.init(image, boxes)
            logger.info("init done!")
            start = True
            flag_lose = False
            continue

        # 双击取消框选
        if draw_bbox.r_double_event():
            d_info.detected = False
            start = False
            continue

        if start is True:
            res = pipeline.update(image)  # track
            logger.info(res)
            # res = xywh2xyxy(rect_pred)
            res = [int(l) for l in res[0]]
            cv2.rectangle(image_show, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 2)

            ## ! 
            # depth = kcf_tracker_h / rect_pred[3] * camera_matrix[1,1]
            # cx = rect_pred[0] - image.shape[1] / 2
            # cy = rect_pred[1] - image.shape[0] / 2
            # d_info.position[0] = depth * cx / camera_matrix[0,0]
            # d_info.position[1] = depth * cy / camera_matrix[1,1]
            # d_info.position[2] = depth
            # d_info.sight_angle[0] = cx / (image.shape[1] / 2) * math.atan((image.shape[1] / 2) / camera_matrix[0,0])
            # d_info.sight_angle[1] = cy / (image.shape[0] / 2) * math.atan((image.shape[0] / 2) / camera_matrix[1,1])
            # d_info.detected = True
            # ## ! 

            # cv2.putText(image, str(state['score']), (res[0] + res[2], res[1] + res[3]), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,0), 1)

            # if state['score'] < 0.5:
            #     count_lose = count_lose + 1
            # else:
            #     count_lose = 0
            if count_lose > 4:
                flag_lose = True

        cv2.putText(image_show, 'Double click to cancel the selection', (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1)
        if flag_lose is True:
            cv2.putText(image_show, 'target lost', (20,40), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
            ## ! 
            d_info.detected = False

        if draw_bbox.show_draw:
            cv2.rectangle(
                image_show, draw_bbox.down_xy, draw_bbox.cur_draw_xy, (0, 255, 0), 2
            )

        cx = int(image_show.shape[1]/2)
        cy = int(image_show.shape[0]/2)
        cv2.line(image_show,(cx-20, cy), (cx+20, cy), (255, 255, 255), 2)
        cv2.line(image_show,(cx, cy-20), (cx, cy+20), (255, 255, 255), 2)
        ## ! 
        pub.publish(d_info)
        cv2.imshow('image', image_show)
        cv2.waitKey(1)
        # rate.sleep()

if __name__ == '__main__':
    data_sub = rospy.get_param('~event_data', '/prophesee/event_data')
    show_sub = rospy.get_param('~event_show', '/prophesee/event_frame')
    config = rospy.get_param('~camera_info', 'src/track_demo/config/tracker/camera_param_gazebo_monocular.yaml')
    # tracker_conf = rospy.get_param('~tracker_conf','/home/yingkai/sdk_ros_warp/src/metavision_player/conf/img_ext_dataset.yaml') 
    tracker_conf = rospy.get_param('~tracker_conf','src/track_demo/config/tracker/img_ext_dataset.yaml') 

    uav_id = rospy.get_param('~uav_id', 1)

    yaml_config_fn = config
    print('Input config file: {}'.format(config))

    # yaml_config = yaml.load(open(yaml_config_fn))
    yaml_config = load(open(yaml_config_fn), Loader=Loader)

    camera_matrix[0,0] = yaml_config['fx']
    camera_matrix[1,1] = yaml_config['fy']
    camera_matrix[2,2] = 1
    camera_matrix[0,2] = yaml_config['x0']
    camera_matrix[1,2] = yaml_config['y0']
    print(camera_matrix)

    distortion_coefficients[0] = yaml_config['k1']
    distortion_coefficients[1] = yaml_config['k2']
    distortion_coefficients[2] = yaml_config['p1']
    distortion_coefficients[3] = yaml_config['p2']
    distortion_coefficients[4] = yaml_config['k3']
    print(distortion_coefficients)

    kcf_tracker_h = yaml_config['kcf_tracker_h']
    print(kcf_tracker_h)

    showImage(data_sub,show_sub, camera_matrix, kcf_tracker_h, uav_id,tracker_conf)
