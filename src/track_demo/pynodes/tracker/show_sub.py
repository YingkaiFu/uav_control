from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import rospy
import cv2
import time


def callback_show(data):
    global s_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    s_image =  cv_image


if __name__ == '__main__':
    global s_image
    # s_image=None
    rospy.init_node('siamrpn_tracker', anonymous=True)
    show_sub = rospy.get_param('~event_show', '/prophesee/event_frame')
    rospy.Subscriber(show_sub, Image, callback_show, queue_size=1)
    begin = False
    while True:
        if not begin:
            time.sleep(1)
            begin = True
        time1 = time.time()
        cv2.imshow('show',s_image)
        print(time.time()-time1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break