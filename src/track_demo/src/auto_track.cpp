#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/Altitude.h>
#include <iostream>
#include <vector>
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "track_demo/Track_Info.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// 一些全局变量

// 悬空高度（追踪小车的高度）
const double h = 2;
// 调整高度的速度（上升或下降）
const double hv = 0.1;
const double rotate_speed = 0.3;
// 控制无人机的速度
geometry_msgs::Twist velocity;

// 无人机当前的高度
double curH;

// 无人机是否已经稳定在空中的标志
bool start = false;
const int centY = 640;
const int centX = 360;

void control_vol(const track_demo::Track_Info::ConstPtr &msg)
{
    velocity.linear.x = 0;
    velocity.linear.y = 0;
    velocity.linear.z = 0;
    velocity.angular.x = 0;
    velocity.angular.y = 0;
    velocity.angular.z = 0;


    // 将无人机发布的图像先转化为灰度图，再进行二值化，就能得到黑白图像，若小车出现，那么在图像内有黑色的像素，否则图像全是白色像素，这也是我将小车改成黑色的原因，若改成其它颜色就不好进行分离
    bool detected = msg->detected;
    int frame = msg->frame;
    float y_dist = msg->y_dist;
    boost::array<int32_t,4> bboxs = msg->bboxs;
    static ros::Time last_find_time = ros::Time::now();
    float x = 0, y = 0;
    if (detected)
    {
        
        y = bboxs[0] + bboxs[2] / 2;
        x = bboxs[1] + bboxs[3] / 2; 
        // double vx = abs(centX - x) / centX;
        double vy = abs(centY - y) / centY *rotate_speed;

        // 经测试，无人机发送的图像的垂直方向是无人机的x方向，图像的水平方向是无人机的y方向
        // 因此，若小车（像素位置）在无人机（像素位置）上方，需要发送一个正的x方向速度，否则要发送一个负方向的速度
        if (y < centY)
            velocity.angular.z = vy;
        else
            velocity.angular.z = -vy;



        ROS_INFO("发布速度 x : %f, y : %f, z : %f", velocity.angular.x, velocity.angular.y, velocity.angular.z);
        // 记录无人机最后一次发现小车的时间，后面有用
        last_find_time = ros::Time::now();
    }
    else
    {
        ros::Time now = ros::Time::now();
        velocity.linear.x = 0;
        velocity.linear.y = 0;
        // 无人机丢失目标五秒内，什么都不操作
        if (now - last_find_time < ros::Duration(5))
        {
            ROS_INFO("没有找到目标...");
            velocity.angular.z = 0;
            velocity.linear.z = 0;
        }
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "auto_track");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");

    ros::Publisher local_vec_pub = nh.advertise<geometry_msgs::Twist>("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);

    ros::Subscriber img_sub = nh.subscribe<track_demo::Track_Info>("/object_detection/siamfcpp_tracker", 1, control_vol);


    // the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30.0);
    ros::Time last_request = ros::Time::now();



    while (ros::ok())
    {
        local_vec_pub.publish(velocity);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
