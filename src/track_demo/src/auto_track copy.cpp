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
const double h = 1.5;
// 调整高度的速度（上升或下降）
const double hv = 0.1;
const double rotate_speed = 0.2;
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
    if (!start)
        return;

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
        ROS_INFO("找到目标位置, x = %f, y = %f", x, y);
        // 将小车（所在像素点）相对无人机（图像中心像素点）的位置归一化到0 ~ 1之间，并以此作为控制无人机的速度，小车离无人机越远，则无人机的速度越大，否则无人机的速度越小
        // double vx = abs(centX - x) / centX;
        double vy = abs(centY - y) / centY *rotate_speed;

        // 经测试，无人机发送的图像的垂直方向是无人机的x方向，图像的水平方向是无人机的y方向
        // 因此，若小车（像素位置）在无人机（像素位置）上方，需要发送一个正的x方向速度，否则要发送一个负方向的速度
        if (y < centY)
            velocity.angular.z = vy;
        else
            velocity.angular.z = -vy;

        // y方向同理
        // if (y < centY) velocity.angular.y = vy;
        // else velocity.angular.z = -vy;

        // 若不给无人机发送z方向的速度，无人机会时上时下，因此通过下面这个代码控制无人机高度，若低于一定高度，就发布z方向的速度，若高于某个高度，就发送一个-z方向的速度，让无人机下降
        if (curH < h - 0.3)
            velocity.linear.z = hv;
        else if (curH < h + 0.3)
            velocity.linear.z = 0;
        else
            velocity.linear.z = (curH - h) * -hv;
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
        // else
        // {
        //     // 无人机丢失目标五秒后，开始向上飞行（扩大视野）来搜寻小车，搜寻的最高高度是无人机跟踪小车高度的两倍，这也是前面代码中控制无人机下降的原因，若无人机在升空过程中发现目标小车，会立刻下降跟踪小车
        //     if (curH < 2 * h - 1)
        //     {
        //         ROS_INFO("上升高度寻找，当前高度为：%.2f", curH);
        //         velocity.linear.z = hv;
        //     }
        //     else
        //     {
        //         if (curH > 2 * h + 1)
        //             velocity.linear.z = -hv;
        //         else
        //             velocity.linear.z = 0;
        //         ROS_INFO("目标丢失。。。");
        //     }
        // }
    }
}

void do_H(const mavros_msgs::Altitude::ConstPtr &msg)
{
    curH = msg->local;
}

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr &msg)
{
    current_state = *msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);
    ros::Publisher local_vec_pub = nh.advertise<geometry_msgs::Twist>("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
    ros::Subscriber img_sub = nh.subscribe<track_demo::Track_Info>("/object_detection/siamfcpp_tracker", 1, control_vol);

    ros::Subscriber height_sub = nh.subscribe<mavros_msgs::Altitude>("/mavros/altitude", 10, do_H);

    // the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30.0);

    // wait for FCU connection
    while (ros::ok() && !current_state.connected)
    {
        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = h;

    velocity.linear.x = 0;
    velocity.linear.y = 0;
    velocity.linear.z = 0;

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    bool takeoff = false;

    while (ros::ok())
    {
        if (!takeoff)
        {
            if (current_state.mode != "OFFBOARD" &&
                (ros::Time::now() - last_request > ros::Duration(2.0)))
            {
                if (set_mode_client.call(offb_set_mode) &&
                    offb_set_mode.response.mode_sent)
                {
                    ROS_INFO("Offboard enabled");
                }
                last_request = ros::Time::now();
            }

            if (!current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(2.0)))
            {
                if (arming_client.call(arm_cmd) &&
                    arm_cmd.response.success)
                {
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }

            if (current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0)))
            {
                takeoff = true;
                ROS_INFO("Vehicle stabled");
                start = true;
                ROS_INFO("开始追踪...");
                last_request = ros::Time::now();
            }

            local_pos_pub.publish(pose);
        }
        else
        {
            local_vec_pub.publish(velocity);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
