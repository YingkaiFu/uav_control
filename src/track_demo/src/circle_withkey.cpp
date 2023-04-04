#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include "KeyboardEvent.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "circle_controller");
    ros::NodeHandle nh;

    // 发布器，发布目标姿态
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // 发布周期为10Hz
    ros::Rate rate(10.0);

    double radius = 2.0; // 圆的半径
    double omega = 0.1; // 角速度

    KeyboardEvent keyboardcontrol;
    char key_now;
    bool move = false;

    while (ros::ok())
    {

        keyboardcontrol.RosWhileLoopRun();
        key_now = keyboardcontrol.GetPressedKey();

        if (key_now == U_KEY_SPACE) {
            move = true;
        } else {
            move = false;
        }

        if (move) {
            double t = ros::Time::now().toSec();
            double x = radius * cos(omega * t);
            double y = radius * sin(omega * t);

            // 发布目标姿态，位置为(x,y,0)，朝向圆心的方向
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            // pose.header.frame_id = "base_footprint";
            pose.pose.position.x = x;
            pose.pose.position.y = y;
            pose.pose.position.z = 3;
            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = sin(omega * t / 2.0);
            pose.pose.orientation.w = cos(omega * t / 2.0);
            pose_pub.publish(pose);
        }

        rate.sleep();
    }

    return 0;
}
