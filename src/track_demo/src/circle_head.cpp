#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "circle_controller");
    ros::NodeHandle nh;

    // 发布器，发布目标姿态
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // 发布周期为10Hz
    ros::Rate rate(20.0);

    double radius = 8.0; // 圆的半径
    double omega = 0.3; // 角速度

    while (ros::ok())
    {
        double t = ros::Time::now().toSec();
        double x = radius * cos(omega * t);
        double y = radius * sin(omega * t);
        double heading = atan2(-y, -x);

        // 发布目标姿态，位置为(x,y,0)，朝向圆心的方向
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "base_footprint";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 10;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = sin(heading / 2.0);
        pose.pose.orientation.w = cos(heading / 2.0);
        pose_pub.publish(pose);

        rate.sleep();
    }

    return 0;
}
