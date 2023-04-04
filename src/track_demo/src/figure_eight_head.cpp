#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <tf/tf.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "figure_eight_controller");
    ros::NodeHandle nh;

    // 发布器，发布目标姿态
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // 发布周期为10Hz
    ros::Rate rate(10.0);

    double radius = 5.0; // 圆的半径
    double omega = 0.5; // 角速度
    double z = 5.0; // 高度

    while (ros::ok())
    {
        double t = ros::Time::now().toSec();
        double x = radius * cos(omega * t);
        double y = radius * sin(2 * omega * t);

        // 计算无人机的朝向
        double vx = -radius * omega * sin(omega * t);
        double vy = 2 * radius * omega * cos(2 * omega * t);
        double yaw = atan2(vy, vx);

        // 发布目标姿态，位置为(x,y,z)，朝向yaw
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "base_footprint";
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = z;
        tf::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        pose_pub.publish(pose);

        rate.sleep();
    }

    return 0;
}
