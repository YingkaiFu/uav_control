#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <math.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_visualizer");
    ros::NodeHandle nh;

    // 发布器，发布Marker消息
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/trajectory_marker", 10);

    // 发布周期为10Hz
    ros::Rate rate(10.0);

    double radius = 5.0; // 圆的半径
    double omega = 0.5; // 角速度
    double z = 2.0; // 高度
    double max_duration = 5.0; // 最大持续时间

    // 创建Marker消息，类型为线条
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.ns = "trajectory";
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    while (ros::ok())
    {
        double t = ros::Time::now().toSec();
        double x = radius * cos(omega * t);
        double y = radius * sin(2 * omega * t);

        // 将当前位置添加到Marker消息中
        geometry_msgs::Point point;
        point.x = x;
        point.y = y;
        point.z = z;
        marker.points.push_back(point);

        // 将持续时间添加到Marker消息中
        std_msgs::ColorRGBA color;
        double duration = ros::Time::now().toSec() - marker.header.stamp.toSec();
        color.a = std::max(0.0, (max_duration - duration) / max_duration);
        marker.colors.push_back(color);

        // 删除旧的位置，以实现轨迹逐渐消失的效果
        while (!marker.colors.empty() && marker.colors.front().a <= 0.0)
        {
            marker.colors.erase(marker.colors.begin());
            marker.points.erase(marker.points.begin());
        }

        // 发布Marker消息
        marker.header.stamp = ros::Time::now();
        marker_pub.publish(marker);

        rate.sleep();
    }

    return 0;
}
