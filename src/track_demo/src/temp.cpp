#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_marker_node");
    ros::NodeHandle nh;

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("drone_marker", 10);

    // 设置Marker的基本参数
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    // 初始化时间戳和顶点列表
    ros::Time start_time = ros::Time::now();
    std::vector<geometry_msgs::PoseStamped> points;

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        // 获取无人机当前位置和时间戳
        geometry_msgs::PoseStamped drone_pose;
        drone_pose.header.stamp = ros::Time::now();
        // 假设获取无人机当前位置的代码为 drone_pose.pose.position = get_drone_position();

        // 将无人机位置添加到Marker的顶点列表中
        points.push_back(drone_pose);

        // 设置Marker的持续时间，即只显示最近的5秒钟的顶点
        ros::Time now = ros::Time::now();
        marker.header.stamp = now;
        marker.lifetime = ros::Duration(0.0);

        // 删除过时的点
        while (!points.empty() && (now - points.front().header.stamp).toSec() > 5.0)
        {
            points.erase(points.begin());
        }

        // 设置Marker的顶点列表
        marker.points.clear();
        for (const auto& pose : points)
        {
            marker.points.push_back(pose.pose.position);
        }

        // 发布Marker消息
        marker_pub.publish(marker);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
