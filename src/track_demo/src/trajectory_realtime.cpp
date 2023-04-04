#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

// 无人机的位置
geometry_msgs::PoseStamped current_pose;
// 设置rviz中的Marker类型为箭头(Arrow)，并设置尺寸和颜色
visualization_msgs::Marker marker;
// 存储无人机的点
std::vector<geometry_msgs::Point> points;


std::vector<ros::Time> point_timestamps;


// 回调函数
void uav_p_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_tragectory_visualization_real_time");
    ros::NodeHandle nh;
    // 订阅无人机的位置
    ros::Subscriber uav_p_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("/mavros/local_position/pose", 10, uav_p_cb);
    // 发布轨迹
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("drone_tragectory_marker_real_time", 10);

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "drone_tragectory_marker_real_time";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;

    marker.scale.x = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    marker.pose.orientation.w = 1;


    // 设置Marker的持续时间为5秒
    // marker.lifetime = ros::Duration(5.0);

    ros::Rate loop_rate(10); // 发布频率为10Hz

    while (ros::ok())
    {
        geometry_msgs::PoseStamped point;
        point.header.stamp = ros::Time::now();
        point.pose.position.x = current_pose.pose.position.x;
        point.pose.position.y = current_pose.pose.position.y;
        point.pose.position.z = current_pose.pose.position.z;
        marker.points.push_back(point.pose.position);
        point_timestamps.push_back(point.header.stamp);

        while (!point_timestamps.empty() && (ros::Time::now() - point_timestamps.front()).toSec() > 5.0)
        {
            marker.points.erase(marker.points.begin());
            point_timestamps.erase(point_timestamps.begin());
        }

        marker_pub.publish(marker);

        ros::spinOnce();
        loop_rate.sleep();
    }
    // ros::spin();
    return 0;
}