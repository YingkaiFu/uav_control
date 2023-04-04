#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

// 无人机的位置
geometry_msgs::PoseStamped current_pose;
// 设置rviz中的Marker类型为箭头(Arrow)，并设置尺寸和颜色
visualization_msgs::Marker marker;
// 存储无人机的点
std::vector<geometry_msgs::Point> points;

// 回调函数
void uav_p_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;

    // 将位姿信息转换为Marker消息类型
    geometry_msgs::Point p;
    p = msg->pose.position;
    points.push_back(p);
    marker.points = points;
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_tragectory_visualization");
    ros::NodeHandle nh;
    // 订阅无人机的位置
    ros::Subscriber uav_p_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("/mavros/local_position/pose", 10, uav_p_cb);
    // 发布轨迹
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("drone_tragectory_marker", 10);

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "drone_tragectory_marker";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.01;
    marker.color.a = 1;
    marker.color.r = 0;
    marker.color.g = 0.0;
    marker.color.b = 1;
    marker.pose.orientation.w = 1;
    marker.points.clear();
    // 设置Marker的持续时间为5秒
    // marker.lifetime = ros::Duration(5.0);

    ros::Rate loop_rate(10); // 发布频率为10Hz

    while (ros::ok())
    {
        // 发布Marker
        marker_pub.publish(marker);

        ros::spinOnce();
        loop_rate.sleep();
    }
    // ros::spin();
    return 0;
}
