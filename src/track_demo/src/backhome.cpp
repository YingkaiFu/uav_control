#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>

// 无人机当前状态信息
mavros_msgs::State current_state;

// 当前位置信息
geometry_msgs::PoseStamped current_pose;

// 设置目标位置
geometry_msgs::PoseStamped setpoint_pose;

// 回调函数，用于接收无人机状态信息
void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

// 回调函数，用于接收无人机位置信息
void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
}
double high = 3;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "back_home_node");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");

    // 订阅无人机状态信息
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
        ("mavros/state", 10, state_cb);

    // 订阅无人机位置信息
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("mavros/local_position/pose", 10, pose_cb);

    // 发布目标位置信息
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
        ("mavros/setpoint_position/local", 10);



    setpoint_pose.pose.position.x = 0;
    setpoint_pose.pose.position.y = 0;
    setpoint_pose.pose.position.z = high;

    ros::Rate rate(20.0);

    // 开始控制无人机回到目标位置
    while (ros::ok()) {

        // 如果无人机已经到达目标位置，则退出循环
        if (fabs(current_pose.pose.position.x) < 0.1 &&
            fabs(current_pose.pose.position.y) < 0.1 &&
            fabs(current_pose.pose.position.z - high) < 0.1) {
            ROS_INFO("无人机回到了(0, 0, %lf)", high);
            break;
        }
        setpoint_pose.header.stamp = ros::Time::now();
        local_pos_pub.publish(setpoint_pose);


        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
