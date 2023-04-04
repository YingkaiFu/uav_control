#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/Altitude.h>
#include <mavros_msgs/PositionTarget.h>
#define waittime 5

geometry_msgs::Twist velocity;
geometry_msgs::TwistStamped uav_velocity;
mavros_msgs::State current_state;
mavros_msgs::PositionTarget uav_velocity_send_linear;
geometry_msgs::TwistStamped uav_velocity_send_angular;


void cmd_cb(const geometry_msgs::Twist::ConstPtr& msg) {
    velocity = *msg;
    // xyz线速度控制
    uav_velocity_send_linear.coordinate_frame = 8;
    uav_velocity_send_linear.header.frame_id = "world";
    uav_velocity_send_linear.header.stamp = ros::Time::now();
    uav_velocity_send_linear.velocity.x = velocity.linear.x;
    uav_velocity_send_linear.velocity.y = velocity.linear.y;
    uav_velocity_send_linear.velocity.z = velocity.linear.z;
    uav_velocity_send_linear.type_mask = mavros_msgs::PositionTarget::IGNORE_PX |
        mavros_msgs::PositionTarget::IGNORE_PY |
        mavros_msgs::PositionTarget::IGNORE_PZ |
        mavros_msgs::PositionTarget::IGNORE_AFX |
        mavros_msgs::PositionTarget::IGNORE_AFY |
        mavros_msgs::PositionTarget::IGNORE_AFZ |
        mavros_msgs::PositionTarget::FORCE |
        mavros_msgs::PositionTarget::IGNORE_YAW |
        mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    // 轴的速度控制
    // uav_velocity_send_angular.twist.linear.x = uav_velocity.twist.linear.x;
    // uav_velocity_send_angular.twist.linear.y = uav_velocity.twist.linear.y;
    // uav_velocity_send_angular.twist.linear.z = uav_velocity.twist.linear.z;
    // uav_velocity_send_angular.twist.angular.x = uav_velocity.twist.angular.x;
    // uav_velocity_send_angular.twist.angular.y = uav_velocity.twist.angular.y;
    uav_velocity_send_angular.twist.angular.z = velocity.angular.z;
}

void uav_v_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    uav_velocity = *msg;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "auto_control");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");
    ros::Subscriber cmd_sub = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, cmd_cb);
    ros::Subscriber uav_v_sub = nh.subscribe<geometry_msgs::TwistStamped>
        ("/mavros/local_position/velocity_body", 10, uav_v_cb);
    // ros::Publisher cmd_pub2px4 = nh.advertise<geometry_msgs::TwistStamped>
    //     ("/mavros/setpoint_velocity/cmd_vel", 10);

    ros::Publisher cmd_pub2px4_linear = nh.advertise<mavros_msgs::PositionTarget>
        ("/mavros/setpoint_raw/local", 10);
    ros::Publisher cmd_pub2px4_angular = nh.advertise<geometry_msgs::TwistStamped>
        ("/mavros/setpoint_velocity/cmd_vel", 10);

    ros::Rate rate(10);

    while (ros::ok())
    {
        cmd_pub2px4_linear.publish(uav_velocity_send_linear);
        cmd_pub2px4_angular.publish(uav_velocity_send_angular);
        ROS_INFO("vx:%.2f vy:%.2f vz:%.2f az:%.2f ",
            uav_velocity_send_linear.velocity.x,
            uav_velocity_send_linear.velocity.y,
            uav_velocity_send_linear.velocity.z,
            uav_velocity_send_angular.twist.angular.z);
        ROS_INFO("uav_vx:%.2f uav_vy:%.2f uav_vz:%.2f uav_az:%.2f ",
            uav_velocity.twist.linear.x,
            uav_velocity.twist.linear.y,
            uav_velocity.twist.linear.z,
            uav_velocity.twist.angular.z);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}