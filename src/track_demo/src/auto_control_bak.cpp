


#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>

geometry_msgs::Twist velocity;
geometry_msgs::TwistStamped uav_velocity;

void cmd_cb(const geometry_msgs::Twist::ConstPtr& msg) {
    velocity = *msg;
}

void uav_v_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    uav_velocity = *msg;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "auto_control");
    ros::NodeHandle nh;
    ros::Subscriber cmd_sub = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, cmd_cb);
    ros::Subscriber uav_v_sub = nh.subscribe<geometry_msgs::TwistStamped>
        ("/mavros/local_position/velocity_body", 10, uav_v_cb);
    ros::Publisher cmd_pub2px4 = nh.advertise<geometry_msgs::Twist>
        ("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);
    ros::Rate rate(10);
    while (ros::ok())
    {
        cmd_pub2px4.publish(velocity);
        ROS_INFO("vx:%.2f vy:%.2f vz:%.2f az:%.2f ",
            velocity.linear.x,
            velocity.linear.y,
            velocity.linear.z,
            velocity.angular.z);
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
