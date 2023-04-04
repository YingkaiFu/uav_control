#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>

geometry_msgs::TwistStamped uav_velocity;
geometry_msgs::PoseStamped uav_pose;
tf::Quaternion quat;
double roll = 0.0;
double pitch = 0.0;
double yaw = 0.0;


void uav_v_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    uav_velocity = *msg;
}
void uav_p_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    uav_pose = *msg;
    // tf::quaternionStampedMsgToTF(uav_pose.pose.orientation, quat);
    tf::Quaternion quat(uav_pose.pose.orientation.x,
        uav_pose.pose.orientation.y,
        uav_pose.pose.orientation.z,
        uav_pose.pose.orientation.w
    );
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
}


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "state_out");
    ros::NodeHandle nh;
    // 订阅速度
    ros::Subscriber uav_v_sub = nh.subscribe<geometry_msgs::TwistStamped>
        ("/mavros/local_position/velocity_body", 10, uav_v_cb);
    // 订阅位置
    ros::Subscriber uav_p_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("/mavros/local_position/pose", 10, uav_p_cb);
    ros::Rate rate(1);
    while (ros::ok())
    {
        ROS_INFO("----------------------------------------");
        ROS_INFO("uav_vx:%.2f uav_vy:%.2f uav_vz:%.2f uav_az:%.2f ",
            uav_velocity.twist.linear.x,
            uav_velocity.twist.linear.y,
            uav_velocity.twist.linear.z,
            uav_velocity.twist.angular.z);
        ROS_INFO("uav_px:%.2f uav_py:%.2f uav_pz:%.2f",
            uav_pose.pose.position.x,
            uav_pose.pose.position.y,
            uav_pose.pose.position.z);
        ROS_INFO("uav_roll:%.2f uav_pitch:%.2f uav_yaw:%.2f",
            roll, pitch, yaw);
        ros::spinOnce();
        rate.sleep();
    }



    return 0;
}

