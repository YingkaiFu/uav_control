#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <iostream>
using namespace std;


#define GREEN "\033[0;1;32m"
#define TAIL "\033[0m"

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
    cout << GREEN << ">>>>>>>>>>>>>>>>>>>>  UAV State  <<<<<<<<<<<<<<<<<<<<" << TAIL << endl;
    cout << GREEN << "UAV_pos [X Y Z] : " << uav_pose.pose.position.x << " [ m ] " << uav_pose.pose.position.y << " [ m ] " << uav_pose.pose.position.z << " [ m ] " << TAIL << endl;
    cout << GREEN << "UAV_vel [X Y Z] : " << uav_velocity.twist.linear.x << " [m/s] " << uav_velocity.twist.linear.y << " [m/s] " << uav_velocity.twist.linear.z << " [m/s] " << TAIL << endl;
    cout << GREEN << "UAV_att [R P Y] : " << roll << " [deg] " << pitch << " [deg] " << yaw << " [deg] " << TAIL << endl;
        ros::spinOnce();
        rate.sleep();
    }



    return 0;
}

