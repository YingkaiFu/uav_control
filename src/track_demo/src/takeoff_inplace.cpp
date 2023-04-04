/**
 * @file offb_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo Classic SITL
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
using namespace std;

mavros_msgs::State current_state;
geometry_msgs::PoseStamped uav_pose;
void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}
void uav_p_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    uav_pose = *msg;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "offb_inplace_node");
    // ros::NodeHandle nh("~");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
        ("mavros/state", 10, state_cb);
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
        ("mavros/setpoint_position/local", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
        ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
        ("mavros/set_mode");
    //订阅位置 
    ros::Subscriber uav_p_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("/mavros/local_position/pose", 10, uav_p_cb);

    float high;
    ros::param::param<float>("~h", high, 3);
    ROS_INFO("high:%f", high);



    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while (ros::ok() && !current_state.connected) {
        ROS_INFO("wait for FCU connection");
        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::PoseStamped pose;


    //send a few setpoints before starting
    // for (int i = 100; ros::ok() && i > 0; --i) {
    //     local_pos_pub.publish(pose);
    //     ros::spinOnce();
    //     rate.sleep();
    // }


    pose.pose.position.x = uav_pose.pose.position.x;
    pose.pose.position.y = uav_pose.pose.position.y;
    pose.pose.position.z = high;
    pose.pose.orientation.x = uav_pose.pose.orientation.x;
    pose.pose.orientation.y = uav_pose.pose.orientation.y;
    pose.pose.orientation.z = uav_pose.pose.orientation.z;
    pose.pose.orientation.w = uav_pose.pose.orientation.w;
    ROS_INFO("px:%f, py:%f, pz:%f",
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    while (ros::ok()) {
        if (current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))) {
            if (set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent) {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        }
        else {
            if (!current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))) {
                if (arming_client.call(arm_cmd) &&
                    arm_cmd.response.success) {
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }
        if (fabs(uav_pose.pose.position.x - pose.pose.position.x) < 0.1 &&
            fabs(uav_pose.pose.position.y - pose.pose.position.y) < 0.1 &&
            fabs(uav_pose.pose.position.z - pose.pose.position.z) < 0.1) {
            ROS_INFO("无人机飞到了(%f, %f, %f)", pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
            offb_set_mode.request.custom_mode = "POSITION";
            if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent) {
                ROS_INFO("Reset to Position mode");
                break;
            }

        }

        local_pos_pub.publish(pose);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}