/**
 * @file offb_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo SITL
 *              功能包：  roscpp std_msgs geometry_msgs mavros_msgs
 *            话题/服务名称                                     操作                                                            消息类型
 *          mavros/cmd/arming                     服务的客户端（进入待机模式）      mavros_msgs::CommandBool
 *          mavros/set_mode                         服务的客户端（设定工作模式）     mavros_msgs/Setmode
 *          mavros/state                                 订阅 无人机状态                                mavros_msgs::State
 *          mavros/setpoint_position/local    发布 无人机控制位置                        geometry_msgs::PoseStamped
 * 
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
using namespace std;

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

int main(int argc, char **argv)
{   
    
    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;
    // 订阅无人机当前状态 
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    // 发布无人机本地位置（控制）
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("mavros/setpoint_position/local", 10);

    // 服务的客户端（设定无人机的模式、状态）
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);
    ROS_INFO("000");
    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
        ROS_INFO("State: %d ",current_state.connected);
    }
    ROS_INFO("111");
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 2;

    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

 // 设定无人机工作模式 offboard
    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";
// 无人机解锁
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

// 记录当前时间
    ros::Time last_request = ros::Time::now();

    ros::Time while_before = ros::Time::now();
    cout << "循环开始的时间是：" << while_before.toSec() << endl;

    while(ros::ok()){
        // 无人机状态设定与判断      
        // 进入while循环后，先循环5s，然后再向客户端发送无人机状态设置的消息
        // set_mode_client.call   arming_client.call 
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
            ros::Time offboard_time = ros::Time::now();
            cout << "offboard_time的时间是：" << offboard_time.toSec() << endl;
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
                ros::Time vehicle_armed_time = ros::Time::now();
                cout << "vehicle_armed_time的时间是：" << vehicle_armed_time.toSec() << endl;
            }
        }

        // 发布位置控制信息
        local_pos_pub.publish(pose);
        // ROS_INFO("111");

        ros::spinOnce();
        rate.sleep();   // 影响消息发布与更新的周期
    }

    return 0;
}