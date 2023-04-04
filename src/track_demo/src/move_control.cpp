#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/Altitude.h>
#define waittime 5

using namespace std;

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void menu() {
    ROS_INFO("请输入选项：");
    ROS_INFO("1.上");
    ROS_INFO("2.下");
    ROS_INFO("3.右转和左移动");
    ROS_INFO("4.左转和右移动");
    ROS_INFO("5.前移动");
    ROS_INFO("6.后移动");
    cout<<"请输入你的选择：";
}

// 控制无人机的速度
geometry_msgs::Twist velocity;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "move_control");
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


    // add
    ros::Publisher local_vec_pub = nh.advertise<geometry_msgs::Twist>
        ("/mavros/setpoint_velocity/cmd_vel_unstamped", 10);

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while (ros::ok() && !current_state.connected) {
        ros::spinOnce();
        rate.sleep();
        ROS_INFO("State: %d ", current_state.connected);
    }

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 10;

    //send a few setpoints before starting
    for (int i = 100; ros::ok() && i > 0; --i) {
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    bool takeoff = false;
    bool start = false;
    int choice = 0;

    while (ros::ok()) {
        if (!takeoff) {
            if (current_state.mode != "OFFBOARD" &&
                (ros::Time::now() - last_request > ros::Duration(waittime))) {
                if (set_mode_client.call(offb_set_mode) &&
                    offb_set_mode.response.mode_sent) {
                    ROS_INFO("Offboard enabled");
                }
                last_request = ros::Time::now();
            }
            else {
                if (!current_state.armed &&
                    (ros::Time::now() - last_request > ros::Duration(waittime))) {
                    if (arming_client.call(arm_cmd) &&
                        arm_cmd.response.success) {
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }

            if (current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(waittime))) {
                takeoff = true;
                start = true;
                ROS_INFO("无人机启动了");
                ROS_INFO("可以开始控制");
                last_request = ros::Time::now();
            }

            local_pos_pub.publish(pose);
        } else {
            menu();
            cin>>choice;
            switch(choice) {
                case 1: 
                    ROS_INFO("上");
                    velocity.linear.x = 0.0;
                    velocity.linear.y = 0.0;
                    velocity.linear.z = 0.2;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = 0;
                    break;
                case 2: 
                    ROS_INFO("下");
                    velocity.linear.x = 0.0;
                    velocity.linear.y = 0.0;
                    velocity.linear.z = -0.2;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = 0;
                    break;
                case 3: 
                    ROS_INFO("右转和左移动");
                    velocity.linear.x = 0.0;
                    velocity.linear.y = 0.2;
                    velocity.linear.z = 0.0;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = -0.1;
                    // velocity.angular.z = 0;
                    break;
                case 4: 
                    ROS_INFO("左转和右移动");
                    velocity.linear.x = 0.0;
                    velocity.linear.y = -0.2;
                    velocity.linear.z = 0.0;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = 0.1;
                    // velocity.angular.z = 0;
                    break;
                case 5: 
                    ROS_INFO("前");
                    velocity.linear.x = 0.2;
                    velocity.linear.y = 0.0;
                    velocity.linear.z = 0.0;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = 0;
                    break;
                case 6: 
                    ROS_INFO("后");
                    velocity.linear.x = -0.2;
                    velocity.linear.y = 0.0;
                    velocity.linear.z = 0.0;
                    velocity.angular.x = 0;
                    velocity.angular.y = 0;
                    velocity.angular.z = 0;
                    break;
                default:
                    ROS_INFO("错误的选择");
            }
            local_vec_pub.publish(velocity);
        }

        

        

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}