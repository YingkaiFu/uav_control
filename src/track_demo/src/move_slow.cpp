#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/Altitude.h>
#include <mavros_msgs/PositionTarget.h>
#include <tf/tf.h>
using namespace std;

// 全局变量


double target_x;        //目标位置
double target_y;
double target_z;
double target_yaw;      //目标偏航角


double kpx_track = 0.4;             //比例系数
double kpy_track = 0.4;
double kpz_track = 0.4;
double kpz_track_yaw = 0.6;



double current_roll = 0.0;          //当前的偏航、俯仰和翻滚
double current_pitch = 0.0;
double current_yaw = 0.0;


geometry_msgs::TwistStamped velocity;                   //发布的速度
geometry_msgs::PoseStamped current_pose;                //无人机的当前位置

// 回调函数
void uav_p_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
    tf::Quaternion quat(current_pose.pose.orientation.x,
        current_pose.pose.orientation.y,
        current_pose.pose.orientation.z,
        current_pose.pose.orientation.w
    );
    tf::Matrix3x3(quat).getRPY(current_roll, current_pitch, current_yaw);
}



int main(int argc, char* argv[])
{
    ros::init(argc, argv, "move_slow");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");
    // 发布速度
    ros::Publisher cmd_pub2px4 = nh.advertise<geometry_msgs::TwistStamped>
        ("/mavros/setpoint_velocity/cmd_vel", 10);
    // 订阅位置
    ros::Subscriber uav_p_sub = nh.subscribe<geometry_msgs::PoseStamped>
        ("/mavros/local_position/pose", 10, uav_p_cb);

    ros::param::param<double>("~x", target_x, 0);
    ros::param::param<double>("~y", target_y, 0);
    ros::param::param<double>("~z", target_z, 3);
    ros::param::param<double>("~yaw", target_yaw, 0);

    while (target_yaw >= 180) {
        target_yaw -= 360;
    }
    while (target_yaw < -180) {
        target_yaw += 360;
    }
    double target_yaw_rad = target_yaw / 360 * 2 * M_PI;

    ROS_INFO("目标点为(x:%f, y:%f, z:%f, yaw:%f)", 
        target_x, target_y, target_z, target_yaw);

    ros::Rate rate(20);
    int n = 0;
    while (ros::ok())
    {
        if (fabs(current_pose.pose.position.x - target_x) < 0.2 &&
            fabs(current_pose.pose.position.y - target_y) < 0.2 &&
            fabs(current_pose.pose.position.z - target_z) < 0.2 &&
            fabs(current_yaw - target_yaw_rad) < 0.03445) {
            ROS_INFO("无人机飞到了(%f, %f, %f, %f)",
                current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z,
                target_yaw);
            cout << "请重新输入(x, y, z, yaw)" << endl;
            cin >> target_x;
            cin >> target_y;
            cin >> target_z;
            cin >> target_yaw;
            while (target_yaw >= 180) {
                target_yaw -= 360;
            }
            while (target_yaw < -180) {
                target_yaw += 360;
            }
            target_yaw_rad = target_yaw / 360 * 2 * M_PI;
            if (target_x == -1) {
                break;
            }
            ROS_INFO("目标点为(x:%f, y:%f, z:%f, yaw:%f)",
                target_x, target_y, target_z, target_yaw);
        }

        double vx = kpx_track * (target_x - current_pose.pose.position.x);
        double vy = kpy_track * (target_y - current_pose.pose.position.y);

        velocity.twist.linear.x = vx;
        velocity.twist.linear.y = vy;
        velocity.twist.linear.z = kpz_track * (target_z - current_pose.pose.position.z);
        velocity.twist.angular.z = kpz_track_yaw * (target_yaw_rad - current_yaw);

        cmd_pub2px4.publish(velocity);
        if (n % 10 == 0) {
            ROS_INFO("vx:%.2f vy:%.2f vz:%.2f az:%.2f ",
                velocity.twist.linear.x,
                velocity.twist.linear.y,
                velocity.twist.linear.z,
                velocity.twist.angular.z
            );
        }
        ros::spinOnce();
        rate.sleep();
        n++;
    }
    return 0;
}


