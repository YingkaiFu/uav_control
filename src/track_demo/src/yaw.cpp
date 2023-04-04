#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>

double yaw = 0.0;

void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch;
    m.getRPY(roll, pitch, yaw);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "get_yaw_node");
    ros::NodeHandle nh;

    ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>("/mavros/imu/data", 10, imuCallback);

    ros::Rate rate(10); // 10Hz

    while (ros::ok()) {
        ROS_INFO("Yaw: %f", yaw);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
