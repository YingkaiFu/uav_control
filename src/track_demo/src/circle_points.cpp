#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "circle_controller");
    ros::NodeHandle nh;

    // 发布器，发布目标姿态
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // 发布周期为20Hz
    ros::Rate rate(20.0);

    double radius = 5.0; // 圆的半径
    double center_x = 0.0;
    double center_y = 0.0;

    // generate the points on the circle
    std::vector<std::pair<double, double>> points;
    for (double t = 0; t < 2 * M_PI; t += 0.01)
    {
        double x = center_x + radius * cos(t);
        double y = center_y + radius * sin(t);
        points.push_back(std::make_pair(x, y));
    }

    geometry_msgs::PoseStamped pose;
    pose.pose.position.z = 2;

    int index = 0;
    while (ros::ok())
    {
        pose.header.stamp = ros::Time::now();

        double target_x = points[index].first;
        double target_y = points[index].second;

        pose.pose.position.x = target_x;
        pose.pose.position.y = target_y;

        pose_pub.publish(pose);

        // update the index to the next point on the circle
        index++;
        if (index >= points.size())
        {
            index = 0;
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}