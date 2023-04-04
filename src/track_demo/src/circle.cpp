// #include <ros/ros.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <math.h>

// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "circle_controller");
//     ros::NodeHandle nh;

//     // 发布器，发布目标姿态
//     ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

//     // 发布周期为10Hz
//     ros::Rate rate(10.0);

//     double radius = 2.0; // 圆的半径
//     double omega = 0.1; // 角速度

//     while (ros::ok())
//     {
//         double t = ros::Time::now().toSec();
//         double x = radius * cos(omega * t);
//         double y = radius * sin(omega * t);

//         // 发布目标姿态，位置为(x,y,0)，朝向圆心的方向
//         geometry_msgs::PoseStamped pose;
//         pose.header.stamp = ros::Time::now();
//         // pose.header.frame_id = "base_footprint";
//         pose.pose.position.x = x;
//         pose.pose.position.y = y;
//         pose.pose.position.z = 3;
//         pose.pose.orientation.x = 0.0;
//         pose.pose.orientation.y = 0.0;
//         pose.pose.orientation.z = sin(omega * t / 2.0);
//         pose.pose.orientation.w = cos(omega * t / 2.0);
//         pose_pub.publish(pose);

//         rate.sleep();
//     }

//     return 0;
// }


// #include <ros/ros.h>
// #include <geometry_msgs/Twist.h>

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "circle");
//     ros::NodeHandle nh;
//     ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1000);

//     double radius = 1.0;
//     double omega = 0.5; // 1 rad/s
//     double center_x = 3.0;
//     double center_y = 3.0;

//     ros::Rate loop_rate(10);

//     while (ros::ok())
//     {
//         geometry_msgs::Twist msg;

//         double current_x, current_y;
//         // TODO: get current position of the drone in Gazebo, using a subscriber to gazebo/model_states topic

//         // compute the desired position on the circle
//         double target_x = center_x + radius * cos(omega * ros::Time::now().toSec());
//         double target_y = center_y + radius * sin(omega * ros::Time::now().toSec());

//         // compute the desired velocity to move towards the target position
//         double dx = target_x - current_x;
//         double dy = target_y - current_y;
//         double distance = sqrt(dx*dx + dy*dy);

//         msg.linear.x = dx / distance;
//         msg.linear.y = dy / distance;
//         msg.angular.z = 0;

//         pub.publish(msg);

//         ros::spinOnce();
//         loop_rate.sleep();
//     }

//     return 0;
// }

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