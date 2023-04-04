#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <tf/tf.h>

int getch()
{
    struct termios oldt, newt;
    int ch;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

    return ch;
}

int kbhit()
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "circle_controller");
    ros::NodeHandle nh;
    setlocale(LC_ALL, "");

    // 发布器，发布目标姿态
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    // 发布周期为20Hz
    ros::Rate rate(20.0);

    // generate target positions for figure-8 trajectory
    std::vector<geometry_msgs::PoseStamped> target_poses;

    double center_x = 0.0;
    double center_y = 0.0;
    double high = 3;

    double point_num = 500;
    double delta_t = 2 * M_PI / point_num;
    double omega = 0.5;
    double radius = 2.0;
    double x, y;
    for (double t = 0; t < point_num; t += delta_t) {
        geometry_msgs::PoseStamped pose;

        // 计算无人机的位置
        x = radius * cos(omega * t);
        y = radius * sin(2 * omega * t);

        pose.pose.position.x = center_x + x;
        pose.pose.position.y = center_y + y;

        // 计算无人机的朝向
        double vx = -radius * omega * sin(omega * t);
        double vy = 2 * radius * omega * cos(2 * omega * t);
        double yaw = atan2(vy, vx);

        tf::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        target_poses.push_back(pose);
    }




    bool publish = false;
    geometry_msgs::PoseStamped pose;
    pose.pose.position.z = high;
    int index = 0;
    ROS_INFO("按下空格开始/暂停8字形运动");
    while (ros::ok())
    {
        if (kbhit()) {
            if (getch() == ' ') {
                publish = !publish;
                if (publish) {
                    ROS_INFO("进行8字形运动");
                }
                else {
                    ROS_INFO("停止8字形运动");
                }
            }
        }

        if (publish) {
            pose.header.stamp = ros::Time::now();

            pose.pose.position.x = target_poses[index].pose.position.x;
            pose.pose.position.y = target_poses[index].pose.position.y;
            pose.pose.orientation.x = target_poses[index].pose.orientation.x;
            pose.pose.orientation.y = target_poses[index].pose.orientation.y;
            pose.pose.orientation.z = target_poses[index].pose.orientation.z;
            pose.pose.orientation.w = target_poses[index].pose.orientation.w;

            pose_pub.publish(pose);

            // update the index to the next point on the circle
            index++;
            if (index >= target_poses.size())
            {
                index = 0;
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}