#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <tf2/LinearMath/Quaternion.h>
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

    double radius = 2.0; // 圆的半径
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
    bool publish = false;
    geometry_msgs::PoseStamped pose;
    pose.pose.position.z = 3;
    int index = 0;
    ROS_INFO("按下空格开始/暂停圆周运动");
    while (ros::ok())
    {
        if (kbhit()) {
            if (getch() == ' ') {
                publish = !publish;
                if (publish) {
                    ROS_INFO("进行圆周运动");
                } else {
                    ROS_INFO("停止圆周运动");
                }
            }
        }

        if (publish) {
            pose.header.stamp = ros::Time::now();

            double target_x = points[index].first;
            double target_y = points[index].second;

            pose.pose.position.x = target_x;
            pose.pose.position.y = target_y;

            // calculate the yaw angle to make the drone face the center
            double yaw = atan2(center_y - target_y, center_x - target_x) + M_PI;
            

            // create a quaternion representing the desired orientation
            tf2::Quaternion q;
            q.setRPY(0, 0, yaw);
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();

            // way2
            // set the target attitude
            // tf::Quaternion q = tf::createQuaternionFromYaw(atan2(-target_y + center_y, -target_x + center_x) + M_PI);
            // pose.pose.orientation.x = q.x();
            // pose.pose.orientation.y = q.y();
            // pose.pose.orientation.z = q.z();
            // pose.pose.orientation.w = q.w();


            pose_pub.publish(pose);

            // update the index to the next point on the circle
            index++;
            if (index >= points.size())
            {
                index = 0;
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}