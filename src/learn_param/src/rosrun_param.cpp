#include "ros/ros.h"
using namespace std;

int main(int argc, char  *argv[])
{
    ros::init(argc, argv, "rosrun_param");
    ros::NodeHandle nh("~");
    std::string username;
    nh.param<std::string>("username", username, "qushanglai");
    ROS_INFO("This is rosrun_param node.");
    // cout << GREEN << "uav_id                    : " << uav_id << " " << TAIL << endl;
    cout << "username:" << username << endl;
    return 0;
}
