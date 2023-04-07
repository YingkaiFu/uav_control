#include "ros/ros.h"
#include "metavision_player/SerializeImage.h"

void doPerson(const metavision_player::SerializeImage::ConstPtr &person_p)
{
    for (auto x : person_p->serialize_image)
    {
        if (x != 0 && x != 1 && x != -1)
            ROS_INFO("%d", x);
    }
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");

    // 1.初始化 ROS 节点
    ros::init(argc, argv, "listener_person");
    // 2.创建 ROS 句柄
    ros::NodeHandle nh;
    // 3.创建订阅对象
    ros::Subscriber sub = nh.subscribe<metavision_player::SerializeImage>("/prophesee/event_data", 1, doPerson);

    // 4.回调函数中处理 person

    // 5.ros::spin();
    ros::spin();
    return 0;
}