#include <ros/ros.h>
#include <std_msgs/String.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>


class HelloWorldNode
{
public:
    HelloWorldNode():
        pub_(nh_.advertise<std_msgs::String>("hello_world_topic", 10)),
        msg_(),
        rate_(10),
        publishing_(false)
    {
        msg_.data = "Hello World!";
    }

    void run()
    {
        while (ros::ok())
        {
            if (kbhit())
            {
                if (getch() == ' ')
                {
                    publishing_ = !publishing_;

                    if (publishing_)
                    {
                        ROS_INFO("Publishing 'Hello World!'");
                    }
                    else
                    {
                        ROS_INFO("Stopped publishing 'Hello World!'");
                    }
                }
            }

            if (publishing_)
            {
                pub_.publish(msg_);
            }

            rate_.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    std_msgs::String msg_;
    ros::Rate rate_;
    bool publishing_;

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
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "hello_world_node");
    HelloWorldNode node;
    node.run();

    ROS_INFO("Shutting down...");

    return 0;
}
