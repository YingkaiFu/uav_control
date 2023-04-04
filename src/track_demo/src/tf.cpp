#include <tf/transform_broadcaster.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "tf_publisher");
    ros::NodeHandle node;

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    tf::Quaternion q;
    q.setRPY(0, 0, 0);
    transform.setRotation(q);

    ros::Rate rate(5.0);
    while (node.ok()) {
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link", "map"));
        rate.sleep();
    }
    return 0;
}
