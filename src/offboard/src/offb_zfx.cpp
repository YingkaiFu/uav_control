/**
 *         功能包：  roscpp std_msgs geometry_msgs mavros_msgs
 *         程序实现内容 ：使无人机飞行一个矩形轨迹后降落 （路径点）
 *          vec_pub.publish(vector);               ——  2021/12/1    Poao
 * 
 *            话题/服务名称                                     操作                                                            消息类型															头文件
 *          mavros/cmd/arming                      服务的客户端（进入待机模式）     mavros_msgs::CommandBool			#include <mavros_msgs/CommandBool.h>	
 *          mavros/set_mode                          服务的客户端（设定工作模式）     mavros_msgs/Setmode				 		#include <mavros_msgs/SetMode.h>
 *          mavros/state                                  订阅 无人机状态                                mavros_msgs::State			             	   #include <mavros_msgs/State.h>		
 *          mavros/local_position/pose          订阅 无人位置消息                            geometry_msgs::PoseStamped			  #include <geometry_msgs/PoseStamped.h>
 * 			mavros/setpoint_velocity/cmd_vel_unstamped    发布 无人机速度		geometry_msgs::Twist			 	       #include<geometry_msgs/Twist.h>
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include<geometry_msgs/Twist.h>
using namespace std;

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

// 订阅的无人机当前位置数据
geometry_msgs::PoseStamped local_pos;
void local_pos_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
    local_pos = *msg;
}


	float err_x = 0;     float err_x0 = 0;		float err_x_err = 0;
	float err_y = 0;	 float err_y0 = 0;		float err_y_err = 0;
	float err_z = 0;	 float err_z0 = 0;		float err_z_err = 0;
	float vel_x = 0;
	float vel_y = 0;
	float vel_z = 0;
	float adj_kp = 1.0;
	float adj_kd = 1.88;    // -0. 2 超调量1 左右   + 0.2 超调量减小
	//  0.5-[0.3]   0.8-[??]  1.0-[比0.8 好一些]  
	//  设定控制无人机的位置
void local_pos_control(float pos_x, float pos_y, float pos_z)
{
	// 误差
	err_x = pos_x - local_pos.pose.position.x;
	err_y = pos_y - local_pos.pose.position.y;
	err_z = pos_z - local_pos.pose.position.z;
	// 误差的误差
	err_x_err = err_x - err_x0;
	err_y_err = err_y - err_y0;
	err_z_err = err_z - err_z0;
	// 保存本次误差
	err_x0 = err_x;
	err_y0 = err_y;
	err_z0 = err_z;

	vel_x = adj_kp * err_x + adj_kd * err_x_err;
	vel_y = adj_kp * err_y + adj_kd * err_y_err;
	vel_z = adj_kp * err_z + adj_kd * err_y_err;

	// 比例控制
	// vel_x = adj_kp * err_x;
	// vel_y = adj_kp * err_y;
	// vel_z = adj_kp * err_z;


	// ROS_INFO("Pose-x: %f",local_pos.pose.position.x);
	// ROS_INFO("Pose-y: %f",local_pos.pose.position.y);
	// ROS_INFO("Pose-z: %f",local_pos.pose.position.z);
}

int main(int argc, char **argv)
{   
    
    ros::init(argc, argv, "offb_cfx");
    ros::NodeHandle nh;
    // 订阅无人机当前状态 
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
     // 订阅无人机当前位置（反馈消息） 
    ros::Subscriber local_pos_sub = nh.subscribe<geometry_msgs::PoseStamped>
            ("mavros/local_position/pose", 10, local_pos_cb);
    // 发布无人机本地位置（控制）
    ros::Publisher vec_pub = nh.advertise<geometry_msgs::Twist>
            ("mavros/setpoint_velocity/cmd_vel_unstamped", 10);

    // 服务的客户端（设定无人机的模式、状态）
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::Twist vector;
    vector.linear.x = 0.0;
	vector.linear.y = 0.0;
	vector.linear.z = 0.0;

    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        vec_pub.publish(vector);
        ros::spinOnce();
        rate.sleep();
    }

 // 设定无人机工作模式 offboard
    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";
// 无人机解锁
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

// 记录当前时间
    ros::Time last_request = ros::Time::now();

//  用于走圈的变量
	int step = 0;
	int sametimes = 0;

	ros::Time while_before = ros::Time::now();
	cout << "循环开始的时间是：" << while_before.toSec() << endl;

    while(ros::ok()){
        // 无人机状态设定与判断      
        // 进入while循环后，先循环5s，然后再向客户端发送无人机状态设置的消息
        // set_mode_client.call   arming_client.call 
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");

				ros::Time offboard_time = ros::Time::now();
				cout << "offboard_time的时间是：" << offboard_time.toSec() << endl;
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");

					ros::Time vehicle_armed_time = ros::Time::now();
					cout << "vehicle_armed_time的时间是：" << vehicle_armed_time.toSec() << endl;

                }
                last_request = ros::Time::now();
            }
            else    //  无人机 Offboard enabled && Vehicle armed 后
            {        //  无人机走矩形 每到达一个点停一会
                     //  z: 0-->10 10-->10 10-->10 10-->10  10-->10 10-->0 
                     //  x: 0-->0   0-->40   40-->40 40-->0    0-->0     0-->0
                     //  y: 0-->0   0-->0     0-->20   20-->20  20-->0   0-->0
                     //  local_pos_pub.publish(pose);
				
				switch (step)
				{
				case 0:
					local_pos_control(0,0,2);  // 速度控制
					vector.linear.x = vel_x;
					vector.linear.y = vel_y;
					vector.linear.z = vel_z;
					if (local_pos.pose.position.z >1.95 && local_pos.pose.position.z <2.05)
					{
						ros::Time step_0 = ros::Time::now();
						if (sametimes == 0)
							cout << "step_0完成的时间是：" << step_0.toSec() << endl;
						if (sametimes == 40)
						{
							step = 1;
							ros::Time step_1 = ros::Time::now();
							cout << "step_1开始的时间是：" << step_1.toSec() << endl;
						}
						else
							sametimes ++;
					}
					else
						sametimes = 0;
					break;

				case 1:
					local_pos_control(20,0,2);
					vector.linear.x = vel_x;
					vector.linear.y = vel_y;
					vector.linear.z = vel_z;
					if (local_pos.pose.position.x >19.95 && local_pos.pose.position.x <20.05)
					{
						ros::Time step_1 = ros::Time::now();
						if (sametimes == 0)
							cout << "step_1完成的时间是：" << step_1.toSec() << endl;
						if (sametimes == 40)
						{
							step = 2;
							ros::Time setp_2 = ros::Time::now();
							cout << "setp_2开始的时间是：" << setp_2.toSec() << endl;
						}
						else
							sametimes ++;
					}
					else
						sametimes = 0;
					break;

				case 2:
					local_pos_control(20,20,2);
					vector.linear.x = vel_x;
					vector.linear.y = vel_y;
					vector.linear.z = vel_z;
					if (local_pos.pose.position.y >19.95 && local_pos.pose.position.y <20.05)
					{
						ros::Time step_2 = ros::Time::now();
						if (sametimes == 0)
							cout << "step_2完成的时间是：" << step_2.toSec() << endl;
						if (sametimes == 40)
						{
							step = 3;
							ros::Time setp_3 = ros::Time::now();
							cout << "setp_3开始的时间是：" << setp_3.toSec() << endl;
						}
						else
							sametimes ++;
					}
					else
						sametimes = 0;
					break;

				case 3:
					local_pos_control(0,20,2);
					vector.linear.x = vel_x;
					vector.linear.y = vel_y;
					vector.linear.z = vel_z;
					if (local_pos.pose.position.x >-0.05 && local_pos.pose.position.x <0.05)
					{
						ros::Time step_3 = ros::Time::now();
						if (sametimes == 0)
							cout << "step_3完成的时间是：" << step_3.toSec() << endl;
						if (sametimes == 40)
						{
							step = 4;
							ros::Time setp_4 = ros::Time::now();
							cout << "setp_4开始的时间是：" << setp_4.toSec() << endl;
						}
						else
							sametimes ++;
					}
					else
						sametimes = 0;
					break;

				case 4:
					local_pos_control(0,0,2);
					vector.linear.x = vel_x;
					vector.linear.y = vel_y;
					vector.linear.z = vel_z;
					if (local_pos.pose.position.y >-0.05 && local_pos.pose.position.y <0.05)
					{
						ros::Time step_4 = ros::Time::now();
						if (sametimes == 0)
							cout << "step_4完成的时间是：" << step_4.toSec() << endl;
						if (sametimes == 40)
						{
							step = 5;
							ros::Time setp_5 = ros::Time::now();
							cout << "setp_5开始的时间是：" << setp_5.toSec() << endl;
						}
						else
							sametimes ++;
					}
					else
						sametimes = 0;
					break; 
				case 5:    // 准备降落
					offb_set_mode.request.custom_mode = "AUTO.LAND";
					if (current_state.mode != "AUTO.LAND" && (ros::Time::now() - last_request > ros::Duration(5.0)))
					{
						if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
						{
							ROS_INFO("AUTO.LAND enabled");
							step = 6;
						}
						last_request = ros::Time::now();
					}
					break;

				default:
					break;
				}
			}

        }

        // 发布位置控制信息
        vec_pub.publish(vector);




        ros::spinOnce();
        rate.sleep();   // 影响消息发布与更新的周期
    }

    return 0;
}