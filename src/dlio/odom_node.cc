/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.h"

int main(int argc, char** argv) {

  ros::init(argc, argv, "dlio_odom_node");
  ros::NodeHandle nh("~");

  dlio::OdomNode node(nh);

  // 多线程回调，参数0表示使用尽可能多的线程来处理消息。如果参数为0，则ROS会根据CPU核心数量自动选择线程数，以实现最佳性能
  ros::AsyncSpinner spinner(0);
  spinner.start();
  node.start();
  ros::waitForShutdown();

  return 0;

}
