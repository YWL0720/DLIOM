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

dlio::OdomNode::OdomNode(ros::NodeHandle node_handle) : nh(node_handle) {

  this->count = 0;
  this->global_map = pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());

  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  this->isam = new gtsam::ISAM2(params);

  this->keyframe_pose_corr = Eigen::Isometry3f::Identity();
  this->icpScore = 1.0;




  // 获取rosparam
  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;
  // 注册雷达回调
  this->lidar_sub = this->nh.subscribe("pointcloud", 1,
      &dlio::OdomNode::callbackPointCloud, this, ros::TransportHints().tcpNoDelay());
  // 注册IMU回调
  this->imu_sub = this->nh.subscribe("imu", 1000,
      &dlio::OdomNode::callbackImu, this, ros::TransportHints().tcpNoDelay());

  this->gps_sub = this->nh.subscribe<sensor_msgs::NavSatFix>("gps", 1000, &dlio::OdomNode::callbackGPSWithoutAlign, this, ros::TransportHints().tcpNoDelay());

  this->odom_pub     = this->nh.advertise<nav_msgs::Odometry>("odom", 1, true);
  this->pose_pub     = this->nh.advertise<geometry_msgs::PoseStamped>("pose", 1, true);
  this->path_pub     = this->nh.advertise<nav_msgs::Path>("path", 1, true);
  this->kf_pose_pub  = this->nh.advertise<geometry_msgs::PoseArray>("kf_pose", 1, true);
  this->kf_cloud_pub = this->nh.advertise<sensor_msgs::PointCloud2>("kf_cloud", 1, true);
  this->deskewed_pub = this->nh.advertise<sensor_msgs::PointCloud2>("deskewed", 1, true);
  this->kf_connect_pub = this->nh.advertise<visualization_msgs::Marker>("kf_connect", 1, true);
  this->loop_constraint_pub = this->nh.advertise<visualization_msgs::Marker>("loop_constraint", 1, true);

  this->global_map_pub = this->nh.advertise<sensor_msgs::PointCloud2>("global_map", 100);
  this->publish_timer = this->nh.createTimer(ros::Duration(0.01), &dlio::OdomNode::publishPose, this);
  this->global_pose_pub = this->nh.advertise<geometry_msgs::PoseArray>("global_odom", 1, true);

  this->gps_pub_test = this->nh.advertise<nav_msgs::Odometry>("odom_gps", 1000);

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->deskewed_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->current_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->current_scan_w = pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->current_scan_lidar = pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->submap_cloud = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);


  this->gicp_tool.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_tool.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_tool.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_tool.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_tool.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_tool.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

//  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
//  this->gicp.setSearchMethodSource(temp, true);
//  this->gicp.setSearchMethodTarget(temp, true);
//  this->gicp_temp.setSearchMethodSource(temp, true);
//  this->gicp_temp.setSearchMethodTarget(temp, true);
//
//  this->gicp_tool.setSearchMethodSource(temp, true);
//  this->gicp_tool.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);
  this->voxel_global.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);



  this->loop_marker.ns = "loop";
  this->loop_marker.id = 0;
  this->loop_marker.type = visualization_msgs::Marker::LINE_LIST;
  this->loop_marker.scale.x = 0.1;
  this->loop_marker.color.r = 1.0;
  this->loop_marker.color.g = 0.0;
  this->loop_marker.color.b = 0.0;
  this->loop_marker.color.a = 1.0;
  this->loop_marker.action = visualization_msgs::Marker::ADD;
  this->loop_marker.pose.orientation.w = 1.0;






  this->mapping_thread = std::thread(&dlio::OdomNode::updateMap, this);
  this->loop_thread = std::thread(&dlio::OdomNode::performLoop, this);

}

dlio::OdomNode::~OdomNode() {}

void dlio::OdomNode::getParams() {

  // Version
  ros::param::param<std::string>("~dlio/version", this->version_, "0.0.0");

  // Frames
  ros::param::param<std::string>("~dlio/frames/odom", this->odom_frame, "odom");
  ros::param::param<std::string>("~dlio/frames/baselink", this->baselink_frame, "base_link");
  ros::param::param<std::string>("~dlio/frames/lidar", this->lidar_frame, "lidar");
  ros::param::param<std::string>("~dlio/frames/imu", this->imu_frame, "imu");
  ros::param::param<bool>("~dlio/pointcloud/dense", this->global_dense, false);
  // Get Node NS and Remove Leading Character
  std::string ns = ros::this_node::getNamespace();
  ns.erase(0,1);

  // Concatenate Frame Name Strings
  this->odom_frame = ns + "/" + this->odom_frame;
  this->baselink_frame = ns + "/" + this->baselink_frame;
  this->lidar_frame = ns + "/" + this->lidar_frame;
  this->imu_frame = ns + "/" + this->imu_frame;

  // Deskew FLag
  ros::param::param<bool>("~dlio/pointcloud/deskew", this->deskew_, true);

  // Gravity
  ros::param::param<double>("~dlio/odom/gravity", this->gravity_, 9.80665);

  // Keyframe Threshold
  ros::param::param<double>("~dlio/odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  ros::param::param<double>("~dlio/odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  ros::param::param<int>("~dlio/odom/submap/keyframe/knn", this->submap_knn_, 10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcc", this->submap_kcc_, 10);
  ros::param::param<bool>("~dlio/odom/submap/useJaccard", this->useJaccard, true);

  // Dense map resolution
  ros::param::param<bool>("~dlio/map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  ros::param::param<bool>("~dlio/map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  ros::param::param<double>("~dlio/odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  ros::param::param<bool>("~dlio/pointcloud/voxelize", this->vf_use_, true);
  ros::param::param<double>("~dlio/odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  ros::param::param<bool>("~dlio/adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<float> t_default{0., 0., 0.};
  std::vector<float> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<float> baselink2imu_t, baselink2imu_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2imu_R.data(), 3, 3);

  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<float> baselink2lidar_t, baselink2lidar_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2lidar_R.data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  ros::param::param<bool>("~dlio/odom/imu/calibration/accel", this->calibrate_accel_, true);
  ros::param::param<bool>("~dlio/odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  ros::param::param<double>("~dlio/odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  ros::param::param<int>("~dlio/odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<float> accel_default{0., 0., 0.}; std::vector<float> prior_accel_bias;
  std::vector<float> gyro_default{0., 0., 0.}; std::vector<float> prior_gyro_bias;

  ros::param::param<bool>("~dlio/odom/imu/approximateGravity", this->gravity_align_, true);
  ros::param::param<bool>("~dlio/imu/calibration", this->imu_calibrate_, true);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<float> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<float> imu_sm;

  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(imu_sm.data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  ros::param::param<int>("~dlio/odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  ros::param::param<int>("~dlio/odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  ros::param::param<double>("~dlio/odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  ros::param::param<int>("~dlio/odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  ros::param::param<double>("~dlio/odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  ros::param::param<double>("~dlio/odom/geo/Kp", this->geo_Kp_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kv", this->geo_Kv_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kq", this->geo_Kq_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kab", this->geo_Kab_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kgb", this->geo_Kgb_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/abias_max", this->geo_abias_max_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/gbias_max", this->geo_gbias_max_, 1.0);


}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::publishPose(const ros::TimerEvent& e) {

  // nav_msgs::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub.publish(this->odom_ros);

  // geometry_msgs::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub.publish(this->pose_ros);

}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub.publish(this->path_ros);

  // transform: odom to baselink
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br.sendTransform(transformStamped);

}

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ (boost::make_shared<pcl::PointCloud<PointType>>());

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub.publish(deskewed_ros);

}

void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, ros::Time timestamp) {

  // Push back
  geometry_msgs::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub.publish(this->kf_pose_ros);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub.publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub.publish(keyframe_cloud_ros);
  }

}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc) {

  // 将点云转换成pcl格式
  pcl::PointCloud<PointType>::Ptr original_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  // 去除NaN值点
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // 设置剪裁框 默认大小为1.0x1.0x1.0
  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // 选择激光雷达传感器的品牌型号
  // automatically detect sensor type
  this->sensor = dlio::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = dlio::SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = dlio::SensorType::VELODYNE;
      break;
    } else if (field.name == "timestamp") {
      this->sensor = dlio::SensorType::HESAI;
      break;
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void dlio::OdomNode::preprocessPoints() {

  // 对点云进行去畸变
  // Deskew the original dlio-type scan
  if (this->deskew_) {
//    std::cout << "Deskew!" << std::endl;
    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {
    // 不去畸变的情况 scan_stamp为消息头的时间
//    std::cout << "No deskew!" << std::endl;
    this->scan_stamp = this->scan_header_stamp.toSec();

    // don't process scans until IMU data is present
    // 第一帧不去畸变
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
      frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});
    // 保存最后一个点的状态作为T_prior
    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    this->current_scan_lidar->clear();
    pcl::transformPointCloud(*this->original_scan, *this->current_scan_lidar, this->extrinsics.baselink2lidar_T);
    this->voxel.setInputCloud(this->current_scan_lidar);
    this->voxel.filter(*this->current_scan_lidar);

    // 转换到结束时刻
    pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // 体素滤波进行降采样
  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_
      (boost::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan));
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::deskewPointcloud() {
  // 去畸变后的点云
  pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_scan_->points.resize(this->original_scan->points.size());

  // 当前帧的参考时间
  // individual point timestamps should be relative to this time
  double sweep_ref_time = this->scan_header_stamp.toSec();

  // 分别用于比较点时间戳，判断时间戳是否相等，获得点的全局时间
  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  // 根据不同的雷达型号，对以上三个函数分别进行定义
  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };

  }
  // 根据时间从小到大的顺序对点云进行排序
  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // 对点云进行索引化并去除时间戳相同的点，得到时间戳唯一的点
  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // 将上面去重后点云的时间戳也保存下来
  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it));
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());
  // 将时间戳中位数作为当前帧的scan_stamp
  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // 第一帧时检查IMU状态
  // don't process scans until IMU data is present
  if (!this->first_valid_scan) 
  {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    // 备份
    this->current_scan_lidar->clear();
    pcl::transformPointCloud(*deskewed_scan_, *this->current_scan_lidar, this->extrinsics.baselink2lidar_T);
    this->voxel.setInputCloud(this->current_scan_lidar);
    this->voxel.filter(*this->current_scan_lidar);

    // 第一帧认为没有畸变
    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }
  // 对于第二帧之后的情况
  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  // IMU积分
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // 如果时间同步存在问题不进行去畸变
  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    ROS_FATAL("Bad time sync between LiDAR and IMU!");

    // 备份
    this->current_scan_lidar->clear();
    pcl::transformPointCloud(*deskewed_scan_, *this->current_scan_lidar, this->extrinsics.baselink2lidar_T);
    this->voxel.setInputCloud(this->current_scan_lidar);
    this->voxel.filter(*this->current_scan_lidar);

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }
  // 将时间中位数作为
  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

  // 去畸变 实际上这里只用到了IMU积分的值
  // 备份

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      // 四维向量表示 前三维为xyz
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }


  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

  this->current_scan_lidar->clear();
  pcl::transformPointCloud(*this->deskewed_scan, *this->current_scan_lidar, this->T_prior.inverse());
  this->voxel.setInputCloud(this->current_scan_lidar);
  this->voxel.filter(*this->current_scan_lidar);

}

void dlio::OdomNode::initializeInputTarget() {
  // 构建当前关键帧
  this->prev_scan_stamp = this->scan_stamp;
  // 保存关键帧的姿态 以及去畸变降采样后的点云
  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  // 保存关键帧消息时间戳
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  // 保存关键帧点云协方差
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  // 初始T-corr为单位的
  this->keyframe_transformations.push_back(this->T_corr);
  this->keyframe_transformations_prior.push_back(this->T_prior);

  this->keyframe_stateT.push_back(this->T);


  std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_his_lidar(this->history_kf_lidar_mutex);
  pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
  pcl::copyPointCloud(*this->current_scan_lidar, *temp);
  this->history_kf_lidar.push_back(temp);
  lock_his_lidar.unlock();

  this->currentFusionState = this->state;
  this->tempKeyframe.pos = this->currentFusionState.p;
  this->tempKeyframe.rot = this->currentFusionState.q;
  this->tempKeyframe.vSim = {1};
  this->tempKeyframe.submap_kf_idx = {0};
  this->tempKeyframe.time = this->scan_stamp;
  this->v_kf_time.push_back(this->scan_stamp);
  pcl::copyPointCloud(*this->current_scan, *this->tempKeyframe.pCloud);
  this->KeyframesInfo.push_back(this->tempKeyframe);

  this->saveFirstKeyframeAndUpdateFactor();


}

void dlio::OdomNode::setInputSource() {
  // 加入kdtree
  this->gicp.setInputSource(this->current_scan);
  // 计算协方差
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void dlio::OdomNode::callbackGPS(const sensor_msgs::NavSatFixConstPtr &gps)
{

    GeographicLib::GeoCoords geoCoords(gps->latitude, gps->longitude);

    Eigen::Matrix3d gps_cov;
    gps_cov << gps->position_covariance[0], gps->position_covariance[1], gps->position_covariance[2],
               gps->position_covariance[3], gps->position_covariance[4], gps->position_covariance[5],
               gps->position_covariance[6], gps->position_covariance[7], gps->position_covariance[8];

    auto get_xy = [&](double& x, double& y) {
        stringstream ss(geoCoords.AltUTMUPSRepresentation(2));
        vector<string> temp;
        string temp_string;
        while(ss >> temp_string)
            temp.push_back(temp_string);

        x = stod(temp[1]);
        y = stod(temp[2]);
    };

    double x, y, z;
    get_xy(x, y);
    z = gps->altitude;

    if (this->dlio_initialized)
    {
        static double initx = x;
        static double inity = y;
        static double initz = z;

        Eigen::Vector3d gps_origin = {x - initx, y - inity, z - initz};
        Eigen::Vector3d gps_map = this->R_M_G * gps_origin + this->t_M_G;

        this->gps_meas.x = gps_map.x();
        this->gps_meas.y = gps_map.y();
        this->gps_meas.z = gps_map.z();
        this->gps_meas.time = gps->header.stamp.toSec();

        std::unique_lock<decltype(this->gps_mutex)> lock_gps(this->gps_mutex);
        this->v_gps_meas.push_back(this->gps_meas);
        lock_gps.unlock();

        {
            nav_msgs::Odometry odom;
            odom.header.stamp = gps->header.stamp;
            odom.header.frame_id = this->odom_frame;
            odom.child_frame_id = this->baselink_frame;
            odom.pose.pose.position.x = this->gps_meas.x;
            odom.pose.pose.position.y = this->gps_meas.y;
            odom.pose.pose.position.z = this->gps_meas.z;
            this->gps_pub_test.publish(odom);
            if (this->gps_init)
            {
//                ROS_INFO("Get GPS coord in UTM frame x = %0.2f , y = %0.2f, z = %0.2f", x - initx, y - inity, z - initz);
//                ROS_INFO("Get GPS coord in MAP frame x = %0.2f , y = %0.2f, z = %0.2f", this->gps_meas.x, this->gps_meas.y, this->gps_meas.z);
            }
        }

        auto gps_distance = [](GPSMeas meas1, GPSMeas meas2) {
            return sqrt((meas1.x - meas2.x) * (meas1.x - meas2.x) +
                        (meas1.y - meas2.y) * (meas1.y - meas2.y) +
                        (meas1.z - meas2.z) * (meas1.z - meas2.z));
        };

        if (!this->gps_init)
        {
            if (gps_distance(this->last_gps_meas, this->gps_meas) < 0.5)
            {
                ROS_WARN("Vehicle do not move, GPS can't initialize");
            }
            else
            {
                this->v_gps_init.push_back(this->gps_meas);
            }

            if (this->v_gps_init.size() > 50 && gps_distance(this->v_gps_meas.front(), this->v_gps_meas.back()) > 10)
            {

                std::unique_lock<std::mutex> lock(this->gps_mutex);
                auto gps_pos = this->v_gps_init;
                auto map_pos = this->v_gps_state;
                lock.unlock();

                // 根据gps_pos的时间戳在body_pos中查找匹配点 并保存到v_match中
                std::vector<GPSMeas> v_match;
                std::unordered_set<int> matched_id;
                std::vector<int> matched_id_order;
                for (int i = 0; i < gps_pos.size(); i++)
                {
                    double time_diff = 1e5;
                    int id = -1;
                    for (int j = 0; j < map_pos.size(); j++)
                    {
                        double diff = abs(gps_pos[i].time - map_pos[j].time);
                        if (diff < time_diff && matched_id.find(j) == matched_id.end())
                        {
                            time_diff = diff;
                            id = j;
                        }
                    }
                    v_match.push_back(map_pos[id]);
                    matched_id.insert(id);
                    matched_id_order.push_back(id);
                }

                // 过滤掉无效点 计算有效中心点
                std::vector<GPSMeas> match_gps;
                std::vector<GPSMeas> match_map;

                Eigen::Vector3d gps_center = {0, 0, 0};
                Eigen::Vector3d map_center = {0, 0, 0};

                for (int i = 0; i < gps_pos.size(); i++)
                {
                    if (matched_id_order[i] != -1)
                    {
                        match_gps.push_back(gps_pos[i]);
                        match_map.push_back(v_match[i]);
                        gps_center += Eigen::Vector3d(gps_pos[i].x, gps_pos[i].y, gps_pos[i].z);
                        map_center += Eigen::Vector3d(v_match[i].x, v_match[i].y, v_match[i].z);
                    }
                }

                gps_center = gps_center / match_gps.size();
                map_center = map_center / match_map.size();

                Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

                for (int i = 0; i < match_gps.size(); i++)
                {
                    Eigen::Vector3d gps_point = {match_gps[i].x, match_gps[i].y, match_gps[i].z};
                    gps_point -= gps_center;
                    Eigen::Vector3d map_point = {match_map[i].x, match_map[i].y, match_map[i].z};
                    map_point -= map_center;

                    W += map_point * gps_point.transpose();
                }

                Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d V = svd.matrixV();
                Eigen::Matrix3d U = svd.matrixU();
                Eigen::Matrix3d E;
                E << 1,0,0,0,1,0,0,0,(U * (V.transpose())).determinant();
                this->R_M_G = U * E * V.transpose();
                this->t_M_G = map_center - this->R_M_G * gps_center;
                this->gps_init = true;


                ROS_INFO("GPS init finish");
                ROS_INFO("R_M_G = \n %0.2f, %0.2f, %0.2f \n %0.2f, %0.2f, %0.2f \n %0.2f, %0.2f, %0.2f",
                         this->R_M_G(0, 0), this->R_M_G(0, 1), this->R_M_G(0, 2),
                         this->R_M_G(1, 0), this->R_M_G(1, 1), this->R_M_G(1, 2),
                         this->R_M_G(2, 0), this->R_M_G(2, 1), this->R_M_G(2, 2));
                ROS_INFO("t_M_G = \n [%0.2f, %0.2f, %0.2f]", this->t_M_G.x(), this->t_M_G.y(), this->t_M_G.z());
            }
        }
        else
        {
            std::lock_guard<std::mutex> lock(this->val_gps_mutex);
            this->v_val_gps.push_back(this->gps_meas);
        }

    }
    else
        return;

}

void dlio::OdomNode::callbackGPSWithoutAlign(const sensor_msgs::NavSatFixConstPtr &gps)
{
    static bool init_origin = false;

    if (this->dlio_initialized)
    {
        Eigen::Vector3d lla = {gps->latitude, gps->longitude, gps->altitude};

        if (!init_origin)
        {
            this->geo_converter.Reset(lla[0], lla[1], lla[2]);
            init_origin = true;
            ROS_INFO("GPS origin init !");
        }

        Eigen::Vector3d ENU = {0, 0, 0};
        this->geo_converter.Forward(lla[0], lla[1], lla[2], ENU[0], ENU[1], ENU[2]);

        this->gps_meas.x = ENU.x();
        this->gps_meas.y = ENU.y();
        this->gps_meas.z = ENU.z();
        this->gps_meas.cov = Eigen::Vector3d(gps->position_covariance[0],
                                             gps->position_covariance[4],
                                             gps->position_covariance[8]).asDiagonal();
        this->gps_meas.time = gps->header.stamp.toSec();

        if (this->gps_meas.cov.trace() < 4.0f && matchGPSWithKf(this->gps_meas))
        {
            std::lock_guard<std::mutex> lock_gps(this->gps_buffer_mutex);
            this->gps_buffer.push_back(this->gps_meas);
        }
    }
}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc) {
  this->count++;
//  std::cout << "==========================================" << std::endl;
//  std::cout << "This is No." << this->count << " frame" << std::endl;
  static double total_time = 0;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();
  this->kf_update = false;
  double then = ros::Time::now().toSec();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = pc->header.stamp.toSec();
  }

  // 检查初始化
  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // 转换点云格式，对点云进行初步的裁剪
  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // 点云预处理 去畸变 降采样
  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    ROS_FATAL("Low number of points in the cloud!");
    return;
  }

  // 计算sparsity
  // Compute Metrics
  this->metrics_thread = std::thread( &dlio::OdomNode::computeMetrics, this );
  this->metrics_thread.detach();

  // 设置自适应参数
  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // 输入该帧点云
  // Set new frame as input source
  this->setInputSource();

  // 将第一帧作为初始关键帧
  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    // 异步执行
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // scan to map求解位姿 并通过几何观测器计算IMU状态
  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();
  // 更新关键帧
  // Update current keyframe poses and map
//  this->updateKeyframes();

  this->saveKeyframeAndUpdateFactor();


  // 构建更新submap
  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // 更新轨迹
  // Update trajectory
  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );
  // 更新时间戳
  // Update time stamps
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // 在ROS内发布出去
  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->publish_thread = std::thread( &dlio::OdomNode::publishToROS, this, published_cloud, this->T_corr );
  this->publish_thread.detach();

  // Update some statistics
  this->comp_times.push_back(ros::Time::now().toSec() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements and publish custom DLIO message
//  this->debug_thread = std::thread( &dlio::OdomNode::debug, this );
//  this->debug_thread.detach();

  this->geo.first_opt_done = true;

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  total_time += time;
//  std::cout << "average time = " << total_time / count * 1000 << " ms" << std::endl;


}

void dlio::OdomNode::callbackImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {
  // 第一帧标志位
  this->first_imu_received = true;
  // 将原始的IMU数据转换到base_link下
  sensor_msgs::Imu::Ptr imu = this->transformImu( imu_raw );
  // 获取IMU的时间戳
  this->imu_stamp = imu->header.stamp;

  // 分别获取角速度和加速度
  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  // 保留第一帧IMU的时间戳
  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu->header.stamp.toSec();
  }

  // IMU静态初始化
  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {
    // 采样数量
    static int num_samples = 0;
    // 角速度和加速度
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;
    // 默认的IMU初始化时间为3秒 时间戳与第一帧IMU时间戳相差3秒以内的都要进行初始化
    if ((imu->header.stamp.toSec() - this->first_imu_stamp) < this->imu_calib_time_) {
      // IMU采样数量+1
      num_samples++;
      // 累加加速度和加速度测量值
      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;
      // 计算角速度和加速度的平均值
      gyro_avg /= num_samples;
      accel_avg /= num_samples;
      // 重力加速度
      Eigen::Vector3f grav_vec (0., 0., this->gravity_);
      // 对重力进行对齐
      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        // 取加速度平均值与ba差的方向作为重力方向 重力大小由参数给出
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        // 计算得到的重力方向与标准z轴的偏转 作为初始的姿态
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));
        // 初始化T的旋转和lidarPose的旋转
        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;
        // 计算欧拉角
        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        // 初始化ba
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {
        // 角速度均值用来初始化bg
        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }
      // IMU初始化完成
      this->imu_calibrated = true;

    }

  } else {
    // IMU初始化已经完成
    double dt = imu->header.stamp.toSec() - this->prev_imu_stamp;
    // 保留本帧IMU与上一帧的时间差
    this->imu_rates.push_back( 1./dt );
    if (dt == 0) { return; }

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu->header.stamp.toSec();
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;
    // 加入IMU缓存
    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      // IMU积分
      this->propagateState();
    }

  }

}

void dlio::OdomNode::getNextPose() {
  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    // 输入submap点云作为target
//    this->gicp.registerInputTarget(this->submap_cloud);
    this->gicp.setInputTarget(this->submap_cloud);
    // Set submap kdtree
    // 设置submao kdtree为target kdtree
//    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  // 进行对齐
  pcl::PointCloud<PointType>::Ptr aligned (boost::make_shared<pcl::PointCloud<PointType>>());
  this->gicp.align(*aligned);
  this->icpScore = float(this->gicp.getFitnessScore());

  // Get final transformation in global frame
  // 得到先验状态修正值
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation


    // 对先验状态进行修正
  this->T = this->T_corr * this->T_prior;

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();



  // Geometric observer update
  this->updateState();

}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {
  // imu_buffer中 front为最新的消息 back为最老的消息 这里等待 使得buffer中最新的消息时间戳大于end_time
  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  // 将last_imu_it移动到end_time处
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  // 将imu_it移动到start_time处
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }
  // IMU测量数据不足的情况
  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }
  // 使用反向迭代器的原因是在imu_buffer中时间顺序是从大到小的 start_time在后 end_time在前
  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  // 获得start_time和end_time对应的 IMU测量数据段 分别保存在begin_imu_it和end_imu_it
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // 两帧IMU之间的时间差
  // Time between first two IMU samples
  double dt = f2.dt;
  // begin_imu与start_time之间的时间差 因为肯定不是完全相等的 还存在一点差异
  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // 前两次IMU采样中的角加速度
  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // 这里的时间关系是
  // ------|-----------|----------|----------
  //      IMU_i    Start_time   IMU_i+1
  // 获得前半段时间内的平均角速度
  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);
  // 将q_init转换到第一帧IMU时的状态 即已知start_time的状态 往前回溯 得到IMU_i的状态 从这里开始积分
  // 角速度为body坐标系下 并不是世界坐标系下 因此采用右乘    q_init = q_init x [1  0.5*\omega*idt]
  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
        q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
        q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
        q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
        q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  // 前两帧IMU之间的平均角速度
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;
  // 得到第二帧IMU时的旋转
  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // 将第一帧和第二帧的加速度转换到世界坐标系
  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // 计算加加速度
  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // 将速度和位置都回溯到第一帧IMU的状态
  // 认为加加速度是常量 有a = kt + c
  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;
  // 获取初始状态
  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  // 加速度为世界坐标系下
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;
    // 相邻两帧IMU之间的时间差
    // Time between IMU samples
    double dt = f.dt;
    // 计算角加速度
    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;
    // 计算平均角速度
    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // 旋转姿态传播
    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    // 根据刚刚传播的状态 将后IMU加速度转换到世界坐标系下
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    // 计算加加速度
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // 对给定时间戳状态进行插值求解
    // -------------|-----------+---------+-----------+-------------|-------------------------
    //             f0           p1        p2          p3             f
    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      // 计算时间差
      double idt = *stamp_it - f0.stamp;
      // f0 与 待插值时间点内的平均角速度
      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;
      // 旋转传播
      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();
      // 位置传播
      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;
      // 写入齐次变换
      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // p和v向下一帧IMU进行传播 q在上面已经算过了
    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

    prev_imu_it = imu_it;

  }

  return imu_se3;

}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;
  // 获取当前的姿态
  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;
  // 将加速度转换到世界坐标系
  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // 世界坐标系下的加速度传播 得到pos "p = vt + 1/2 a t^2"
  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);
  // 世界坐标系下的速度传播 v = v + at
  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // 姿态传播
  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

  if (!this->gps_init)
  {
      std::lock_guard<std::mutex> lock(this->gps_mutex);
      this->v_gps_state.push_back(GPSMeas(this->state.p.x(), this->state.p.y(), this->state.p.z(), this->imu_stamp.toSec()));
  }

  this->currentFusionState = this->state;
  this->currentFusionT = Eigen::Isometry3f::Identity();
  this->currentFusionT.translate(this->currentFusionState.p);
  this->currentFusionT.rotate(this->currentFusionState.q);

}

sensor_msgs::Imu::Ptr dlio::OdomNode::transformImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {
  sensor_msgs::Imu::Ptr imu (new sensor_msgs::Imu);
  // 复制消息头
  // Copy header
  imu->header = imu_raw->header;
  // 获取第一帧IMU的时间戳
  static double prev_stamp = imu->header.stamp.toSec();
  // 计算当前帧IMU与上一帧IMU之间的时间差
  double dt = imu->header.stamp.toSec() - prev_stamp;
  prev_stamp = imu->header.stamp.toSec();
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  // 获取角速度
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);
  // 获取base_link下的角速度
  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;
  // 用base_link下的角速度赋值
  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  // 获取加速度
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);
  // 考虑旋转
  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;
  // 考虑平移
  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;
  // 赋值
  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness() {

  // 遍历计算每个点的range
  // compute range of points
  std::vector<float> ds;

  for (int i = 0; i <= this->original_scan->points.size(); i++) {
    float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                        pow(this->original_scan->points[i].y, 2));
    ds.push_back(d);
  }

  // 这里用的是没有去畸变的点云 取距离的中位数
  // 找到range中位数 并将其置于ds索引为ds.size()/2的位置 其之前都比该值小 之后都比该值大
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = ds[ds.size()/2];
  static float median_prev = median_curr;
  // 论文中的公式 用来描述sparsity
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  // push
  this->metrics.spaciousness.push_back( median_lpf );

}

void dlio::OdomNode::computeDensity() {
  // 这里的方法与sparsity类似
  // TODO source_density_的含义不太清楚
  // 这个密度指的是当前帧每个点到其最近五个点的平均距离
  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );

}

// 计算关键帧所在位置点构成点云的凸包
void dlio::OdomNode::computeConvexHull() {

  // 需要至少有四个关键帧
  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // cloud存储的是每个关键帧pose的位置xyz
  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // 输入点云
  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // 重建凸包
  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->convex_hull.reconstruct(*convex_points);

  // 得到凸包点的索引
  pcl::PointIndices::Ptr convex_hull_point_idx = pcl::PointIndices::Ptr (boost::make_shared<pcl::PointIndices>());
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::computeConcaveHull() {
  // 至少有5个关键帧
  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = pcl::PointIndices::Ptr (boost::make_shared<pcl::PointIndices>());
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode:: updateKeyframes() {
  // calculate difference in pose and rotation to all poses in trajectory
  static int frame_num = 0;
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;
  // 遍历所有关键帧
  for (const auto& k : this->keyframes) {
    // 计算当前帧位置与关键帧位置的差值
    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                          pow(this->state.p[1] - k.first.first[1], 2) +
                          pow(this->state.p[2] - k.first.first[2], 2) );

    // 如果与该关键帧的距离小于阈值的1.5倍
    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    // 筛选出最近的关键帧
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;

  }
  // 获取最近关键帧的位姿
  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;
  // 这不就是closest_d么？
  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  // 计算旋转的差异
  Eigen::Quaternionf dq;
  // 四元数的点积可以用来描述两个旋转的相似程度 点积绝对值越大 则两个四元数代表的角位移越相似 而点积的正负则代表了二者的方向 当小于0时 方向相反
  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }
  // 解算出旋转角度
  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  // 当前帧与最近关键帧的距离偏角大于阈值
  // spaciousness keyframing
  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  // 距离虽然不够 但是旋转角度够 并且与当前帧距离小于1.5倍阈值的关键帧 只有一个
  // rotational exploration keyframing
  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  // 距离不够 或小于0.5m的
  // check for nearby keyframes
  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  } else if (abs(dd) <= 0.5) {
    newKeyframe = false;
  }

  // 添加关键帧
  if (newKeyframe) {
    if (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
    {
        // update keyframe vector
        std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
        pcl::copyPointCloud(*this->current_scan, *this->current_scan_w);
        this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan_w));
        this->keyframe_timestamps.push_back(this->scan_header_stamp);
        this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
        this->keyframe_transformations.push_back(this->T_corr);
        lock.unlock();

        std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_his_lidar(this->history_kf_lidar_mutex);
        pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*this->current_scan_lidar, *temp);
//        std::cout << " updateKeyframes: " << this->current_scan_lidar->size() << std::endl;
        this->history_kf_lidar.push_back(temp);
        lock_his_lidar.unlock();

        // 检查是否存在闭环候选 当存在闭环候选 并且该帧为关键帧时，才进行闭环
        std::unique_lock<decltype(this->loop_info_mutex)> lock_loop(this->loop_info_mutex);
        if (this->curr_loop_info.loop_candidate)
        {
            this->curr_loop_info.loop_candidate = false;
            this->curr_loop_info.loop = true;
            std::shared_ptr<nano_gicp::CovarianceList> normals_temp(std::make_shared<nano_gicp::CovarianceList>());
            this->curr_loop_info.current_kf = this->keyframes.back();
            std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
            this->curr_loop_info.current_id = this->keyframes.size() - 1;
            for (int i = 0; i < this->curr_loop_info.candidate_key.size(); i++)
            {
                normals_temp->insert( std::end(*normals_temp),
                                      std::begin(*(this->keyframe_normals[this->curr_loop_info.candidate_key[i]])),
                                      std::end(*(this->keyframe_normals[this->curr_loop_info.candidate_key[i]])) );
                this->curr_loop_info.candidate_frame_normals.push_back(normals_temp);
                normals_temp->clear();
            }


            visualization_msgs::Marker loop_marker;
            loop_marker.ns = "line_extraction";
            loop_marker.header.stamp = this->imu_stamp;
            loop_marker.header.frame_id = this->odom_frame;
            loop_marker.id = 0;
            loop_marker.type = visualization_msgs::Marker::LINE_LIST;
            loop_marker.scale.x = 0.1;
            loop_marker.color.r = 1.0;
            loop_marker.color.g = 0.0;
            loop_marker.color.b = 0.0;
            loop_marker.color.a = 1.0;
            loop_marker.action = visualization_msgs::Marker::ADD;
            loop_marker.pose.orientation.w = 1.0;

            geometry_msgs::Point point1;
            point1.x = this->curr_loop_info.current_kf.first.first[0];
            point1.y = this->curr_loop_info.current_kf.first.first[1];
            point1.z = this->curr_loop_info.current_kf.first.first[2];

            geometry_msgs::Point point2;
            point2.x = this->curr_loop_info.candidate_frame[0].first.first[0];
            point2.y = this->curr_loop_info.candidate_frame[0].first.first[1];
            point2.z = this->curr_loop_info.candidate_frame[0].first.first[2];

            loop_marker.points.push_back(point1);
            loop_marker.points.push_back(point2);

            std::cout << "**********push loop**********" << std::endl;
            this->loop_constraint_pub.publish(loop_marker);

            lock.unlock();

        }
        lock_loop.unlock();


        std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
        pcl::copyPointCloud(*this->current_scan_w, *this->tempKeyframe.pCloud);
        this->tempKeyframe.time = this->scan_stamp;
        this->tempKeyframe.rot = this->state.q;
        this->tempKeyframe.pos = this->state.p;
        this->KeyframesInfo.push_back(this->tempKeyframe);


        lock_temp.unlock();

        this->kf_update = true;
    }




  }
  frame_num++;

}

void dlio::OdomNode::setAdaptiveParams() {
  // 获取当前帧点云的spacious
  // Spaciousness
  float sp = this->metrics.spaciousness.back();
  // 进行限制 判断属于小环境还是大环境
  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  // 关键帧距离阈值？
  this->keyframe_thresh_dist_ = sp;

  // Density
  // 根据点云密度设置icp最大临近距离 用以判断是否是一对合理的匹配
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

// 筛选出与当前帧最近的k个关键帧
void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}
/**
 * @brief 构建submap
 * @param vehicle_state
 * @note 该函数较为重要 对于第一帧而言 由于num_processed_keyframes=1 所以该函数只完成了将第一帧的点云加入submap 构建gicp target
 *
 */
void dlio::OdomNode::buildSubmap(State vehicle_state) {
  static int count = 1;
  static double total_time = 0;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  // 存储关键帧id的vector
  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();
  // 这里计算的距离是什么含义可能得看imu后才能知道
  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  // 遍历刚刚遍历过的关键帧
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    // 当前帧状态与遍历关键帧位置差
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // 在上述关键帧中筛选出与当前帧位置最近的submap_knn_个关键帧 submap_kf_idx_curr
  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // 计算关键帧位置点云凸包 keyframe_convex， num_processed_keyframes至少大于等于4时执行
  // get convex hull indices
  this->computeConvexHull();

  // 读取刚刚得到的凸包  将这些凸包关键帧与当前关键帧的距离添加到convex_ds中
  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // 在从上述凸包关键帧中找出与当前帧位置最近的submap_kcv_个关键帧 submap_kf_idx_curr
  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // 计算关键帧位置点云凸包 keyframe_concave， num_processed_keyframes至少大于等于5时执行
  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }
  //  在从上述凸包关键帧中找出与当前帧位置最近的submap_kcc_个关键帧 submap_kf_idx_curr
  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);
  // 对关键帧索引vector进行升序排序 并将重复的元素移动至vector末尾 随后将其删除
  // concatenate all submap clouds and normals
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // 检查当前搜索得到的关键帧vector是否与前一帧相同
  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){
    std::cout << "submap_kf.size() = " << this->submap_kf_idx_curr.size() << std::endl;
    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();
    // 如果不同 更新submap及其协方差
    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ (boost::make_shared<pcl::PointCloud<PointType>>());
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();
    // 设置gicp的目标点云为刚刚搜索的submap 构建submap kdtree
    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  total_time += time;
//  std::cout << "average time = " << total_time / count * 1000 << " ms" << std::endl;
  count++;
}

void dlio::OdomNode::buildSubmapViaJaccard(dlio::OdomNode::State vehicle_state)
{
    static int count = 1;
    static double total_time = 0;
    static int last_loop_id = -1;

    // 当前构建submap的关键帧索引 开始时进行清除
    this->submap_kf_idx_curr.clear();
    std::vector<pcl::PointCloud<PointType>::ConstPtr> kf_pc;
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    std::vector<float> ds;
    std::vector<int> keyframe_nn;
    // 遍历关键帧 先筛选出距离比较近的关键帧
    for (int i = 0; i < this->num_processed_keyframes; i++)
    {
        kf_pc.push_back(this->keyframes[i].second);
        // 当前状态与关键帧状态之间的距离
        float d = sqrt(pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                       pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                       pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2));
        ds.push_back(d);
        keyframe_nn.push_back(i);
    }
    lock.unlock();

    // 进行排序 获得距离前submap_knn的关键帧
    this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);
    // 与每个关键帧交集点的数量
    std::vector<int> intersection_nums;
    intersection_nums.reserve(this->submap_kf_idx_curr.size());
    // 与每个关键帧并集点的数量
    std::vector<int> union_nums;
    union_nums.reserve(this->submap_kf_idx_curr.size());
    // 与每个关键帧的交并比
    this->similarity.clear();

    std::vector<int> submap_kf_idx_curr_final;
    int id;
    float dis;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // 不够3帧的不进行筛选
    if (this->submap_kf_idx_curr.size() > 3)
    {
        // 遍历根据位置筛选后的关键帧 submao_kf_idx_curr里存的是对应关键帧的id
        for (int i = 0; i < this->submap_kf_idx_curr.size(); i++)
        {
            // 将候选关键帧点云加入八叉树
            pcl::octree::OctreePointCloudSearch<PointType>::Ptr octree(new pcl::octree::OctreePointCloudSearch<PointType>(0.25));
            octree->setInputCloud(kf_pc[this->submap_kf_idx_curr[i]]);
            octree->addPointsFromInputCloud();

            intersection_nums[i] = 0;
            // 遍历当前帧点云
            for (int j = 0; j < this->current_scan->size(); j++)
            {
                octree->approxNearestSearch(this->current_scan->points[j], id, dis);
                if (dis < 0.5)
                {
                    intersection_nums[i]++;
                }

            }
            union_nums[i] = this->current_scan->size() + kf_pc[this->submap_kf_idx_curr[i]]->size() - intersection_nums[i];
            this->similarity.push_back( float(intersection_nums[i]) / float(union_nums[i]) );
//            std::cout << "similarity = " << similarity[i] << std::endl;
            if (this->similarity[i] > 0.1)
                submap_kf_idx_curr_final.push_back(this->submap_kf_idx_curr[i]);

            // loop
            if ((this->num_processed_keyframes - this->submap_kf_idx_curr[i]) > 30 && (this->num_processed_keyframes - last_loop_id) > 10)
            {
                lock.lock();
                float d = sqrt(pow(vehicle_state.p[0] - this->keyframes[this->submap_kf_idx_curr[i]].first.first[0], 2) +
                               pow(vehicle_state.p[1] - this->keyframes[this->submap_kf_idx_curr[i]].first.first[1], 2));
                lock.unlock();
                if (d < 10)
                {
                    std::cout << "**************Build map find loop ************" << std::endl;
                    std::unique_lock<decltype(this->loop_info_mutex)> lock_loop(this->loop_info_mutex);
                    this->curr_loop_info.loop_candidate = true;
                    this->curr_loop_info.candidate_key.push_back(this->submap_kf_idx_curr[i]);
                    this->curr_loop_info.candidate_sim.push_back(this->similarity[i]);
                    this->curr_loop_info.candidate_dis.push_back(d);
                    this->curr_loop_info.candidate_frame.push_back(this->keyframes[this->submap_kf_idx_curr[i]]);
                    lock_loop.unlock();
                    last_loop_id = this->num_processed_keyframes;
                }
            }


        }
        // 如果筛选后的关键帧大于3帧 那么就用筛选后的子地图 如果不够3帧 则用相似度最大的三帧
        if (submap_kf_idx_curr_final.size() > 3)
        {
            this->submap_kf_idx_curr = submap_kf_idx_curr_final;
            // 更新tempKF
            std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
//            pcl::copyPointCloud(*this->current_scan, this->tempKeyframe.pCloud);
            this->tempKeyframe.submap_kf_idx = this->submap_kf_idx_curr;
            this->tempKeyframe.vSim = this->similarity;
            lock_temp.unlock();
        }
        else
        {
            std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end(),
                      [&](int a, int b){
                int index1 = std::distance(this->submap_kf_idx_curr.begin(), std::find(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end(), a));
                int index2 = std::distance(this->submap_kf_idx_curr.begin(), std::find(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end(), b));
                return this->similarity[index1] > this->similarity[index2];
            });
            this->submap_kf_idx_curr.resize(3);

            std::sort(this->similarity.begin(), this->similarity.end(), [](float a, float b) {return a > b;});
            this->similarity.resize(3);

            // 更新tempKF
            std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
            this->tempKeyframe.submap_kf_idx = this->submap_kf_idx_curr;
            this->tempKeyframe.vSim = this->similarity;
            lock_temp.unlock();

        }
//        std::cout << "jaccard size = " << submap_kf_idx_curr_final.size() << std::endl;
    }
    else
    {
        std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
        this->tempKeyframe.submap_kf_idx = this->submap_kf_idx_curr;
        this->tempKeyframe.vSim = {};
        lock_temp.unlock();
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    total_time += time;
//    std::cout << "average time = " << total_time * 1000 / count << " ms" << std::endl;
    count++;

    std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
    auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
    this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

    std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
    std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());



    // 检查submap是否需要更新
    if (this->submap_kf_idx_curr != this->submap_kf_idx_prev)
    {
//        std::cout << "submap_kf.size() = " << this->submap_kf_idx_curr.size() << std::endl;
        this->submap_hasChanged = true;
        this->pauseSubmapBuildIfNeeded();
        pcl::PointCloud<PointType>::Ptr submap_cloud_(new pcl::PointCloud<PointType>());
        std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());;

        std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_his(this->history_kf_lidar_mutex);
        auto his = this->history_kf_lidar;
        lock_his.unlock();

        for (auto k : this->submap_kf_idx_curr)
        {
            pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);

            lock.lock();

            Eigen::Isometry3f temp_T = Eigen::Isometry3f::Identity();
            temp_T.translate(this->keyframes[k].first.first);
            temp_T.rotate(this->keyframes[k].first.second);
            pcl::transformPointCloud(*his[k], *temp_cloud, temp_T.matrix());
            *submap_cloud_ += *temp_cloud;
//            *submap_cloud_ += *this->keyframes[k].second;
            lock.unlock();

            this->gicp_tool.setInputSource(temp_cloud);
            this->gicp_tool.calculateSourceCovariances();

            submap_normals_->insert(std::end(*submap_normals_), std::begin(*(this->gicp_tool.getSourceCovariances())),
                                    std::end(*(this->gicp_tool.getSourceCovariances())));
        }
        this->submap_cloud = submap_cloud_;
        this->submap_normals = submap_normals_;

        this->pauseSubmapBuildIfNeeded();

        this->gicp_temp.setInputTarget(this->submap_cloud);
        this->submap_kdtree = this->gicp_temp.target_kdtree_;
        this->submap_kf_idx_prev = this->submap_kf_idx_curr;
    }
}

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {
  // transform the new keyframe(s) and associated covariance list(s)
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  // 遍历关键帧 num_processed_keyframes初始化为0
  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    // 关键帧的pair<<V3F, QF>, pcl::Ptr>
    // 拿到关键帧对应的点云
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    // 拿到关键帧的协方差
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    // 拿到关键帧的齐次变换
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe (boost::make_shared<pcl::PointCloud<PointType>>());
    // 转换到世界坐标系下
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);
    // 对协方差进行修正
    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;
    // 将关键帧在ROS内发布出去
//    this->publish_keyframe_thread = std::thread( &dlio::OdomNode::publishKeyframe, this, this->keyframes[i], this->keyframe_timestamps[i] );
//    this->publish_keyframe_thread.detach();
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();
  // 构建submap
  if (this->useJaccard)
    this->buildSubmapViaJaccard(vehicle_state);
  else
    this->buildSubmap(vehicle_state);


    // 可视化
    visualization_msgs::Marker kf_connect_marker;
    kf_connect_marker.ns = "line_extraction";
    kf_connect_marker.header.stamp = this->imu_stamp;
    kf_connect_marker.header.frame_id = this->odom_frame;
    kf_connect_marker.id = 0;
    kf_connect_marker.type = visualization_msgs::Marker::LINE_LIST;
    kf_connect_marker.scale.x = 0.1;
    kf_connect_marker.color.r = 1.0;
    kf_connect_marker.color.g = 1.0;
    kf_connect_marker.color.b = 1.0;
    kf_connect_marker.color.a = 1.0;
    kf_connect_marker.action = visualization_msgs::Marker::ADD;
    kf_connect_marker.pose.orientation.w = 1.0;
    lock.lock();
    for (auto i : this->submap_kf_idx_curr)
    {
        geometry_msgs::Point point1;
        point1.x = vehicle_state.p[0];
        point1.y = vehicle_state.p[1];
        point1.z = vehicle_state.p[2];
        kf_connect_marker.points.push_back(point1);

        geometry_msgs::Point point2;
        point2.x = this->keyframes[i].first.first.x();
        point2.y = this->keyframes[i].first.first.y();
        point2.z = this->keyframes[i].first.first.z();
        kf_connect_marker.points.push_back(point2);
    }
    lock.unlock();
    if (kf_connect_marker.points.size() > 0)
        this->kf_connect_pub.publish(kf_connect_marker);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void dlio::OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::mapping()
{
    // 全局地图
    pcl::PointCloud<PointType>::Ptr cloud_map(new pcl::PointCloud<PointType>);

    pcl::VoxelGrid<PointType> voxelGrid;
    if (this->global_dense)
        voxelGrid.setLeafSize(0.1, 0.1, 0.1);
    else
        voxelGrid.setLeafSize(0.25, 0.25, 0.25);
    // GTSAM初始化
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(params);

    // 当前处理关键帧的索引
    int processed_keyframes_num = 0;
    // 历史关键帧信息
    int history_keyframes_num = 0;
    std::vector<KeyframeInfo> history_keyframes;
    bool find_loop = false;

    while (this->nh.ok())
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // 获取历史关键帧信息
        std::unique_lock<decltype(this->tempKeyframe_mutex)> lock(this->tempKeyframe_mutex);
        history_keyframes_num = this->KeyframesInfo.size();
        history_keyframes = this->KeyframesInfo;
        lock.unlock();

        // 没有新的关键帧插入 跳过
        std::unique_lock<decltype(this->loop_factor_mutex)> lock_factor(this->loop_factor_mutex);
        if (processed_keyframes_num >= history_keyframes_num && !this->curr_factor_info.loop)
        {
            lock_factor.unlock();
            continue;
        }
        else
            lock_factor.unlock();


        // 初始第一帧
        if (processed_keyframes_num == 0)
        {
            gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
                    (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
            gtSAMgraph.addPrior(0, state2Pose3(history_keyframes[0].rot, history_keyframes[0].pos), priorNoise);
            initialEstimate.insert(0, state2Pose3(history_keyframes[0].rot, history_keyframes[0].pos));
            processed_keyframes_num++;

        }
        else if (processed_keyframes_num != history_keyframes_num)
        {

            // 当前帧匹配的子地图关键帧索引
            std::vector<int> curr_submap_id = history_keyframes[processed_keyframes_num].submap_kf_idx;
            // 当前帧匹配的子地图关键帧相似度
            std::vector<float> curr_sim = history_keyframes[processed_keyframes_num].vSim;
            // 当前帧的位姿
            gtsam::Pose3 poseTo = state2Pose3(history_keyframes[processed_keyframes_num].rot,
                                              history_keyframes[processed_keyframes_num].pos);
            // 遍历与当前帧匹配的子地图关键帧
            for (int i = 0; i < curr_submap_id.size(); i++)
            {
                // 子地图关键帧的位姿
                gtsam::Pose3 poseFrom = state2Pose3(history_keyframes[curr_submap_id[i]].rot,
                                                    history_keyframes[curr_submap_id[i]].pos);
                // 噪声权重
                double weight = 1.0;

                if (curr_sim.size() > 0)
                {
                    // 对相似度进行归一化
                    auto max_sim = std::max_element(curr_sim.begin(), curr_sim.end());
                    for (auto &i: curr_sim)
                        i = i / *max_sim;
                    double sim = curr_sim[i] >= 1 ? 0.99 : curr_sim[i] < 0 ? 0 : curr_sim[i];

                    // 相似度越大 噪声权重越小
                    weight = 1 - sim;
//                    std::cout << "weight = " << weight << std::endl;
                }
                // 添加相邻/不相邻里程计因子
                gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
                        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished() * weight);
                gtSAMgraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(curr_submap_id[i],
                                                                              processed_keyframes_num,
                                                                              poseFrom.between(poseTo), odometryNoise);
            }
            initialEstimate.insert(processed_keyframes_num, poseTo);
            processed_keyframes_num++;

        }

        // loop
        lock_factor.lock();
        loop_factor_info loop_factor;
        if (this->curr_factor_info.loop)
        {
            find_loop = true;
            loop_factor = curr_factor_info;
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
                    (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());

            gtsam::Pose3 poseFrom = state2Pose3(Eigen::Quaternionf(loop_factor.T_target.rotation()),
                                                loop_factor.T_target.translation());

            gtsam::Pose3 poseTo = state2Pose3(Eigen::Quaternionf(loop_factor.T_current.rotation()),
                                              loop_factor.T_current.translation());

            gtSAMgraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(loop_factor.factor_id.second,
                                                                          loop_factor.factor_id.first,
                                                                          poseFrom.between(poseTo), odometryNoise);
            std:: cout << "***********Add loop factor***********" << std::endl;
            this->curr_factor_info.loop = false;
            lock_factor.unlock();
        }
        else
            lock_factor.unlock();

//        gtSAMgraph.print("GTSAM Graph:\n");

        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (find_loop)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }


        gtSAMgraph.resize(0);
        initialEstimate.clear();
        iSAMCurrentEstimate = isam->calculateEstimate();

//        std::cout << "========================================" << std::endl;
//        std::cout << "Before rot = " << history_keyframes[history_keyframes.size() - 1].rot.w() << " "
//                  << history_keyframes[history_keyframes.size() - 1].rot.x() << " "
//                  << history_keyframes[history_keyframes.size() - 1].rot.y() << " "
//                  << history_keyframes[history_keyframes.size() - 1].rot.z() << " "
//                  << "Before pos = "
//                  << history_keyframes[history_keyframes.size() - 1].pos.x() << " "
//                  << history_keyframes[history_keyframes.size() - 1].pos.y() << " "
//                  << history_keyframes[history_keyframes.size() - 1].pos.z() << " " << std::endl;
//
//
//        std::cout << "After rot = "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().w() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().x() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().y() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().z() << " "
//                  << "After pos = "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().x() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().y() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().z() << " "
//                  << std::endl;
//        std::cout << "========================================" << std::endl;




        // 发布地图
        if (find_loop)
        {
            cloud_map->points.clear();
            std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
            for (int i = 0; i < iSAMCurrentEstimate.size() ; i++)
            {
                pcl::PointCloud<PointType>::Ptr curr_kf(new pcl::PointCloud<PointType>);
                Eigen::Isometry3f curr_T = Eigen::Isometry3f::Identity();

                Eigen::Quaternionf q = {
                        iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().w(),
                        iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().x(),
                        iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().y(),
                        iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().z()};
                curr_T.translate(iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().cast<float>());
                curr_T.rotate(q);


                std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_kf_his(this->history_kf_lidar_mutex);
                pcl::transformPointCloud(*this->history_kf_lidar[i], *curr_kf, curr_T.matrix());
                lock_kf_his.unlock();

                *cloud_map += *curr_kf;
                voxelGrid.setInputCloud(cloud_map);
                voxelGrid.filter(*cloud_map);

                // 更新历史关键帧
                this->keyframes[i].first.first = iSAMCurrentEstimate.at<gtsam::Pose3>(
                        i).translation().vector().cast<float>();
                this->keyframes[i].first.second.w() = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
                this->keyframes[i].first.second.x() = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
                this->keyframes[i].first.second.y() = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
                this->keyframes[i].first.second.z() = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().z();

            }
            lock_kf.unlock();

            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*cloud_map, map_msg);
            map_msg.header.stamp = ros::Time::now();
            map_msg.header.frame_id = this->odom_frame;

            this->global_map_pub.publish(map_msg);
            find_loop = false;
        }
        else
        {
            pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr curr_kf(new pcl::PointCloud<PointType>);
            Eigen::Isometry3f last_T = Eigen::Isometry3f::Identity();
            Eigen::Isometry3f curr_T = Eigen::Isometry3f::Identity();

            Eigen::Quaternionf q = {
                    iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().w(),
                    iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().x(),
                    iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().y(),
                    iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().z()};
            curr_T.translate(iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().cast<float>());
            curr_T.rotate(q);

            Eigen::Quaternionf last_q = {history_keyframes[history_keyframes.size() - 1].rot.w(),
                                         history_keyframes[history_keyframes.size() - 1].rot.x(),
                                         history_keyframes[history_keyframes.size() - 1].rot.y(),
                                         history_keyframes[history_keyframes.size() - 1].rot.z()};
            last_T.translate(history_keyframes[history_keyframes.size() - 1].pos);
            last_T.rotate(last_q);



            std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_kf_his(this->history_kf_lidar_mutex);
            pcl::transformPointCloud(*this->history_kf_lidar[iSAMCurrentEstimate.size() - 1], *curr_kf, last_T.matrix());
            lock_kf_his.unlock();


            voxelGrid.setInputCloud(curr_kf);
            voxelGrid.filter(*curr_kf);
            *cloud_map += *curr_kf;
            voxelGrid.setInputCloud(cloud_map);
            voxelGrid.filter(*cloud_map);


            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*cloud_map, map_msg);
            map_msg.header.stamp = this->scan_header_stamp;
            map_msg.header.frame_id = this->odom_frame;

            this->global_map_pub.publish(map_msg);

        }

        // 更新关键帧位姿
        this->global_pose.poses.clear();
        for (int i = 0; i < this->iSAMCurrentEstimate.size(); i++)
        {
            geometry_msgs::Pose p;
            p.position.x = iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[0];
            p.position.y = iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[1];
            p.position.z = iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[2];

            p.orientation.w = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
            p.orientation.x = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
            p.orientation.y = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
            p.orientation.z = iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().z();

            this->global_pose.poses.push_back(p);
            this->global_pose.header.stamp = ros::Time::now();
            this->global_pose.header.frame_id = this->odom_frame;
            this->global_pose_pub.publish(this->global_pose);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        std::cout << "Back end cost time = " << time*1000 << " ms" << std::endl;
    }

}

gtsam::Pose3 dlio::OdomNode::state2Pose3(Eigen::Quaternionf rot, Eigen::Vector3f pos)
{
    rot.normalize();
    return gtsam::Pose3(gtsam::Rot3(rot.cast<double>()), gtsam::Point3(pos.cast<double>()));
}

void dlio::OdomNode::performLoop()
{
    loop_info current_loop_info;

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_*2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    pcl::VoxelGrid<PointType> voxel_loop;
    voxel_loop.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);



    while(this->nh.ok())
    {
        std::unique_lock<decltype(this->loop_info_mutex)> lock_loop(this->loop_info_mutex);
        current_loop_info = this->curr_loop_info;
        lock_loop.unlock();

        if (current_loop_info.loop)
        {
            // 选取相似度最大的候选关键帧
            int max_id = std::max_element(current_loop_info.candidate_sim.begin(), current_loop_info.candidate_sim.end()) - current_loop_info.candidate_sim.begin();

            std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_his(this->history_kf_lidar_mutex);
            auto his_lidar = this->history_kf_lidar;
            lock_his.unlock();

            std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
            auto kfs = this->keyframes;
            lock_kf.unlock();

            std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_kf_info(this->tempKeyframe_mutex);
            KeyframeInfo current_loop_kf_info = this->KeyframesInfo[current_loop_info.candidate_key[max_id]];
            lock_kf_info.unlock();

            // 将当前帧点云转换到最新的世界坐标系下
            pcl::PointCloud<PointType>::Ptr current_cloud(new pcl::PointCloud<PointType>);
            Eigen::Isometry3f tempT = Eigen::Isometry3f::Identity();
            tempT.translate(kfs[current_loop_info.current_id].first.first);
            tempT.rotate(kfs[current_loop_info.current_id].first.second);
            pcl::transformPointCloud(*his_lidar[current_loop_info.current_id], *current_cloud, tempT.matrix());


            // 添加闭环候选帧的点云和normal
            pcl::PointCloud<PointType>::Ptr loop_candidate_map(new pcl::PointCloud<PointType>);

            pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
            tempT = Eigen::Isometry3f::Identity();
            tempT.translate(kfs[current_loop_info.candidate_key[max_id]].first.first);
            tempT.rotate(kfs[current_loop_info.candidate_key[max_id]].first.second);
            pcl::transformPointCloud(*his_lidar[current_loop_info.candidate_key[max_id]], *temp_cloud, tempT.matrix());

            *loop_candidate_map += *temp_cloud;

            // 添加闭环候选帧子地图的点云和normal

            for (int i = 0; i < current_loop_kf_info.submap_kf_idx.size(); i++)
            {
                tempT = Eigen::Isometry3f::Identity();
                tempT.translate(kfs[current_loop_kf_info.submap_kf_idx[i]].first.first);
                tempT.rotate(kfs[current_loop_kf_info.submap_kf_idx[i]].first.second);
                pcl::transformPointCloud(*his_lidar[current_loop_kf_info.submap_kf_idx[i]], *temp_cloud, tempT.matrix());
                *loop_candidate_map += *temp_cloud;
            }

            voxel_loop.setInputCloud(loop_candidate_map);
            voxel_loop.filter(*loop_candidate_map);

            icp.setInputSource(current_cloud);
            icp.setInputTarget(loop_candidate_map);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);


            if (icp.hasConverged() == false)
            {
                std::cout << "Perform loop find icp not converged" << std::endl;
                lock_loop.lock();
                this->curr_loop_info.reset();
                lock_loop.unlock();
                continue;
            }

            auto T_c = icp.getFinalTransformation();

            // 闭环优化前当前关键帧的位姿
            Eigen::Isometry3f T_before = Eigen::Isometry3f::Identity();
            T_before.translate(current_loop_info.current_kf.first.first);
            T_before.rotate(current_loop_info.current_kf.first.second);
            // 优化后的当前关键帧位姿
            auto T_after = T_c * T_before;
            float score = icp.getFitnessScore();

            std::cout << "=========================" << std::endl;
            std::cout << "After loop pos = " << T_after.translation() << std::endl;
//            std::cout << "After loop rot = " << T_after.rotation() << std::endl;
            std::cout << "=========================" << std::endl;


            // 闭环候选关键帧的位姿
            Eigen::Isometry3f T_candidate = Eigen::Isometry3f::Identity();
            T_candidate.translate(current_loop_info.candidate_frame[max_id].first.first);
            T_candidate.rotate(current_loop_info.candidate_frame[max_id].first.second);

            std::unique_lock<decltype(this->loop_factor_mutex)> lock_factor(this->loop_factor_mutex);
            this->curr_factor_info.loop = true;
            this->curr_factor_info.T_current = Eigen::Isometry3f::Identity();
            this->curr_factor_info.T_current.translate(T_after.translation());
            this->curr_factor_info.T_current.rotate(T_after.rotation());
            this->curr_factor_info.T_target = T_candidate;
            this->curr_factor_info.dis = current_loop_info.candidate_dis[max_id];
            this->curr_factor_info.factor_id = std::make_pair(current_loop_info.current_id, current_loop_info.candidate_key[max_id]);
            this->curr_factor_info.sim = current_loop_info.candidate_sim[max_id] * score * (1.0 / (this->curr_factor_info.factor_id.first - this->curr_factor_info.factor_id.second));
            lock_factor.unlock();

            std::cout << "*************Finished loop icp*************" << std::endl;

            lock_loop.lock();
            this->curr_loop_info.reset();
            lock_loop.unlock();


        }

    }

}

bool dlio::OdomNode::isKeyframe()
{
    static int frame_num = 0;
    float closest_d = std::numeric_limits<float>::infinity();
    int closest_idx = 0;
    int keyframes_idx = 0;

    int num_nearby = 0;
    for (const auto& k : this->keyframes)
    {
        // 计算当前帧位置与关键帧位置的差值
        // calculate distance between current pose and pose in keyframes
        float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                              pow(this->state.p[1] - k.first.first[1], 2) +
                              pow(this->state.p[2] - k.first.first[2], 2) );

        // 如果与该关键帧的距离小于阈值的1.5倍
        // count the number nearby current pose
        if (delta_d <= this->keyframe_thresh_dist_ * 1.5)
        {
            ++num_nearby;
        }

        // store into variable
        // 筛选出最近的关键帧
        if (delta_d < closest_d)
        {
            closest_d = delta_d;
            closest_idx = keyframes_idx;
        }

        keyframes_idx++;

    }
    // 获取最近关键帧的位姿
    // get closest pose and corresponding rotation
    Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
    Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;
    // 这不就是closest_d么？
    // calculate distance between current pose and closest pose from above
    float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                     pow(this->state.p[1] - closest_pose[1], 2) +
                     pow(this->state.p[2] - closest_pose[2], 2) );

    // calculate difference in orientation using SLERP
    // 计算旋转的差异
    Eigen::Quaternionf dq;
    // 四元数的点积可以用来描述两个旋转的相似程度 点积绝对值越大 则两个四元数代表的角位移越相似 而点积的正负则代表了二者的方向 当小于0时 方向相反
    if (this->state.q.dot(closest_pose_r) < 0.)
    {
        Eigen::Quaternionf lq = closest_pose_r;
        lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
        dq = this->state.q * lq.inverse();
    }
    else
    {
        dq = this->state.q * closest_pose_r.inverse();
    }
    // 解算出旋转角度
    double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
    double theta_deg = theta_rad * (180.0/M_PI);

    // update keyframes
    bool newKeyframe = false;

    // 当前帧与最近关键帧的距离偏角大于阈值
    // spaciousness keyframing
    if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_)
    {
        newKeyframe = true;
    }

    // 距离虽然不够 但是旋转角度够 并且与当前帧距离小于1.5倍阈值的关键帧 只有一个
    // rotational exploration keyframing
    if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1)
    {
        newKeyframe = true;
    }

    // 距离不够 或小于0.5m的
    // check for nearby keyframes
    if (abs(dd) <= this->keyframe_thresh_dist_)
    {
        newKeyframe = false;
    }
    else if (abs(dd) <= 0.5)
    {
        newKeyframe = false;
    }

    return newKeyframe;
}

void dlio::OdomNode::addOdomFactor()
{
    static int num_factor = 0;

    std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
    KeyframeInfo current_kf_info = this->tempKeyframe;
    lock_temp.unlock();

    // 初始第一帧
    if (num_factor == 0)
    {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished());
        this->gtSAMgraph.addPrior(0, state2Pose3(current_kf_info.rot, current_kf_info.pos), priorNoise);
        this->initialEstimate.insert(0, state2Pose3(current_kf_info.rot, current_kf_info.pos));
    }
    else
    {
        // 当前帧匹配的子地图关键帧索引
        std::vector<int> curr_submap_id = current_kf_info.submap_kf_idx;
        // 当前帧匹配的子地图关键帧相似度
        std::vector<float> curr_sim = current_kf_info.vSim;
        // 当前帧的位姿
        gtsam::Pose3 poseTo = state2Pose3(this->state.q,
                                          this->state.p);
        // 遍历与当前帧匹配的子地图关键帧
        for (int i = 0; i < curr_submap_id.size(); i++)
        {
            // 子地图关键帧的位姿
            std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
            gtsam::Pose3 poseFrom = state2Pose3(this->keyframes[curr_submap_id[i]].first.second,
                                                this->keyframes[curr_submap_id[i]].first.first);
            lock_kf.unlock();
            // 噪声权重
            double weight = 1.0 * this->icpScore;

            if (curr_sim.size() > 0)
            {
                // 对相似度进行归一化
                auto max_sim = std::max_element(curr_sim.begin(), curr_sim.end());
                for (auto &it: curr_sim)
                    it = it / *max_sim;
                double sim = curr_sim[i] >= 1 ? 0.99 : curr_sim[i] < 0 ? 0 : curr_sim[i];

                // 相似度越大 噪声权重越小
                weight = 1 - sim;
            }
            // 添加相邻/不相邻里程计因子
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
                    (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished() * weight);
            this->gtSAMgraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(curr_submap_id[i],
                                                                          num_factor,poseFrom.between(poseTo), odometryNoise);
        }
//        std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
//        gtsam::Pose3 poseFrom = state2Pose3(this->keyframes[num_factor - 1].first.second,
//                                            this->keyframes[num_factor - 1].first.first);
//        lock_kf.unlock();
//        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
//                (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
//        this->gtSAMgraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(num_factor - 1,
//                                                                            num_factor,poseFrom.between(poseTo), odometryNoise);

        this->initialEstimate.insert(num_factor, poseTo);
    }
    num_factor++;

}

void dlio::OdomNode::addGPSFactor()
{
    static int last_val_gps_size = 0;
    std::unique_lock<std::mutex> lock(this->val_gps_mutex);
    int current_size = this->v_val_gps.size();
    lock.unlock();

    // 有未处理的GPS消息 且历史GPS大于2帧
    if (current_size > last_val_gps_size && current_size > 1)
    {
        last_val_gps_size = current_size;
        lock.lock();
        auto current_gps_meas = this->v_val_gps[current_size - 1];
        auto last_gps_meas = this->v_val_gps[current_size - 2];
        lock.unlock();

        double current_gps_time = current_gps_meas.time;
        // 筛选出与当前处理GPS时间戳最近的关键帧id
        double min_time_diff = 10e5;
        int matched_id = -1;
        for (int i = 0; i < this->v_kf_time.size(); i++)
        {
            double time_diff = abs(current_gps_time - this->v_kf_time[i]);
            if (time_diff < min_time_diff)
            {
                min_time_diff = time_diff;
                matched_id = i;
            }
        }

        // 得到与当前GPS匹配的关键帧
        if (matched_id != -1 && this->gps_node_id.find(matched_id) == this->gps_node_id.end())
        {
            this->gps_node_id.insert(matched_id);
            // 用匀速运动插值
            // TODO IMU积分插值
            Eigen::Vector3d begin_point = {last_gps_meas.x, last_gps_meas.y, last_gps_meas.z};
            Eigen::Vector3d end_point = {current_gps_meas.x, current_gps_meas.y, current_gps_meas.z};
            double begin_time = last_gps_meas.time;
            double end_time = current_gps_time;
            double kf_time = this->v_kf_time[matched_id];

            Eigen::Vector3d kf_point = {0, 0, 0};
            kf_point = begin_point + (end_point - begin_point) / (end_time - begin_time) * (kf_time - begin_time);

            // TODO 协方差未转换到UTM下
            gtsam::Vector Vector3(3);
            Vector3 << 1.0, 1.0, 1.0;
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gpsFactor(matched_id, gtsam::Vector3(kf_point), gps_noise);
            ROS_INFO("Add a gps factor");
            this->gtSAMgraph.add(gpsFactor);
        }
    }
    else
        return;
}


bool dlio::OdomNode::matchGPSWithKf(GPSMeas& gps)
{
    double current_gps_time = gps.time;
    // 筛选出与当前处理GPS时间戳最近的关键帧id
    double min_time_diff = 10e5;
    int matched_id = -1;
    for (int i = 0; i < this->v_kf_time.size(); i++)
    {
        double time_diff = abs(current_gps_time - this->v_kf_time[i]);
        if (time_diff < min_time_diff)
        {
            min_time_diff = time_diff;
            matched_id = i;
        }
    }

    if (matched_id != -1 && this->gps_node_id.find(matched_id) == this->gps_node_id.end())
    {
        this->gps_node_id.insert(matched_id);
        gps.mathced_id = matched_id;
        return true;
    }
    else
        return false;
}


void dlio::OdomNode::addGPSFactorWithoutAlign()
{
    static bool GPS_align = false;
    static int last_gps_size = 0;

    std::unique_lock<std::mutex> lock_gps_buffer(this->gps_buffer_mutex);
    auto gps_buffer_copy = this->gps_buffer;
    int val_gps_num = this->gps_buffer.size();
    lock_gps_buffer.unlock();

    if (!GPS_align)
    {
        if (val_gps_num < 20)
        {
            ROS_INFO("GNSS not init, now: %d, hope: 20", val_gps_num);
            return;
        }

        for (auto gps : gps_buffer_copy)
        {
            gtsam::Vector3 noise;
            noise << gps.cov(0, 0), gps.cov(1, 1), gps.cov(2, 2);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(noise);
            gtsam::GPSFactor gps_factor(gps.mathced_id, gtsam::Vector3(gps.x, gps.y, gps.z), gps_noise);
            this->gtSAMgraph.add(gps_factor);
        }

        last_gps_size = val_gps_num;

        GPS_align = true;

        this->isLoop = true;
        ROS_INFO("GNSS init success !");
    }
    else
    {
        if (val_gps_num > last_gps_size)
        {
            last_gps_size = val_gps_num;
            auto gps = gps_buffer_copy.back();
            gtsam::Vector3 noise;
            noise << gps.cov(0, 0), gps.cov(1, 1), gps.cov(2, 2);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(noise);
            gtsam::GPSFactor gps_factor(gps.mathced_id, gtsam::Vector3(gps.x, gps.y, gps.z), gps_noise);
            this->gtSAMgraph.add(gps_factor);
            ROS_INFO("Add a GPS factor");
        }
        else
        {
            return;
        }

    }
}


void dlio::OdomNode::loopVisual()
{
    this->loop_marker.header.stamp = this->imu_stamp;
    this->loop_marker.header.frame_id = this->odom_frame;
    std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
    for (auto two : this->history_loop_id)
    {
        geometry_msgs::Point point1;
        point1.x = this->keyframes[two.first].first.first.x();
        point1.y = this->keyframes[two.first].first.first.y();
        point1.z = this->keyframes[two.first].first.first.z();

        geometry_msgs::Point point2;
        point2.x = this->keyframes[two.second].first.first.x();
        point2.y = this->keyframes[two.second].first.first.y();
        point2.z = this->keyframes[two.second].first.first.z();

        this->loop_marker.points.push_back(point1);
        this->loop_marker.points.push_back(point2);
    }
    lock_kf.unlock();

    this->loop_constraint_pub.publish(this->loop_marker);
}


void dlio::OdomNode::addLoopFactor()
{
    std::unique_lock<decltype(this->loop_factor_mutex)> lock_loop_factor(this->loop_factor_mutex);
    loop_factor_info current_loop_factor_info = this->curr_factor_info;
    lock_loop_factor.unlock();

    this->isLoop = false;

    if (current_loop_factor_info.loop)
    {
        this->isLoop = true;
        std::cout << "Back end processing loop" << std::endl;

        gtsam::noiseModel::Diagonal::shared_ptr loopNoise = gtsam::noiseModel::Diagonal::Variances(
                (gtsam::Vector(6) << 1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6).finished() * current_loop_factor_info.sim);

        gtsam::Pose3 poseFrom = state2Pose3(Eigen::Quaternionf(current_loop_factor_info.T_target.rotation()),
                                            current_loop_factor_info.T_target.translation());

        gtsam::Pose3 poseTo = state2Pose3(Eigen::Quaternionf(current_loop_factor_info.T_current.rotation()),
                                          current_loop_factor_info.T_current.translation());

        this->gtSAMgraph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(current_loop_factor_info.factor_id.second,
                                                                         current_loop_factor_info.factor_id.first,
                                                                            poseFrom.between(poseTo),
                                                                            loopNoise);

        this->history_loop_id.push_back(std::make_pair(current_loop_factor_info.factor_id.first, current_loop_factor_info.factor_id.second));

        lock_loop_factor.lock();
        this->curr_factor_info.loop = false;
        lock_loop_factor.unlock();
    }
    else
    {
        return;
    }

    this->loopVisual();
}

void dlio::OdomNode::correctPoses()
{
    if (this->isLoop)
    {
        this->global_pose.poses.clear();
        std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
        for (int i = 0; i < this->iSAMCurrentEstimate.size(); i++)
        {
            this->keyframes[i].first.first = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().cast<float>();
            this->keyframes[i].first.second = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().cast<float>();

            geometry_msgs::Pose p;
            p.position.x = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[0];
            p.position.y = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[1];
            p.position.z = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).translation().vector().cast<float>()[2];

            p.orientation.w = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
            p.orientation.x = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
            p.orientation.y = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
            p.orientation.z = this->iSAMCurrentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().z();
            this->global_pose.poses.push_back(p);
        }
        this->f.open("/home/ywl/trajectory.txt", std::ios::out);
        this->f.precision(9);
        this->f.setf(std::ios::fixed);

        for (int i = 0; i < this->global_pose.poses.size(); i++)
        {
            geometry_msgs::Point position = this->global_pose.poses[i].position;
            geometry_msgs::Quaternion orientation = this->global_pose.poses[i].orientation;
            double time = this->keyframe_timestamps[i].toSec();
            this->f << time << " " << position.x << " " << position.y << " " << position.z << " " << orientation.x <<
              " " << orientation.y << " " << orientation.z << " " << orientation.w << std::endl;
        }
        this->f.close();
        ROS_INFO("Finished trajectory");

        lock_kf.unlock();



    }
    else
    {
        geometry_msgs::Pose p;
        p.position.x = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).translation().vector().cast<float>()[0];
        p.position.y = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).translation().vector().cast<float>()[1];
        p.position.z = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).translation().vector().cast<float>()[2];

        p.orientation.w = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().w();
        p.orientation.x = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().x();
        p.orientation.y = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().y();
        p.orientation.z = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().z();
        this->global_pose.poses.push_back(p);

    }
    this->global_pose.header.stamp = ros::Time::now();
    this->global_pose.header.frame_id = this->odom_frame;
    this->global_pose_pub.publish(this->global_pose);
    this->lidarPose.p = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).translation().cast<float>();
    this->lidarPose.q = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().cast<float>();
//    this->state.p = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).translation().cast<float>();
//    this->state.q = this->iSAMCurrentEstimate.at<gtsam::Pose3>(this->iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().cast<float>();
    this->updateState();

}

void dlio::OdomNode::updateMap()
{
    while (this->nh.ok())
    {

        std::unique_lock<decltype(this->update_map_info_mutex)> lock_update_map(this->update_map_info_mutex);
        if (this->update_map_info.size() == 0)
        {
            lock_update_map.unlock();
            continue;
        }
        auto map_info = this->update_map_info.front();
        this->update_map_info.pop();
        lock_update_map.unlock();

        gtsam::Values currentEstimate = map_info.second;
        if (map_info.first)
        {
            std::cout << "start update map" << std::endl;
            std::unique_lock<decltype(this->history_kf_lidar_mutex)> lock_kf_his(this->history_kf_lidar_mutex);
            auto his = this->history_kf_lidar;
            lock_kf_his.unlock();
            this->global_map->clear();
            for (int i = 0; i < currentEstimate.size() ; i++)
            {
                pcl::PointCloud<PointType>::Ptr curr_kf(new pcl::PointCloud<PointType>);
                Eigen::Isometry3f curr_T = Eigen::Isometry3f::Identity();

                Eigen::Quaternionf q =currentEstimate.at<gtsam::Pose3>(i).rotation().toQuaternion().cast<float>();
                curr_T.translate(currentEstimate.at<gtsam::Pose3>(i).translation().cast<float>());
                curr_T.rotate(q);

                pcl::transformPointCloud(*his[i], *curr_kf, curr_T.matrix());

                *this->global_map += *curr_kf;
                this->voxel_global.setInputCloud(this->global_map);
                this->voxel_global.filter(*this->global_map);
            }

            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*this->global_map, map_msg);
            map_msg.header.stamp = ros::Time::now();
            map_msg.header.frame_id = this->odom_frame;

            this->global_map_pub.publish(map_msg);
            std::cout << "finish update map" << std::endl;

        }
        else
        {
            pcl::PointCloud<PointType>::Ptr curr_kf(new pcl::PointCloud<PointType>);
            Eigen::Isometry3f curr_T = Eigen::Isometry3f::Identity();

            Eigen::Quaternionf q =currentEstimate.at<gtsam::Pose3>(currentEstimate.size() - 1).rotation().toQuaternion().cast<float>();
            curr_T.translate(currentEstimate.at<gtsam::Pose3>(currentEstimate.size() - 1).translation().cast<float>());
            curr_T.rotate(q);

            std::unique_lock<decltype(this->keyframes_mutex)> lock_kf_his(this->keyframes_mutex);
            pcl::transformPointCloud(*this->history_kf_lidar[currentEstimate.size() - 1], *curr_kf, curr_T.matrix()) ;
            lock_kf_his.unlock();


            this->voxel_global.setInputCloud(curr_kf);
            this->voxel_global.filter(*curr_kf);
            *this->global_map += *curr_kf;
            this->voxel_global.setInputCloud(this->global_map);
            this->voxel_global.filter(*this->global_map);


            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(*this->global_map, map_msg);
            map_msg.header.stamp = this->scan_header_stamp;
            map_msg.header.frame_id = this->odom_frame;

            this->global_map_pub.publish(map_msg);

        }
    }

}

void dlio::OdomNode::updateCurrentInfo()
{
    // 更新关键帧信息
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->currentFusionState.p, this->currentFusionState.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.push_back(this->T_corr);
    this->keyframe_transformations_prior.push_back(this->T_prior);
    this->v_kf_time.push_back(this->scan_stamp);
    lock.unlock();

    std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
    pcl::copyPointCloud(*this->current_scan_lidar, *this->tempKeyframe.pCloud);
    this->tempKeyframe.rot = this->currentFusionState.q;
    this->tempKeyframe.pos = this->currentFusionState.p;
    this->KeyframesInfo.push_back(this->tempKeyframe);
    this->keyframe_stateT.push_back(this->currentFusionT.matrix());
    lock_temp.unlock();

    pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
    pcl::copyPointCloud(*this->current_scan_lidar, *temp);
    this->history_kf_lidar.push_back(temp);


    std::unique_lock<decltype(this->loop_info_mutex)> lock_loop(this->loop_info_mutex);
    if (this->curr_loop_info.loop_candidate)
    {
        this->curr_loop_info.loop_candidate = false;
        this->curr_loop_info.loop = true;
        std::unique_lock<decltype(this->keyframes_mutex)> lock_kf(this->keyframes_mutex);
        this->curr_loop_info.current_kf = this->keyframes.back();
        this->curr_loop_info.current_id = this->keyframes.size() - 1;
        lock_kf.unlock();
    }
    lock_loop.unlock();
}

void dlio::OdomNode::saveKeyframeAndUpdateFactor()
{

    if (this->isKeyframe())
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        this->updateCurrentInfo();

        // 添加里程计因子
        this->addOdomFactor();

        // 添加GPS因子
        this->addGPSFactorWithoutAlign();

        // 添加闭环因子
        this->addLoopFactor();

//        gtSAMgraph.print("GTSAM Graph:\n");


        this->isam->update(this->gtSAMgraph, this->initialEstimate);
        this->isam->update();

        if (this->isLoop)
        {
            this->isam->update();
            this->isam->update();
            this->isam->update();
        }

        this->gtSAMgraph.resize(0);
        this->initialEstimate.clear();
        this->iSAMCurrentEstimate = this->isam->calculateEstimate();


        std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
        KeyframeInfo his_info = this->tempKeyframe;
        lock_temp.unlock();
//        std::cout << "========================================" << std::endl;
//        std::cout << "Before rot = " << his_info.rot.w() << " "
//                  << his_info.rot.x() << " "
//                  << his_info.rot.y() << " "
//                  << his_info.rot.z() << " "
//                  << "Before pos = "
//                  << his_info.pos.x() << " "
//                  << his_info.pos.y() << " "
//                  << his_info.pos.z() << " " << std::endl;
//
//
//        std::cout << "After rot = "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().w() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().x() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().y() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().z() << " "
//                  << "After pos = "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().x() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().y() << " "
//                  << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().z() << " "
//                  << std::endl;
//        std::cout << "========================================" << std::endl;



        this->correctPoses();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        std::cout << "Back end cost " << time * 1000 << " ms" << std::endl;

        auto loop_copy = this->isLoop;
        auto iSAMCurrentEstimate_copy  = this->iSAMCurrentEstimate;
        std::unique_lock<decltype(this->update_map_info_mutex)> lock_update_map(this->update_map_info_mutex);
        this->update_map_info.push(std::make_pair(loop_copy, iSAMCurrentEstimate_copy));
        lock_update_map.unlock();

    }
    else
    {
        return;
    }

}

void dlio::OdomNode::saveFirstKeyframeAndUpdateFactor()
{
    // 添加里程计因子
    this->addOdomFactor();

    this->isam->update(this->gtSAMgraph, this->initialEstimate);
    this->isam->update();

    this->gtSAMgraph.resize(0);
    this->initialEstimate.clear();
    this->iSAMCurrentEstimate = this->isam->calculateEstimate();

    std::unique_lock<decltype(this->tempKeyframe_mutex)> lock_temp(this->tempKeyframe_mutex);
    KeyframeInfo his_info = this->tempKeyframe;
    lock_temp.unlock();
    std::cout << "========================================" << std::endl;
    std::cout << "Before rot = " << his_info.rot.w() << " "
              << his_info.rot.x() << " "
              << his_info.rot.y() << " "
              << his_info.rot.z() << " "
              << "Before pos = "
              << his_info.pos.x() << " "
              << his_info.pos.y() << " "
              << his_info.pos.z() << " " << std::endl;


    std::cout << "After rot = "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().w() << " "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().x() << " "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().y() << " "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).rotation().toQuaternion().z() << " "
              << "After pos = "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().x() << " "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().y() << " "
              << iSAMCurrentEstimate.at<gtsam::Pose3>(iSAMCurrentEstimate.size() - 1).translation().z() << " "
              << std::endl;
    std::cout << "========================================" << std::endl;



    this->correctPoses();
    auto loop_copy = this->isLoop;
    auto iSAMCurrentEstimate_copy  = this->iSAMCurrentEstimate;
    std::unique_lock<decltype(this->update_map_info_mutex)> lock_update_map(this->update_map_info_mutex);
    this->update_map_info.push(std::make_pair(loop_copy, iSAMCurrentEstimate_copy));
    lock_update_map.unlock();


}