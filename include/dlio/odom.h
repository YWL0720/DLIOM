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

#include "dlio/dlio.h"

class dlio::OdomNode
{

public:
    OdomNode(ros::NodeHandle node_handle);
    ~OdomNode();

    void start();
    void performLoop();
    void saveKeyframeAndUpdateFactor();
    bool isKeyframe();
    void addOdomFactor();
    void addLoopFactor();
    void addGPSFactor();
    void loopVisual();
    void correctPoses();
    void updateMap();
    void updateCurrentInfo();
    void saveFirstKeyframeAndUpdateFactor();

private:
    ros::NodeHandle nh;
    ros::Timer publish_timer;

    // Subscribers
    ros::Subscriber lidar_sub;
    ros::Subscriber imu_sub;
    ros::Subscriber gps_sub;

    // Publishers
    ros::Publisher odom_pub;
    ros::Publisher pose_pub;
    ros::Publisher path_pub;
    ros::Publisher kf_pose_pub;
    ros::Publisher kf_cloud_pub;
    ros::Publisher deskewed_pub;
    ros::Publisher kf_connect_pub;
    ros::Publisher global_map_pub;
    ros::Publisher global_pose_pub;
    ros::Publisher loop_constraint_pub;

    ros::Publisher gps_pub_test;

    // ROS Msgs
    nav_msgs::Odometry odom_ros;
    geometry_msgs::PoseStamped pose_ros;
    nav_msgs::Path path_ros;
    geometry_msgs::PoseArray kf_pose_ros;
    geometry_msgs::PoseArray global_pose;

    bool global_dense;
    // Flags
    std::atomic<bool> dlio_initialized;
    std::atomic<bool> first_valid_scan;
    std::atomic<bool> first_imu_received;
    std::atomic<bool> imu_calibrated;
    std::atomic<bool> submap_hasChanged;
    std::atomic<bool> gicp_hasConverged;
    std::atomic<bool> deskew_status;
    std::atomic<int> deskew_size;

    // Threads
    std::thread publish_thread;
    std::thread publish_keyframe_thread;
    std::thread metrics_thread;
    std::thread debug_thread;
    std::thread mapping_thread;
    std::thread loop_thread;

    // struct
    struct GPSMeas
    {
        GPSMeas() { mathced_id = -1; };
        GPSMeas(double x_, double y_, double z_, double time_) : x(x_), y(y_), z(z_), time(time_) { mathced_id = -1; };
        double x;
        double y;
        double z;
        double time;
        Eigen::Matrix3d cov;
        int mathced_id;
    };
    GPSMeas gps_meas;
    GPSMeas last_gps_meas;

    struct KeyframeInfo
    {
        pcl::PointCloud<PointType>::Ptr pCloud = boost::make_shared<pcl::PointCloud<PointType>>();
        Eigen::Vector3f pos;
        Eigen::Quaternionf rot;
        std::vector<float> vSim = {};
        std::vector<int> submap_kf_idx = {};
        double time;
    };
    KeyframeInfo tempKeyframe;

    struct loop_info
    {
        bool loop = false;
        bool loop_candidate = false;
        int current_id = -1;
        std::vector<int> candidate_key;
        std::vector<float> candidate_dis;
        std::vector<float> candidate_sim;
        std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                pcl::PointCloud<PointType>::ConstPtr>> candidate_frame;
        std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                pcl::PointCloud<PointType>::ConstPtr> current_kf;
        std::vector<std::shared_ptr<nano_gicp::CovarianceList>> candidate_frame_normals;
        struct State;
        struct ImuMeas;
        void reset()
        {
            loop = false;
            loop_candidate = false;
            candidate_frame.clear();
            candidate_frame_normals.clear();
            candidate_key.clear();
            candidate_dis.clear();
            candidate_sim.clear();
        }
    };
    loop_info curr_loop_info;

    struct loop_factor_info
    {
        // curr target
        bool loop = false;
        // curr target
        std::pair<int, int> factor_id;
        Eigen::Isometry3f T_current;
        Eigen::Isometry3f T_target;
        float sim;
        float dis;

    };
    loop_factor_info curr_factor_info;

    struct ImuMeas
    {
        double stamp;
        double dt; // defined as the difference between the current and the previous measurement
        Eigen::Vector3f ang_vel;
        Eigen::Vector3f lin_accel;
    };
    ImuMeas imu_meas;



    static bool comparatorImu(ImuMeas m1, ImuMeas m2)
    {
        return (m1.stamp < m2.stamp);
    };

    // Geometric Observer
    struct Geo
    {
        bool first_opt_done;
        std::mutex mtx;
        double dp;
        double dq_deg;
        Eigen::Vector3f prev_p;
        Eigen::Quaternionf prev_q;
        Eigen::Vector3f prev_vel;
    };
    Geo geo;

    // State Vector
    struct ImuBias
    {
        Eigen::Vector3f gyro;
        Eigen::Vector3f accel;
    };

    struct Frames
    {
        Eigen::Vector3f b;
        Eigen::Vector3f w;
    };

    struct Velocity
    {
        Frames lin;
        Frames ang;
    };

    struct State
    {
        Eigen::Vector3f p; // position in world frame
        Eigen::Quaternionf q; // orientation in world frame
        Velocity v;
        ImuBias b; // imu biases in body frame
    };
    State state;
    State currentFusionState;


    struct Pose
    {
        Eigen::Vector3f p; // position in world frame
        Eigen::Quaternionf q; // orientation in world frame
    };
    Pose lidarPose;
    Pose imuPose;

    // Metrics
    struct Metrics
    {
        std::vector<float> spaciousness;
        std::vector<float> density;
    };
    Metrics metrics;

    struct Extrinsics
    {
        struct SE3
        {
            Eigen::Vector3f t;
            Eigen::Matrix3f R;
        };
        SE3 baselink2imu;
        SE3 baselink2lidar;
        Eigen::Matrix4f baselink2imu_T;
        Eigen::Matrix4f baselink2lidar_T;
    };
    Extrinsics extrinsics;

    // mutex
    std::mutex mtx_imu;
    std::mutex global_map_update_mutex;
    std::mutex update_map_info_mutex;
    std::mutex gps_buffer_mutex;
    std::mutex val_gps_mutex;
    std::mutex gps_mutex;
    std::mutex keyframes_mutex;
    std::mutex tempKeyframe_mutex;
    std::mutex history_kf_lidar_mutex;
    std::mutex loop_info_mutex;
    std::mutex loop_factor_mutex;
    std::mutex main_loop_running_mutex;
    std::condition_variable cv_imu_stamp;

    // lidar
    dlio::SensorType sensor;
    // Keyframes
    pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
    int num_processed_keyframes;
    pcl::ConvexHull<PointType> convex_hull;
    pcl::ConcaveHull<PointType> concave_hull;
    std::vector<int> keyframe_convex;
    std::vector<int> keyframe_concave;
    std::vector<double> v_kf_time;
    std::vector<KeyframeInfo> KeyframesInfo;
    std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
            pcl::PointCloud<PointType>::ConstPtr>> keyframes;
    std::vector<ros::Time> keyframe_timestamps;
    std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations_prior;
    pcl::PointCloud<PointType>::Ptr current_scan_lidar;
    std::vector<pcl::PointCloud<PointType>::Ptr> history_pointcloud_lidar;
    std::vector<pcl::PointCloud<PointType>::Ptr> history_kf_lidar;

    // Point Clouds
    pcl::PointCloud<PointType>::ConstPtr original_scan;       // 原始点云
    pcl::PointCloud<PointType>::ConstPtr deskewed_scan;       // 去畸变后的点云
    pcl::PointCloud<PointType>::ConstPtr current_scan;        // 去畸变且降采样后的点云
    pcl::PointCloud<PointType>::Ptr current_scan_w;        // 去畸变且降采样后的点云

    // Submap
    pcl::PointCloud<PointType>::ConstPtr submap_cloud;
    std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
    std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;
    std::vector<float> similarity;
    std::vector<int> submap_kf_idx_curr;
    std::vector<int> submap_kf_idx_prev;

    // imu
    boost::circular_buffer<ImuMeas> imu_buffer;
    ros::Time imu_stamp;
    double first_imu_stamp;
    double prev_imu_stamp;
    double imu_dp, imu_dq_deg;


    // GNSS
    Eigen::Matrix3d R_M_G = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_M_G = Eigen::Vector3d::Zero();
    GeographicLib::LocalCartesian geo_converter;
    std::vector<GPSMeas> v_gps_meas;
    std::vector<GPSMeas> v_gps_init;
    std::vector<GPSMeas> v_gps_state;
    std::vector<GPSMeas> gps_buffer;
    std::unordered_set<int> gps_node_id;
    std::vector<GPSMeas> v_val_gps;
    bool gps_init = false;

    // loop
    bool isLoop;
    visualization_msgs::Marker loop_marker;
    std::vector<std::pair<int, int>> history_loop_id;

    // global map
    pcl::PointCloud<PointType>::Ptr global_map;
    std::condition_variable global_map_update_cv;
    pcl::VoxelGrid<PointType> voxel_global;
    std::queue<std::pair<bool, gtsam::Values>> update_map_info;
    bool global_map_update_finish = true;


    // Frames
    std::string odom_frame;
    std::string baselink_frame;
    std::string lidar_frame;
    std::string imu_frame;

    // Preprocessing
    pcl::CropBox<PointType> crop;
    pcl::VoxelGrid<PointType> voxel;

    // file
    std::fstream f;

    // Timestamps
    ros::Time scan_header_stamp;
    double scan_stamp;
    double prev_scan_stamp;
    double scan_dt;
    std::vector<double> comp_times;
    std::vector<double> imu_rates;
    std::vector<double> lidar_rates;

    // Transformations
    Eigen::Matrix4f T, T_prior, T_corr;
    Eigen::Quaternionf q_final;
    Eigen::Vector3f origin;
    Eigen::Isometry3f currentFusionT;
    std::vector<Eigen::Matrix4f> keyframe_stateT;

    // Trajectory
    std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
    double length_traversed;

    // GICP
    nano_gicp::NanoGICP<PointType, PointType> gicp;
    nano_gicp::NanoGICP<PointType, PointType> gicp_temp;
    nano_gicp::NanoGICP<PointType, PointType> gicp_tool;
    float icpScore;

    // gtasm
    gtsam::ISAM2 *isam;
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::Values iSAMCurrentEstimate;
    Eigen::MatrixXd poseCovariance;


    void getParams();

    void callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc);
    void callbackImu(const sensor_msgs::Imu::ConstPtr& imu);
    void callbackGPS(const sensor_msgs::NavSatFixConstPtr& gps);
    void callbackGPSWithoutAlign(const sensor_msgs::NavSatFixConstPtr& gps);
    bool matchGPSWithKf(GPSMeas& gps);

    void getTransformBetweenMapAndGPS(std::vector<GPSMeas>& gps_init, std::vector<GPSMeas>& gps_state);


    void publishPose(const ros::TimerEvent& e);

    void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
    void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
    void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, ros::Time timestamp);

    void getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc);
    void preprocessPoints();
    void deskewPointcloud();
    void initializeInputTarget();
    void setInputSource();

    void initializeDLIO();

    void getNextPose();
    bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);
    void propagateGICP();

    void propagateState();
    void updateState();

    void setAdaptiveParams();
    void setKeyframeCloud();

    void computeMetrics();
    void computeSpaciousness();
    void computeDensity();

    void computeJaccard();

    sensor_msgs::Imu::Ptr transformImu(const sensor_msgs::Imu::ConstPtr& imu);

    void updateKeyframes();
    void computeConvexHull();
    void computeConcaveHull();
    void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
    void buildSubmap(State vehicle_state);


    void buildSubmapViaJaccard(State vehicle_state);

    void buildKeyframesAndSubmap(State vehicle_state);
    void pauseSubmapBuildIfNeeded();


    gtsam::Pose3 state2Pose3(Eigen::Quaternionf rot, Eigen::Vector3f pos);

    void debug();


    bool new_submap_is_ready;
    std::future<void> submap_future;
    std::condition_variable submap_build_cv;
    bool main_loop_running;
    bool useJaccard;
    double first_scan_stamp;
    double elapsed_time;

    std::string cpu_type;
    std::vector<double> cpu_percents;
    clock_t lastCPU, lastSysCPU, lastUserCPU;
    int numProcessors;

    // Parameters
    bool kf_update;
    std::string version_;
    int num_threads_;

    bool deskew_;

    double gravity_;

    bool adaptive_params_;

    double obs_submap_thresh_;
    double obs_keyframe_thresh_;
    double obs_keyframe_lag_;

    double keyframe_thresh_dist_;
    double keyframe_thresh_rot_;

    int submap_knn_;
    int submap_kcv_;
    int submap_kcc_;
    double submap_concave_alpha_;

    bool densemap_filtered_;
    bool wait_until_move_;

    double crop_size_;

    bool vf_use_;
    double vf_res_;

    bool imu_calibrate_;
    bool calibrate_gyro_;
    bool calibrate_accel_;
    bool gravity_align_;
    double imu_calib_time_;
    int imu_buffer_size_;
    Eigen::Matrix3f imu_accel_sm_;

    int gicp_min_num_points_;
    int gicp_k_correspondences_;
    double gicp_max_corr_dist_;
    int gicp_max_iter_;
    double gicp_transformation_ep_;
    double gicp_rotation_ep_;
    double gicp_init_lambda_factor_;

    double geo_Kp_;
    double geo_Kv_;
    double geo_Kq_;
    double geo_Kab_;
    double geo_Kgb_;
    double geo_abias_max_;
    double geo_gbias_max_;



};
