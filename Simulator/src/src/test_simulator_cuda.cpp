#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_simulator/DynamicObjectState.h>
#include <sensor_simulator/DynamicObjectStateArray.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <chrono>
#include "maps.hpp"
#include "dynamic_actor.hpp"

using namespace raycast;

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // 读取camera参数
        camera = new CameraParams();
        camera->fx = config["camera"]["fx"].as<float>();
        camera->fy = config["camera"]["fy"].as<float>();
        camera->cx = config["camera"]["cx"].as<float>();
        camera->cy = config["camera"]["cy"].as<float>();
        camera->image_width = config["camera"]["image_width"].as<int>();
        camera->image_height = config["camera"]["image_height"].as<int>();
        camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
        quat_bc = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
        const auto camera_position = config["camera"]["position_body"];
        camera_position_body = Eigen::Vector3f(
            camera_position[0].as<float>(), camera_position[1].as<float>(), camera_position[2].as<float>());
        Eigen::Matrix3f rotation_native_from_optical;
        rotation_native_from_optical << 0, 0, 1,
                                       -1, 0, 0,
                                        0,-1, 0;
        quat_body_from_optical = quat_bc * Eigen::Quaternionf(rotation_native_from_optical);

        // 读取lidar参数
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        ros::NodeHandle private_nh("~");
        private_nh.param("render_lidar", render_lidar, render_lidar);
        private_nh.param("render_depth", render_depth, render_depth);
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        depth_pub_duration = ros::Duration(1 / depth_fps);
        lidar_pub_duration = ros::Duration(1 / lidar_fps);
        
        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string instance_topic = config["dynamic_instance_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();
        lidar_body_topic = config["lidar_body_topic"].as<std::string>();
        lidar_optical_topic = config["lidar_optical_topic"].as<std::string>();
        lidar_world_topic = config["lidar_world_topic"].as<std::string>();
        std::string camera_info_topic = config["camera_info_topic"].as<std::string>();
        dynamic_gt_topic = config["dynamic_gt_topic"].as<std::string>();
        dynamic_collision_topic = config["dynamic_collision_topic"].as<std::string>();
        dynamic_marker_topic = config["dynamic_marker_topic"].as<std::string>();
        publish_dynamic_markers = config["publish_dynamic_markers"].as<bool>();
        uav_collision_radius = config["uav_collision_radius"].as<float>();
        if (!std::isfinite(uav_collision_radius) || uav_collision_radius <= 0.0f) {
            throw std::invalid_argument("uav_collision_radius must be finite and positive");
        }
        world_frame_id = config["world_frame_id"].as<std::string>();
        body_frame_id = config["body_frame_id"].as<std::string>();
        camera_optical_frame_id = config["camera_optical_frame_id"].as<std::string>();
        std::string scenario_file = config["dynamic_scenario_file"].as<std::string>();
        private_nh.param<std::string>("dynamic_scenario_file", scenario_file, scenario_file);
        dynamic_scenario = dynamic_sim::DynamicScenario::load(scenario_file);
        ROS_INFO("Dynamic scenario %s seed=%u enabled=%s from %s",
                 dynamic_scenario.scenarioId().c_str(), dynamic_scenario.seed(),
                 dynamic_scenario.enabled() ? "true" : "false", scenario_file.c_str());

        // 读取地图参数
        bool use_random_map = config["random_map"].as<bool>();
        private_nh.param("random_map", use_random_map, use_random_map);
        private_nh.param<std::string>("ply_file", ply_file, ply_file);
        std::string authority_artifact =
            config["authority_artifact"]
            ? config["authority_artifact"].as<std::string>() : "";
        private_nh.param<std::string>(
            "authority_artifact", authority_artifact, authority_artifact);
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (!authority_artifact.empty()) {
            grid_map = new GridMap(authority_artifact);
            ROS_INFO(
                "Static geometry authority %s hash=%s occupied_voxels=%d",
                authority_artifact.c_str(),
                grid_map->authorityHash().c_str(),
                grid_map->occupiedVoxelCount());
        }
        else if (use_random_map) {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX      = sizeX;
            info.sizeY      = sizeY;
            info.sizeZ      = sizeZ;
            info.seed       = seed;
            info.scale      = scale;
            info.cloud      = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(type);
        }
        else {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
                throw std::runtime_error("could not read PLY file: " + ply_file);
            }
        }
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "world";

        std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
        printf("2.Mapping... \n");
        if (authority_artifact.empty())
            grid_map = new GridMap(cloud, resolution, occupy_threshold);
        
        next_depth_pub_time = ros::Time::now();
        next_lidar_pub_time = ros::Time::now();

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        instance_pub_ = nh_.advertise<sensor_msgs::Image>(instance_topic, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        point_cloud_body_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_body_topic, 1);
        point_cloud_optical_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_optical_topic, 1);
        point_cloud_world_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_world_topic, 1);
        camera_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(camera_info_topic, 1);
        dynamic_gt_pub_ = nh_.advertise<sensor_simulator::DynamicObjectStateArray>(dynamic_gt_topic, 1);
        dynamic_collision_pub_ = nh_.advertise<std_msgs::Bool>(dynamic_collision_topic, 1);
        if (publish_dynamic_markers) {
            dynamic_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(dynamic_marker_topic, 1);
        }
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void renderDepthCallback(const ros::Time stamp);

    void renderLidarCallback(const ros::Time stamp);

    void timerMapCallback(const ros::TimerEvent &);

    void publishDynamicGroundTruth(
        const ros::Time& stamp, double scenario_time,
        const std::vector<dynamic_sim::ActorState>& actors,
        const std::vector<dynamic_sim::RenderDiagnostic>& diagnostics);

private:
    bool render_depth{false};
    bool render_lidar{false};
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    Eigen::Quaternionf quat_body_from_optical;
    Eigen::Vector3f pos;
    Eigen::Vector3f camera_position_body;

    CameraParams* camera;
    LidarParams* lidar;
    GridMap* grid_map;
    sensor_msgs::PointCloud2 output;
    
    ros::NodeHandle nh_;
    ros::Publisher image_pub_, instance_pub_, point_cloud_pub_;
    ros::Publisher point_cloud_body_pub_, point_cloud_optical_pub_, point_cloud_world_pub_;
    ros::Publisher camera_info_pub_;
    ros::Publisher dynamic_gt_pub_, dynamic_collision_pub_, dynamic_marker_pub_;
    ros::Publisher pcl_pub;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_, timer_map_;

    ros::Time next_depth_pub_time, next_lidar_pub_time;
    ros::Duration depth_pub_duration, lidar_pub_duration;
    double depth_time{0.0}, lidar_time{0.0};
    int depth_count{0}, lidar_count{0};
    std::string lidar_body_topic, lidar_optical_topic, lidar_world_topic;
    std::string world_frame_id, body_frame_id, camera_optical_frame_id;
    std::string dynamic_gt_topic, dynamic_collision_topic, dynamic_marker_topic;
    bool publish_dynamic_markers{false};
    float uav_collision_radius{0.3f};
    dynamic_sim::DynamicScenario dynamic_scenario;
    ros::Time scenario_start_stamp;
    ros::Time last_odom_stamp;
    bool scenario_clock_initialized{false};
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    // mocka::Maps map;
};



void SensorSimulator::renderDepthCallback(const ros::Time stamp) {
    if (!render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    Eigen::Vector3f camera_position_world = pos + quat * camera_position_body;
    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                             camera_position_world.x(), camera_position_world.y(), camera_position_world.z());
    cv::Mat depth_image;
    renderDepthImage(grid_map, camera, T_wc, depth_image);
    const double scenario_time = scenario_clock_initialized
        ? std::max(0.0, (stamp - scenario_start_stamp).toSec()) : 0.0;
    const auto actor_states = dynamic_scenario.sample(scenario_time);
    cv::Mat instance_image;
    const auto render_diagnostics = dynamic_sim::renderActorsIntoDepth(
        depth_image, instance_image, actor_states, camera_position_world, quat_wc,
        camera->fx, camera->fy, camera->cx, camera->cy, camera->max_depth_dist);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time += elapsed.count();
    depth_count++;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp;
    cv_image.header.frame_id = camera_optical_frame_id;
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
    sensor_msgs::Image ros_instance;
    cv_bridge::CvImage cv_instance;
    cv_instance.header = ros_image.header;
    cv_instance.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
    cv_instance.image = instance_image;
    cv_instance.toImageMsg(ros_instance);
    instance_pub_.publish(ros_instance);

    sensor_msgs::CameraInfo camera_info;
    camera_info.header = ros_image.header;
    camera_info.width = camera->image_width;
    camera_info.height = camera->image_height;
    camera_info.distortion_model = "plumb_bob";
    camera_info.D.assign(5, 0.0);
    std::fill(camera_info.K.begin(), camera_info.K.end(), 0.0);
    std::fill(camera_info.R.begin(), camera_info.R.end(), 0.0);
    std::fill(camera_info.P.begin(), camera_info.P.end(), 0.0);
    camera_info.K[0] = camera->fx;
    camera_info.K[2] = camera->cx;
    camera_info.K[4] = camera->fy;
    camera_info.K[5] = camera->cy;
    camera_info.K[8] = 1.0;
    camera_info.R[0] = camera_info.R[4] = camera_info.R[8] = 1.0;
    camera_info.P[0] = camera->fx;
    camera_info.P[2] = camera->cx;
    camera_info.P[5] = camera->fy;
    camera_info.P[6] = camera->cy;
    camera_info.P[10] = 1.0;
    camera_info_pub_.publish(camera_info);
    publishDynamicGroundTruth(stamp, scenario_time, actor_states, render_diagnostics);
}

void SensorSimulator::publishDynamicGroundTruth(
    const ros::Time& stamp, double scenario_time,
    const std::vector<dynamic_sim::ActorState>& actors,
    const std::vector<dynamic_sim::RenderDiagnostic>& diagnostics) {
    if (actors.size() != diagnostics.size()) {
        throw std::logic_error("actor/diagnostic size mismatch");
    }
    sensor_simulator::DynamicObjectStateArray array;
    array.header.stamp = stamp;
    array.header.frame_id = world_frame_id;
    array.scenario_id = dynamic_scenario.scenarioId();
    array.seed = dynamic_scenario.seed();
    array.scenario_time = scenario_time;
    bool any_collision = false;
    visualization_msgs::MarkerArray markers;
    for (std::size_t index = 0; index < actors.size(); ++index) {
        const auto& actor = actors[index];
        const auto& diagnostic = diagnostics[index];
        sensor_simulator::DynamicObjectState message;
        message.header = array.header;
        message.object_id = actor.object_id;
        message.object_type = dynamic_sim::shapeName(actor.shape);
        message.active = actor.active;
        message.visible = diagnostic.visible;
        message.occluded = diagnostic.occluded;
        message.inside_image = diagnostic.inside_image;
        message.collision = dynamic_sim::actorCollidesWithUav(actor, pos, uav_collision_radius);
        any_collision = any_collision || message.collision;
        message.position_world.x = actor.position_world.x();
        message.position_world.y = actor.position_world.y();
        message.position_world.z = actor.position_world.z();
        message.velocity_world.x = actor.velocity_world.x();
        message.velocity_world.y = actor.velocity_world.y();
        message.velocity_world.z = actor.velocity_world.z();
        message.acceleration_world.x = actor.acceleration_world.x();
        message.acceleration_world.y = actor.acceleration_world.y();
        message.acceleration_world.z = actor.acceleration_world.z();
        message.radius = actor.radius;
        message.height = actor.height;
        message.projected_u = diagnostic.projected_u;
        message.projected_v = diagnostic.projected_v;
        message.expected_surface_depth = diagnostic.expected_surface_depth;
        message.observed_depth = diagnostic.observed_depth;
        message.depth_error = diagnostic.depth_error;
        message.rendered_pixel_count = diagnostic.rendered_pixel_count;
        array.objects.push_back(message);

        if (publish_dynamic_markers && actor.active) {
            visualization_msgs::Marker shape;
            shape.header = array.header;
            shape.ns = "dynamic_actor";
            shape.id = static_cast<int>(2 * actor.object_id);
            shape.type = actor.shape == dynamic_sim::Shape::Sphere
                ? visualization_msgs::Marker::SPHERE : visualization_msgs::Marker::CYLINDER;
            shape.action = visualization_msgs::Marker::ADD;
            shape.pose.position = message.position_world;
            shape.pose.orientation.w = 1.0;
            shape.scale.x = shape.scale.y = 2.0f * actor.radius;
            shape.scale.z = actor.height;
            shape.color.r = message.visible ? 0.1f : 1.0f;
            shape.color.g = message.visible ? 0.9f : 0.5f;
            shape.color.b = 0.1f;
            shape.color.a = 0.8f;
            shape.lifetime = depth_pub_duration * 2.0;
            markers.markers.push_back(shape);

            visualization_msgs::Marker velocity;
            velocity.header = array.header;
            velocity.ns = "dynamic_velocity";
            velocity.id = static_cast<int>(2 * actor.object_id + 1);
            velocity.type = visualization_msgs::Marker::ARROW;
            velocity.action = visualization_msgs::Marker::ADD;
            geometry_msgs::Point start = message.position_world;
            geometry_msgs::Point finish = start;
            finish.x += actor.velocity_world.x();
            finish.y += actor.velocity_world.y();
            finish.z += actor.velocity_world.z();
            velocity.points = {start, finish};
            velocity.scale.x = 0.06;
            velocity.scale.y = 0.12;
            velocity.scale.z = 0.12;
            velocity.color.r = 0.1;
            velocity.color.g = 0.3;
            velocity.color.b = 1.0;
            velocity.color.a = 0.9;
            velocity.lifetime = depth_pub_duration * 2.0;
            markers.markers.push_back(velocity);
        }
    }
    array.uav_collision = any_collision;
    dynamic_gt_pub_.publish(array);
    std_msgs::Bool collision;
    collision.data = any_collision;
    dynamic_collision_pub_.publish(collision);
    if (publish_dynamic_markers) dynamic_marker_pub_.publish(markers);
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    if (pcl_pub.getNumSubscribers() > 0)
        pcl_pub.publish(output);    
}

void SensorSimulator::renderLidarCallback(const ros::Time stamp) {
    if (!render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time += elapsed.count();
    lidar_count++;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
    if (point_cloud_pub_.getNumSubscribers() > 0) {
        ROS_WARN_THROTTLE(5.0,
            "Deprecated /lidar_points has body-frame values but historical frame_id=odom; use %s or %s",
            lidar_body_topic.c_str(), lidar_optical_topic.c_str());
    }

    sensor_msgs::PointCloud2 body_output;
    pcl::toROSMsg(lidar_points, body_output);
    body_output.header.stamp = stamp;
    body_output.header.frame_id = body_frame_id;
    point_cloud_body_pub_.publish(body_output);

    pcl::PointCloud<pcl::PointXYZ> optical_points;
    pcl::PointCloud<pcl::PointXYZ> world_points;
    optical_points.points.reserve(lidar_points.size());
    world_points.points.reserve(lidar_points.size());
    for (const auto& point : lidar_points.points) {
        const Eigen::Vector3f point_body(point.x, point.y, point.z);
        const Eigen::Vector3f point_optical =
            quat_body_from_optical.inverse() * (point_body - camera_position_body);
        const Eigen::Vector3f point_world = quat * point_body + pos;
        // The optical topic is a camera-facing perception input. A 360-degree
        // lidar also has points behind the camera; do not label those as a
        // forward optical cloud.
        if (point_optical.z() > 0.0f) {
            optical_points.emplace_back(point_optical.x(), point_optical.y(), point_optical.z());
        }
        world_points.emplace_back(point_world.x(), point_world.y(), point_world.z());
    }
    optical_points.width = optical_points.size();
    optical_points.height = 1;
    optical_points.is_dense = true;
    world_points.width = world_points.size();
    world_points.height = 1;
    world_points.is_dense = true;

    sensor_msgs::PointCloud2 optical_output;
    pcl::toROSMsg(optical_points, optical_output);
    optical_output.header.stamp = stamp;
    optical_output.header.frame_id = camera_optical_frame_id;
    point_cloud_optical_pub_.publish(optical_output);

    sensor_msgs::PointCloud2 world_output;
    pcl::toROSMsg(world_points, world_output);
    world_output.header.stamp = stamp;
    world_output.header.frame_id = world_frame_id;
    point_cloud_world_pub_.publish(world_output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    if (!last_odom_stamp.isZero() && msg->header.stamp <= last_odom_stamp) {
        throw std::runtime_error("odometry simulation time must be strictly increasing");
    }
    last_odom_stamp = msg->header.stamp;
    if (!scenario_clock_initialized) {
        scenario_start_stamp = msg->header.stamp;
        scenario_clock_initialized = true;
    } else if (msg->header.stamp < scenario_start_stamp) {
        throw std::runtime_error("odometry simulation time moved backwards");
    }
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    geometry_msgs::TransformStamped world_from_body;
    world_from_body.header.stamp = msg->header.stamp;
    world_from_body.header.frame_id = world_frame_id;
    world_from_body.child_frame_id = body_frame_id;
    world_from_body.transform.translation.x = pos.x();
    world_from_body.transform.translation.y = pos.y();
    world_from_body.transform.translation.z = pos.z();
    world_from_body.transform.rotation = msg->pose.pose.orientation;

    geometry_msgs::TransformStamped body_from_optical;
    body_from_optical.header.stamp = msg->header.stamp;
    body_from_optical.header.frame_id = body_frame_id;
    body_from_optical.child_frame_id = camera_optical_frame_id;
    body_from_optical.transform.translation.x = camera_position_body.x();
    body_from_optical.transform.translation.y = camera_position_body.y();
    body_from_optical.transform.translation.z = camera_position_body.z();
    body_from_optical.transform.rotation.x = quat_body_from_optical.x();
    body_from_optical.transform.rotation.y = quat_body_from_optical.y();
    body_from_optical.transform.rotation.z = quat_body_from_optical.z();
    body_from_optical.transform.rotation.w = quat_body_from_optical.w();
    tf_broadcaster_.sendTransform(world_from_body);
    tf_broadcaster_.sendTransform(body_from_optical);

    ros::Time tnow = ros::Time::now();

    // 避免仿真odom消息中断，导致时间差太大
    if (fabs((tnow - next_depth_pub_time).toSec()) > 10 * depth_pub_duration.toSec())
        next_depth_pub_time = tnow;
    if (fabs((tnow - next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration.toSec())
        next_lidar_pub_time = tnow;

    if (tnow >= next_depth_pub_time){
        next_depth_pub_time += depth_pub_duration;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time){
        next_lidar_pub_time += lidar_pub_duration;
        renderLidarCallback(msg->header.stamp);
    }
    ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration || render_duration > lidar_pub_duration){
        // Performance reference: should take < 1 ms on 3060 GPU & Ubuntu 20.04
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time / (depth_count + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time / (lidar_count + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
