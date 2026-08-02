#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <stdexcept>
#include <memory>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

void prepareFreshPath(const std::string &path, bool print=false)
{
    if (fs::exists(path))
        throw std::runtime_error("refusing to delete or reuse existing path: " + path);
    fs::create_directories(path);
    if (print)
        std::cout << "Created new dataset directory: " << path << std::endl;
}

void savePointCloudAsPLY(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::string &path)
{
    if (pcl::io::savePLYFileBinary(path, *cloud) == -1)
        std::cerr << "Failed to save ply file to " << path << std::endl;
}

void saveDepthAs16BitPNG(const cv::Mat &depth_float, float max_depth_dist, const std::string &filepath)
{
    cv::Mat depth_scaled;
    depth_scaled = depth_float / max_depth_dist; // 归一化0~1

    // clip [0,1]
    cv::threshold(depth_scaled, depth_scaled, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(depth_scaled, depth_scaled, 0.0, 0.0, cv::THRESH_TOZERO);

    // 转成uint16
    depth_scaled.convertTo(depth_scaled, CV_16UC1, 65535.0);

    cv::imwrite(filepath, depth_scaled);
}

Eigen::Quaternionf RPY2Quat(float roll_deg, float pitch_deg, float yaw_deg)
{
    float roll = roll_deg * M_PI / 180.0f;
    float pitch = pitch_deg * M_PI / 180.0f;
    float yaw = yaw_deg * M_PI / 180.0f;
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
    return yawAngle * pitchAngle * rollAngle;
}

void printProgressBar(int current, int total, int bar_width = 50)
{
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%";
    std::cout.flush();
}

int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
    std::string output_override;
    bool overwrite = false;
    bool confirmed = false;
    int map_seed_override = -1;
    int pose_seed_override = -1;
    int actor_seed_override = -1;
    int env_num_override = -1;
    int image_num_override = -1;
    std::string authority_map_root;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--save-path" && i + 1 < argc) output_override = argv[++i];
        else if (argument == "--map-seed" && i + 1 < argc) map_seed_override = std::stoi(argv[++i]);
        else if (argument == "--pose-seed" && i + 1 < argc) pose_seed_override = std::stoi(argv[++i]);
        else if (argument == "--actor-seed" && i + 1 < argc) actor_seed_override = std::stoi(argv[++i]);
        else if (argument == "--env-num" && i + 1 < argc) env_num_override = std::stoi(argv[++i]);
        else if (argument == "--image-num" && i + 1 < argc) image_num_override = std::stoi(argv[++i]);
        else if (argument == "--authority-map-root" && i + 1 < argc) authority_map_root = argv[++i];
        else if (argument == "--overwrite") overwrite = true;
        else if (argument == "--yes") confirmed = true;
        else throw std::invalid_argument("unknown or incomplete argument: " + argument);
    }

    // 1. 相机参数
    CameraParams camera;
    camera.fx = config["camera"]["fx"].as<float>();
    camera.fy = config["camera"]["fy"].as<float>();
    camera.cx = config["camera"]["cx"].as<float>();
    camera.cy = config["camera"]["cy"].as<float>();
    camera.image_width = config["camera"]["image_width"].as<int>();
    camera.image_height = config["camera"]["image_height"].as<int>();
    camera.max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
    camera.normalize_depth = config["camera"]["normalize_depth"].as<bool>();
    float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
    Eigen::AngleAxisf angle_axis(pitch, Eigen::Vector3f::UnitY());
    Eigen::Quaternionf quat_bc(angle_axis);

    // 2. 地图参数
    float resolution = config["resolution"].as<float>();
    int occupy_threshold = config["occupy_threshold"].as<int>();
    int map_seed = config["map_seed"] ? config["map_seed"].as<int>() : config["seed"].as<int>();
    int pose_seed = config["pose_seed"] ? config["pose_seed"].as<int>() : map_seed + 100000;
    int actor_seed = config["actor_seed"] ? config["actor_seed"].as<int>() : map_seed + 200000;
    if (map_seed_override >= 0) map_seed = map_seed_override;
    if (pose_seed_override >= 0) pose_seed = pose_seed_override;
    if (actor_seed_override >= 0) actor_seed = actor_seed_override;
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    double scale = 1 / resolution;
    sizeX *= scale;
    sizeY *= scale;
    sizeZ *= scale;

    // 3. 数据集参数
    std::string final_path = output_override.empty() ? config["save_path"].as<std::string>() : output_override;
    final_path = fs::absolute(final_path).lexically_normal().string();
    if (fs::exists(final_path)) {
        std::cout << "Existing output target: " << final_path << std::endl;
        if (!overwrite) throw std::runtime_error("output exists; pass --overwrite explicitly");
        if (!confirmed) {
            std::cout << "Type OVERWRITE to replace the listed target: ";
            std::string response; std::cin >> response;
            if (response != "OVERWRITE") throw std::runtime_error("overwrite not confirmed");
        }
    }
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::string save_path = final_path + ".staging-" + std::to_string(nonce) + "/";
    prepareFreshPath(save_path, true);
    int env_num = config["env_num"].as<int>();
    int image_num = config["image_num"].as<int>();
    if (env_num_override >= 0) env_num = env_num_override;
    if (image_num_override >= 0) image_num = image_num_override;
    if (env_num <= 0 || image_num < 2)
        throw std::invalid_argument("env-num must be positive and image-num must be at least two");
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();

    // 中心对齐，计算偏移量
    int dataset_num = env_num * image_num;
    float x_min = -x_range / 2.0f;
    float y_min = -y_range / 2.0f;

    std::cout << "地图范围 (m): "
              << "X: [" << -sizeX * resolution / 2.0 << ", " << sizeX * resolution / 2.0 << "], "
              << "Y: [" << -sizeY * resolution / 2.0 << ", " << sizeY * resolution / 2.0 << "], "
              << "Z: [" << 0 << ", " << sizeZ * resolution << "]" << std::endl;

    std::cout << "采集范围 (m): "
              << "X: [" << x_min << ", " << x_min + x_range << "], "
              << "Y: [" << y_min << ", " << y_min + y_range << "], "
              << "Z: [" << z_min << ", " << z_max << "]" << std::endl;

    std::cout << "角度范围 (度): "
              << "Roll: [" << -roll_range << ", " << roll_range << "], "
              << "Pitch: [" << -pitch_range << ", " << pitch_range << "], "
              << "Yaw: [0, 360]" << std::endl;

    // 收集所有数据
    // Explicit pose seed is part of the dataset contract; random_device is forbidden.
    std::mt19937 generator(static_cast<std::mt19937::result_type>(pose_seed));
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f); // 均值0，标准差1
    std::uniform_real_distribution<float> uniform_uniform(0.0f, 1.0f);
    std::vector<std::string> authority_hashes;
    for (int map_i = 0; map_i < env_num; ++map_i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX = sizeX;
        info.sizeY = sizeY;
        info.sizeZ = sizeZ;
        info.seed = map_seed + map_i; // 每个环境使用不同的显式随机种子
        info.scale = scale;
        info.cloud = cloud;

        mocka::Maps map;
        map.setParam(config);
        map.setInfo(info);
        map.generate(config["maze_type"].as<int>());

        // New versioned datasets may load map-<id>/canonical occupancy.
        // Legacy generation remains available but is explicitly non-authoritative.
        std::unique_ptr<GridMap> grid_map;
        if (authority_map_root.empty())
            grid_map.reset(new GridMap(cloud, resolution, occupy_threshold));
        else
            grid_map.reset(new GridMap(
                authority_map_root + "/map-" + std::to_string(map_i)));
        authority_hashes.push_back(grid_map->authorityHash());

        // 保存地图 (先滤波)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareFreshPath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        for (int image_i = 0; image_i < image_num; ++image_i)
        {
            Eigen::Vector3f pos;
            float dist;
            do{
                pos.x() = x_min + uniform_uniform(generator) * x_range;
                pos.y() = y_min + uniform_uniform(generator) * y_range;
                pos.z() = z_min + uniform_uniform(generator) * (z_max - z_min);
                pcl::PointXYZ searchPoint(pos.x(), pos.y(), pos.z());
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                int found_num = kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
                dist = sqrt(pointNKNSquaredDistance[0]);
            } while (dist < safe_dist);

            float roll = normal_distribution(generator) * roll_range / 3.0f;   // 3 * sigmoid = range
            float pitch = normal_distribution(generator) * pitch_range / 3.0f; // 3 * sigmoid = range
            float yaw = uniform_uniform(generator) * 360.0f;

            Eigen::Quaternionf quat = RPY2Quat(roll, pitch, yaw);
            Eigen::Quaternionf quat_wc = quat * quat_bc;

            cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                                     pos.x(), pos.y(), pos.z());

            cv::Mat depth_image;
            renderDepthImage(grid_map.get(), &camera, T_wc, depth_image);

            std::string filename = image_path + "/img_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, filename);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        grid_map->freeGridMap();
    }

    std::ofstream metadata(save_path + "generation_metadata.yaml");
    metadata << "completion_status: complete\n"
             << "map_seed: " << map_seed << "\n"
             << "pose_seed: " << pose_seed << "\n"
             << "actor_seed: " << actor_seed << "\n"
             << "env_num: " << env_num << "\n"
             << "image_num: " << image_num << "\n"
             << "static_geometry_authority_version: "
             << (authority_map_root.empty()
                 ? "none_legacy_non_authoritative"
                 : "static_geometry_authority_v1") << "\n"
             << "authority_map_root: " << authority_map_root << "\n"
             << "authority_hashes:\n";
    for (size_t index = 0; index < authority_hashes.size(); ++index)
        metadata << "  " << index << ": \""
                 << authority_hashes[index] << "\"\n";
    metadata
             << "reachability_status: pending_external_validation\n";
    metadata.close();
    if (fs::exists(final_path)) {
        const std::string backup = final_path + ".backup-" + std::to_string(nonce);
        fs::rename(final_path, backup);
        std::cout << "Recoverable overwrite backup: " << backup << std::endl;
    }
    fs::rename(fs::path(save_path).parent_path(), final_path);
    std::cout << "\nDataset generation completed atomically: " << final_path << std::endl;

    return 0;
}
