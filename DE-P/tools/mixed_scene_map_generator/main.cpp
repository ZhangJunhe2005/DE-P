#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "maps.hpp"

namespace fs = std::filesystem;

namespace {

std::string argument(int argc, char **argv, const std::string &name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  throw std::invalid_argument("missing argument " + name);
}

void requireFinite(const pcl::PointCloud<pcl::PointXYZ> &cloud) {
  if (cloud.empty()) throw std::runtime_error("generated point cloud is empty");
  for (const auto &point : cloud) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)
        || !std::isfinite(point.z)) {
      throw std::runtime_error("generated point cloud contains non-finite data");
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const fs::path config_path = fs::absolute(argument(argc, argv, "--config"));
    const fs::path output_path = fs::absolute(argument(argc, argv, "--output"));
    if (fs::exists(output_path)) {
      throw std::runtime_error("refusing to overwrite " + output_path.string());
    }
    const YAML::Node config = YAML::LoadFile(config_path.string());
    const int type = config["maze_type"].as<int>();
    const int seed = config["seed"].as<int>();
    const double resolution = config["resolution"].as<double>();
    if (type < 1 || type > 7 || std::abs(resolution - 0.1) > 1e-12) {
      throw std::invalid_argument("unsupported maze_type or resolution");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    mocka::Maps::BasicInfo info;
    info.sizeX = static_cast<int>(
        std::llround(config["x_length"].as<double>() / resolution));
    info.sizeY = static_cast<int>(
        std::llround(config["y_length"].as<double>() / resolution));
    info.sizeZ = static_cast<int>(
        std::llround(config["z_length"].as<double>() / resolution));
    info.seed = seed;
    info.scale = 1.0 / resolution;
    info.cloud = cloud;

    mocka::Maps map;
    map.setParam(config);
    map.setInfo(info);
    map.generate(type);

    const auto translate = config["post_translate"];
    const float tx = translate ? translate[0].as<float>() : 0.0f;
    const float ty = translate ? translate[1].as<float>() : 0.0f;
    const float tz = translate ? translate[2].as<float>() : 0.0f;
    if (tx != 0.0f || ty != 0.0f || tz != 0.0f) {
      for (auto &point : *cloud) {
        point.x += tx;
        point.y += ty;
        point.z += tz;
      }
    }
    // A versioned profile may add a deterministic thin-wall annex.  The
    // original map remains untouched; the annex supplies a bounded, connected
    // free-space region where the frozen natural-occlusion constructor can
    // place a camera and an actor on opposite sides of a one-voxel surface.
    if (config["occlusion_annex"]
        && config["occlusion_annex"].as<bool>()) {
      const float resolution_f = static_cast<float>(resolution);
      const float half_x = .5f * config["x_length"].as<float>();
      const float panel_x = half_x + 5.0f;
      const float annex_length = config["occlusion_annex_length_m"]
          ? config["occlusion_annex_length_m"].as<float>() : 20.0f;
      const float panel_width = config["occlusion_panel_width_m"]
          ? config["occlusion_panel_width_m"].as<float>() : 0.2f;
      const float panel_half_y =
          .5f * std::max(resolution_f, panel_width - resolution_f);
      const float panel_height = 6.0f;
      for (float y = -panel_half_y; y <= panel_half_y + 1e-6f;
           y += resolution_f) {
        for (float z = 0.0f; z <= panel_height + 1e-6f;
             z += resolution_f) {
          cloud->emplace_back(panel_x, y, z);
        }
      }
      // Floor strip connects the annex to the original map's free space.
      for (float x = half_x; x <= panel_x + annex_length + 1e-6f;
           x += resolution_f) {
        for (float y = -6.0f; y <= 6.0f + 1e-6f; y += resolution_f) {
          cloud->emplace_back(x, y, 0.0f);
        }
      }
      // Non-authoritative visualization is optional, but canonical bounds
      // need deterministic clearance beyond both panel faces.
      cloud->emplace_back(panel_x + annex_length, -6.0f, 8.0f);
      cloud->emplace_back(panel_x + annex_length, 6.0f, 8.0f);
    }
    requireFinite(*cloud);

    fs::create_directories(output_path.parent_path());
    const fs::path staging = output_path.string() + ".tmp";
    if (fs::exists(staging)) fs::remove(staging);
    std::ofstream stream(staging, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot create output");
    const std::uint64_t count = cloud->size();
    stream.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto &point : *cloud) {
      const float xyz[3] = {point.x, point.y, point.z};
      stream.write(reinterpret_cast<const char *>(xyz), sizeof(xyz));
    }
    stream.flush();
    if (!stream) throw std::runtime_error("failed while writing output");
    stream.close();
    fs::rename(staging, output_path);
    std::cout << "{\"status\":\"PASS\",\"maze_type\":" << type
              << ",\"seed\":" << seed << ",\"point_count\":" << count
              << ",\"output\":\"" << output_path.string() << "\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MIXED_SCENE_MAP_GENERATOR_ERROR: " << error.what() << "\n";
    return 2;
  }
}
