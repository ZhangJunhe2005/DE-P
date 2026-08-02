#ifndef SENSOR_SIMULATOR_DYNAMIC_ACTOR_HPP
#define SENSOR_SIMULATOR_DYNAMIC_ACTOR_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dynamic_sim {

enum class Shape { Sphere, VerticalCylinder };
enum class Trajectory { Stationary, Linear, DelayedLinear, WaypointPingPong };

struct ActorState {
    std::uint32_t object_id{0};
    Shape shape{Shape::Sphere};
    bool active{false};
    Eigen::Vector3f position_world{Eigen::Vector3f::Zero()};
    Eigen::Vector3f velocity_world{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration_world{Eigen::Vector3f::Zero()};
    float radius{0.0f};
    float height{0.0f};
};

struct RenderDiagnostic {
    bool visible{false};
    bool occluded{false};
    bool inside_image{false};
    float projected_u{-1.0f};
    float projected_v{-1.0f};
    float expected_surface_depth{0.0f};
    float observed_depth{0.0f};
    float depth_error{0.0f};
    std::uint32_t rendered_pixel_count{0};
};

class DynamicActor {
public:
    static DynamicActor fromYaml(const YAML::Node& node);
    ActorState sample(double scenario_time) const;
    std::uint32_t id() const { return object_id_; }
    bool enabled() const { return enabled_; }

private:
    std::uint32_t object_id_{0};
    Shape shape_{Shape::Sphere};
    Trajectory trajectory_{Trajectory::Stationary};
    bool enabled_{true};
    float radius_{0.0f};
    float height_{0.0f};
    double start_time_{0.0};
    double end_time_{0.0};
    float waypoint_speed_{0.0f};
    Eigen::Vector3f initial_position_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f linear_velocity_{Eigen::Vector3f::Zero()};
    std::vector<Eigen::Vector3f> waypoints_;
};

class DynamicScenario {
public:
    static DynamicScenario load(const std::string& path);
    std::vector<ActorState> sample(double scenario_time) const;
    const std::string& scenarioId() const { return scenario_id_; }
    std::uint32_t seed() const { return seed_; }
    bool enabled() const { return enabled_; }

private:
    bool enabled_{false};
    std::string scenario_id_{"no_target"};
    std::uint32_t seed_{0};
    std::vector<DynamicActor> actors_;
};

std::string shapeName(Shape shape);

std::vector<RenderDiagnostic> renderActorsIntoDepth(
    cv::Mat& depth_image,
    cv::Mat& instance_image,
    const std::vector<ActorState>& states,
    const Eigen::Vector3f& camera_position_world,
    const Eigen::Quaternionf& rotation_world_from_native_camera,
    float fx, float fy, float cx, float cy,
    float max_depth);

bool actorCollidesWithUav(
    const ActorState& actor,
    const Eigen::Vector3f& uav_position_world,
    float uav_radius);

}  // namespace dynamic_sim

#endif
