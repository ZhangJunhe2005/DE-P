#include "dynamic_actor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace dynamic_sim {
namespace {

Eigen::Vector3f vector3(const YAML::Node& node, const std::string& name) {
    if (!node || !node.IsSequence() || node.size() != 3) {
        throw std::invalid_argument(name + " must contain exactly three finite values");
    }
    Eigen::Vector3f value(node[0].as<float>(), node[1].as<float>(), node[2].as<float>());
    if (!value.allFinite()) throw std::invalid_argument(name + " contains NaN/Inf");
    return value;
}

float positive(const YAML::Node& node, const std::string& name) {
    if (!node) throw std::invalid_argument("missing " + name);
    const float value = node.as<float>();
    if (!std::isfinite(value) || value <= 0.0f) {
        throw std::invalid_argument(name + " must be finite and positive");
    }
    return value;
}

bool solveQuadratic(float a, float b, float c, float& near_value, float& far_value) {
    if (std::abs(a) < 1e-9f) return false;
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return false;
    const float root = std::sqrt(std::max(0.0f, discriminant));
    near_value = (-b - root) / (2.0f * a);
    far_value = (-b + root) / (2.0f * a);
    if (near_value > far_value) std::swap(near_value, far_value);
    return true;
}

float sphereDepth(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
                  const Eigen::Vector3f& center, float radius) {
    const Eigen::Vector3f offset = origin - center;
    float near_value, far_value;
    if (!solveQuadratic(direction.squaredNorm(), 2.0f * direction.dot(offset),
                        offset.squaredNorm() - radius * radius, near_value, far_value)) {
        return std::numeric_limits<float>::infinity();
    }
    if (near_value > 0.0f) return near_value;
    if (far_value > 0.0f) return far_value;
    return std::numeric_limits<float>::infinity();
}

float cylinderDepth(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
                    const Eigen::Vector3f& center, float radius, float height) {
    float best = std::numeric_limits<float>::infinity();
    const float ox = origin.x() - center.x();
    const float oy = origin.y() - center.y();
    float near_value, far_value;
    if (solveQuadratic(direction.x() * direction.x() + direction.y() * direction.y(),
                       2.0f * (direction.x() * ox + direction.y() * oy),
                       ox * ox + oy * oy - radius * radius, near_value, far_value)) {
        for (float candidate : {near_value, far_value}) {
            if (candidate <= 0.0f) continue;
            const float z = origin.z() + candidate * direction.z();
            if (z >= center.z() - 0.5f * height && z <= center.z() + 0.5f * height) {
                best = std::min(best, candidate);
            }
        }
    }
    if (std::abs(direction.z()) > 1e-9f) {
        for (float z_cap : {center.z() - 0.5f * height, center.z() + 0.5f * height}) {
            const float candidate = (z_cap - origin.z()) / direction.z();
            if (candidate <= 0.0f) continue;
            const Eigen::Vector3f point = origin + candidate * direction;
            if (std::hypot(point.x() - center.x(), point.y() - center.y()) <= radius) {
                best = std::min(best, candidate);
            }
        }
    }
    return best;
}

}  // namespace

DynamicActor DynamicActor::fromYaml(const YAML::Node& node) {
    if (!node || !node.IsMap()) throw std::invalid_argument("actor must be a map");
    DynamicActor actor;
    actor.object_id_ = node["id"].as<std::uint32_t>();
    if (actor.object_id_ == 0) throw std::invalid_argument("actor id must be positive");
    actor.enabled_ = node["enabled"] ? node["enabled"].as<bool>() : true;
    const std::string shape = node["shape"].as<std::string>();
    if (shape == "sphere") actor.shape_ = Shape::Sphere;
    else if (shape == "vertical_cylinder") actor.shape_ = Shape::VerticalCylinder;
    else throw std::invalid_argument("actor shape must be sphere or vertical_cylinder");
    actor.radius_ = positive(node["radius"], "actor radius");
    actor.height_ = actor.shape_ == Shape::VerticalCylinder
        ? positive(node["height"], "cylinder height") : 2.0f * actor.radius_;
    actor.initial_position_ = vector3(node["initial_position_world"], "initial_position_world");
    const YAML::Node trajectory = node["trajectory"];
    if (!trajectory || !trajectory.IsMap()) throw std::invalid_argument("missing actor trajectory");
    const std::string type = trajectory["type"].as<std::string>();
    if (type == "stationary") actor.trajectory_ = Trajectory::Stationary;
    else if (type == "linear") actor.trajectory_ = Trajectory::Linear;
    else if (type == "delayed_linear") actor.trajectory_ = Trajectory::DelayedLinear;
    else if (type == "waypoint_ping_pong") actor.trajectory_ = Trajectory::WaypointPingPong;
    else throw std::invalid_argument("unsupported trajectory type: " + type);
    actor.start_time_ = trajectory["start_time"].as<double>();
    actor.end_time_ = trajectory["end_time"].as<double>();
    if (!std::isfinite(actor.start_time_) || !std::isfinite(actor.end_time_) ||
        actor.start_time_ < 0.0 || actor.end_time_ <= actor.start_time_) {
        throw std::invalid_argument("trajectory requires 0 <= start_time < end_time");
    }
    if (actor.trajectory_ == Trajectory::Linear || actor.trajectory_ == Trajectory::DelayedLinear) {
        actor.linear_velocity_ = vector3(trajectory["velocity_world"], "velocity_world");
    }
    if (actor.trajectory_ == Trajectory::WaypointPingPong) {
        const YAML::Node waypoints = trajectory["waypoints_world"];
        if (!waypoints || !waypoints.IsSequence() || waypoints.size() < 2) {
            throw std::invalid_argument("waypoint_ping_pong needs at least two waypoints");
        }
        for (const auto& waypoint : waypoints) {
            actor.waypoints_.push_back(vector3(waypoint, "waypoint"));
        }
        actor.waypoint_speed_ = positive(trajectory["speed"], "waypoint speed");
    }
    return actor;
}

ActorState DynamicActor::sample(double scenario_time) const {
    ActorState state;
    state.object_id = object_id_;
    state.shape = shape_;
    state.radius = radius_;
    state.height = height_;
    state.position_world = initial_position_;
    state.active = enabled_ && scenario_time >= start_time_ && scenario_time <= end_time_;
    const double clamped_time = std::clamp(scenario_time, start_time_, end_time_);
    const float elapsed = static_cast<float>(clamped_time - start_time_);
    if (trajectory_ == Trajectory::Linear || trajectory_ == Trajectory::DelayedLinear) {
        state.position_world = initial_position_ + elapsed * linear_velocity_;
        if (state.active) state.velocity_world = linear_velocity_;
    } else if (trajectory_ == Trajectory::WaypointPingPong) {
        std::vector<float> lengths;
        float total = 0.0f;
        for (std::size_t index = 1; index < waypoints_.size(); ++index) {
            lengths.push_back((waypoints_[index] - waypoints_[index - 1]).norm());
            total += lengths.back();
        }
        if (total <= 1e-6f) throw std::runtime_error("waypoint path has zero total length");
        float distance = std::fmod(elapsed * waypoint_speed_, 2.0f * total);
        bool reverse = distance > total;
        if (reverse) distance = 2.0f * total - distance;
        for (std::size_t index = 0; index < lengths.size(); ++index) {
            if (distance <= lengths[index] || index + 1 == lengths.size()) {
                Eigen::Vector3f direction = (waypoints_[index + 1] - waypoints_[index]).normalized();
                state.position_world = waypoints_[index] + direction * std::min(distance, lengths[index]);
                if (state.active) state.velocity_world = (reverse ? -1.0f : 1.0f) * waypoint_speed_ * direction;
                break;
            }
            distance -= lengths[index];
        }
    }
    return state;
}

DynamicScenario DynamicScenario::load(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node node = root["dynamic_scenario"];
    if (!node || !node.IsMap()) throw std::invalid_argument("missing dynamic_scenario map");
    DynamicScenario scenario;
    scenario.enabled_ = node["enabled"].as<bool>();
    scenario.scenario_id_ = node["scenario_id"].as<std::string>();
    scenario.seed_ = node["seed"].as<std::uint32_t>();
    if (scenario.scenario_id_.empty()) throw std::invalid_argument("scenario_id cannot be empty");
    const YAML::Node actors = node["actors"];
    if (!actors || !actors.IsSequence()) throw std::invalid_argument("actors must be a sequence");
    std::set<std::uint32_t> ids;
    for (const auto& actor_node : actors) {
        DynamicActor actor = DynamicActor::fromYaml(actor_node);
        if (!ids.insert(actor.id()).second) throw std::invalid_argument("duplicate actor id");
        scenario.actors_.push_back(actor);
    }
    if (!scenario.enabled_ && !scenario.actors_.empty()) {
        throw std::invalid_argument("disabled no-target scenario must not define actors");
    }
    return scenario;
}

std::vector<ActorState> DynamicScenario::sample(double scenario_time) const {
    std::vector<ActorState> result;
    if (!enabled_) return result;
    result.reserve(actors_.size());
    for (const auto& actor : actors_) result.push_back(actor.sample(scenario_time));
    return result;
}

std::string shapeName(Shape shape) {
    return shape == Shape::Sphere ? "sphere" : "vertical_cylinder";
}

std::vector<RenderDiagnostic> renderActorsIntoDepth(
    cv::Mat& depth_image, cv::Mat& instance_image,
    const std::vector<ActorState>& states,
    const Eigen::Vector3f& camera_position_world,
    const Eigen::Quaternionf& rotation_world_from_native_camera,
    float fx, float fy, float cx, float cy, float max_depth) {
    if (depth_image.type() != CV_32FC1) throw std::invalid_argument("depth image must be CV_32FC1");
    instance_image = cv::Mat(depth_image.rows, depth_image.cols, CV_32SC1, cv::Scalar(0));
    std::vector<RenderDiagnostic> diagnostics(states.size());
    const Eigen::Quaternionf rotation_native_from_world = rotation_world_from_native_camera.inverse();
    for (std::size_t index = 0; index < states.size(); ++index) {
        const auto& actor = states[index];
        if (!actor.active) continue;
        const Eigen::Vector3f center_native = rotation_native_from_world *
            (actor.position_world - camera_position_world);
        if (center_native.x() > 0.0f) {
            diagnostics[index].projected_u = cx - fx * center_native.y() / center_native.x();
            diagnostics[index].projected_v = cy - fy * center_native.z() / center_native.x();
            diagnostics[index].inside_image = diagnostics[index].projected_u >= 0.0f &&
                diagnostics[index].projected_u < depth_image.cols &&
                diagnostics[index].projected_v >= 0.0f && diagnostics[index].projected_v < depth_image.rows;
        }
        for (int v = 0; v < depth_image.rows; ++v) {
            for (int u = 0; u < depth_image.cols; ++u) {
                const Eigen::Vector3f direction_native(
                    1.0f, -(u - cx) / fx, -(v - cy) / fy);
                const Eigen::Vector3f direction_world = rotation_world_from_native_camera * direction_native;
                const float candidate = actor.shape == Shape::Sphere
                    ? sphereDepth(camera_position_world, direction_world, actor.position_world, actor.radius)
                    : cylinderDepth(camera_position_world, direction_world, actor.position_world,
                                    actor.radius, actor.height);
                if (std::isfinite(candidate) && candidate > 0.0f && candidate < max_depth &&
                    candidate < depth_image.at<float>(v, u)) {
                    depth_image.at<float>(v, u) = candidate;
                    instance_image.at<std::int32_t>(v, u) =
                        static_cast<std::int32_t>(actor.object_id);
                }
            }
        }
    }
    for (std::size_t index = 0; index < states.size(); ++index)
        diagnostics[index].rendered_pixel_count = static_cast<std::uint32_t>(
            cv::countNonZero(instance_image == static_cast<std::int32_t>(
                states[index].object_id)));
    for (std::size_t index = 0; index < states.size(); ++index) {
        auto& diagnostic = diagnostics[index];
        diagnostic.visible = states[index].active && diagnostic.rendered_pixel_count > 0;
        diagnostic.occluded = states[index].active && diagnostic.inside_image && !diagnostic.visible;
        if (!diagnostic.inside_image) continue;
        const int u = std::clamp(static_cast<int>(std::lround(diagnostic.projected_u)), 0, depth_image.cols - 1);
        const int v = std::clamp(static_cast<int>(std::lround(diagnostic.projected_v)), 0, depth_image.rows - 1);
        const Eigen::Vector3f center_direction_native(1.0f, -(u - cx) / fx, -(v - cy) / fy);
        const Eigen::Vector3f center_direction_world = rotation_world_from_native_camera * center_direction_native;
        diagnostic.expected_surface_depth = states[index].shape == Shape::Sphere
            ? sphereDepth(camera_position_world, center_direction_world,
                          states[index].position_world, states[index].radius)
            : cylinderDepth(camera_position_world, center_direction_world,
                            states[index].position_world, states[index].radius, states[index].height);
        diagnostic.observed_depth = depth_image.at<float>(v, u);
        diagnostic.depth_error = std::isfinite(diagnostic.expected_surface_depth)
            ? std::abs(diagnostic.expected_surface_depth - diagnostic.observed_depth) : 0.0f;
    }
    return diagnostics;
}

bool actorCollidesWithUav(const ActorState& actor,
                          const Eigen::Vector3f& uav_position_world,
                          float uav_radius) {
    if (!actor.active) return false;
    if (actor.shape == Shape::Sphere) {
        return (actor.position_world - uav_position_world).norm() <= actor.radius + uav_radius;
    }
    const float radial = std::hypot(actor.position_world.x() - uav_position_world.x(),
                                    actor.position_world.y() - uav_position_world.y());
    const float radial_gap = std::max(0.0f, radial - actor.radius);
    const float vertical_gap = std::max(0.0f,
        std::abs(actor.position_world.z() - uav_position_world.z()) - 0.5f * actor.height);
    return std::hypot(radial_gap, vertical_gap) <= uav_radius;
}

}  // namespace dynamic_sim
