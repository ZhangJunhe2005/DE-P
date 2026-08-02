"""ROS-free dynamic perception with lazy optional-dependency imports.

Keeping this package initializer light is what lets the static DEP network run
without importing scikit-learn or FilterPy.
"""

from importlib import import_module

__all__ = [
    "CameraModel",
    "ClusterObservation",
    "DynamicPerception",
    "DynamicPerceptionConfig",
    "DynamicPerceptionResult",
    "DynamicTrack",
    "DynamicContext",
    "DynamicTrackSummary",
    "Pose",
    "ProjectedDynamicTrack",
    "build_dynamic_attention",
    "depth_to_pointcloud", "make_depth_frame", "DepthFrame",
    "image_to_feature_coordinates",
    "project_world_points_to_image",
]

_EXPORTS = {
    "DynamicContext": (".context", "DynamicContext"),
    "DynamicTrackSummary": (".context", "DynamicTrackSummary"),
    "depth_to_pointcloud": (".camera_model", "depth_to_pointcloud"),
    "make_depth_frame": (".camera_model", "make_depth_frame"),
    "DepthFrame": (".types", "DepthFrame"),
    "DynamicPerception": (".dynamic_perception", "DynamicPerception"),
    "build_dynamic_attention": (".attention", "build_dynamic_attention"),
    "image_to_feature_coordinates": (".projection", "image_to_feature_coordinates"),
    "project_world_points_to_image": (".projection", "project_world_points_to_image"),
}
for _name in ("CameraModel", "ClusterObservation", "DynamicPerceptionConfig",
              "DynamicPerceptionResult", "DynamicTrack", "Pose", "ProjectedDynamicTrack"):
    _EXPORTS[_name] = (".types", _name)


def __getattr__(name):
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value
