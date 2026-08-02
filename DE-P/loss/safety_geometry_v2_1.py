"""Safety Evaluator V2.1 geometry.

Dynamic geometry is deliberately re-exported from V2 unchanged.  Only the
static final verifier is replaced by the canonical occupancy authority.
"""

from loss.safety_geometry_v2 import (  # noqa: F401
    actor_physical_clearance,
    continuous_sphere_segment_gap,
    directional_covariance_margin,
    finite_vertical_cylinder_signed_gap,
    maximum_eigenvalue_margin,
    planning_clearance,
    sphere_signed_gap,
    static_physical_clearance,
)
from loss.static_continuous_authority_v1 import (  # noqa: F401
    CertificateState,
    StaticAuthorityCertificate,
    certify_bezier_authority,
    combine_authority_certificates,
    line_bezier_control_points,
    quadratic_bezier_control_points,
    quintic_bezier_control_points,
)

