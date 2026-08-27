"""Core APIs for Pathline Template Matching."""

from .encoder import IndependentFMT3DConfig, encode_independent_fmt_3d
from .integration import compute_pathlines_3d_batch
from .library import MatchResult, TemplateLibrary
from .primitives import (
    centered_xyz,
    generate_seeding_grid_3d,
    integrate_cross_primitives_3d,
    integrate_multiscale_primitives_3d,
)
from .vector_field import UnsteadyVectorField3D

__all__ = [
    "IndependentFMT3DConfig",
    "MatchResult",
    "TemplateLibrary",
    "UnsteadyVectorField3D",
    "centered_xyz",
    "compute_pathlines_3d_batch",
    "encode_independent_fmt_3d",
    "generate_seeding_grid_3d",
    "integrate_cross_primitives_3d",
    "integrate_multiscale_primitives_3d",
]

__version__ = "0.1.0"
