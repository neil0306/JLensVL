"""JLensVL — a Jacobian-Lens (J-Lens) observer for vision-language models."""
from . import viz
from .mlx_backend import MLXJLens          # import-safe; mlx/torch loaded lazily

# The torch-backed classes need torch + jlens; in an MLX-only env (no torch) they
# stay None so `from jlensvl import MLXJLens` still works on Apple Silicon.
try:
    from .core import JLensVL
    from .prompt_helper import PromptHelper
    from .retrieval_lens import RetrievalIndex, RetrievalLens, build_index
    from .prompt_studio import PromptStudio
except ImportError:  # pragma: no cover
    JLensVL = None
    PromptHelper = None
    RetrievalIndex = None
    RetrievalLens = None
    build_index = None
    PromptStudio = None

# Vision-tower J-Lens (torch only; no `jlens` engine dependency). Guard so MLX-only envs
# and installs without transformers' Qwen3.5 vision model still import the package.
try:
    from .vision_lens import VisionJLens, VisionJacobianLens
except ImportError:  # pragma: no cover
    VisionJLens = None
    VisionJacobianLens = None

# Cross-modal Jacobian (P3) + vision-side causal swap (P4); torch only.
try:
    from .cross_modal import (CrossModalJacobianLens, fit_cross_modal_jacobian,
                              combine_cross_modal, cross_modal_jacobian_rows)
    from .interventions import LensIntervention, VisionLensIntervention
except ImportError:  # pragma: no cover
    CrossModalJacobianLens = None
    fit_cross_modal_jacobian = None
    combine_cross_modal = None
    cross_modal_jacobian_rows = None
    LensIntervention = None
    VisionLensIntervention = None

__all__ = ["JLensVL", "PromptHelper", "PromptStudio", "RetrievalIndex", "RetrievalLens",
           "build_index", "VisionJLens", "VisionJacobianLens", "MLXJLens", "viz",
           "CrossModalJacobianLens", "fit_cross_modal_jacobian", "combine_cross_modal",
           "cross_modal_jacobian_rows", "LensIntervention", "VisionLensIntervention"]
__version__ = "0.4.0"
