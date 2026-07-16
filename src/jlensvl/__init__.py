"""JLensVL — a Jacobian-Lens (J-Lens) observer for vision-language models."""
from . import viz
from .mlx_backend import MLXJLens          # import-safe; mlx/torch loaded lazily

# The torch-backed classes need torch + jlens; in an MLX-only env (no torch) they
# stay None so `from jlensvl import MLXJLens` still works on Apple Silicon.
try:
    from .core import JLensVL
    from .prompt_helper import PromptHelper
except ImportError:  # pragma: no cover
    JLensVL = None
    PromptHelper = None

# Vision-tower J-Lens (torch only; no `jlens` engine dependency). Guard so MLX-only envs
# and installs without transformers' Qwen3.5 vision model still import the package.
try:
    from .vision_lens import VisionJLens, VisionJacobianLens
except ImportError:  # pragma: no cover
    VisionJLens = None
    VisionJacobianLens = None

__all__ = ["JLensVL", "PromptHelper", "MLXJLens",
           "VisionJLens", "VisionJacobianLens", "viz"]
__version__ = "0.4.0"
