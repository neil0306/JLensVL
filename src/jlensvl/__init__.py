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

__all__ = ["JLensVL", "PromptHelper", "MLXJLens", "viz"]
__version__ = "0.3.0"
