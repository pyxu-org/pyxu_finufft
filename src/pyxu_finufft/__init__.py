import importlib.metadata

try:
    __version__ = importlib.metadata.version("pyxu_eigh")
except ImportError:
    __version__ = "unknown"

from .math import eigh

from .operator import Flip, NullFunc


__all__ = (
    "eigh",
    "Flip",
    "NullFunc",
)
