"""Runtime package for Fashion 3.1.2.

The heavyweight zero-one-n engine imports CLIP/transformers and is loaded lazily
so lightweight helpers such as spatial constraints remain importable in minimal
CI and documentation checks.
"""

__all__ = ["Fashion312ZeroOneNFunctionalRuntime"]


def __getattr__(name):
    if name == "Fashion312ZeroOneNFunctionalRuntime":
        from .engine import Fashion312ZeroOneNFunctionalRuntime
        return Fashion312ZeroOneNFunctionalRuntime
    raise AttributeError(name)
