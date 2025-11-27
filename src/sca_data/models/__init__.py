import importlib
import typing

if typing.TYPE_CHECKING:
    from . import audio, downloads, events


def __getattr__(name: str):
    if name in {"audio", "downloads", "events"}:
        return importlib.import_module("." + name, __package__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return  __all__

__all__ = ["audio", "downloads", "events"]
