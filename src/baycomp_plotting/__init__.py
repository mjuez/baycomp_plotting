from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .plotting import Color, dens, tern

try:
    __version__ = _version("baycomp_plotting")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["Color", "dens", "tern", "__version__"]
