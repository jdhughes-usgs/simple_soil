"""
The simple_soil package consists of a set of Python scripts to ...

simple_soil is an open-source project and any assistance is welcomed.

"""
# See CITATION.cff for authors
__author__ = "simple_soil Team"

from .version import __version__  # isort:skip
from . import base, utils

# from .mbase import run_model, which

__all__ = [
    "__author__",
    "__version__",
    "base",
    "utils",
]
