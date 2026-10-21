"""ampzip: training-free simplification of spinor-helicity amplitudes.

Subpackages: ``simplifier`` (CLI, input gate, backends), ``search`` (the best-first search),
``verify`` (finite-field equality), ``webservice`` (the HTTP front end). The move-execution
engine is the extension module ``ampzip_engine`` built from ``ampzip/engine``.
"""
from ._version import __version__  # noqa: F401
