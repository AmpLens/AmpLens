"""ampzip: training-free simplification of spinor-helicity amplitudes.

``simplifier`` holds the CLI, the input check and the backends, ``search`` the best-first
search, ``verify`` the finite-field test and ``webservice`` the HTTP service. The moves are
executed by the extension module ``ampzip_engine`` built from ``ampzip/engine``.
"""
from ._version import __version__  # noqa: F401
