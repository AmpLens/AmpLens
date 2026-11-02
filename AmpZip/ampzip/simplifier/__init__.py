"""AmpZip, a multi-algorithm spinor-helicity amplitude simplifier.

The entry point is `python -m ampzip ...`, implemented in `simplifier/cli.py`.

Backends:
  - sft_greedy       SFT-greedy, a single deterministic policy rollout
  - sft_best_of_n    SFT best-of-N, one greedy and N-1 stochastic rollouts
  - cds_oneshot      CDS one-shot, a single sequence-to-sequence decode
  - model_free_gbfs  best-first search on the engine, the paper's gbfs, with the variants
                     gbfs_c to gbfs_f

docs/LIMITS.md and `simplifier/limits.py` list the limits.
"""

# Suppress benign warnings from the dependencies that fire on every CLI invocation. The
# pkg_resources deprecation fires because torch, sympy and gdown still use the old setuptools
# API. The torch nested_tensor prototype warning fires from PyTorch's TransformerEncoder fast
# path with key_padding_mask, an internal API that is not yet stable, and behaviour is
# unaffected. The torch.load `weights_only` FutureWarning fires from spinorhelicity's model
# loader in `spinorhelicity/model/__init__.py`, where weights_only=True cannot be passed
# because CDS checkpoints carry full pickled state, the encoder, decoder, optimizer and env
# config. The SFT loader already passes
#      weights_only=True explicitly so it doesn't fire there.
# Filtered at package import so they take effect before any third-party
# import (torch / sympy) triggers them.
import warnings as _warnings
_warnings.filterwarnings("ignore",
                         message=r".*pkg_resources is deprecated.*",
                         category=UserWarning)
_warnings.filterwarnings("ignore",
                         message=r".*pkg_resources is deprecated.*",
                         category=DeprecationWarning)
_warnings.filterwarnings("ignore",
                         message=r".*PyTorch API of nested tensors.*",
                         category=UserWarning)
_warnings.filterwarnings("ignore",
                         message=r".*weights_only.*",
                         category=FutureWarning)
del _warnings

from .._version import __version__  # noqa: E402,F401
