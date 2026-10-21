"""AmpZip — multi-algorithm spinor-helicity amplitude simplifier.

Entry: `python -m ampzip ...` (see `simplifier/cli.py`).

Backends:
  - sft_greedy       SFT-greedy (single deterministic policy rollout)
  - sft_best_of_n    SFT best-of-N (1 greedy + N-1 stochastic rollouts; NOT beam search)
  - cds_oneshot      CDS one-shot (Dersy et al. seq2seq, single decode)
  - model_free_gbfs  model-free best-first search on the Rust engine (the paper's gbfs; gbfs_c … gbfs_f
                     are its variants)

Limits: see docs/LIMITS.md and `simplifier/limits.py`.
"""

# Suppress benign upstream warnings that fire on every CLI invocation:
#   1) pkg_resources deprecation — torch / sympy / gdown still use the old
#      setuptools API. Remove filter once upstream migrates.
#   2) torch nested_tensor prototype warning — fires from PyTorch's
#      TransformerEncoder fast-path (nn.MultiheadAttention with key_padding_mask).
#      Internal PyTorch API not yet stable; behavior is unaffected.
#   3) torch.load `weights_only` FutureWarning — fires from spinorhelicity's
#      model loader (`spinorhelicity/model/__init__.py`); we can't pass
#      weights_only=True there because CDS checkpoints carry full pickled state
#      (encoder + decoder + optimizer + env config). SFT loader already passes
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
