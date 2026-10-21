"""Randomized finite-field checks for spinor-helicity expressions.

Arithmetic is exact; agreement at finitely many points is probabilistic evidence.
equal/zero return True for sampled agreement and False for a modular mismatch;
resampling exhaustion raises RuntimeError and leaves the comparison inconclusive.

Public API:
    from ampzip.verify import equal, zero, DEFAULT_PRIMES, DEFAULT_K
"""
from .field import eval_modp, modinv, DegeneratePoint
from .kinematics import sample_onshell_modp, vals_from_point
from .verifier import equal, zero, DEFAULT_PRIMES, DEFAULT_K

__all__ = [
    "eval_modp", "modinv", "DegeneratePoint",
    "sample_onshell_modp", "vals_from_point",
    "equal", "zero", "DEFAULT_PRIMES", "DEFAULT_K",
]
