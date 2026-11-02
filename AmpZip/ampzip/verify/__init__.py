"""Randomized finite-field checks for spinor-helicity expressions.

Arithmetic is exact, and agreement at finitely many points is probabilistic evidence.
equal and zero return True for sampled agreement and False for a modular mismatch.
Resampling exhaustion raises RuntimeError and leaves the comparison inconclusive.

Public API:
    from ampzip.verify import equal, zero, DEFAULT_PRIMES, DEFAULT_K
"""
from .field import eval_modp, modinv, DegeneratePoint
from .kinematics import sample_onshell_modp, vals_from_point
from .verifier import equal, zero, compare, primes_for, DEFAULT_PRIMES, DEFAULT_K

__all__ = [
    "eval_modp", "modinv", "DegeneratePoint",
    "sample_onshell_modp", "vals_from_point",
    "equal", "zero", "compare", "primes_for", "DEFAULT_PRIMES", "DEFAULT_K",
]
