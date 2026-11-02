"""Input checks before parsing.

The web service applies the rules and limits of `ampzip.simplifier.input_limits`, the same
as the command line, with the bounds set by the `WEBSVC_MAX_*` variables (config.py).

See docs/SECURITY.md for public deployment guidance.
"""

from __future__ import annotations

from ..simplifier.input_limits import InputLimitError, InputLimits, check_input

# The name the handlers use for the error. `reasons` lists the messages.
UnsafeInputError = InputLimitError


def prescreen(raw: str, limits: InputLimits) -> None:
    """Raise `UnsafeInputError` when `raw` breaks a rule or exceeds a limit."""
    check_input(raw, limits)
