"""Input validation and checkpoint size checks.

Parses the input string and checks its atoms, symbols, coefficients and leg labels.
It also checks the number of external legs and checkpoint size limits. Returns a
`ValidationResult`. Callers must not run any backend when `result.ok` is False.
"""

from __future__ import annotations
from typing import Tuple, Optional, Set
import sympy as sp

from .limits import (
    SFT_LIMITS,
    ALLOWED_ATOMS,
    KNOWN_REJECTED_ATOMS,
    KNOWN_REJECTED_SYMBOLS,
    SUPPORTED_N_POINTS,
)
from .result import ValidationResult, ValidationIssue
from .parse import parse_expr, ParseError


# The three size checks use the SFT input limits defined in limits.py, which are
# also applied to CDS. They do not constrain the model-free GBFS backends. With
# `enforce_size_limits=False`, the size tags are still reported for routing but do
# not make the result invalid. `ENGINE_LIMIT_TAGS` are also used for routing. Other
# `LIMIT_*` tags describe invalid inputs and always fail validation.
SIZE_LIMIT_TAGS = frozenset({"LIMIT_TERMS", "LIMIT_BRACKETS_PER_TERM", "LIMIT_BRACKETS_TOTAL"})
# Inputs the learned backends accept but the search engine cannot represent: non-integer
# rational coefficients (the engine's reciprocal needs a unit coefficient) and the imaginary
# unit. Like the size tags these do not fail validation; the routers drop the search family
# with a note (and fail only when nothing is left to run).
ENGINE_LIMIT_TAGS = frozenset({"LIMIT_ENGINE_COEFF"})


def _get_bracket_classes():
    """The package's own ab/sb atoms (ampzip.atoms); kept as the accessor every caller uses."""
    from ..atoms import bracket_classes  # noqa: WPS433
    return bracket_classes()


def _walk_atoms(expr: sp.Expr) -> Tuple[Set[str], Set[str]]:
    """Return (function_head_names, symbol_names) appearing in expr."""
    func_names: Set[str] = set()
    sym_names: Set[str] = set()
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function):
            func_names.add(type(sub).__name__)
        elif isinstance(sub, sp.Symbol):
            sym_names.add(sub.name)
    return func_names, sym_names


def _all_bracket_calls(expr: sp.Expr, ab_cls, sb_cls):
    """Yield every (head_name, args_tuple) for ab/sb calls in expr."""
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, (ab_cls, sb_cls)):
            yield type(sub).__name__, tuple(sub.args)


def _count_terms(expr: sp.Expr) -> int:
    """Number of summands at the top level (sympy.Add.args). Non-Add → 1."""
    if isinstance(expr, sp.Add):
        return len(expr.args)
    return 1


def _term_iter(expr: sp.Expr):
    if isinstance(expr, sp.Add):
        yield from expr.args
    else:
        yield expr


def _brackets_in(expr: sp.Expr, ab_cls, sb_cls) -> int:
    """Count occurrences of ab/sb in a single term (powers count multiplicity)."""
    n = 0
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Pow):
            base, exp_ = sub.args
            if isinstance(base, (ab_cls, sb_cls)) and exp_.is_integer:
                # only count once below via preorder; but Pow includes base too,
                # so subtract the implicit single-count to add the exponent count.
                # Simpler: handle Pow explicitly and skip the inner base in main loop.
                pass
    # Simpler implementation: flatten via Mul.args
    return _count_brackets_flat(expr, ab_cls, sb_cls)


def _count_brackets_flat(expr: sp.Expr, ab_cls, sb_cls) -> int:
    """Count ab/sb factor multiplicity in a single term."""
    # For a Mul: each arg may be Pow(bracket, n) or bracket itself.
    if isinstance(expr, sp.Mul):
        factors = expr.args
    else:
        factors = (expr,)
    n = 0
    for f in factors:
        if isinstance(f, (ab_cls, sb_cls)):
            n += 1
        elif isinstance(f, sp.Pow):
            base, exp_ = f.args
            if isinstance(base, (ab_cls, sb_cls)) and exp_.is_Integer:
                n += abs(int(exp_))
        # ignore nested Adds and other; they would be unusual in a single term
    return n


def _distinct_brackets(expr: sp.Expr, ab_cls, sb_cls) -> Set[Tuple[str, int, int]]:
    out: Set[Tuple[str, int, int]] = set()
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, (ab_cls, sb_cls)):
            args = sub.args
            if len(args) == 2 and all(a.is_Integer for a in args):
                i, j = int(args[0]), int(args[1])
                # Normalize order (ab is antisymmetric in args; same bracket up to sign)
                key = (type(sub).__name__, min(i, j), max(i, j))
                out.add(key)
    return out


def _numeric_constant_issues(expr: sp.Expr) -> list:
    """Reject Float and transcendental NumberSymbols (pi, E, EulerGamma, …).
    Accept Integer, Rational, ImaginaryUnit (I). Returns list of ValidationIssue."""
    issues = []
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Float):
            issues.append(ValidationIssue(
                tag="LIMIT_COEFF_CLASS",
                message=f"float coefficient encountered: {sub}",
                detail={"value": str(sub)},
            ))
        elif isinstance(sub, sp.NumberSymbol):
            # sp.pi, sp.E, sp.EulerGamma, sp.GoldenRatio, sp.Catalan
            issues.append(ValidationIssue(
                tag="LIMIT_COEFF_CLASS",
                message=f"transcendental constant '{sub}' is outside the "
                        f"training distribution",
                detail={"constant": str(sub)},
            ))
    return issues


def _engine_coefficient_issues(expr: sp.Expr) -> list:
    """What the Rust search engine refuses although validation accepts it: a rational
    coefficient that is not an integer (poly.rs `ratfn_recip`: 'reciprocal with non-unit
    coefficient') and the imaginary unit (parse.rs: unknown identifier 'I')."""
    issues = []
    seen = set()
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Rational) and not isinstance(sub, sp.Integer):
            key = ("rational", str(sub))
            if key not in seen:
                seen.add(key)
                issues.append(ValidationIssue(
                    tag="LIMIT_ENGINE_COEFF",
                    message=f"rational coefficient {sub} is not representable by the search "
                            f"engine (it keeps integer coefficients only); the learned backends "
                            f"accept it",
                    detail={"value": str(sub), "engine_reason": "reciprocal with non-unit coefficient"},
                ))
        elif sub is sp.I:
            if ("I", "I") not in seen:
                seen.add(("I", "I"))
                issues.append(ValidationIssue(
                    tag="LIMIT_ENGINE_COEFF",
                    message="the imaginary unit I is not representable by the search engine; "
                            "the learned backends accept it",
                    detail={"value": "I", "engine_reason": "unknown identifier 'I'"},
                ))
    return issues


def _algebraic_pow_issues(expr: sp.Expr, ab_cls, sb_cls) -> list:
    """Reject Pow with non-integer exponent (sqrt, cube root, etc.) — out of
    training distribution. Returns list of ValidationIssue."""
    issues = []
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Pow):
            base, exp_ = sub.args
            if not exp_.is_Integer:
                # tolerated exception: ab/sb base ** non-integer — still illegal
                base_repr = type(base).__name__
                issues.append(ValidationIssue(
                    tag="LIMIT_COEFF_CLASS",
                    message=f"non-integer power encountered: {sub} "
                            f"(base={base_repr}, exp={exp_}); "
                            f"algebraic roots / fractional exponents are outside "
                            f"the training distribution",
                    detail={"expr": str(sub), "exp": str(exp_)},
                ))
    return issues


def validate(raw_input: str, override_n_point: Optional[int] = None,
             enforce_size_limits: bool = True) -> ValidationResult:
    """Parse + check input. Never raises — returns ValidationResult.

    `enforce_size_limits` defaults to True. When False, checkpoint size issues
    (`SIZE_LIMIT_TAGS`) are still reported but do not make `ok` False. CLI and web
    job handlers pass False and use these issues to filter the requested backends,
    preserving GBFS runs that do not use these checkpoints. Numerical checks also
    pass False because they evaluate expressions without a model.

    `ENGINE_LIMIT_TAGS` are reported for routing regardless of this flag.
    All other `LIMIT_*` issues make `ok` False.
    """
    issues = []

    ab_cls, sb_cls = _get_bracket_classes()
    local_dict = {"ab": ab_cls, "sb": sb_cls, "I": sp.I}

    # 1) Parse. Lenient: unknown calls and names come back as undefined Functions / Symbols
    #    so the atom checks below report them under their own tags; nothing is executed.
    try:
        expr = parse_expr(raw_input, local_dict, lenient=True)
    except ParseError as e:
        issues.append(ValidationIssue(
            tag="LIMIT_PARSE",
            message=f"could not parse input: {e}",
        ))
        return ValidationResult(ok=False, inferred_n_point=None,
                                canonical_expr_str=None, issues=issues)

    canonical_str = str(expr)

    # 2) Atom whitelist
    func_names, sym_names = _walk_atoms(expr)
    unknown_funcs = func_names - ALLOWED_ATOMS
    for fname in unknown_funcs:
        if fname in KNOWN_REJECTED_ATOMS:
            issues.append(ValidationIssue(
                tag="LIMIT_ATOM_REJECTED",
                message=f"atom '{fname}' is recognized but not supported: "
                        f"{KNOWN_REJECTED_ATOMS[fname]}",
                detail={"atom": fname},
            ))
        else:
            issues.append(ValidationIssue(
                tag="LIMIT_ATOM_UNKNOWN",
                message=f"atom '{fname}' is not in the allowlist {{ab, sb}}",
                detail={"atom": fname},
            ))

    for sname in sym_names:
        if sname in KNOWN_REJECTED_SYMBOLS:
            issues.append(ValidationIssue(
                tag="LIMIT_SYMBOL_REJECTED",
                message=f"symbol '{sname}' is recognized but not supported: "
                        f"{KNOWN_REJECTED_SYMBOLS[sname]}",
                detail={"symbol": sname},
            ))
        else:
            issues.append(ValidationIssue(
                tag="LIMIT_SYMBOL_UNKNOWN",
                message=f"symbol '{sname}' is not allowed (no symbolic coefficients)",
                detail={"symbol": sname},
            ))

    # 3) Coefficients (Float + NumberSymbol) + non-integer Pow exponents (sqrt etc.)
    issues.extend(_numeric_constant_issues(expr))
    issues.extend(_algebraic_pow_issues(expr, ab_cls, sb_cls))
    issues.extend(_engine_coefficient_issues(expr))

    # If atoms/symbols/coefficients already failed, don't infer n_point or count
    if any(i.tag.startswith("LIMIT_ATOM") or i.tag.startswith("LIMIT_SYMBOL")
           or i.tag == "LIMIT_COEFF_CLASS"
           for i in issues):
        return ValidationResult(ok=False, inferred_n_point=None,
                                canonical_expr_str=canonical_str, issues=issues)

    # 4) Leg labels: each bracket arg must be a positive integer
    max_leg = 0
    for head, args in _all_bracket_calls(expr, ab_cls, sb_cls):
        if len(args) != 2:
            issues.append(ValidationIssue(
                tag="LIMIT_LEG_LABEL",
                message=f"{head}{args} has {len(args)} args; expected 2",
            ))
            continue
        for a in args:
            if not a.is_Integer or a <= 0:
                issues.append(ValidationIssue(
                    tag="LIMIT_LEG_LABEL",
                    message=f"{head}{args} contains non-positive-integer arg {a!r}",
                ))
                break
        else:
            # A self-bracket `ab(i,i)` or `sb(i,i)` is identically zero, but it can
            # remain unevaluated in the SymPy expression used by the numerical
            # check. `_flat_subs` supplies only `i != j` substitutions, so
            # `lambdify` can leave an undefined `ab` or `sb` call. The resulting
            # evaluation error causes `/api/numerical_check` to return HTTP 500.
            # Reject self-brackets here for CLI inputs, jobs and numerical checks.
            if args[0] == args[1]:
                issues.append(ValidationIssue(
                    tag="LIMIT_LEG_LABEL",
                    message=f"{head}{args} is a self-bracket (equal legs) — "
                            f"<ii>=[ii]=0, not a valid input",
                ))
                continue
            for a in args:
                max_leg = max(max_leg, int(a))

    if any(i.tag == "LIMIT_LEG_LABEL" for i in issues):
        return ValidationResult(ok=False, inferred_n_point=None,
                                canonical_expr_str=canonical_str, issues=issues)

    # 5) n_point inference / override check
    inferred_n_point = max_leg if max_leg else None
    if override_n_point is not None:
        if override_n_point not in SUPPORTED_N_POINTS:
            issues.append(ValidationIssue(
                tag="LIMIT_N_POINT",
                message=f"--n-point {override_n_point} not in {SUPPORTED_N_POINTS}",
            ))
            return ValidationResult(ok=False, inferred_n_point=inferred_n_point,
                                    canonical_expr_str=canonical_str, issues=issues)
        if inferred_n_point and inferred_n_point > override_n_point:
            issues.append(ValidationIssue(
                tag="LIMIT_LEG_OUT_OF_RANGE",
                message=f"expression uses leg {inferred_n_point} but --n-point={override_n_point}",
            ))
            return ValidationResult(ok=False, inferred_n_point=inferred_n_point,
                                    canonical_expr_str=canonical_str, issues=issues)
        n_point = override_n_point
    else:
        if inferred_n_point is None or inferred_n_point not in SUPPORTED_N_POINTS:
            issues.append(ValidationIssue(
                tag="LIMIT_N_POINT",
                message=f"inferred n_point={inferred_n_point} not in {SUPPORTED_N_POINTS}; "
                        f"use --n-point {{4,5,6}} to override or rewrite the expression",
            ))
            return ValidationResult(ok=False, inferred_n_point=inferred_n_point,
                                    canonical_expr_str=canonical_str, issues=issues)
        n_point = inferred_n_point

    # 6) Size budget
    L = SFT_LIMITS[n_point]
    n_terms = _count_terms(expr)
    if n_terms > L.max_terms:
        issues.append(ValidationIssue(
            tag="LIMIT_TERMS",
            message=f"{n_terms} terms > {L.max_terms} (n_point={n_point})",
            detail={"n_terms": n_terms, "max_terms": L.max_terms},
        ))

    for k, term in enumerate(_term_iter(expr)):
        nb = _count_brackets_flat(term, ab_cls, sb_cls)
        if nb > L.max_brackets_per_term:
            issues.append(ValidationIssue(
                tag="LIMIT_BRACKETS_PER_TERM",
                message=f"term #{k} has {nb} brackets > {L.max_brackets_per_term} (n_point={n_point})",
                detail={"term_index": k, "n_brackets": nb,
                        "max_brackets_per_term": L.max_brackets_per_term},
            ))

    distinct = _distinct_brackets(expr, ab_cls, sb_cls)
    if len(distinct) > L.max_brackets:
        issues.append(ValidationIssue(
            tag="LIMIT_BRACKETS_TOTAL",
            message=f"{len(distinct)} distinct brackets > {L.max_brackets} (n_point={n_point})",
            detail={"n_distinct": len(distinct), "max_brackets": L.max_brackets},
        ))

    # Size tags make validation fail only when `enforce_size_limits=True`.
    # Engine tags are reported for routing. All other LIMIT_* tags always fail.
    ok = not any(
        i.tag.startswith("LIMIT_") and i.tag not in ENGINE_LIMIT_TAGS
        and (enforce_size_limits or i.tag not in SIZE_LIMIT_TAGS)
        for i in issues
    )
    return ValidationResult(ok=ok, inferred_n_point=n_point,
                            canonical_expr_str=canonical_str, issues=issues)
