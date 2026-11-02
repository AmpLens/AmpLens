"""Input validation and checkpoint size checks.

Parses the input string and checks its atoms, symbols, coefficients and leg labels.
It also checks the number of external legs and checkpoint size limits. Returns a
`ValidationResult`, and no backend runs when `result.ok` is False.
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


# The three size checks use the SFT input limits of limits.py, which also apply to CDS. They
# do not constrain the gbfs family. With `enforce_size_limits=False` the size tags are still
# reported for routing but do not make the result invalid. `ENGINE_LIMIT_TAGS` are also used
# for routing. The other `LIMIT_*` tags describe invalid inputs and always fail validation.
SIZE_LIMIT_TAGS = frozenset({"LIMIT_TERMS", "LIMIT_BRACKETS_PER_TERM", "LIMIT_BRACKETS_TOTAL"})
# Inputs the neural baselines accept but the search engine cannot represent, non-integer
# rational coefficients, since the engine's reciprocal needs a unit coefficient, and the
# imaginary unit. Like the size tags these do not fail validation. The routers drop the search
# family with a note and fail only when nothing is left to run.
ENGINE_LIMIT_TAGS = frozenset({"LIMIT_ENGINE_COEFF"})


def _get_bracket_classes():
    """The package's own ab/sb atoms from ampzip.atoms, the accessor every caller uses."""
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
    """The number of top-level summands, sympy.Add.args, which is 1 for anything but an Add."""
    if isinstance(expr, sp.Add):
        return len(expr.args)
    return 1


def _term_iter(expr: sp.Expr):
    if isinstance(expr, sp.Add):
        yield from expr.args
    else:
        yield expr


def _brackets_in(expr: sp.Expr, ab_cls, sb_cls) -> int:
    """Count the ab/sb occurrences in a single term, a power counting its exponent."""
    return _count_brackets_flat(expr, ab_cls, sb_cls)


def _count_brackets_flat(expr: sp.Expr, ab_cls, sb_cls) -> int:
    """Count the ab/sb factor multiplicity in a single term."""
    # Each factor of a Mul may be a bracket or a Pow of one.
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
        # Nested Adds and other factors are ignored. They are unusual in a single term.
    return n


def _distinct_brackets(expr: sp.Expr, ab_cls, sb_cls) -> Set[Tuple[str, int, int]]:
    out: Set[Tuple[str, int, int]] = set()
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, (ab_cls, sb_cls)):
            args = sub.args
            if len(args) == 2 and all(a.is_Integer for a in args):
                i, j = int(args[0]), int(args[1])
                # Normalise the order. A bracket is antisymmetric, so it is the same up to sign.
                key = (type(sub).__name__, min(i, j), max(i, j))
                out.add(key)
    return out


def _numeric_constant_issues(expr: sp.Expr) -> list:
    """Reject Float and the transcendental NumberSymbols pi, E, EulerGamma and the like, and
    accept Integer, Rational and the imaginary unit. Returns a list of ValidationIssue."""
    issues = []
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Float):
            issues.append(ValidationIssue(
                tag="LIMIT_COEFF_CLASS",
                message=f"float coefficient encountered: {sub}",
                detail={"value": str(sub)},
            ))
        elif isinstance(sub, sp.NumberSymbol):
            # sp.pi, sp.E, sp.EulerGamma, sp.GoldenRatio and sp.Catalan
            issues.append(ValidationIssue(
                tag="LIMIT_COEFF_CLASS",
                message=f"transcendental constant '{sub}' is outside the "
                        f"training distribution",
                detail={"constant": str(sub)},
            ))
    return issues


def _engine_coefficient_issues(expr: sp.Expr) -> list:
    """What the search engine refuses although validation accepts it, a rational coefficient
    that is not an integer, which `ratfn_recip` in poly.rs rejects as a reciprocal with a
    non-unit coefficient, and the imaginary unit."""
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
    """Reject a Pow with a non-integer exponent such as a square or cube root, which is outside
    the training distribution. Returns a list of ValidationIssue."""
    issues = []
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Pow):
            base, exp_ = sub.args
            if not exp_.is_Integer:
                # An ab/sb base with a non-integer exponent is illegal too.
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
    """Parse and check the input, returning a ValidationResult.

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

    # 1) Parse leniently. Unknown calls and names come back as undefined Functions and
    #    Symbols so the atom checks below report them under their own tags.
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

    # 3) Coefficients, Float and NumberSymbol, and non-integer Pow exponents such as sqrt
    issues.extend(_numeric_constant_issues(expr))
    issues.extend(_algebraic_pow_issues(expr, ab_cls, sb_cls))
    issues.extend(_engine_coefficient_issues(expr))

    # If the atoms, symbols or coefficients already failed, infer no n_point and count nothing.
    if any(i.tag.startswith("LIMIT_ATOM") or i.tag.startswith("LIMIT_SYMBOL")
           or i.tag == "LIMIT_COEFF_CLASS"
           for i in issues):
        return ValidationResult(ok=False, inferred_n_point=None,
                                canonical_expr_str=canonical_str, issues=issues)

    # 4) Leg labels. Each bracket argument must be a positive integer.
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
            # A self-bracket `ab(i,i)` is zero but stays an unevaluated call in SymPy,
            # which the backends would treat as an ordinary atom.
            if args[0] == args[1]:
                issues.append(ValidationIssue(
                    tag="LIMIT_LEG_LABEL",
                    message=f"{head}{args} has equal legs, and <ii>=[ii]=0",
                ))
                continue
            for a in args:
                max_leg = max(max_leg, int(a))

    if any(i.tag == "LIMIT_LEG_LABEL" for i in issues):
        return ValidationResult(ok=False, inferred_n_point=None,
                                canonical_expr_str=canonical_str, issues=issues)

    # 5) n_point inference and the override check
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
