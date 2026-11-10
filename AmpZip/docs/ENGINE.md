# Rust engine and Python API

`ampzip_engine` is the Rust engine used by AmpZip. It applies Schouten, momentum-conservation and momentum-squared identities, enumerates moves, and evaluates expressions over finite fields using exact arithmetic.

## Source layout

```text
parse.rs   Parse bracket expressions into syntax trees (Ast)
poly.rs    Rational-function algebra: num / (den * product(den_opaque))
eval.rs    Evaluate Ast and RatFn expressions over finite fields
ident.rs   Build identities and apply bracket substitutions
lib.rs     Python bindings and parallel move application
```

The engine stores expressions as polynomial numerators over monomial denominators. Denominators may also contain polynomial factors, represented by `opaque_den(S)` for `1/S`. Common factors are cancelled when possible.

## Building and testing

The engine is distributed as `ampzip-engine` and imported as `ampzip_engine`. To build it from source, install a Rust toolchain and run the following command from the AmpZip directory.

```bash
python -m pip install ./ampzip/engine
```

Run the Rust checks:

```bash
cd ampzip/engine
cargo clippy --all-targets --release -- -D warnings
cargo test --release
cd ../..
```

After installing AmpZip and its development dependencies as described in the [README](../README.md), run the Python tests:

```bash
python tests/run_tests.py
```

## Python API

### Module functions

- `version() -> str` returns the engine version.
- `parses(expr) -> bool` checks syntax only. A successful parse does not guarantee that `Expr` can represent the input.
- `canonical(expr) -> str` combines like terms, cancels common factors and returns the canonical expression string.
- `apply(expr, square, i, j, identity, params, n) -> str` replaces every occurrence of the selected bracket in the numerator and returns the resulting expression string.
- `eval_onshell(expr, lam, lamt, p) -> int` evaluates the expression exactly modulo the prime `p` at the supplied spinors.
- `parse_ast(expr, params=None) -> Ast` parses the expression without expanding it. `params` lists the identifiers admitted as named parameters.
- `sqrt_mod(a, p) -> int` returns a square root of `a` modulo the odd prime `p`, or raises when there is none.

For `a = parse_ast(expr)`, `a.eval(lam, lamt, p, params=None)` returns the field value and `a.eval_pair(lam, lamt, p, params=None)` returns `(numerator, denominator)` modulo `p` without division. `params` maps each named parameter to its value. `a.needs()` returns `(has_I, sqrt_primes, param_names)`. The evaluator accepts integers of any size, decimals read as exact rationals, `I`, `sqrt(n)` and the named parameters, each mapped into the field by a ring homomorphism, so `I` needs `p = 1 (mod 4)` and every prime under a square root must be a quadratic residue modulo `p`. A zero denominator must be treated as an undefined evaluation.

For identity application, `square=False` selects `ab(i,j)` and `square=True` selects `sb(i,j)`. `n` is the number of external particles. The `identity` and `params` arguments use the following encoding:

| Identity | `params` |
|---|---|
| `schouten` | `[pk, pl]` |
| `momentum` | `[pout1, pout2]` |
| `momentum_sq` | `mom_add`, a list of additional particle labels |

### Expression objects

`Expr` keeps the expression in Rust between moves.

- `Expr(expr, max_terms=0)` parses the input and cancels common factors.
- `e.to_str() -> str` returns the canonical expression string. `repr(e)` wraps this string in `Expr(...)`.
- `e.apply(square, i, j, identity, params, n, term_slot=0, max_terms=0) -> Expr` returns a new `Expr` without modifying the original. `term_slot=0` replaces all occurrences of the selected bracket in the numerator. `term_slot=k>0` replaces one bracket factor in the k-th numerator term in engine order.
- `e.num_terms() -> int` returns the number of numerator terms.
- `e.distinct_brackets() -> [(square, i, j)]` lists the distinct brackets in the numerator.
- `e.is_zero() -> bool` reports whether the numerator is zero.
- `e.eval_onshell(lam, lamt, p) -> int` evaluates the stored expression exactly modulo `p`.
- `e.term_monos() -> [str]` lists numerator monomials without their coefficients.
- `e.terms() -> [(mono_str, coeff)]` lists numerator monomials with their coefficients.
- `e.term_powers() -> [(coeff, [(square, i, j, exp), ...])]` returns the numerator terms as bracket powers.
- `e.den_brackets() -> [(square, i, j, exp)]` returns the bracket powers in the monomial part of the denominator.
- `e.den_opaque_terms()` returns each polynomial denominator factor as a list of terms in the same format as `term_powers()`.

`e.bracket_count()` returns the raw numerator bracket count. `e.factored_bracket_count()` counts brackets after extracting the monomial factor common to all numerator terms. Both counts include multiplicity and exclude denominator brackets. The default search uses `(e.factored_bracket_count(), e.num_terms())`.

`max_terms` limits the numerator term count during input expansion and identity application. A value of `0` uses `AMPZIP_MAX_TERMS`, which defaults to 50,000.

### Move enumeration and parallel application

Moves have the form `(square, i, j, term_slot, identity, params)`. The engine enumerates the same moves as `ampzip.search.actions.legal_moves`: for each bracket present in the numerator, it offers whole-expression moves and moves within each term containing that bracket. Degenerate identity applications are rejected when the moves are applied.

- `e.legal_moves(n)` returns a Python list of moves.
- `e.legal_move_list(n) -> MoveList` keeps the moves in Rust for use with `apply_slice()`. `len(ml)` gives the number of moves, and `ml.get(k)` returns the move at zero-based index `k`.
- `e.apply_one(square, i, j, identity, params, n, term_slot=0, max_terms=0)` applies one move and returns the result record described below. It raises `ValueError` when `e.apply()` would.
- `e.apply_slice(moves, offset, limit, n, max_terms=0, threads=0)` applies a slice of a `MoveList` and returns one record or `None` per move, in slice order. `limit=0` selects all remaining moves. `None` indicates an application that would raise `ValueError` in `e.apply()`.
- `e.apply_batch(moves, n, max_terms=0, threads=0)` applies a Python list of moves and returns the same records as `apply_slice()`, in input order.
- `e.apply_slice_coded(moves, offset, limit, n, max_terms=0, threads=0)` returns `(code, record)` for each move. Code `0` indicates success, `1` an invalid identity application or term slot, `2` an exceeded term limit, and `3` coefficient overflow. Rejected moves have `record=None`.

Each successful application returns:

```text
(child, key, factored_bracket_count, bracket_count, num_terms)
```

`child` is an `Expr`. `key` is the 16-byte hash returned by `child.key()`, used to identify previously visited expressions. Use `child.to_str()` to obtain the expression string.

`threads=0` uses Rust's `available_parallelism()` to select the worker count, or one worker if the query fails.

Moves are applied in parallel, and results are returned in input order. The search processes these results in order to preserve the search ordering. The returned expression may differ if `--gbfs-max-seconds` is reached or the search is cancelled.

## Adding an identity

Implement the replacement in `ident::build_identity`. To make the identity available to the search, also update the Rust and Python move menus, the identity mapping in `apply_batch`, and the search's identity encoding. Test both whole-expression and per-term application, and check that the results preserve the input's value over finite fields.

## Usage notes

- Term slots follow the engine's canonical term order, which may differ from SymPy's term order. Do not use the SFT checkpoints' term indices directly with `Expr`.
- Recompute the available moves after each application because the terms and their indices may change.
- Record the engine version when saving derivations, since changes in term ordering can affect replay.
- `Expr.apply()` raises `ValueError` if the selected term does not exist or does not contain the selected bracket.
- `Expr` uses signed 128-bit integer coefficients. Coefficient overflow raises a `ValueError`. During search, moves that overflow are skipped.
