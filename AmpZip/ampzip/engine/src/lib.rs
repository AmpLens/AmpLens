//! ampzip_engine — the move-execution engine's PyO3 surface (thin).
//!
//! Architecture:
//!   parse  — sympy `str()`  ->  Ast            (`parse.rs`)
//!   poly   — `RatFn`: num / (den * product(den_opaque)),
//!            arithmetic, cancellation, string output   (`poly.rs`)
//!   eval   — exact mod-p on-shell evaluation of an Ast   (`eval.rs`)
//!   ident  — Ast->RatFn, the 3 identities, bracket substitution  (`ident.rs`)
mod eval;
mod ident;
mod parse;
mod poly;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};

/// Raise if a coefficient overflow was flagged during the preceding algebra. Callers
/// must `poly::overflow_check_reset()` *before* the computation to clear stale state.
fn check_overflow() -> PyResult<()> {
    if poly::overflow_check_reset() {
        return Err(PyValueError::new_err(
            "i128 coefficient overflow — promote to bigint",
        ));
    }
    Ok(())
}

/// Raise if a whole-expression or single-term rewrite exceeded its numerator term
/// limit. The search handles this ValueError by skipping the move. Callers must
/// call `poly::term_limit_hit_reset()` before each application to clear stale state.
fn check_term_limit() -> PyResult<()> {
    if poly::term_limit_hit_reset() {
        return Err(PyValueError::new_err(
            "numerator term count exceeded AMPZIP_MAX_TERMS during apply — move skipped",
        ));
    }
    Ok(())
}

/// Why a move was rejected, as returned by `apply_core` / `Expr::apply_slice_coded`.
/// 0 = applied, 1 = degenerate identity or illegal term slot, 2 = term cap breached,
/// 3 = coefficient overflow. The search counts these per cause (`*_skips` in its result).
const REJECT_DEGENERATE: u8 = 1;
const REJECT_TERM_LIMIT: u8 = 2;
const REJECT_OVERFLOW: u8 = 3;

/// Shared core of `Expr::apply` with no Python types in the signature, so it can run off the
/// GIL inside a worker thread (see `Expr::apply_batch`). Returns `Err(code)` in exactly the
/// cases where `Expr::apply` raises a `ValueError` — degenerate identity, illegal term slot,
/// `max_terms` breach, coefficient overflow — which is the case the search already skips with
/// `except ValueError: continue`; the code says which.
///
/// Thread-safety of the two `thread_local!` flags in `poly.rs` (`OVERFLOW`, `TERM_LIMIT_HIT`):
/// reset -> compute -> check all happen inside THIS function, hence on ONE thread, so a worker
/// only ever observes its own flags. No locking is needed and no flag can leak between moves —
/// that is what makes a threaded batch bit-identical to the sequential loop.
#[allow(clippy::too_many_arguments)] // mirrors `Expr::apply`, whose args are the RL action encoding
fn apply_core(
    rf: &poly::RatFn,
    square: bool,
    i: u16,
    j: u16,
    identity: &str,
    params: &[u16],
    n: u16,
    term_slot: usize,
    max_terms: usize,
) -> Result<poly::RatFn, u8> {
    poly::overflow_check_reset();
    poly::term_limit_hit_reset();
    let ret = ident::build_identity(square, i, j, identity, params, n).map_err(|_| REJECT_DEGENERATE)?;
    let (target, _) = poly::make_bracket(square, i, j);
    let mut out = if term_slot == 0 {
        ident::substitute(rf, target, &ret, max_terms)
    } else {
        ident::substitute_term(rf, target, &ret, term_slot - 1, max_terms).map_err(|_| REJECT_DEGENERATE)?
    };
    // Bail on a substitution blow-up BEFORE the costly cancel, exactly as `apply` does.
    if poly::term_limit_hit_reset() {
        return Err(REJECT_TERM_LIMIT);
    }
    poly::ratfn_cancel(&mut out);
    if poly::overflow_check_reset() {
        return Err(REJECT_OVERFLOW);
    }
    Ok(out)
}

/// `itertools.combinations(pool, r)` — same yield ORDER as CPython (lexicographic by index),
/// which `legal_moves` depends on to reproduce the shipped action ordering. `r == 0` yields one
/// empty selection; `r > pool.len()` yields none.
fn combinations(pool: &[u16], r: usize) -> Vec<Vec<u16>> {
    let n = pool.len();
    let mut out: Vec<Vec<u16>> = Vec::new();
    if r > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        out.push(idx.iter().map(|&k| pool[k]).collect());
        // rightmost index that can still advance
        let mut i = r;
        let mut found = false;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - r {
                found = true;
                break;
            }
        }
        if !found {
            return out;
        }
        idx[i] += 1;
        for j in i + 1..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// The `(identity, params)` menu for bracket `bk(i,j)`: **C(n-2,2) Schouten + 2(n-1) momentum +
/// 2^(n-2) momentum_sq** = 11/19/32 entries at n=4/5/6. Port of
/// `ampzip.search.actions.identity_menu`, including its order
/// (Schouten, momentum, momentum_sq).
///
/// Independent of the bracket TYPE — the Python version also takes `square` and ignores it.
fn identity_menu(i: u16, j: u16, n: u16) -> Vec<(&'static str, Vec<u16>)> {
    let other: Vec<u16> = (1..=n).filter(|&m| m != i && m != j).collect();
    let mut menu: Vec<(&'static str, Vec<u16>)> = Vec::new();
    for c in combinations(&other, 2) {
        menu.push(("schouten", c));
    }
    for pout1 in [i, j] {
        menu.push(("momentum", vec![pout1, pout1]));
        for &pout2 in &other {
            menu.push(("momentum", vec![pout1, pout2]));
        }
    }
    for size in 0..=other.len() {
        for c in combinations(&other, size) {
            menu.push(("momentum_sq", c));
        }
    }
    menu
}

/// A bracket with its exponent: `(square, i, j, exponent)`.
type BracketPow = (bool, u16, u16, u32);
/// A numerator term: `(coefficient, brackets)`.
type TermPow = (i128, Vec<BracketPow>);
/// A legal move as Python sees it: `(square, i, j, term_slot, identity, params)`.
type Move = (bool, u16, u16, usize, String, Vec<u16>);
/// The same move in Rust, with the identity as a static tag (no per-move allocation).
type MoveRs = (bool, u16, u16, usize, &'static str, Vec<u16>);
/// One batch-apply result: `(child, key, factored_bracket_count, bracket_count,
/// num_terms)` — everything the search reads off a child, computed off the GIL.
/// What a worker computes for one child: the handle, its dedup key, and the three counts.
type ChildRec = (Expr, [u8; 16], u64, u64, usize);
/// The same record as handed to Python (the key as `bytes`).
type ChildPy = (Expr, Py<PyBytes>, u64, u64, usize);

/// Dedup key of a canonical form: blake2b with a 16-byte digest over the canonical string's
/// UTF-8, exactly `hashlib.blake2b(s.encode(), digest_size=16).digest()` in Python, so keys
/// computed here and keys the search computed in Python before this existed are the same bytes.
fn ratfn_key(rf: &poly::RatFn) -> [u8; 16] {
    let mut h = Blake2b::<U16>::new();
    h.update(poly::ratfn_to_string(rf).as_bytes());
    h.finalize().into()
}

fn child_py(py: Python<'_>, rec: ChildRec) -> ChildPy {
    let (e, k, a, b, c) = rec;
    (e, PyBytes::new(py, &k).unbind(), a, b, c)
}

/// Enumerate every legal move of `rf` (see `Expr::legal_moves` for the contract).
fn enumerate_moves(rf: &poly::RatFn, n: u16) -> Vec<MoveRs> {
    let mut present = std::collections::BTreeSet::new();
    for m in rf.num.keys() {
        for bk in m.keys() {
            present.insert((bk.square, bk.i, bk.j));
        }
    }
    if present.is_empty() {
        return Vec::new();
    }
    let per_term: Vec<std::collections::BTreeSet<(bool, u16, u16)>> = rf
        .num
        .keys()
        .map(|m| m.keys().map(|bk| (bk.square, bk.i, bk.j)).collect())
        .collect();
    let mut moves: Vec<MoveRs> = Vec::new();
    for (square, i, j) in present {
        let menu = identity_menu(i, j, n);
        for (ident, params) in &menu {
            moves.push((square, i, j, 0, ident, params.clone()));
        }
        for (k, bset) in per_term.iter().enumerate() {
            if bset.contains(&(square, i, j)) {
                for (ident, params) in &menu {
                    moves.push((square, i, j, k + 1, ident, params.clone()));
                }
            }
        }
    }
    moves
}

/// An opaque, resident list of a node's legal moves.
///
/// Why this exists rather than just handing Python the list: materialising the moves as Python
/// tuples is what dominates enumeration. Measured on a 3000-character 6pt node (10 528 moves),
/// building them in Rust and converting takes **24.8 ms** against **16.5 ms** for the pure-Python
/// enumeration in `ampzip.search.actions` — the Python version wins because it reuses one
/// `params` list object across every term slot, while the conversion allocates a fresh tuple,
/// `str` and `list` per move. Keeping the list opaque skips that entirely: the search slices it
/// by index (`Expr::apply_slice`) and only ever materialises a move for a child it actually
/// keeps (`MoveList::get`).
#[pyclass]
struct MoveList {
    moves: Vec<MoveRs>,
}

#[pymethods]
impl MoveList {
    fn __len__(&self) -> usize {
        self.moves.len()
    }

    /// Materialise the `k`'th move as the usual
    /// `(square, i, j, term_slot, identity, params)` tuple.
    fn get(&self, k: usize) -> PyResult<Move> {
        let m = self
            .moves
            .get(k)
            .ok_or_else(|| PyValueError::new_err("move index out of range"))?;
        Ok((m.0, m.1, m.2, m.3, m.4.to_string(), m.5.clone()))
    }
}

/// A parsed expression kept as its syntax tree — NOT canonicalised, so a product of sums
/// is never expanded. The verifier evaluates it at on-shell points as a fraction pair.
#[pyclass]
struct Ast {
    ast: parse::Ast,
}

#[pymethods]
impl Ast {
    /// Exact value at an on-shell point, `num * den^{-1}` (raises on a vanishing `den`).
    fn eval(&self, lam: Vec<Vec<u64>>, lamt: Vec<Vec<u64>>, p: u64) -> PyResult<u64> {
        eval::eval_ast(&self.ast, &lam, &lamt, p).map_err(PyValueError::new_err)
    }

    /// Exact value at an on-shell point as `(num, den)` in F_p with no division performed
    /// (see `eval::eval_pair`); `den` may be 0 and is for the caller to treat as degenerate.
    fn eval_pair(&self, lam: Vec<Vec<u64>>, lamt: Vec<Vec<u64>>, p: u64) -> PyResult<(u64, u64)> {
        eval::eval_pair(&self.ast, &lam, &lamt, p).map_err(PyValueError::new_err)
    }
}

/// Parse a sympy-style string into a resident `Ast` (no canonicalisation, no expansion).
#[pyfunction]
fn parse_ast(expr: &str) -> PyResult<Ast> {
    Ok(Ast { ast: parse::parse(expr).map_err(PyValueError::new_err)? })
}

/// Sanity entry point.
#[pyfunction]
fn version() -> String {
    format!("ampzip_engine {}", env!("CARGO_PKG_VERSION"))
}

/// True if `expr` parses cleanly.
#[pyfunction]
fn parses(expr: &str) -> bool {
    parse::parse(expr).is_ok()
}

/// Exact F_p value at an on-shell point (brackets = 2x2 minors of lam/lamt).
#[pyfunction]
fn eval_onshell(expr: &str, lam: Vec<Vec<u64>>, lamt: Vec<Vec<u64>>, p: u64) -> PyResult<u64> {
    let ast = parse::parse(expr).map_err(PyValueError::new_err)?;
    eval::eval_ast(&ast, &lam, &lamt, p).map_err(PyValueError::new_err)
}

/// Parse into a `RatFn`, cancel common monomial factors and exactly divisible
/// opaque factors, then return its canonical string representation.
#[pyfunction]
fn canonical(expr: &str) -> PyResult<String> {
    poly::overflow_check_reset();
    let ast = parse::parse(expr).map_err(PyValueError::new_err)?;
    let mut rf = ident::ast_to_ratfn(&ast).map_err(PyValueError::new_err)?;
    poly::ratfn_cancel(&mut rf);
    check_overflow()?;
    Ok(poly::ratfn_to_string(&rf))
}

/// Apply one identity to the numerator bracket `bk(i,j)` of type `square`
/// (False -> ab, True -> sb), whole-expression. `identity` in
/// {"schouten","momentum","momentum_sq"}; `params` = [pk,pl] | [pout1,pout2] |
/// mom_add. Returns the resulting expression string (value-equal to the input).
#[pyfunction]
fn apply(expr: &str, square: bool, i: u16, j: u16, identity: &str, params: Vec<u16>, n: u16) -> PyResult<String> {
    poly::overflow_check_reset();
    poly::term_limit_hit_reset();
    let ast = parse::parse(expr).map_err(PyValueError::new_err)?;
    let rf = ident::ast_to_ratfn(&ast).map_err(PyValueError::new_err)?;
    let ret = ident::build_identity(square, i, j, identity, &params, n).map_err(PyValueError::new_err)?;
    let (target, _) = poly::make_bracket(square, i, j);
    let mut out = ident::substitute(&rf, target, &ret, 0); // 0 => process-default term cap
    check_term_limit()?;
    poly::ratfn_cancel(&mut out);
    check_overflow()?;
    Ok(poly::ratfn_to_string(&out))
}

/// A resident expression handle: keeps the canonical `RatFn` in Rust so a search/MCTS
/// loop can apply many moves without the parse + to_string round-trip on every step.
/// Immutable — every mutator returns a NEW `Expr` (cheap BTreeMap clone), which suits an
/// MCTS tree where parent states persist. Invariant: `rf` is always cancelled.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct Expr {
    rf: poly::RatFn,
}

#[pymethods]
impl Expr {
    /// Parse a sympy-style string once into the canonical form. `max_terms` bounds the
    /// numerator term count while the input is expanded (0 = the process default
    /// `AMPZIP_MAX_TERMS`); a breach raises a ValueError instead of expanding without bound.
    #[new]
    #[pyo3(signature = (expr, max_terms=0))]
    fn new(expr: &str, max_terms: usize) -> PyResult<Self> {
        poly::overflow_check_reset();
        let ast = parse::parse(expr).map_err(PyValueError::new_err)?;
        let limit = if max_terms > 0 { max_terms } else { poly::term_limit() };
        let mut rf = ident::ast_to_ratfn_capped(&ast, limit).map_err(PyValueError::new_err)?;
        poly::ratfn_cancel(&mut rf);
        check_overflow()?;
        Ok(Expr { rf })
    }

    fn __repr__(&self) -> String {
        format!("Expr({})", poly::ratfn_to_string(&self.rf))
    }

    /// Canonical sympy-style string of the current expression.
    fn to_str(&self) -> String {
        poly::ratfn_to_string(&self.rf)
    }

    /// The search's dedup key: blake2b-128 of the canonical string, computed here so the
    /// string itself never has to be built as a Python object. Byte-identical to
    /// `hashlib.blake2b(expr.to_str().encode(), digest_size=16).digest()`.
    fn key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &ratfn_key(&self.rf))
    }

    /// `apply` plus everything the search reads off the child, in one call:
    /// `(child, key, factored_bracket_count, bracket_count, num_terms)` — the sequential
    /// counterpart of one `apply_batch` record. Raises exactly when `apply` would.
    #[pyo3(signature = (square, i, j, identity, params, n, term_slot=0, max_terms=0))]
    #[allow(clippy::too_many_arguments)]
    fn apply_one(
        &self,
        py: Python<'_>,
        square: bool,
        i: u16,
        j: u16,
        identity: &str,
        params: Vec<u16>,
        n: u16,
        term_slot: usize,
        max_terms: usize,
    ) -> PyResult<ChildPy> {
        let e = self.apply(square, i, j, identity, params, n, term_slot, max_terms)?;
        let k = ratfn_key(&e.rf);
        let fbc = e.factored_bracket_count();
        let bc = e.bracket_count();
        let nt = e.num_terms();
        Ok((e, PyBytes::new(py, &k).unbind(), fbc, bc, nt))
    }

    /// Apply one identity to the numerator bracket `bk(i,j)` (square: False -> ab, True
    /// -> sb) and return a NEW handle. `term_slot=0` rewrites the whole expression (all
    /// occurrences); `term_slot=k>0` peels ONE occurrence from the engine's k-1'th
    /// numerator term (engine canonical order). `identity` in
    /// {"schouten","momentum","momentum_sq"}; `params` = [pk,pl] | [pout1,pout2] | mom_add.
    #[pyo3(signature = (square, i, j, identity, params, n, term_slot=0, max_terms=0))]
    #[allow(clippy::too_many_arguments)] // mirrors the RL action encoding
    fn apply(
        &self,
        square: bool,
        i: u16,
        j: u16,
        identity: &str,
        params: Vec<u16>,
        n: u16,
        term_slot: usize,
        // Per-call numerator term cap for whole-expression and single-term rewrites.
        // 0 uses the process default `AMPZIP_MAX_TERMS` (see poly::term_limit).
        // Passed explicitly so the caller can set it per rollout. The environment
        // value is cached per thread, and runner threads are reused across jobs.
        max_terms: usize,
    ) -> PyResult<Expr> {
        poly::overflow_check_reset();
        poly::term_limit_hit_reset();
        let ret = ident::build_identity(square, i, j, identity, &params, n).map_err(PyValueError::new_err)?;
        let (target, _) = poly::make_bracket(square, i, j);
        let mut out = if term_slot == 0 {
            ident::substitute(&self.rf, target, &ret, max_terms)
        } else {
            ident::substitute_term(&self.rf, target, &ret, term_slot - 1, max_terms).map_err(PyValueError::new_err)?
        };
        // Raise on a substitution blow-up BEFORE the costly cancel, so a single pathological
        // move aborts cheaply and the search skips it (existing `except ValueError: continue`).
        check_term_limit()?;
        poly::ratfn_cancel(&mut out);
        check_overflow()?;
        Ok(Expr { rf: out })
    }

    /// Every legal `(square, i, j, term_slot, identity, params)` move for this expression:
    /// `term_slot=0` (whole expression) for each bracket present in the numerator, plus
    /// `term_slot=k+1` for each numerator term that carries it. Port of
    /// `ampzip.search.actions.legal_moves`, same order — that Python version builds the
    /// per-term bracket sets by regex-scanning `term_monos()`, i.e. it formats every monomial
    /// to a string and re-parses it; here the sets come straight off the canonical `Mono`.
    ///
    /// No identity-specific masking: all 11/19/32 menu entries are offered once the bracket is
    /// present, and degenerate ones are rejected when the engine applies them.
    ///
    /// This materialises every move as a Python tuple, which for a large node costs more than
    /// the enumeration itself — prefer `legal_move_list` + `apply_slice` on the search's hot
    /// path. This interface also allows comparison with the Python move enumerator.
    fn legal_moves(&self, n: u16) -> Vec<Move> {
        enumerate_moves(&self.rf, n)
            .into_iter()
            .map(|m| (m.0, m.1, m.2, m.3, m.4.to_string(), m.5))
            .collect()
    }

    /// The same enumeration, kept resident in Rust as an opaque `MoveList` — see that type for
    /// why the search wants this instead of the tuple list.
    fn legal_move_list(&self, n: u16) -> MoveList {
        MoveList { moves: enumerate_moves(&self.rf, n) }
    }

    /// Apply `moves[offset .. offset+limit]` to this expression on worker threads. Same
    /// contract and same return shape as `apply_batch`, one entry per move in slice order;
    /// `limit=0` means "to the end". The move that produced result `k` is `offset + k`, so a
    /// caller only has to materialise `moves.get(offset + k)` for the children it keeps.
    #[pyo3(signature = (moves, offset, limit, n, max_terms=0, threads=0))]
    #[allow(clippy::too_many_arguments)] // slice bounds + the apply args, all caller-supplied
    fn apply_slice(
        &self,
        py: Python<'_>,
        moves: &MoveList,
        offset: usize,
        limit: usize,
        n: u16,
        max_terms: usize,
        threads: usize,
    ) -> Vec<Option<ChildPy>> {
        let all = &moves.moves;
        let start = offset.min(all.len());
        let end = if limit == 0 { all.len() } else { (start + limit).min(all.len()) };
        self.apply_moves_par(py, &all[start..end], n, max_terms, threads)
    }

    /// `apply_slice` with the rejection cause: one `(code, child)` per move in slice order,
    /// `code` 0 = applied (`child` present), 1 = degenerate identity / illegal term slot,
    /// 2 = term cap breached, 3 = coefficient overflow (`child` None). Lets the search count
    /// skipped moves by cause without re-applying them on the calling thread.
    #[pyo3(signature = (moves, offset, limit, n, max_terms=0, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn apply_slice_coded(
        &self,
        py: Python<'_>,
        moves: &MoveList,
        offset: usize,
        limit: usize,
        n: u16,
        max_terms: usize,
        threads: usize,
    ) -> Vec<(u8, Option<ChildPy>)> {
        let all = &moves.moves;
        let start = offset.min(all.len());
        let end = if limit == 0 { all.len() } else { (start + limit).min(all.len()) };
        self.apply_moves_par_coded(py, &all[start..end], n, max_terms, threads)
            .into_iter()
            .map(|(code, rec)| (code, rec.map(|r| child_py(py, r))))
            .collect()
    }

    /// Apply MANY moves to this expression on worker threads, returning one entry per move
    /// **in input order**: `None` where the sequential `apply()` would raise, else
    /// `(child, key, factored_bracket_count, bracket_count, num_terms)` — `key` is the
    /// 16-byte dedup digest (`Expr.key`), so no canonical string ever crosses the GIL.
    ///
    /// The four child quantities are computed inside the worker, so the caller never pays
    /// `to_str()` (a 20k-character format on a deep 6pt child) or the O(terms x brackets)
    /// `factored_bracket_count()` under the GIL.
    ///
    /// The GIL is released for the whole batch. Work is handed out through an atomic counter
    /// rather than by static sharding, because per-move cost is heavy-tailed (measured on a 6pt
    /// node: median 1.5 ms, max 2.8 s) and contiguous shards would leave most workers idle.
    /// `threads=0` -> `available_parallelism()`, which honours the process's CPU affinity mask
    /// and cgroup quota, so it respects a `taskset` / SLURM / container allocation.
    ///
    /// Callers must keep the ORDER-SENSITIVE bookkeeping (budget accounting, dedup, best
    /// tracking, goal break) serial and replay it over this list in order — then the result is
    /// bit-identical to applying the moves one at a time. See `ampzip/search/gbfs.py`.
    #[pyo3(signature = (moves, n, max_terms=0, threads=0))]
    fn apply_batch(
        &self,
        py: Python<'_>,
        moves: Vec<Move>,
        n: u16,
        max_terms: usize,
        threads: usize,
    ) -> Vec<Option<ChildPy>> {
        let owned: Vec<MoveRs> = moves
            .iter()
            .map(|(sq, i, j, ts, ident, p)| {
                // The identity tag has to become 'static to share the worker core with
                // `apply_slice`; an unknown one is passed through as "" and `apply_core`
                // rejects it exactly as `build_identity` would.
                let tag: &'static str = match ident.as_str() {
                    "schouten" => "schouten",
                    "momentum" => "momentum",
                    "momentum_sq" => "momentum_sq",
                    _ => "",
                };
                (*sq, *i, *j, *ts, tag, p.clone())
            })
            .collect();
        self.apply_moves_par(py, &owned, n, max_terms, threads)
    }

    /// Number of additive terms in the numerator (canonical form). Re-read after every
    /// apply — substitution merges/splits terms so indices shift.
    fn num_terms(&self) -> usize {
        self.rf.num.len()
    }

    /// Raw numerator bracket count: the sum of bracket powers over all numerator
    /// monomials. Denominator brackets are excluded.
    fn bracket_count(&self) -> u64 {
        self.rf
            .num
            .keys()
            .map(|m| m.values().map(|e| *e as u64).sum::<u64>())
            .sum()
    }

    /// Factor-aware search heuristic: numerator bracket multiplicity AFTER pulling the
    /// common monomial factor out of the numerator sum (ONE level, like sympy
    /// `factor_terms` — no recursion into partial-overlap subgroups). A bracket carried
    /// by EVERY term is counted once, not once per term, so
    /// `ab(1,2)ab(2,3)ab(3,4) + ab(1,2)ab(2,3)ab(4,5)` scores 4, not 6. Equals
    /// `bracket_count()` for a single term. Denominator excluded, as in `bracket_count`.
    ///
    /// Closed form: `raw - (num_terms - 1) * common_total`, where `raw` is
    /// `bracket_count()` and `common_total` is the sum over brackets of the min exponent
    /// across all terms (a common bracket must appear in every term, hence in the first).
    fn factored_bracket_count(&self) -> u64 {
        let nterms = self.rf.num.len();
        let raw: u64 = self
            .rf
            .num
            .keys()
            .map(|m| m.values().map(|e| *e as u64).sum::<u64>())
            .sum();
        if nterms <= 1 {
            return raw;
        }
        // Candidates for the common factor are the first term's brackets: a bracket
        // shared by every term must appear there. For each, min exponent over all terms.
        let mut common_total = 0u64;
        if let Some((first, _)) = self.rf.num.iter().next() {
            for (bk, &e0) in first {
                let mut mn = e0;
                for m in self.rf.num.keys() {
                    let p = *m.get(bk).unwrap_or(&0);
                    if p < mn {
                        mn = p;
                    }
                    if mn == 0 {
                        break;
                    }
                }
                common_total += mn as u64;
            }
        }
        raw - (nterms as u64 - 1) * common_total
    }

    /// Distinct brackets present in the numerator, as `(square, i, j)`, canonical order —
    /// for building an action mask.
    fn distinct_brackets(&self) -> Vec<(bool, u16, u16)> {
        let mut set = std::collections::BTreeSet::new();
        for m in self.rf.num.keys() {
            for bk in m.keys() {
                set.insert((bk.square, bk.i, bk.j));
            }
        }
        set.into_iter().collect()
    }

    fn is_zero(&self) -> bool {
        self.rf.is_zero()
    }

    /// Exact F_p value at an on-shell point, computed straight from the resident RatFn
    /// (no re-parse). Agrees with the stateless `eval_onshell`.
    fn eval_onshell(&self, lam: Vec<Vec<u64>>, lamt: Vec<Vec<u64>>, p: u64) -> PyResult<u64> {
        eval::eval_ratfn_modp(&self.rf, &lam, &lamt, p).map_err(PyValueError::new_err)
    }

    /// Per-term monomial strings (coefficient stripped), engine canonical order. The
    /// constant term is reported as "1". Used by the Python move enumerator and
    /// action mask to identify the brackets in each numerator term.
    fn term_monos(&self) -> Vec<String> {
        self.rf
            .num
            .keys()
            .map(|m| {
                let s = poly::mono_to_string(m);
                if s.is_empty() {
                    "1".to_string()
                } else {
                    s
                }
            })
            .collect()
    }

    /// Per-term `(monomial_string, coefficient)`, engine canonical order.
    fn terms(&self) -> Vec<(String, i128)> {
        self.rf
            .num
            .iter()
            .map(|(m, c)| {
                let s = poly::mono_to_string(m);
                (if s.is_empty() { "1".to_string() } else { s }, *c)
            })
            .collect()
    }

    /// Structured numerator: per term `(coeff, [(square, i, j, exponent), ...])`, engine
    /// canonical order. For building observation tensors without string parsing.
    fn term_powers(&self) -> Vec<TermPow> {
        self.rf
            .num
            .iter()
            .map(|(m, c)| (*c, m.iter().map(|(bk, e)| (bk.square, bk.i, bk.j, *e)).collect()))
            .collect()
    }

    /// Denominator brackets as `(square, i, j, exponent)`, canonical order (always a
    /// single monomial; empty if the expression is a polynomial).
    fn den_brackets(&self) -> Vec<BracketPow> {
        self.rf.den.iter().map(|(bk, e)| (bk.square, bk.i, bk.j, *e)).collect()
    }

    /// Opaque (sum) denominator factors, each as its list of terms
    /// `[(coeff, [(square, i, j, exponent), ...]), ...]`. Empty for the common monomial
    /// case; populated only for B2 opaque poles (e.g. the spurious pole of eq 5.28).
    fn den_opaque_terms(&self) -> Vec<Vec<TermPow>> {
        self.rf
            .den_opaque
            .iter()
            .map(|poly| {
                poly.iter()
                    .map(|(m, c)| (*c, m.iter().map(|(bk, e)| (bk.square, bk.i, bk.j, *e)).collect()))
                    .collect()
            })
            .collect()
    }

    /// Denominator-aware search bias: how much of the (monomial) denominator is NOT yet
    /// cancellable against the numerator. For each denominator bracket b^e, the cancellable
    /// amount is the min over numerator terms of b's exponent there (a bracket cancels only
    /// if EVERY term carries it); the remaining gap is sum of max(0, e - that_min). 0 means
    /// the whole denominator can cancel now. Mirrors `ratfn_cancel`'s inner loop, computed
    /// in Rust so the den-aware `value_fn` stays off the Python hot path (was ~2.3x wall).
    fn den_match_remaining(&self) -> u64 {
        let mut rem = 0u64;
        for (bk, &e) in &self.rf.den {
            let mut mn = u32::MAX;
            for m in self.rf.num.keys() {
                let p = *m.get(bk).unwrap_or(&0);
                if p < mn {
                    mn = p;
                }
                if mn == 0 {
                    break;
                }
            }
            if mn == u32::MAX {
                mn = 0; // empty numerator (degenerate; rf is cancelled so this is rare)
            }
            rem += e.saturating_sub(mn) as u64;
        }
        rem
    }
}

impl Expr {
    /// Worker core shared by `apply_batch` and `apply_slice`: apply every move in `moves` off
    /// the GIL and return the results **in input order**. See `apply_batch` for the contract.
    fn apply_moves_par(
        &self,
        py: Python<'_>,
        moves: &[MoveRs],
        n: u16,
        max_terms: usize,
        threads: usize,
    ) -> Vec<Option<ChildPy>> {
        self.apply_moves_par_coded(py, moves, n, max_terms, threads)
            .into_iter()
            .map(|(_, rec)| rec.map(|r| child_py(py, r)))
            .collect()
    }

    /// The worker loop behind `apply_slice` / `apply_batch` / `apply_slice_coded`: one
    /// `(code, record)` per move in input order, computed off the GIL.
    fn apply_moves_par_coded(
        &self,
        py: Python<'_>,
        moves: &[MoveRs],
        n: u16,
        max_terms: usize,
        threads: usize,
    ) -> Vec<(u8, Option<ChildRec>)> {
        let nthreads = if threads == 0 {
            std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)
        } else {
            threads.max(1)
        }
        .min(moves.len().max(1));
        let rf = &self.rf;
        // pyo3 0.29 renamed `Python::allow_threads` -> `Python::detach`.
        let slots: Vec<(u8, Option<ChildRec>)> = py.detach(move || {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let parts: Vec<Vec<(usize, (u8, Option<ChildRec>))>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..nthreads)
                    .map(|_| {
                        let next = &next;
                        s.spawn(move || {
                            let mut out: Vec<(usize, (u8, Option<ChildRec>))> = Vec::new();
                            loop {
                                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if k >= moves.len() {
                                    break;
                                }
                                let (sq, i, j, ts, ident, params) = &moves[k];
                                let r = match apply_core(rf, *sq, *i, *j, ident, params, n, *ts, max_terms) {
                                    Ok(out_rf) => {
                                        let e = Expr { rf: out_rf };
                                        let s = ratfn_key(&e.rf);
                                        let fbc = e.factored_bracket_count();
                                        let bc = e.bracket_count();
                                        let nt = e.num_terms();
                                        (0u8, Some((e, s, fbc, bc, nt)))
                                    }
                                    Err(code) => (code, None),
                                };
                                out.push((k, r));
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut slots: Vec<(u8, Option<ChildRec>)> = Vec::new();
            slots.resize_with(moves.len(), || (REJECT_DEGENERATE, None));
            for part in parts {
                for (k, r) in part {
                    slots[k] = r;
                }
            }
            slots
        });
        slots
    }
}

#[pymodule]
fn ampzip_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(parses, m)?)?;
    m.add_function(wrap_pyfunction!(eval_onshell, m)?)?;
    m.add_function(wrap_pyfunction!(canonical, m)?)?;
    m.add_function(wrap_pyfunction!(apply, m)?)?;
    m.add_class::<Expr>()?;
    m.add_class::<MoveList>()?;
    m.add_function(wrap_pyfunction!(parse_ast, m)?)?;
    m.add_class::<Ast>()?;
    Ok(())
}
