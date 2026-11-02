//! Canonical rational-function representation over spinor brackets.
//!
//! A spinor-helicity expression is represented as `num / (den * product(den_opaque))`,
//! with an integer-coefficient polynomial numerator, a monomial denominator part
//! and optional polynomial denominator factors.
//! `ratfn_cancel` cancels common monomial factors and attempts exact division
//! by opaque factors. It does not compute a general polynomial GCD. This module
//! contains algebraic operations only, with no parsing or Python interface.
use std::cell::Cell;
use std::collections::BTreeMap;

thread_local! {
    /// Set by checked coefficient arithmetic when an i128 op overflows, read and
    /// cleared at the PyO3 boundary in lib.rs. The internal signatures stay infallible
    /// and overflow still surfaces. It goes away when coefficients move to bigint.
    static OVERFLOW: Cell<bool> = const { Cell::new(false) };
}

/// Flag a coefficient overflow on the current thread. The checked arithmetic in this
/// module and in `ident.rs` calls it.
pub(crate) fn set_overflow() {
    OVERFLOW.with(|f| f.set(true));
}

/// Return whether an overflow was flagged since the last call, and clear it.
pub(crate) fn overflow_check_reset() -> bool {
    OVERFLOW.with(|f| f.replace(false))
}

thread_local! {
    /// Set when a whole-expression or single-term rewrite exceeds its numerator
    /// term limit. Read and cleared at the PyO3 boundary, as for `OVERFLOW`. This
    /// allows an application to stop when intermediate expressions grow too large,
    /// for example while constructing `ret^power`.
    static TERM_LIMIT_HIT: Cell<bool> = const { Cell::new(false) };
    /// Cached per-thread term cap, 0 until it is read from the environment.
    static TERM_LIMIT: Cell<usize> = const { Cell::new(0) };
}

/// Flag a term-limit violation on the current thread during a whole-expression or
/// single-term rewrite.
pub(crate) fn set_term_limit_hit() {
    TERM_LIMIT_HIT.with(|f| f.set(true));
}

/// Return whether a term-limit blow-up was flagged since the last call, and clear it.
pub(crate) fn term_limit_hit_reset() -> bool {
    TERM_LIMIT_HIT.with(|f| f.replace(false))
}

/// Default numerator term limit used during parsing and rewriting when no explicit
/// limit is supplied. During a rewrite, exceeding the limit rejects the move so the
/// search can continue. The cap controls combinatorial growth such as `ret^p`, but
/// may also exclude useful intermediate expressions. A positive integer in
/// `AMPZIP_MAX_TERMS` overrides 50_000. Unset, invalid or zero values fall back to
/// 50_000. The value is read once per thread and cached.
pub(crate) fn term_limit() -> usize {
    TERM_LIMIT.with(|c| {
        let v = c.get();
        if v != 0 {
            return v;
        }
        let lim = std::env::var("AMPZIP_MAX_TERMS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(50_000);
        c.set(lim);
        lim
    })
}

/// A bracket, canonicalized so that `i < j`. The caller carries the antisymmetry sign
/// through [`make_bracket`]. `square=false` is the angle bracket `ab` and `true` is `sb`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct Bracket {
    pub square: bool,
    pub i: u16,
    pub j: u16,
}

/// Canonicalize `bracket(a,b)` to `i<j` and return `(bracket, sign)`, with sign 0 when
/// `a == b` and the bracket is identically zero.
pub(crate) fn make_bracket(square: bool, a: u16, b: u16) -> (Bracket, i128) {
    use std::cmp::Ordering::*;
    match a.cmp(&b) {
        Less => (Bracket { square, i: a, j: b }, 1),
        Greater => (Bracket { square, i: b, j: a }, -1),
        Equal => (Bracket { square, i: a, j: b }, 0),
    }
}

/// A monomial, bracket -> positive exponent.
pub(crate) type Mono = BTreeMap<Bracket, u32>;
/// A polynomial, monomial -> nonzero integer coefficient.
pub(crate) type Poly = BTreeMap<Mono, i128>;

/// A rational function `num / (den * prod(den_opaque))`.
///
/// `den` is a monomial, a product of brackets. `den_opaque` lists the sum-denominator
/// factors, kept symbolic and never expanded into `den`, which supports spurious poles
/// such as `<2|(6+1)|5]` of eq 5.28.
///
/// `den_opaque` is empty for every input with a monomial denominator, which is every case
/// of the public sets. When it is empty, every RatFn operation runs its monomial-only path,
/// guarded by `den_opaque.is_empty()`, so the common case is byte-identical and pays one
/// `Vec::is_empty()` check.
#[derive(Clone, Debug)]
pub(crate) struct RatFn {
    pub num: Poly,
    pub den: Mono,
    pub den_opaque: Vec<Poly>,
}

// --- low-level poly and mono helpers, shared with ident.rs --- //
pub(crate) fn poly_add_term(poly: &mut Poly, mono: Mono, coeff: i128) {
    if coeff == 0 {
        return;
    }
    let slot = poly.entry(mono).or_insert(0);
    match slot.checked_add(coeff) {
        Some(v) => *slot = v,
        None => {
            set_overflow();
            *slot = slot.wrapping_add(coeff);
        }
    }
}
/// `poly_add_term` without the overflow check, for `ratfn_cancel`, which only re-adds
/// coefficients that already coexisted in one polynomial and so cannot overflow.
fn poly_add_term_unchecked(poly: &mut Poly, mono: Mono, coeff: i128) {
    if coeff == 0 {
        return;
    }
    *poly.entry(mono).or_insert(0) += coeff;
}
pub(crate) fn poly_prune(poly: &mut Poly) {
    poly.retain(|_, c| *c != 0);
}
fn mono_mul(a: &Mono, b: &Mono) -> Mono {
    let mut r = a.clone();
    for (k, e) in b {
        *r.entry(*k).or_insert(0) += e;
    }
    r
}
fn mono_lcm(a: &Mono, b: &Mono) -> Mono {
    let mut r = a.clone();
    for (k, e) in b {
        let cur = r.entry(*k).or_insert(0);
        if *e > *cur {
            *cur = *e;
        }
    }
    r
}
fn mono_div(a: &Mono, b: &Mono) -> Mono {
    let mut r = a.clone();
    for (k, e) in b {
        *r.entry(*k).or_insert(0) -= e; // callers guarantee a >= b (lcm)
    }
    r.retain(|_, e| *e != 0);
    r
}

impl RatFn {
    pub(crate) fn zero() -> RatFn {
        RatFn { num: Poly::new(), den: Mono::new(), den_opaque: Vec::new() }
    }
    pub(crate) fn is_zero(&self) -> bool {
        self.num.is_empty()
    }
    pub(crate) fn from_int(n: i128) -> RatFn {
        let mut num = Poly::new();
        if n != 0 {
            num.insert(Mono::new(), n);
        }
        RatFn { num, den: Mono::new(), den_opaque: Vec::new() }
    }
    pub(crate) fn from_mono_coeff(mono: Mono, coeff: i128) -> RatFn {
        let mut num = Poly::new();
        if coeff != 0 {
            num.insert(mono, coeff);
        }
        RatFn { num, den: Mono::new(), den_opaque: Vec::new() }
    }
    /// 1 / den, where `den` is a monomial.
    pub(crate) fn recip_of_mono(den: &Mono) -> RatFn {
        let mut num = Poly::new();
        num.insert(Mono::new(), 1);
        RatFn { num, den: den.clone(), den_opaque: Vec::new() }
    }
}

pub(crate) fn ratfn_add(a: &RatFn, b: &RatFn) -> RatFn {
    if a.is_zero() {
        return b.clone();
    }
    if b.is_zero() {
        return a.clone();
    }
    let lcm = mono_lcm(&a.den, &b.den);
    let fa = mono_div(&lcm, &a.den);
    let fb = mono_div(&lcm, &b.den);
    // The fast path of the common case, no sum-denominator factors, is the monomial-LCM add.
    if a.den_opaque.is_empty() && b.den_opaque.is_empty() {
        let mut num = Poly::new();
        for (m, c) in &a.num {
            poly_add_term(&mut num, mono_mul(m, &fa), *c);
        }
        for (m, c) in &b.num {
            poly_add_term(&mut num, mono_mul(m, &fb), *c);
        }
        poly_prune(&mut num);
        return RatFn { num, den: lcm, den_opaque: Vec::new() };
    }
    // With sum-denominator factors the common denominator is the monomial LCM times the
    // multiset union of the factors. Each distinct factor is treated as an indivisible prime.
    // Each side's numerator is multiplied by the monomial cofactor and by the factors the
    // other side contributes that this side lacks, the multiset difference.
    let union = opaque_union(&a.den_opaque, &b.den_opaque);
    let a_missing = opaque_diff(&union, &a.den_opaque);
    let b_missing = opaque_diff(&union, &b.den_opaque);
    let mut pa = poly_scale_mono(&a.num, &fa);
    for s in &a_missing {
        pa = poly_mul(&pa, s);
    }
    let mut pb = poly_scale_mono(&b.num, &fb);
    for s in &b_missing {
        pb = poly_mul(&pb, s);
    }
    let mut num = Poly::new();
    for (m, c) in pa.iter().chain(pb.iter()) {
        poly_add_term(&mut num, m.clone(), *c);
    }
    poly_prune(&mut num);
    RatFn { num, den: lcm, den_opaque: union }
}

/// Polynomial product with checked coefficients.
fn poly_mul(a: &Poly, b: &Poly) -> Poly {
    let mut r = Poly::new();
    for (ma, ca) in a {
        for (mb, cb) in b {
            let c = match ca.checked_mul(*cb) {
                Some(v) => v,
                None => {
                    set_overflow();
                    ca.wrapping_mul(*cb)
                }
            };
            poly_add_term(&mut r, mono_mul(ma, mb), c);
        }
    }
    r
}

/// Multiply every monomial of `a` by `m`.
fn poly_scale_mono(a: &Poly, m: &Mono) -> Poly {
    if m.is_empty() {
        return a.clone();
    }
    let mut r = Poly::new();
    for (ma, ca) in a {
        poly_add_term(&mut r, mono_mul(ma, m), *ca);
    }
    r
}

/// Multiset union of two opaque-factor lists, each distinct polynomial kept to the larger
/// of its two multiplicities. The lists hold at most a few factors, so linear scans suffice.
fn opaque_union(a: &[Poly], b: &[Poly]) -> Vec<Poly> {
    let mut out = a.to_vec();
    let mut done: Vec<&Poly> = Vec::new();
    for sb in b {
        if done.contains(&sb) {
            continue; // each distinct factor of b handled once
        }
        done.push(sb);
        let ca = a.iter().filter(|x| *x == sb).count();
        let cb = b.iter().filter(|x| *x == sb).count();
        for _ in ca..cb {
            out.push(sb.clone());
        }
    }
    out
}

/// Multiset difference `union - a`, the opaque factors with multiplicity that `a` lacks.
fn opaque_diff(union: &[Poly], a: &[Poly]) -> Vec<Poly> {
    let mut used = vec![false; a.len()];
    let mut out = Vec::new();
    for u in union {
        let mut matched = false;
        for (idx, x) in a.iter().enumerate() {
            if !used[idx] && x == u {
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            out.push(u.clone());
        }
    }
    out
}

pub(crate) fn ratfn_mul(a: &RatFn, b: &RatFn) -> RatFn {
    if a.is_zero() || b.is_zero() {
        return RatFn::zero();
    }
    let mut num = Poly::new();
    for (ma, ca) in &a.num {
        for (mb, cb) in &b.num {
            let c = match ca.checked_mul(*cb) {
                Some(v) => v,
                None => {
                    set_overflow();
                    ca.wrapping_mul(*cb)
                }
            };
            poly_add_term(&mut num, mono_mul(ma, mb), c);
        }
    }
    poly_prune(&mut num);
    // The sum-denominator factors of a product multiply, so both operands' lists are
    // concatenated. They are empty in the common monomial case and nothing is allocated.
    let den_opaque = if a.den_opaque.is_empty() && b.den_opaque.is_empty() {
        Vec::new()
    } else {
        let mut v = a.den_opaque.clone();
        v.extend(b.den_opaque.iter().cloned());
        v
    };
    RatFn { num, den: mono_mul(&a.den, &b.den), den_opaque }
}

pub(crate) fn ratfn_recip(a: &RatFn) -> Result<RatFn, String> {
    if a.is_zero() {
        return Err("division by zero".to_string());
    }
    if !a.den_opaque.is_empty() {
        // 1/a would move the sum factors into the numerator. Only a negative power of a
        // sum-denominator expression reaches this, which no supported input produces.
        return Err("reciprocal of expression with opaque denominator factor".to_string());
    }
    if a.num.len() != 1 {
        return Err("reciprocal of non-monomial (denominator would be a sum)".to_string());
    }
    let (m, c) = a.num.iter().next().unwrap();
    if *c != 1 && *c != -1 {
        return Err(format!("reciprocal with non-unit coefficient {}", c));
    }
    // 1 / (c * m / a.den) = c * a.den / m with c = ±1.
    let mut num = Poly::new();
    num.insert(a.den.clone(), *c);
    Ok(RatFn { num, den: m.clone(), den_opaque: Vec::new() })
}

/// Cancel common bracket powers between the numerator and the denominator, like terms
/// being combined already. First the monomial part in `cancel_mono`, then, if present, the
/// sum factors in `cancel_opaque`, the construct-then-cancel step.
pub(crate) fn ratfn_cancel(rf: &mut RatFn) {
    if rf.num.is_empty() {
        rf.den = Mono::new();
        rf.den_opaque.clear();
        return;
    }
    cancel_mono(rf);
    if !rf.den_opaque.is_empty() {
        cancel_opaque(rf);
    }
}

/// Cancel common bracket powers between the numerator and the monomial denominator.
fn cancel_mono(rf: &mut RatFn) {
    if rf.den.is_empty() {
        return;
    }
    let mut canc: BTreeMap<Bracket, u32> = BTreeMap::new();
    for (bk, &d) in &rf.den {
        let mut mn = u32::MAX;
        for m in rf.num.keys() {
            let e = *m.get(bk).unwrap_or(&0);
            if e < mn {
                mn = e;
            }
            if mn == 0 {
                break;
            }
        }
        let c = d.min(mn);
        if c > 0 {
            canc.insert(*bk, c);
        }
    }
    if canc.is_empty() {
        return;
    }
    for (bk, c) in &canc {
        *rf.den.get_mut(bk).unwrap() -= c;
    }
    rf.den.retain(|_, e| *e != 0);
    let mut newnum = Poly::new();
    for (m, c) in &rf.num {
        let mut m2 = m.clone();
        for (bk, cc) in &canc {
            if let Some(e) = m2.get_mut(bk) {
                *e -= cc;
            }
        }
        m2.retain(|_, e| *e != 0);
        poly_add_term_unchecked(&mut newnum, m2, *c);
    }
    poly_prune(&mut newnum);
    rf.num = newnum;
}

/// For each sum factor `S` in the denominator, attempt exact free-polynomial division of
/// the numerator by `S`. On a zero remainder the numerator becomes the quotient and `S` is
/// dropped, the factor having been constructed and cancelled. A factor that does not
/// divide, a physical pole or one not yet constructed, is kept. After any cancellation the
/// quotient may share brackets with the monomial denominator, so `cancel_mono` runs again.
fn cancel_opaque(rf: &mut RatFn) {
    let opaque = std::mem::take(&mut rf.den_opaque);
    let mut remaining = Vec::new();
    let mut changed = false;
    for s in opaque {
        if let Some(q) = poly_exact_div(&rf.num, &s) {
            rf.num = q;
            changed = true;
        } else {
            remaining.push(s);
        }
    }
    rf.den_opaque = remaining;
    if changed {
        if rf.num.is_empty() {
            rf.den = Mono::new();
            rf.den_opaque.clear();
            return;
        }
        cancel_mono(rf);
    }
}

/// Graded-lexicographic ordering key for a monomial, the total degree and then the sorted
/// `(bracket, exp)` list. It gives a well-defined leading term, so polynomial division is exact.
fn mono_key(m: &Mono) -> (u32, Vec<(Bracket, u32)>) {
    (m.values().sum(), m.iter().map(|(b, e)| (*b, *e)).collect())
}

/// Leading `(monomial, coeff)` of a polynomial under `mono_key`, or None when empty.
fn poly_leading(p: &Poly) -> Option<(Mono, i128)> {
    p.iter()
        .max_by(|a, b| mono_key(a.0).cmp(&mono_key(b.0)))
        .map(|(m, c)| (m.clone(), *c))
}

/// True iff monomial `a` divides `b`, every bracket power in `a` being at most that in `b`.
fn mono_divides(a: &Mono, b: &Mono) -> bool {
    a.iter().all(|(k, e)| b.get(k).is_some_and(|be| be >= e))
}

/// Exact quotient `num / s` as free multivariate polynomials over the brackets, or `None`
/// when `s` does not divide `num` exactly, by a nonzero remainder or a non-divisible
/// coefficient. Standard leading-term reduction under the `mono_key` order.
fn poly_exact_div(num: &Poly, s: &Poly) -> Option<Poly> {
    let (s_lm, s_lc) = poly_leading(s)?;
    let mut r = num.clone();
    let mut q = Poly::new();
    while let Some((r_lm, r_lc)) = poly_leading(&r) {
        if !mono_divides(&s_lm, &r_lm) || r_lc % s_lc != 0 {
            return None;
        }
        let qm = mono_div(&r_lm, &s_lm); // exact: mono_divides guarantees r_lm >= s_lm
        let qc = r_lc / s_lc;
        for (sm, sc) in s {
            let prod = match qc.checked_mul(*sc) {
                Some(v) => v,
                None => {
                    set_overflow();
                    qc.wrapping_mul(*sc)
                }
            };
            poly_add_term(&mut r, mono_mul(sm, &qm), -prod);
        }
        poly_prune(&mut r);
        poly_add_term(&mut q, qm, qc);
    }
    Some(q)
}

/// Build a monomial from a list of `(square,a,b)` brackets, `None` when a bracket with
/// a==b is identically zero. Returns `(mono, product_of_signs)`.
pub(crate) fn mono_from(brs: &[(bool, u16, u16)]) -> Option<(Mono, i128)> {
    let mut m = Mono::new();
    let mut sign = 1i128;
    for &(sq, a, b) in brs {
        let (bk, s) = make_bracket(sq, a, b);
        if s == 0 {
            return None;
        }
        sign *= s;
        *m.entry(bk).or_insert(0) += 1;
    }
    Some((m, sign))
}

fn bracket_to_string(b: &Bracket) -> String {
    format!("{}({}, {})", if b.square { "sb" } else { "ab" }, b.i, b.j)
}
pub(crate) fn mono_to_string(m: &Mono) -> String {
    let mut parts = Vec::new();
    for (bk, e) in m {
        if *e == 1 {
            parts.push(bracket_to_string(bk));
        } else {
            parts.push(format!("{}**{}", bracket_to_string(bk), e));
        }
    }
    parts.join("*")
}

pub(crate) fn ratfn_to_string(rf: &RatFn) -> String {
    if rf.num.is_empty() {
        return "0".to_string();
    }
    let mut s = String::new();
    for (idx, (m, c)) in rf.num.iter().enumerate() {
        let (neg, ac) = if *c < 0 { (true, -*c) } else { (false, *c) };
        let body = if m.is_empty() {
            format!("{}", ac)
        } else if ac == 1 {
            mono_to_string(m)
        } else {
            format!("{}*{}", ac, mono_to_string(m))
        };
        if idx == 0 {
            if neg {
                s.push('-');
            }
            s.push_str(&body);
        } else {
            s.push_str(if neg { " - " } else { " + " });
            s.push_str(&body);
        }
    }
    let mut out = if rf.den.is_empty() {
        s
    } else {
        format!("({})/({})", s, mono_to_string(&rf.den))
    };
    // Sum factors ride in the denominator as `* opaque_den(S)`, which parses back to 1/S,
    // so the round trip is exact. In the common case the list is empty and `out` is unchanged.
    for sp in &rf.den_opaque {
        let inner = ratfn_to_string(&RatFn { num: sp.clone(), den: Mono::new(), den_opaque: Vec::new() });
        out = format!("({})*opaque_den({})", out, inner);
    }
    out
}
