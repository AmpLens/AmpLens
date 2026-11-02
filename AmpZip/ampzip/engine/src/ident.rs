//! The rewrite layer. It converts a parsed `Ast` into the canonical `RatFn`, builds the
//! replacement for each identity and substitutes a bracket.
use crate::parse::Ast;
use crate::poly::*;

/// Evaluate a parsed expression into the canonical rational-function form under the
/// process-default term cap of [`ast_to_ratfn_capped`].
pub(crate) fn ast_to_ratfn(a: &Ast) -> Result<RatFn, String> {
    ast_to_ratfn_capped(a, term_limit())
}

fn parse_cap_err(limit: usize) -> String {
    format!(
        "numerator term count exceeded the term cap ({}) during parse — expanding this input is \
         not bounded by the search budget; raise max_terms or rewrite the input",
        limit
    )
}

/// Evaluate a parsed expression into the canonical rational-function form, refusing to let any
/// intermediate numerator grow past `limit` terms. Expanding the input, a high power of a sum
/// or a product of long sums, happens before any search budget applies and holds the GIL, so
/// it is bounded here by the per-move term cap the search uses, `Expr.new(expr, max_terms=...)`
/// with the default `AMPZIP_MAX_TERMS` of 50000. Inputs that fit the cap evaluate as before.
pub(crate) fn ast_to_ratfn_capped(a: &Ast, limit: usize) -> Result<RatFn, String> {
    match a {
        Ast::Int(n) => Ok(RatFn::from_int(*n as i128)),
        Ast::Big(d) => d
            .parse::<i128>()
            .map(RatFn::from_int)
            .map_err(|_| format!("integer coefficient {} exceeds the engine's i128 range", d)),
        Ast::ImagUnit => Err("the rewrite engine takes integer coefficients only (I)".to_string()),
        Ast::Sqrt(q) => Err(format!("the rewrite engine takes integer coefficients only (sqrt({}))", q)),
        Ast::Param(n) => Err(format!("the rewrite engine takes integer coefficients only (parameter '{}')", n)),
        Ast::Bracket { square, i, j } => {
            let (bk, s) = make_bracket(*square, *i as u16, *j as u16);
            if s == 0 {
                return Ok(RatFn::zero());
            }
            let mut m = Mono::new();
            m.insert(bk, 1);
            Ok(RatFn::from_mono_coeff(m, s))
        }
        Ast::Neg(x) => {
            let mut r = ast_to_ratfn_capped(x, limit)?;
            for c in r.num.values_mut() {
                *c = match c.checked_neg() {
                    Some(v) => v,
                    None => {
                        set_overflow();
                        c.wrapping_neg()
                    }
                };
            }
            Ok(r)
        }
        Ast::Add(xs) => {
            let mut acc = RatFn::zero();
            for x in xs {
                acc = ratfn_add(&acc, &ast_to_ratfn_capped(x, limit)?);
                if acc.num.len() > limit {
                    return Err(parse_cap_err(limit));
                }
            }
            Ok(acc)
        }
        Ast::Mul(xs) => {
            let mut acc = RatFn::from_int(1);
            for x in xs {
                acc = ratfn_mul(&acc, &ast_to_ratfn_capped(x, limit)?);
                if acc.num.len() > limit {
                    return Err(parse_cap_err(limit));
                }
            }
            Ok(acc)
        }
        Ast::Pow(b, e) => {
            let rb = ast_to_ratfn_capped(b, limit)?;
            let base = if *e >= 0 { rb } else { ratfn_recip(&rb)? };
            // `ratfn_pow` step by step, so a blow-up is caught at the first power that
            // crosses the cap, before the whole tower is expanded.
            let mut r = RatFn::from_int(1);
            for _ in 0..e.unsigned_abs() {
                r = ratfn_mul(&r, &base);
                if r.num.len() > limit {
                    return Err(parse_cap_err(limit));
                }
            }
            Ok(r)
        }
        Ast::OpaqueDen(x) => {
            // opaque_den(S) = 1/S, with S kept symbolic as a denominator factor.
            let rx = ast_to_ratfn_capped(x, limit)?;
            if !rx.den.is_empty() || !rx.den_opaque.is_empty() {
                return Err("opaque_den(...) argument must be a plain polynomial".to_string());
            }
            if rx.num.is_empty() {
                return Err("opaque_den(0) is division by zero".to_string());
            }
            let mut num = Poly::new();
            num.insert(Mono::new(), 1);
            Ok(RatFn { num, den: Mono::new(), den_opaque: vec![rx.num] })
        }
    }
}

/// The replacement `RatFn` for the target bracket `bk(i,j)` of type `square`, mirroring
/// `spinorhelicity/environment/spin_helicity_env.py`. `params` is [pk, pl] for schouten,
/// [pout1, pout2] for momentum and the variable-length mom_add for momentum_sq.
pub(crate) fn build_identity(
    square: bool,
    i: u16,
    j: u16,
    identity: &str,
    params: &[u16],
    n: u16,
) -> Result<RatFn, String> {
    // Wrap the numerator `terms` over a single-bracket denominator bk_den(da,db), folding
    // the denominator's antisymmetry sign into the coefficients.
    let make = |terms: Vec<(Mono, i128)>, den_sq: bool, da: u16, db: u16| -> RatFn {
        let (dbk, ds) = make_bracket(den_sq, da, db);
        let mut num = Poly::new();
        for (m, c) in terms {
            // 1/(ds*den) = ds/den with ds = ±1. Checked for completeness, c being ±1 here.
            let cc = match c.checked_mul(ds) {
                Some(v) => v,
                None => {
                    set_overflow();
                    c.wrapping_mul(ds)
                }
            };
            poly_add_term(&mut num, m, cc);
        }
        poly_prune(&mut num);
        let mut den = Mono::new();
        if !num.is_empty() {
            den.insert(dbk, 1);
        }
        RatFn { num, den, den_opaque: Vec::new() }
    };

    match identity {
        "schouten" => {
            // (bk(i,pl)*bk(pk,j) + bk(i,pk)*bk(j,pl)) / bk(pk,pl)
            let (pk, pl) = (params[0], params[1]);
            let mut terms = Vec::new();
            if let Some((m, s)) = mono_from(&[(square, i, pl), (square, pk, j)]) {
                terms.push((m, s));
            }
            if let Some((m, s)) = mono_from(&[(square, i, pk), (square, j, pl)]) {
                terms.push((m, s));
            }
            Ok(make(terms, square, pk, pl))
        }
        "momentum" => {
            let (pout1, pout2) = (params[0], params[1]);
            let (larg, rarg) = if !square { (pout1, pout2) } else { (pout2, pout1) };
            let in_arg = if pout1 == i { j } else { i };
            let sgn: i128 = if (!square && pout1 == i) || (square && pout1 == j) { 1 } else { -1 };
            let mut terms = Vec::new();
            for k in 1..=n {
                if k == in_arg || k == larg || k == rarg {
                    continue;
                }
                if let Some((m, s)) = mono_from(&[(false, larg, k), (true, k, rarg)]) {
                    terms.push((m, -sgn * s));
                }
            }
            let (da, db) = if !square { (in_arg, pout2) } else { (pout2, in_arg) };
            Ok(make(terms, !square, da, db))
        }
        "momentum_sq" => {
            let mut in_lhs = vec![false; (n + 1) as usize];
            for &m in params {
                in_lhs[m as usize] = true;
            }
            in_lhs[i as usize] = true;
            in_lhs[j as usize] = true;
            let lhs: Vec<u16> = (1..=n).filter(|k| in_lhs[*k as usize]).collect();
            let rhs: Vec<u16> = (1..=n).filter(|k| !in_lhs[*k as usize]).collect();
            let mut terms = Vec::new();
            for a in 0..rhs.len() {
                for b in (a + 1)..rhs.len() {
                    if let Some((m, s)) = mono_from(&[(false, rhs[a], rhs[b]), (true, rhs[a], rhs[b])]) {
                        terms.push((m, s));
                    }
                }
            }
            for a in 0..lhs.len() {
                for b in (a + 1)..lhs.len() {
                    let (l1, l2) = (lhs[a], lhs[b]);
                    let both = (l1 == i || l1 == j) && (l2 == i || l2 == j);
                    if both {
                        continue;
                    }
                    if let Some((m, s)) = mono_from(&[(false, l1, l2), (true, l1, l2)]) {
                        terms.push((m, -s));
                    }
                }
            }
            Ok(make(terms, !square, i, j))
        }
        other => Err(format!("unknown identity '{}'", other)),
    }
}

/// Substitute `ret` for every occurrence of `target` in the numerator of `expr`. The result
/// retains the monomial and sum-denominator factors of `expr`, and `ret` can introduce
/// additional monomial denominator factors. The caller cancels.
pub(crate) fn substitute(expr: &RatFn, target: Bracket, ret: &RatFn, max_terms: usize) -> RatFn {
    // Bound the intermediate size. `pows[k] = ret^k` and the accumulator explode
    // combinatorially when `target` carries a high power. On a breach, flag it and return a
    // sentinel. The apply() boundary raises before the costly cancel and the search skips
    // this one move, whose child would be discarded anyway. `max_terms > 0` is an explicit
    // per-call cap and 0 falls back to the process default of term_limit().
    let limit = if max_terms > 0 { max_terms } else { term_limit() };
    let mut maxp = 0u32;
    for m in expr.num.keys() {
        if let Some(p) = m.get(&target) {
            maxp = maxp.max(*p);
        }
    }
    let mut pows = Vec::with_capacity((maxp + 1) as usize);
    pows.push(RatFn::from_int(1));
    for k in 1..=maxp {
        let next = ratfn_mul(&pows[(k - 1) as usize], ret);
        if next.num.len() > limit {
            set_term_limit_hit();
            return RatFn::zero();
        }
        pows.push(next);
    }
    let mut acc = RatFn::zero();
    for (m, c) in &expr.num {
        let p = *m.get(&target).unwrap_or(&0);
        let mut rest = m.clone();
        rest.remove(&target);
        let term = RatFn::from_mono_coeff(rest, *c);
        let term = if p > 0 { ratfn_mul(&term, &pows[p as usize]) } else { term };
        acc = ratfn_add(&acc, &term);
        if acc.num.len() > limit {
            set_term_limit_hit();
            return RatFn::zero();
        }
    }
    let mut out = ratfn_mul(&acc, &RatFn::recip_of_mono(&expr.den));
    // The numerator-only rewrite leaves any sum-denominator factor untouched, so it is
    // carried onto the result, where the apply boundary's `ratfn_cancel` attempts to cancel it.
    out.den_opaque = expr.den_opaque.clone();
    out
}

/// Apply `ret` to one occurrence of `target` in the `term_idx`-th numerator term in engine
/// canonical order, the BTreeMap order, leaving every other term unchanged. This mirrors the
/// Python `apply_identity_to_term(..., power_to_apply=1)` but indexes terms by the engine's
/// own canonical order, which differs from SymPy's `.args` order. It does no cancellation.
/// The caller cancels at the boundary, exactly like [`substitute`].
///
/// Where the Python reference silently returns the input, this errors on an out-of-range
/// index or a target bracket absent from the selected term, since the move mask proposes
/// only legal `(bracket, term_slot)` pairs.
///
/// It is bounded by the term cap of [`substitute`], `max_terms` with 0 meaning the process
/// default. A breach flags the term limit and returns the zero sentinel, which the apply
/// boundary turns into the skipped-move error exactly as for a whole-expression rewrite.
pub(crate) fn substitute_term(
    expr: &RatFn,
    target: Bracket,
    ret: &RatFn,
    term_idx: usize,
    max_terms: usize,
) -> Result<RatFn, String> {
    let limit = if max_terms > 0 { max_terms } else { term_limit() };
    let (target_mono, coeff) = match expr.num.iter().nth(term_idx) {
        Some((m, c)) => (m.clone(), *c),
        None => {
            return Err(format!(
                "term_idx {} out of range (numerator has {} terms)",
                term_idx,
                expr.num.len()
            ))
        }
    };
    let p = *target_mono.get(&target).unwrap_or(&0);
    if p == 0 {
        return Err(format!("target bracket absent from term {}", term_idx));
    }
    // The moved term is coeff * target_mono with one `target` peeled * ret, over expr.den.
    // `ret` may carry its own monomial denominator, such as Schouten's /bk(pk,pl). The
    // resulting denominator expr.den*ret.den is reconciled with `rest` by ratfn_add.
    let mut rest_mono = target_mono.clone();
    match rest_mono.get_mut(&target) {
        Some(e) if *e > 1 => *e -= 1,
        _ => {
            rest_mono.remove(&target);
        }
    }
    let base = RatFn::from_mono_coeff(rest_mono, coeff);
    let moved = ratfn_mul(&ratfn_mul(&base, ret), &RatFn::recip_of_mono(&expr.den));
    if moved.num.len() > limit {
        set_term_limit_hit();
        return Ok(RatFn::zero());
    }
    // `rest` holds every other numerator term, over the original denominator, untouched.
    let mut rest_num = Poly::new();
    for (idx, (m, c)) in expr.num.iter().enumerate() {
        if idx != term_idx {
            rest_num.insert(m.clone(), *c);
        }
    }
    let rest = RatFn {
        num: rest_num,
        den: expr.den.clone(),
        den_opaque: Vec::new(),
    };
    // `rest` and `moved` share the same sum-denominator factors, a whole-expression
    // multiplier the per-term rewrite leaves untouched. They are factored out of the add
    // and reattached after.
    let mut out = ratfn_add(&rest, &moved);
    if out.num.len() > limit {
        set_term_limit_hit();
        return Ok(RatFn::zero());
    }
    out.den_opaque = expr.den_opaque.clone();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn rf(s: &str) -> RatFn {
        ast_to_ratfn(&parse(s).unwrap()).unwrap()
    }
    fn canon(mut r: RatFn) -> String {
        ratfn_cancel(&mut r);
        ratfn_to_string(&r)
    }

    // ab(1,2) -> schouten with [3,4], ab(1,2) = (ab(1,4)*ab(3,2)+ab(1,3)*ab(4,2))/ab(3,4).
    fn schouten_12() -> (Bracket, RatFn) {
        let (t, _) = make_bracket(false, 1, 2);
        (t, build_identity(false, 1, 2, "schouten", &[3, 4], 4).unwrap())
    }

    #[test]
    fn term_idx_out_of_range_errors() {
        let e = rf("ab(1,2)*ab(3,4) + ab(1,3)*ab(2,4)");
        let (t, ret) = schouten_12();
        assert!(substitute_term(&e, t, &ret, 5, 0).is_err());
    }

    #[test]
    fn target_absent_from_term_errors() {
        // Term 0 in canonical order may contain ab(1,2). Find one that does not.
        let e = rf("ab(1,2)*ab(3,4) + ab(1,3)*ab(2,4)");
        let (t, ret) = schouten_12();
        let absent = e
            .num
            .keys()
            .position(|m| !m.contains_key(&t))
            .expect("expected a term without ab(1,2)");
        assert!(substitute_term(&e, t, &ret, absent, 0).is_err());
    }

    #[test]
    fn single_occurrence_matches_whole() {
        // ab(1,2) appears in exactly one term at power 1, so per-term equals whole-expression.
        let e = rf("ab(1,2)*ab(3,4) + ab(1,3)*ab(2,4)");
        let (t, ret) = schouten_12();
        let idx = e.num.keys().position(|m| m.contains_key(&t)).unwrap();
        let per = substitute_term(&e, t, &ret, idx, 0).unwrap();
        let whole = substitute(&e, t, &ret, 0);
        assert_eq!(canon(per), canon(whole));
    }

    #[test]
    fn power_two_peels_exactly_one() {
        // For ab(1,2)**2, per-term peels one and gives ab(1,2)*ret while whole gives ret**2.
        // They differ, and per-term must equal ab(1,2)*ret.
        let e = rf("ab(1,2)**2");
        let (t, ret) = schouten_12();
        let per_s = canon(substitute_term(&e, t, &ret, 0, 0).unwrap());
        let expected = ratfn_mul(&rf("ab(1,2)"), &ret);
        assert_eq!(per_s, canon(expected));
        let whole = substitute(&e, t, &ret, 0);
        assert_ne!(per_s, canon(whole));
    }

    // --- sum-denominator factors ---

    #[test]
    fn opaque_den_cancels_when_numerator_divisible() {
        // (S * ab(2,4)) * opaque_den(S) = ab(2,4), with S = ab(1,2)+ab(1,3).
        let e = "(ab(1,2)+ab(1,3))*ab(2,4)*opaque_den(ab(1,2)+ab(1,3))";
        assert_eq!(canon(rf(e)), "ab(2, 4)");
    }

    #[test]
    fn opaque_den_kept_when_not_divisible() {
        // S does not divide the numerator, so the factor must survive cancel.
        let mut r = rf("ab(2,4)*opaque_den(ab(1,2)+ab(1,3))");
        ratfn_cancel(&mut r);
        assert_eq!(r.den_opaque.len(), 1);
        assert!(ratfn_to_string(&r).contains("opaque_den"));
    }

    #[test]
    fn opaque_factors_combine_in_add() {
        // In a*opaque_den(P) + b*opaque_den(Q) the distinct factors give a union of size 2,
        // neither dividing the combined numerator, so both are kept.
        let mut r = rf("ab(3,4)*opaque_den(ab(1,2)+ab(1,3)) + ab(2,4)*opaque_den(ab(1,4)+ab(2,3))");
        ratfn_cancel(&mut r);
        assert_eq!(r.den_opaque.len(), 2);
    }

    #[test]
    fn opaque_den_shared_factor_not_duplicated() {
        // Both terms carry the same factor S, so the union keeps one copy of S.
        let mut r = rf("ab(3,4)*opaque_den(ab(1,2)+ab(1,3)) + ab(2,4)*opaque_den(ab(1,2)+ab(1,3))");
        ratfn_cancel(&mut r);
        assert_eq!(r.den_opaque.len(), 1);
    }
}
