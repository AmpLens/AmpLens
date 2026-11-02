//! Exact evaluation of an `Ast` or `RatFn` over F_p at an on-shell point.
//! Brackets are 2x2 minors of the spinors `lam`/`lamt`, matching
//! `ampzip.verify.vals_from_point`. The finite-field verifier uses `eval_pair`
//! through `Ast.eval_pair` to compare expressions without division.
use crate::parse::Ast;
use crate::poly::{Mono, Poly, RatFn};
use std::collections::HashMap;

/// Values of the named parameters at one evaluation point.
pub(crate) type Params = HashMap<String, u64>;

pub(crate) fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}
pub(crate) fn powmod(mut base: u64, mut e: u64, p: u64) -> u64 {
    base %= p;
    let mut r = 1u64 % p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    r
}
pub(crate) fn modinv(a: u64, p: u64) -> Result<u64, String> {
    if a.is_multiple_of(p) {
        return Err("inverse of 0 (degenerate point)".to_string());
    }
    Ok(powmod(a, p - 2, p))
}
/// A decimal integer of any length reduced mod p.
pub(crate) fn big_mod(digits: &str, p: u64) -> u64 {
    digits.bytes().fold(0u64, |acc, b| (mulmod(acc, 10, p) + (b - b'0') as u64) % p)
}

/// A square root of `a` mod the odd prime `p` by Tonelli-Shanks. The result depends only on
/// `a` and `p`, so every occurrence of the same root agrees. Err when `a` has no root mod `p`.
pub(crate) fn sqrt_mod(a: u64, p: u64) -> Result<u64, String> {
    let a = a % p;
    if a == 0 {
        return Ok(0);
    }
    if p == 2 {
        return Ok(a);
    }
    if powmod(a, (p - 1) / 2, p) != 1 {
        return Err(format!("{} has no square root mod {}", a, p));
    }
    if p % 4 == 3 {
        return Ok(powmod(a, (p + 1) / 4, p));
    }
    // p - 1 = q * 2^s with q odd; z is the smallest non-residue.
    let mut q = p - 1;
    let mut s_ = 0u32;
    while q % 2 == 0 {
        q /= 2;
        s_ += 1;
    }
    let mut z = 2u64;
    while powmod(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s_;
    let mut c = powmod(z, q, p);
    let mut t = powmod(a, q, p);
    let mut r = powmod(a, (q + 1) / 2, p);
    while t != 1 {
        let mut i = 0u32;
        let mut t2 = t;
        while t2 != 1 {
            t2 = mulmod(t2, t2, p);
            i += 1;
        }
        let b = powmod(c, 1u64 << (m - i - 1), p);
        m = i;
        c = mulmod(b, b, p);
        t = mulmod(t, c, p);
        r = mulmod(r, b, p);
    }
    Ok(r)
}

fn param_value(name: &str, params: &Params) -> Result<u64, String> {
    params.get(name).copied().ok_or_else(|| format!("no value supplied for parameter '{}'", name))
}

pub(crate) fn minor(square: bool, i: usize, j: usize, lam: &[Vec<u64>], lamt: &[Vec<u64>], p: u64) -> Result<u64, String> {
    let v = if square { lamt } else { lam };
    if i > v.len() || j > v.len() {
        return Err(format!("index out of range ({},{})", i, j));
    }
    let (a0, a1) = (v[i - 1][0] % p, v[i - 1][1] % p);
    let (b0, b1) = (v[j - 1][0] % p, v[j - 1][1] % p);
    Ok((mulmod(a0, b1, p) + p - mulmod(a1, b0, p)) % p)
}

pub(crate) fn eval_ast(a: &Ast, lam: &[Vec<u64>], lamt: &[Vec<u64>], p: u64, params: &Params) -> Result<u64, String> {
    match a {
        Ast::Int(n) => Ok((((n % p as i64) + p as i64) % p as i64) as u64),
        Ast::Big(d) => Ok(big_mod(d, p)),
        Ast::ImagUnit => sqrt_mod(p - 1, p).map_err(|_| format!("I needs a prime p = 1 (mod 4); {} is not", p)),
        Ast::Sqrt(q) => sqrt_mod(*q, p).map_err(|_| format!("sqrt({}) needs a prime where {} is a square; {} is not", q, q, p)),
        Ast::Param(name) => param_value(name, params),
        Ast::Bracket { square, i, j } => minor(*square, *i, *j, lam, lamt, p),
        Ast::Neg(x) => Ok((p - eval_ast(x, lam, lamt, p, params)?) % p),
        Ast::Add(xs) => {
            let mut s = 0u64;
            for x in xs {
                s = (s + eval_ast(x, lam, lamt, p, params)?) % p;
            }
            Ok(s)
        }
        Ast::Mul(xs) => {
            let mut s = 1u64 % p;
            for x in xs {
                s = mulmod(s, eval_ast(x, lam, lamt, p, params)?, p);
            }
            Ok(s)
        }
        Ast::Pow(b, e) => {
            let bv = eval_ast(b, lam, lamt, p, params)?;
            if *e >= 0 {
                Ok(powmod(bv, *e as u64, p))
            } else {
                Ok(powmod(modinv(bv, p)?, (-*e) as u64, p))
            }
        }
        Ast::OpaqueDen(x) => Ok(modinv(eval_ast(x, lam, lamt, p, params)?, p)?),
    }
}

/// Value of an `Ast` at an on-shell point as a fraction `(num, den)` in F_p, with no division
/// anywhere: sums cross-multiply, products multiply componentwise, a negative power or an
/// `opaque_den` swaps the two. A vanishing `den` is returned as it is, and the caller treats
/// the point as degenerate, as the verifier does.
pub(crate) fn eval_pair(a: &Ast, lam: &[Vec<u64>], lamt: &[Vec<u64>], p: u64, params: &Params) -> Result<(u64, u64), String> {
    let one = 1u64 % p;
    match a {
        Ast::Int(n) => Ok(((((n % p as i64) + p as i64) % p as i64) as u64, one)),
        Ast::Big(_) | Ast::ImagUnit | Ast::Sqrt(_) | Ast::Param(_) => Ok((eval_ast(a, lam, lamt, p, params)?, one)),
        Ast::Bracket { square, i, j } => Ok((minor(*square, *i, *j, lam, lamt, p)?, one)),
        Ast::Neg(x) => {
            let (n, d) = eval_pair(x, lam, lamt, p, params)?;
            Ok(((p - n) % p, d))
        }
        Ast::Add(xs) => {
            let (mut n, mut d) = (0u64, one);
            for x in xs {
                let (n2, d2) = eval_pair(x, lam, lamt, p, params)?;
                n = (mulmod(n, d2, p) + mulmod(n2, d, p)) % p;
                d = mulmod(d, d2, p);
            }
            Ok((n, d))
        }
        Ast::Mul(xs) => {
            let (mut n, mut d) = (one, one);
            for x in xs {
                let (n2, d2) = eval_pair(x, lam, lamt, p, params)?;
                n = mulmod(n, n2, p);
                d = mulmod(d, d2, p);
            }
            Ok((n, d))
        }
        Ast::Pow(b, e) => {
            let (n, d) = eval_pair(b, lam, lamt, p, params)?;
            if *e >= 0 {
                Ok((powmod(n, *e as u64, p), powmod(d, *e as u64, p)))
            } else {
                Ok((powmod(d, (-*e) as u64, p), powmod(n, (-*e) as u64, p)))
            }
        }
        Ast::OpaqueDen(x) => {
            let (n, d) = eval_pair(x, lam, lamt, p, params)?;
            Ok((d, n))
        }
    }
}

/// Product of bracket minors `Π minor(bk)^e` over a monomial, mod p.
fn eval_mono(m: &Mono, lam: &[Vec<u64>], lamt: &[Vec<u64>], p: u64) -> Result<u64, String> {
    let mut acc = 1u64 % p;
    for (bk, e) in m {
        let v = minor(bk.square, bk.i as usize, bk.j as usize, lam, lamt, p)?;
        acc = mulmod(acc, powmod(v, *e as u64, p), p);
    }
    Ok(acc)
}

/// `Σ coeff * eval_mono(mono)` over a polynomial, mod p.
fn eval_poly(poly: &Poly, lam: &[Vec<u64>], lamt: &[Vec<u64>], p: u64) -> Result<u64, String> {
    let mut acc = 0u64;
    for (m, c) in poly {
        let cm = (((*c % p as i128) + p as i128) % p as i128) as u64;
        if cm == 0 {
            continue;
        }
        acc = (acc + mulmod(cm, eval_mono(m, lam, lamt, p)?, p)) % p;
    }
    Ok(acc)
}

/// Exact evaluation of a `RatFn` over F_p: `eval_poly(num)` multiplied by the
/// modular inverse of `eval_mono(den) * product(eval_poly(f) for f in den_opaque)`.
/// This evaluates a resident `Expr` without parsing its string representation
/// again and must agree with `eval_ast` on the same expression.
pub(crate) fn eval_ratfn_modp(
    rf: &RatFn,
    lam: &[Vec<u64>],
    lamt: &[Vec<u64>],
    p: u64,
) -> Result<u64, String> {
    let num = eval_poly(&rf.num, lam, lamt, p)?;
    if num == 0 || (rf.den.is_empty() && rf.den_opaque.is_empty()) {
        return Ok(num);
    }
    let mut den = if rf.den.is_empty() { 1 % p } else { eval_mono(&rf.den, lam, lamt, p)? };
    for s in &rf.den_opaque {
        den = mulmod(den, eval_poly(s, lam, lamt, p)?, p);
    }
    Ok(mulmod(num, modinv(den, p)?, p))
}

/// Whether `I` occurs, the primes under square roots and the parameter names, each list
/// sorted without duplicates.
pub(crate) fn needs(a: &Ast) -> (bool, Vec<u64>, Vec<String>) {
    fn walk(a: &Ast, imag: &mut bool, roots: &mut Vec<u64>, names: &mut Vec<String>) {
        match a {
            Ast::ImagUnit => *imag = true,
            Ast::Sqrt(q) => roots.push(*q),
            Ast::Param(n) => names.push(n.clone()),
            Ast::Neg(x) | Ast::Pow(x, _) | Ast::OpaqueDen(x) => walk(x, imag, roots, names),
            Ast::Add(xs) | Ast::Mul(xs) => xs.iter().for_each(|x| walk(x, imag, roots, names)),
            Ast::Int(_) | Ast::Big(_) | Ast::Bracket { .. } => {}
        }
    }
    let (mut imag, mut roots, mut names) = (false, Vec::new(), Vec::new());
    walk(a, &mut imag, &mut roots, &mut names);
    roots.sort_unstable();
    roots.dedup();
    names.sort();
    names.dedup();
    (imag, roots, names)
}
