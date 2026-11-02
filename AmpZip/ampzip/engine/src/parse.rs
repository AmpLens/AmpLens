//! Parse sympy `str()` output over the brackets `ab(i,j)` / `sb(i,j)` into an AST.
//! The grammar, parsed by recursive descent, is expr = ±term {(+|-) term},
//! term = factor {(*|/) factor}, factor = atom [** [±]int] and
//! atom = number | ab/sb(i,j) | opaque_den(expr) | sqrt(int) | I | name | (expr) | -atom.
//!
//! Besides integers, the evaluator in `eval.rs` accepts integers of any size, decimals read
//! as exact rationals, `I`, `sqrt(n)` and named parameters listed by the caller. `sqrt(n)`
//! is rewritten as `k * sqrt(q1) * sqrt(q2) ...` over the primes of the square-free part
//! of n, so that equal radicals written differently evaluate alike. The rewrite engine in
//! `ident.rs` takes integer coefficients only.

#[derive(Debug, Clone)]
pub(crate) enum Ast {
    Int(i64),
    /// A non-negative integer beyond i64, kept as decimal digits.
    Big(String),
    Bracket { square: bool, i: usize, j: usize },
    Neg(Box<Ast>),
    Add(Vec<Ast>),
    Mul(Vec<Ast>),
    Pow(Box<Ast>, i64),
    /// `opaque_den(S)` = 1/S with the sum `S` kept symbolic, an opaque denominator factor.
    OpaqueDen(Box<Ast>),
    /// The imaginary unit.
    ImagUnit,
    /// The square root of a prime.
    Sqrt(u64),
    /// A named parameter whose value the caller supplies at evaluation.
    Param(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    /// Integer digits beyond i64.
    Big(String),
    /// A decimal literal as its digits and the power of ten to divide by, so `1.5e-3` is
    /// ("15", 4).
    Dec(String, i64),
    Ab,
    Sb,
    OpaqueDen,
    Sqrt,
    Imag,
    Name(String),
    LP,
    RP,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    DStar,
}

fn tokenize(s: &str, params: &[String]) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut out = Vec::new();
    while i < n {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < n && (b[i + 1] as char).is_ascii_digit()) {
            let start = i;
            while i < n && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            let mut frac = 0i64;
            let mut digits = s[start..i].to_string();
            if i < n && b[i] as char == '.' {
                i += 1;
                let fstart = i;
                while i < n && (b[i] as char).is_ascii_digit() {
                    i += 1;
                }
                frac = (i - fstart) as i64;
                digits.push_str(&s[fstart..i]);
            }
            let mut exp = 0i64;
            // An `e[+-]digits` suffix is an exponent, as in `1.0e-3`.
            if i < n && (b[i] as char == 'e' || b[i] as char == 'E') {
                let mut j = i + 1;
                let mut sign = 1i64;
                if j < n && (b[j] as char == '+' || b[j] as char == '-') {
                    sign = if b[j] as char == '-' { -1 } else { 1 };
                    j += 1;
                }
                if j < n && (b[j] as char).is_ascii_digit() {
                    let estart = j;
                    while j < n && (b[j] as char).is_ascii_digit() {
                        j += 1;
                    }
                    exp = sign * s[estart..j].parse::<i64>().map_err(|_| "bad exponent".to_string())?;
                    i = j;
                }
            }
            if frac == 0 && exp == 0 {
                let digits = digits.trim_start_matches('0');
                let digits = if digits.is_empty() { "0" } else { digits };
                match digits.parse::<i64>() {
                    Ok(v) => out.push(Tok::Int(v)),
                    Err(_) => out.push(Tok::Big(digits.to_string())),
                }
            } else {
                out.push(Tok::Dec(digits, frac - exp));
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < n && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_') {
                i += 1;
            }
            match &s[start..i] {
                "ab" => out.push(Tok::Ab),
                "sb" => out.push(Tok::Sb),
                "opaque_den" => out.push(Tok::OpaqueDen),
                "sqrt" => out.push(Tok::Sqrt),
                "I" => out.push(Tok::Imag),
                other if params.iter().any(|q| q == other) => out.push(Tok::Name(other.to_string())),
                other => return Err(format!("unknown identifier '{}'", other)),
            }
        } else {
            match c {
                '(' => out.push(Tok::LP),
                ')' => out.push(Tok::RP),
                ',' => out.push(Tok::Comma),
                '+' => out.push(Tok::Plus),
                '-' => out.push(Tok::Minus),
                '/' => out.push(Tok::Slash),
                '*' => {
                    if i + 1 < n && b[i + 1] as char == '*' {
                        out.push(Tok::DStar);
                        i += 1;
                    } else {
                        out.push(Tok::Star);
                    }
                }
                _ => return Err(format!("unexpected char '{}'", c)),
            }
            i += 1;
        }
    }
    Ok(out)
}

/// Returns `(k, [q...])` with `n = k^2 * q1 * q2 * ...` and distinct primes `q`.
fn square_free_split(mut n: u64) -> (u64, Vec<u64>) {
    let mut k = 1u64;
    let mut primes = Vec::new();
    let mut q = 2u64;
    while q * q <= n {
        let mut e = 0u32;
        while n % q == 0 {
            n /= q;
            e += 1;
        }
        if e > 0 {
            k *= q.pow(e / 2);
            if e % 2 == 1 {
                primes.push(q);
            }
        }
        q += if q == 2 { 1 } else { 2 };
    }
    if n > 1 {
        primes.push(n);
    }
    (k, primes)
}

/// `sqrt(n)` as `k * sqrt(q1) * ...` over the primes of the square-free part of `n`.
fn sqrt_ast(n: u64) -> Ast {
    if n == 0 || n == 1 {
        return Ast::Int(n as i64);
    }
    let (k, primes) = square_free_split(n);
    let mut factors = Vec::new();
    if k != 1 {
        factors.push(Ast::Int(k as i64));
    }
    factors.extend(primes.into_iter().map(Ast::Sqrt));
    if factors.len() == 1 {
        factors.pop().unwrap()
    } else {
        Ast::Mul(factors)
    }
}

fn big_or_int(digits: &str) -> Ast {
    match digits.parse::<i64>() {
        Ok(v) => Ast::Int(v),
        Err(_) => Ast::Big(digits.to_string()),
    }
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        match self.next() {
            Some(ref g) if g == t => Ok(()),
            other => Err(format!("expected {:?}, got {:?}", t, other)),
        }
    }
    fn expect_int(&mut self) -> Result<i64, String> {
        match self.next() {
            Some(Tok::Int(v)) => Ok(v),
            other => Err(format!("expected integer, got {:?}", other)),
        }
    }
    fn parse_expr(&mut self) -> Result<Ast, String> {
        let mut terms = Vec::new();
        let mut neg = false;
        match self.peek() {
            Some(Tok::Plus) => {
                self.next();
            }
            Some(Tok::Minus) => {
                self.next();
                neg = true;
            }
            _ => {}
        }
        let first = self.parse_term()?;
        terms.push(if neg { Ast::Neg(Box::new(first)) } else { first });
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.next();
                    terms.push(self.parse_term()?);
                }
                Some(Tok::Minus) => {
                    self.next();
                    terms.push(Ast::Neg(Box::new(self.parse_term()?)));
                }
                _ => break,
            }
        }
        Ok(if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            Ast::Add(terms)
        })
    }
    fn parse_term(&mut self) -> Result<Ast, String> {
        let mut factors = vec![self.parse_factor()?];
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.next();
                    factors.push(self.parse_factor()?);
                }
                Some(Tok::Slash) => {
                    self.next();
                    factors.push(Ast::Pow(Box::new(self.parse_factor()?), -1));
                }
                _ => break,
            }
        }
        Ok(if factors.len() == 1 {
            factors.pop().unwrap()
        } else {
            Ast::Mul(factors)
        })
    }
    fn parse_factor(&mut self) -> Result<Ast, String> {
        let base = self.parse_atom()?;
        if let Some(Tok::DStar) = self.peek() {
            self.next();
            Ok(Ast::Pow(Box::new(base), self.parse_exponent()?))
        } else {
            Ok(base)
        }
    }
    fn parse_exponent(&mut self) -> Result<i64, String> {
        let paren = matches!(self.peek(), Some(Tok::LP));
        if paren {
            self.next();
        }
        let mut sign = 1i64;
        match self.peek() {
            Some(Tok::Minus) => {
                self.next();
                sign = -1;
            }
            Some(Tok::Plus) => {
                self.next();
            }
            _ => {}
        }
        let v = self.expect_int()?;
        if paren {
            self.expect(&Tok::RP)?;
        }
        Ok(sign * v)
    }
    fn parse_atom(&mut self) -> Result<Ast, String> {
        match self.peek().cloned() {
            Some(Tok::Minus) => {
                self.next();
                Ok(Ast::Neg(Box::new(self.parse_atom()?)))
            }
            Some(Tok::Int(v)) => {
                self.next();
                Ok(Ast::Int(v))
            }
            Some(Tok::Big(d)) => {
                self.next();
                Ok(Ast::Big(d))
            }
            Some(Tok::Dec(digits, scale)) => {
                self.next();
                // digits / 10^scale. A negative scale multiplies instead.
                let d = digits.trim_start_matches('0');
                let mant = big_or_int(if d.is_empty() { "0" } else { d });
                if scale == 0 {
                    return Ok(mant);
                }
                let mut ten = String::from("1");
                ten.push_str(&"0".repeat(scale.unsigned_abs() as usize));
                let power = big_or_int(&ten);
                Ok(if scale > 0 {
                    Ast::Mul(vec![mant, Ast::Pow(Box::new(power), -1)])
                } else {
                    Ast::Mul(vec![mant, power])
                })
            }
            Some(Tok::Imag) => {
                self.next();
                Ok(Ast::ImagUnit)
            }
            Some(Tok::Name(name)) => {
                self.next();
                Ok(Ast::Param(name))
            }
            Some(Tok::Sqrt) => {
                self.next();
                self.expect(&Tok::LP)?;
                let v = match self.next() {
                    Some(Tok::Int(v)) if v >= 0 => v as u64,
                    other => return Err(format!("sqrt() takes a non-negative integer, got {:?}", other)),
                };
                self.expect(&Tok::RP)?;
                Ok(sqrt_ast(v))
            }
            Some(Tok::Ab) | Some(Tok::Sb) => {
                let square = matches!(self.next(), Some(Tok::Sb));
                self.expect(&Tok::LP)?;
                let i = self.expect_int()?;
                self.expect(&Tok::Comma)?;
                let j = self.expect_int()?;
                self.expect(&Tok::RP)?;
                if i < 1 || j < 1 {
                    return Err(format!("bracket index < 1: ({},{})", i, j));
                }
                Ok(Ast::Bracket {
                    square,
                    i: i as usize,
                    j: j as usize,
                })
            }
            Some(Tok::OpaqueDen) => {
                self.next();
                self.expect(&Tok::LP)?;
                let e = self.parse_expr()?;
                self.expect(&Tok::RP)?;
                Ok(Ast::OpaqueDen(Box::new(e)))
            }
            Some(Tok::LP) => {
                self.next();
                let e = self.parse_expr()?;
                self.expect(&Tok::RP)?;
                Ok(e)
            }
            other => Err(format!("unexpected token {:?}", other)),
        }
    }
}

pub(crate) fn parse(s: &str) -> Result<Ast, String> {
    parse_with_params(s, &[])
}

/// Parses `s`, admitting the listed identifiers as named parameters.
pub(crate) fn parse_with_params(s: &str, params: &[String]) -> Result<Ast, String> {
    let toks = tokenize(s, params)?;
    let mut p = Parser { toks, pos: 0 };
    let ast = p.parse_expr()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens from {}", p.pos));
    }
    Ok(ast)
}
