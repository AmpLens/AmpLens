//! Parse sympy `str()` output over the brackets `ab(i,j)` / `sb(i,j)` into an AST.
//! Grammar (recursive descent): expr = ±term {(+|-) term}; term = factor {(*|/)
//! factor}; factor = atom [** [±]int];
//! atom = int | ab/sb(i,j) | opaque_den(expr) | (expr) | -atom.

#[derive(Debug, Clone)]
pub(crate) enum Ast {
    Int(i64),
    Bracket { square: bool, i: usize, j: usize },
    Neg(Box<Ast>),
    Add(Vec<Ast>),
    Mul(Vec<Ast>),
    Pow(Box<Ast>, i64),
    /// `opaque_den(S)` = 1/S with the sum `S` kept symbolic (an opaque denominator factor).
    OpaqueDen(Box<Ast>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    Ab,
    Sb,
    OpaqueDen,
    LP,
    RP,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    DStar,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut out = Vec::new();
    while i < n {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let start = i;
            while i < n && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Int(s[start..i].parse().map_err(|_| "bad int".to_string())?));
        } else if c.is_ascii_alphabetic() {
            let start = i;
            while i < n && ((b[i] as char).is_ascii_alphabetic() || b[i] as char == '_') {
                i += 1;
            }
            match &s[start..i] {
                "ab" => out.push(Tok::Ab),
                "sb" => out.push(Tok::Sb),
                "opaque_den" => out.push(Tok::OpaqueDen),
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
    let toks = tokenize(s)?;
    let mut p = Parser { toks, pos: 0 };
    let ast = p.parse_expr()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens from {}", p.pos));
    }
    Ok(ast)
}
