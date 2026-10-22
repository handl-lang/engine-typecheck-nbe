//! Dependent-structure checking for the tagged-type engine (phase-1 spec
//! WS-C; ADR-0152/0154 — port of the `synth_data`/`refine_scrutinee_`/
//! `synth_match_arms`/`synth_transport`/`synth_j` fragment of the Handl
//! typecheck model).
//!
//! Encoding (tagged `Value`s, mirroring the model):
//!
//! - `type.family{name, base}` — a declared inductive former; `base` is a
//!   `type.array{elements}` of the parameter sorts.
//! - `type.family_app{name, args}` — a saturated family application;
//!   `args` are *index terms*.
//! - `type.ctor_sig{ctor, family, params, args, result, bare}` — a
//!   constructor signature spec; [`instantiate_ctor`] re-elaborates it
//!   per use into a `type.function` chain ending in `family_app` (fresh
//!   index metavariable per parameter).
//! - Index terms — `Value` numbers, `{"__meta__": id}` metas,
//!   `{"__tag__": "var", "name": n}` rigid variables, `{"__tag__": "Bin",
//!   "op": kind, "l", "r"}` and `{"__tag__": "Un", "op", "operand"}`
//!   operators. Index equality is definitional via evaluation
//!   ([`eval_index`]) plus the bounded solver [`solve_index`] — no
//!   bespoke rewriting beyond the model's canonicalization.
//! - `Path{frm, to}` — the proof-relevant index-equality witness bound
//!   as `eq_<name>` in `results` by branch refinement.

use handl_language::{Ast, AstNode, AstRef, LangError, Language};
use hnl_engine_typecheck::TypecheckDiagnostic;
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::{
    extract_meta_id, is_assign_node, meta_value, role_child_ref, NbeTypecheckEngine,
    TypecheckContext,
};

/// Build a tagged type value `{"__tag__": tag, "__registry__": id, …fields}`.
fn make_tag(ctx: &TypecheckContext, tag: &str, fields: Map<String, Value>) -> Value {
    let mut m = fields;
    m.insert("__tag__".to_string(), Value::String(tag.to_string()));
    m.insert(
        "__registry__".to_string(),
        Value::String(ctx.registry.registry_id().to_string()),
    );
    Value::Object(m)
}

fn tag_of(v: &Value) -> Option<&str> {
    v.get("__tag__").and_then(Value::as_str)
}

fn var_term(name: &str) -> Value {
    let mut m = Map::new();
    m.insert("__tag__".to_string(), Value::String("var".to_string()));
    m.insert("name".to_string(), Value::String(name.to_string()));
    Value::Object(m)
}

fn bin_term(op: &str, l: Value, r: Value) -> Value {
    let mut m = Map::new();
    m.insert("__tag__".to_string(), Value::String("Bin".to_string()));
    m.insert("op".to_string(), Value::String(op.to_string()));
    m.insert("l".to_string(), l);
    m.insert("r".to_string(), r);
    Value::Object(m)
}

fn un_term(op: &str, operand: Value) -> Value {
    let mut m = Map::new();
    m.insert("__tag__".to_string(), Value::String("Un".to_string()));
    m.insert("op".to_string(), Value::String(op.to_string()));
    m.insert("operand".to_string(), operand);
    Value::Object(m)
}

fn path_val(frm: Value, to: Value) -> Value {
    let mut m = Map::new();
    m.insert("__tag__".to_string(), Value::String("Path".to_string()));
    m.insert("frm".to_string(), frm);
    m.insert("to".to_string(), to);
    Value::Object(m)
}

fn num_val(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn children_of(node: &AstNode) -> Option<&Vec<Value>> {
    node.payload.get("children").and_then(|c| c.as_array())
}

fn child_at(ast: &Ast, node: &AstNode, i: usize) -> Option<AstRef> {
    children_of(node)?
        .get(i)?
        .as_u64()
        .map(|n| AstRef::new(n as usize))
}

fn node_name(node: &AstNode) -> Option<String> {
    node.payload
        .as_str()
        .map(str::to_string)
        .or_else(|| {
            node.payload
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

// ---------------------------------------------------------------------------
// Index-term algebra
// ---------------------------------------------------------------------------

/// Replace solved metas with their solutions (deep).
pub(crate) fn zonk_index(v: &Value, ctx: &TypecheckContext) -> Value {
    if let Some(id) = extract_meta_id(v) {
        if let Some(sol) = ctx.metas.get(id as usize).and_then(|o| o.as_ref()) {
            return zonk_index(sol, ctx);
        }
        return v.clone();
    }
    if let Value::Object(m) = v {
        if m.len() == 1 {
            return v.clone();
        }
        let mut out = Map::new();
        for (k, x) in m {
            out.insert(k.clone(), zonk_index(x, ctx));
        }
        return Value::Object(out);
    }
    if let Value::Array(xs) = v {
        return Value::Array(xs.iter().map(|x| zonk_index(x, ctx)).collect());
    }
    v.clone()
}

/// Canonicalize one `Bin` node: constant folding, additive/multiplicative
/// unit laws, commutative canonical order (closed operand on the right).
fn canon_bin(op: &str, l: Value, r: Value) -> Value {
    let (ln, rn) = (num_val(&l), num_val(&r));
    // x + 0 → x ; x - 0 → x ; x * 1 → x ; x * 0 → 0 ; x / 1 → x
    if let Some(r0) = rn {
        match op {
            "binary.add" | "binary.sub" if r0 == 0.0 => return l,
            "binary.mul" if r0 == 1.0 => return l,
            "binary.mul" if r0 == 0.0 => return Value::from(0),
            "binary.div" | "binary.mod" if r0 == 1.0 => return l,
            _ => {}
        }
    }
    if let Some(l0) = ln {
        match op {
            "binary.add" if l0 == 0.0 => return r,
            "binary.mul" if l0 == 1.0 => return r,
            "binary.mul" if l0 == 0.0 => return Value::from(0),
            _ => {}
        }
    }
    // Closed × open commutes to open × closed (`2 * ?n` → `Bin{mul,?n,2}`).
    if ln.is_some() && rn.is_none() && matches!(op, "binary.add" | "binary.mul") {
        return bin_term(op, r, l);
    }
    bin_term(op, l, r)
}

/// Evaluate an index term: resolve `var{name}` through `results`
/// (self-bindings `var{n} → var{n}` are the rigid fixed point), zonk
/// metas, fold closed `Bin`/`Un` arithmetic, canonicalize the rest.
pub(crate) fn eval_index(v: &Value, ctx: &TypecheckContext) -> Value {
    eval_index_fuel(v, ctx, 64)
}

fn eval_index_fuel(v: &Value, ctx: &TypecheckContext, fuel: u32) -> Value {
    if fuel == 0 {
        return v.clone();
    }
    if let Some(id) = extract_meta_id(v) {
        if let Some(sol) = ctx.metas.get(id as usize).and_then(|o| o.as_ref()) {
            return eval_index_fuel(sol, ctx, fuel - 1);
        }
        return v.clone();
    }
    if tag_of(v) == Some("var") {
        let name = v.get("name").and_then(Value::as_str).unwrap_or("");
        if let Some(bound) = ctx.results.get(name) {
            // A `var{n} ↦ var{n}` self-binding is the rigid fixed point.
            if bound == v {
                return v.clone();
            }
            return eval_index_fuel(bound, ctx, fuel - 1);
        }
        return v.clone();
    }
    if tag_of(v) == Some("Bin") {
        let op = v.get("op").and_then(Value::as_str).unwrap_or("");
        let l = eval_index_fuel(v.get("l").unwrap_or(&Value::Null), ctx, fuel - 1);
        let r = eval_index_fuel(v.get("r").unwrap_or(&Value::Null), ctx, fuel - 1);
        if let (Some(a), Some(b)) = (num_val(&l), num_val(&r)) {
            let res = match op {
                "binary.add" => Some(a + b),
                "binary.sub" => Some(a - b),
                "binary.mul" => Some(a * b),
                "binary.div" if b != 0.0 => Some(a / b),
                "binary.mod" if b != 0.0 => Some(a % b),
                "binary.pow" if b >= 0.0 && b.fract() == 0.0 => {
                    Some((0..b as u64).fold(1.0, |acc, _| acc * a))
                }
                "binary.eq" => return Value::from(a == b),
                _ => None,
            };
            if let Some(n) = res {
                return Value::from(n);
            }
        }
        return canon_bin(op, l, r);
    }
    if tag_of(v) == Some("Un") {
        let op = v.get("op").and_then(Value::as_str).unwrap_or("");
        let o = eval_index_fuel(v.get("operand").unwrap_or(&Value::Null), ctx, fuel - 1);
        if op == "unary.neg" {
            if let Some(n) = num_val(&o) {
                return Value::from(-n);
            }
        }
        return un_term(op, o);
    }
    v.clone()
}

/// The closed numeric value of a term, if it evaluates to a numeral.
fn closed_num(v: &Value, ctx: &TypecheckContext) -> Option<f64> {
    num_val(&eval_index(v, ctx))
}

/// Whether `name` occurs in term `v` (guards the `var` rewrite against
/// cyclic `results` bindings).
fn var_occurs(name: &str, v: &Value) -> bool {
    if tag_of(v) == Some("var") {
        return v.get("name").and_then(Value::as_str) == Some(name);
    }
    if let Value::Array(xs) = v {
        return xs.iter().any(|x| var_occurs(name, x));
    }
    if let Value::Object(m) = v {
        return m.values().any(|x| var_occurs(name, x));
    }
    false
}

/// The integer `e`-th root of `n`, if one exists (`i^e == n`), else
/// `None` — a *decided* absence (search bounded by `n`, `i^e ≥ i`).
fn iroot(n: i64, e: u64) -> Option<i64> {
    if n < 0 || e == 0 {
        return None;
    }
    let mut i: i64 = 0;
    loop {
        let p = i.checked_pow(e as u32)?;
        if p == n {
            return Some(i);
        }
        if p > n || i > n {
            return None;
        }
        i += 1;
    }
}

/// Outcome of solving `term ≡ numeral target`.
enum Solve {
    /// Equation holds (possibly after solving metas).
    Ok,
    /// Provably unsatisfiable (distinct numerals, non-divisible product,
    /// absent root, occurs violation).
    Decided,
    /// Cannot decide — the term is symbolic beyond the solver fragment.
    Stuck,
}

/// Solve `term ≡ target` (the model's `solve_index_`): linear `X + c`,
/// `X - c`, `c - X`; `X * c` by exact division (non-divisible is
/// *decided*); `X * X` and `X ** k` by integer root; metas solve
/// (occurs-guarded); anything else is stuck.
fn solve_index(term: &Value, target: i64, ctx: &mut TypecheckContext) -> Solve {
    let t = eval_index(term, ctx);
    if let Some(n) = num_val(&t) {
        return if n as i64 == target && n == target as f64 {
            Solve::Ok
        } else {
            Solve::Decided
        };
    }
    if let Some(id) = extract_meta_id(&t) {
        if var_occurs(&format!("__meta__{id}"), &Value::from(target)) {
            return Solve::Decided;
        }
        ctx.solve_meta(id, Value::from(target));
        return Solve::Ok;
    }
    if tag_of(&t) != Some("Bin") {
        return Solve::Stuck;
    }
    let op = t.get("op").and_then(Value::as_str).unwrap_or("");
    let l = eval_index(t.get("l").unwrap_or(&Value::Null), ctx);
    let r = eval_index(t.get("r").unwrap_or(&Value::Null), ctx);
    let ln = num_val(&l);
    let rn = num_val(&r);
    match op {
        "binary.add" => {
            if let Some(c) = rn {
                solve_index(&l, target - c as i64, ctx)
            } else if let Some(c) = ln {
                solve_index(&r, target - c as i64, ctx)
            } else {
                Solve::Stuck
            }
        }
        "binary.sub" => {
            if let Some(c) = rn {
                solve_index(&l, target + c as i64, ctx)
            } else if let Some(c) = ln {
                solve_index(&r, c as i64 - target, ctx)
            } else {
                Solve::Stuck
            }
        }
        "binary.mul" => {
            if let Some(c) = rn {
                if c == 0.0 {
                    return if target == 0 { Solve::Ok } else { Solve::Decided };
                }
                if c.fract() == 0.0 && target % c as i64 == 0 {
                    solve_index(&l, target / c as i64, ctx)
                } else {
                    Solve::Decided
                }
            } else if let Some(c) = ln {
                if c == 0.0 {
                    return if target == 0 { Solve::Ok } else { Solve::Decided };
                }
                if c.fract() == 0.0 && target % c as i64 == 0 {
                    solve_index(&r, target / c as i64, ctx)
                } else {
                    Solve::Decided
                }
            } else if l == r {
                // X * X = t → integer square root.
                match iroot(target, 2) {
                    Some(s) => solve_index(&l, s, ctx),
                    None => Solve::Decided,
                }
            } else {
                Solve::Stuck
            }
        }
        "binary.pow" => {
            if let Some(e) = rn {
                if e.fract() == 0.0 && e >= 1.0 {
                    match iroot(target, e as u64) {
                        Some(s) => solve_index(&l, s, ctx),
                        None => Solve::Decided,
                    }
                } else {
                    Solve::Stuck
                }
            } else {
                Solve::Stuck
            }
        }
        _ => Solve::Stuck,
    }
}

/// Index-term unification: `a ≡ b` — metas solve, rigids bind through
/// `results`, closed terms compare, open `Bin`s go through
/// [`solve_index`]. Returns `(ok, decided_failure)`.
fn unify_index(a: &Value, b: &Value, ctx: &mut TypecheckContext) -> (bool, bool) {
    let ea = eval_index(a, ctx);
    let eb = eval_index(b, ctx);
    if ea == eb {
        return (true, false);
    }
    if let Some(id) = extract_meta_id(&ea) {
        ctx.solve_meta(id, eb);
        return (true, false);
    }
    if let Some(id) = extract_meta_id(&eb) {
        ctx.solve_meta(id, ea);
        return (true, false);
    }
    if tag_of(&ea) == Some("var") {
        let name = ea.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        if !name.is_empty() && !var_occurs(&name, &eb) {
            ctx.bind_result(name.clone(), eb.clone());
            ctx.bind_result(format!("eq_{name}"), path_val(ea.clone(), eb));
        }
        return (true, false);
    }
    if tag_of(&eb) == Some("var") {
        let name = eb.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        if !name.is_empty() && !var_occurs(&name, &ea) {
            ctx.bind_result(name.clone(), ea.clone());
            ctx.bind_result(format!("eq_{name}"), path_val(eb.clone(), ea));
        }
        return (true, false);
    }
    let (an, bn) = (num_val(&ea), num_val(&eb));
    if let Some(t) = bn {
        if t.fract() == 0.0 {
            return match solve_index(&ea, t as i64, ctx) {
                Solve::Ok => (true, false),
                Solve::Decided => (false, true),
                Solve::Stuck => (false, false),
            };
        }
    }
    if let Some(t) = an {
        if t.fract() == 0.0 {
            return match solve_index(&eb, t as i64, ctx) {
                Solve::Ok => (true, false),
                Solve::Decided => (false, true),
                Solve::Stuck => (false, false),
            };
        }
    }
    (false, false)
}

/// Deep-substitute `to` for `frm` inside an index term (transport).
fn subst_index(term: &Value, frm: &Value, to: &Value) -> Value {
    if term == frm {
        return to.clone();
    }
    match term {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                out.insert(k.clone(), subst_index(v, frm, to));
            }
            Value::Object(out)
        }
        Value::Array(xs) => Value::Array(xs.iter().map(|v| subst_index(v, frm, to)).collect()),
        _ => term.clone(),
    }
}

// ---------------------------------------------------------------------------
// `data` payload parsing (mirror of the desugar native's parse)
// ---------------------------------------------------------------------------

/// A family parameter: name, optional annotation node, `nat` flag for the
/// bare-constructor base-index convention.
#[derive(Debug, Clone)]
pub(crate) struct DepParam {
    name: String,
    node: Option<usize>,
    nat: bool,
}

/// A ctor signature parsed from a `data` node.
#[derive(Debug, Clone)]
pub(crate) struct DepCtor {
    name: String,
    /// `(arg_name, type-expr node idx)` — `node` absent for untyped args.
    args: Vec<(String, Option<usize>)>,
    /// The declared result type-expr node (`Vec(n + 1, a)`), if annotated.
    result: Option<usize>,
    /// Bare `var` constructor (the `0`-base-index bootstrap convention).
    bare: bool,
}

/// Parse a `data` node's `(type_name, params, ctors)` — roles first,
/// positional fallback (same shapes as `desugar.rs::parse_data_payload`).
pub(crate) fn parse_data_node(
    lang: &Language,
    node: &AstNode,
    ast: &Ast,
) -> Option<(String, Vec<DepParam>, Vec<DepCtor>)> {
    let children = children_of(node)?;
    // Hand-built payload: {"name": ..., "constructors": [...]} — no AST
    // params; ctor args carry inline `{name, type}` strings.
    if node.payload.get("constructors").is_some() {
        let name = node.payload.get("name")?.as_str()?.to_string();
        return Some((name, Vec::new(), Vec::new()));
    }
    if children.is_empty() {
        return None;
    }
    let (name_idx, cons_idx) = if children.len() >= 2 {
        (children[0].as_u64()? as usize, children.last()?.clone())
    } else {
        return None;
    };
    let name_node = ast.get(AstRef::new(name_idx))?;
    let (name, params) = parse_family_head(lang, name_node, ast)?;
    let cons_node = ast.get(AstRef::new(cons_idx.as_u64()? as usize))?;
    let mut ctors = Vec::new();
    collect_ctors(lang, cons_node, ast, &mut ctors);
    Some((name, params, ctors))
}

fn parse_family_head(
    lang: &Language,
    node: &AstNode,
    ast: &Ast,
) -> Option<(String, Vec<DepParam>)> {
    match node.kind.as_str() {
        "var" => Some((node.payload.as_str()?.to_string(), Vec::new())),
        "call" => {
            let kids = children_of(node)?;
            let callee = ast.get(AstRef::new(kids.first()?.as_u64()? as usize))?;
            let name = callee.payload.as_str()?.to_string();
            let mut params = Vec::new();
            for cv in kids.iter().skip(1) {
                let p = ast.get(AstRef::new(cv.as_u64()? as usize))?;
                params.push(parse_param(lang, p, ast)?);
            }
            Some((name, params))
        }
        _ => None,
    }
}

/// `type-annot(var n, T)` → `{name: n, node: T}`; bare `var p` →
/// `{name: p, node: None}`.
fn parse_param(lang: &Language, node: &AstNode, ast: &Ast) -> Option<DepParam> {
    if node.kind == "type-annot" || role_child_ref(lang, node, "annot").is_some() {
        let kids = children_of(node)?;
        let subj = lang
            .child_role_index(&node.kind, "subject")
            .ok()
            .flatten()
            .unwrap_or(0);
        let annot = lang
            .child_role_index(&node.kind, "annot")
            .ok()
            .flatten()
            .unwrap_or(1);
        let name = node_name(ast.get(AstRef::new(kids.get(subj)?.as_u64()? as usize))?)?;
        let ty_idx = kids.get(annot)?.as_u64()? as usize;
        let nat = node_name(ast.get(AstRef::new(ty_idx))?).as_deref() == Some("Nat");
        return Some(DepParam {
            name,
            node: Some(ty_idx),
            nat,
        });
    }
    let name = node.payload.as_str()?.to_string();
    Some(DepParam {
        name,
        node: None,
        nat: false,
    })
}

fn collect_ctors(lang: &Language, node: &AstNode, ast: &Ast, out: &mut Vec<DepCtor>) {
    match node.kind.as_str() {
        "alt" => {
            if let Some(kids) = children_of(node) {
                for c in kids {
                    if let Some(n) = ast.get(AstRef::new(c.as_u64().unwrap_or(u64::MAX) as usize))
                    {
                        collect_ctors(lang, n, ast, out);
                    }
                }
            }
        }
        "type-annot" => {
            let kids = children_of(node).cloned().unwrap_or_default();
            let subj = lang
                .child_role_index(&node.kind, "subject")
                .ok()
                .flatten()
                .unwrap_or(0);
            let annot = lang
                .child_role_index(&node.kind, "annot")
                .ok()
                .flatten()
                .unwrap_or(1);
            let cons_idx = kids.get(subj).and_then(Value::as_u64).map(|v| v as usize);
            let result_idx = kids.get(annot).and_then(Value::as_u64).map(|v| v as usize);
            if let Some(ci) = cons_idx {
                if let Some(cn) = ast.get(AstRef::new(ci)) {
                    collect_ctor_leaf(cn, ast, out, result_idx, false);
                }
            }
        }
        _ => collect_ctor_leaf(node, ast, out, None, node.kind == "var"),
    }
}

fn collect_ctor_leaf(
    node: &AstNode,
    ast: &Ast,
    out: &mut Vec<DepCtor>,
    result: Option<usize>,
    bare: bool,
) {
    match node.kind.as_str() {
        "var" => out.push(DepCtor {
            name: node.payload.as_str().unwrap_or("").to_string(),
            args: Vec::new(),
            result,
            bare: true,
        }),
        "call" => {
            let Some(kids) = children_of(node) else { return };
            let Some(callee_idx) = kids.first().and_then(Value::as_u64) else {
                return;
            };
            let Some(callee) = ast.get(AstRef::new(callee_idx as usize)) else {
                return;
            };
            let name = callee.payload.as_str().unwrap_or("").to_string();
            let mut args = Vec::new();
            for (i, cv) in kids.iter().enumerate().skip(1) {
                let Some(arg_idx) = cv.as_u64() else { continue };
                let Some(arg) = ast.get(AstRef::new(arg_idx as usize)) else {
                    continue;
                };
                if arg.kind == "type-annot" {
                    let ak = children_of(arg).cloned().unwrap_or_default();
                    let an = ak
                        .first()
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(node_name)
                        .unwrap_or_else(|| format!("_{}", i - 1));
                    let tn = ak.get(1).and_then(Value::as_u64).map(|v| v as usize);
                    args.push((an, tn));
                } else {
                    args.push((format!("_{}", i - 1), Some(arg_idx as usize)));
                }
            }
            out.push(DepCtor {
                name,
                args,
                result,
                bare,
            });
        }
        _ => {}
    }
}

/// Strict positivity over the ctor-arg AST (checker-owned, ADR-0152 §4):
/// the defined former must not occur in a *domain* position. `occurs` is
/// any occurrence; a `call(->, dom, cod)`/`function` node's domain child
/// is the negative position.
fn occurs_name(ast: &Ast, node: &AstNode, name: &str) -> bool {
    if node.kind == "var" {
        return node.payload.as_str() == Some(name);
    }
    if let Some(kids) = children_of(node) {
        return kids.iter().any(|c| {
            c.as_u64()
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .is_some_and(|n| occurs_name(ast, n, name))
        });
    }
    if let Some(b) = node.payload.get("body").and_then(Value::as_u64) {
        if let Some(n) = ast.get(AstRef::new(b as usize)) {
            return occurs_name(ast, n, name);
        }
    }
    false
}

fn positivity_ok(ast: &Ast, node: &AstNode, name: &str) -> bool {
    match node.kind.as_str() {
        "function" => {
            // params child is the domain — the former must not occur.
            let kids = children_of(node).cloned().unwrap_or_default();
            let dom = kids.first().and_then(Value::as_u64);
            let cod = kids.get(1).and_then(Value::as_u64);
            let dom_bad = dom
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .is_some_and(|n| occurs_name(ast, n, name));
            if dom_bad {
                return false;
            }
            cod.and_then(|i| ast.get(AstRef::new(i as usize)))
                .is_none_or(|n| positivity_ok(ast, n, name))
        }
        "call" => {
            let kids = children_of(node).cloned().unwrap_or_default();
            let callee_is_arrow = kids
                .first()
                .and_then(Value::as_u64)
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .is_some_and(|n| {
                    n.kind == "var" && (n.payload.as_str() == Some("->") || n.payload.as_str() == Some("~>"))
                });
            if callee_is_arrow && kids.len() >= 3 {
                let dom_bad = kids
                    .get(1)
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .is_some_and(|n| occurs_name(ast, n, name));
                if dom_bad {
                    return false;
                }
                return kids
                    .get(2)
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .is_none_or(|n| positivity_ok(ast, n, name));
            }
            // Other applications: args are positive positions.
            kids.iter().all(|c| {
                c.as_u64()
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .is_none_or(|n| positivity_ok(ast, n, name))
            })
        }
        _ => {
            if let Some(kids) = children_of(node) {
                return kids.iter().all(|c| {
                    c.as_u64()
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .is_none_or(|n| positivity_ok(ast, n, name))
                });
            }
            true
        }
    }
}

impl NbeTypecheckEngine {
    /// Interpret an annotation/type-expression node into a type value —
    /// the dependent counterpart of `interpret_type_expr`: declared
    /// families/ctors resolve through Γ (and `results` for index vars),
    /// `call` covers former applications, `binary.*`/`unary.*` evaluate
    /// as index terms, and `Path`/`Family`/`Pi`/`Sigma`/`Forall` are the
    /// dependent surface formers.
    pub(crate) fn dep_synth_type(
        &self,
        lang: &Language,
        ast: &Ast,
        idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast.get(AstRef::new(idx)).ok_or_else(|| {
            LangError::engine(format!("dep_synth_type: node {} out of bounds", idx))
        })?;
        match node.kind.as_str() {
            "paren" => {
                let inner = child_at(ast, node, 0)
                    .ok_or_else(|| LangError::engine("dep_synth_type: empty paren"))?;
                self.dep_synth_type(lang, ast, inner.index(), ctx)
            }
            "type-annot" => {
                let annot = role_child_ref(lang, node, "annot")
                    .or_else(|| child_at(ast, node, 1))
                    .ok_or_else(|| LangError::engine("dep_synth_type: type-annot missing annot"))?;
                self.dep_synth_type(lang, ast, annot.index(), ctx)
            }
            "value" | "lit" => Ok(node.payload.clone()),
            "var" => self.dep_synth_var_type(lang, ast, node, ctx),
            "field-access" => {
                let kids = children_of(node)
                    .ok_or_else(|| LangError::engine("dep_synth_type: field-access no children"))?;
                let field_idx = kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
                    LangError::engine("dep_synth_type: field-access missing field child")
                })? as usize;
                let field = node_name(ast.get(AstRef::new(field_idx)).ok_or_else(|| {
                    LangError::engine("dep_synth_type: field node out of bounds")
                })?)
                .ok_or_else(|| LangError::engine("dep_synth_type: field name missing"))?;
                self.dep_builtin_or_tag(&field, ctx)
            }
            "array" => {
                let inner = child_at(ast, node, 0)
                    .ok_or_else(|| LangError::engine("dep_synth_type: empty array"))?;
                let el = self.dep_synth_type(lang, ast, inner.index(), ctx)?;
                let mut m = Map::new();
                m.insert("element".to_string(), el);
                Ok(make_tag(ctx, "type.array", m))
            }
            "function" => {
                let kids = children_of(node)
                    .ok_or_else(|| LangError::engine("dep_synth_type: function no children"))?;
                let params_idx = role_child_ref(lang, node, "params")
                    .or_else(|| child_at(ast, node, 0))
                    .ok_or_else(|| LangError::engine("dep_synth_type: function missing params"))?;
                let body_idx = role_child_ref(lang, node, "body")
                    .or_else(|| child_at(ast, node, 1))
                    .ok_or_else(|| LangError::engine("dep_synth_type: function missing body"))?;
                // The params child is a type expression (or a paren-wrapped
                // params list — the last var is the domain for `A -> B`).
                let dom_idx = if let Some(p) = ast.get(params_idx) {
                    children_of(p)
                        .and_then(|k| k.last())
                        .and_then(Value::as_u64)
                        .map(|i| AstRef::new(i as usize))
                        .unwrap_or(params_idx)
                } else {
                    params_idx
                };
                let dom = self.dep_synth_type(lang, ast, dom_idx.index(), ctx)?;
                let cod = self.dep_synth_type(lang, ast, body_idx.index(), ctx)?;
                let mut m = Map::new();
                m.insert("param".to_string(), dom);
                m.insert("body".to_string(), cod);
                Ok(make_tag(ctx, "type.function", m))
            }
            "call" => self.dep_synth_call_type(lang, ast, node, ctx),
            "absurd" | "if" | "match" | "object" | "index" => {
                // Non-type positions in annotations — leave them open.
                let mut m = Map::new();
                Ok(make_tag(ctx, "type.unknown", std::mem::take(&mut m)))
            }
            _ => {
                if node.kind.starts_with("binary.") || node.kind.starts_with("unary.") {
                    let t = self.node_to_index_term(lang, ast, node, ctx);
                    Ok(eval_index(&t, ctx))
                } else if node.kind == "transport" {
                    let mut m = Map::new();
                    Ok(make_tag(ctx, "type.unknown", std::mem::take(&mut m)))
                } else {
                    let mut m = Map::new();
                    Ok(make_tag(ctx, "type.unknown", std::mem::take(&mut m)))
                }
            }
        }
    }

    /// `var` in type position — ctor-sig instantiation, family →
    /// `family_app` at fresh metas, `results` index terms, Γ bindings,
    /// then the builtin table (`Nat`/`Bool`/`Type`/`Pi`/…).
    fn dep_synth_var_type(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let name_owned = node_name(node)
            .ok_or_else(|| LangError::engine("dep_synth_type: var payload is not a string"))?;
        let name = name_owned.as_str();
        if let Some(bound) = ctx.lookup(name).cloned() {
            match tag_of(&bound) {
                Some("type.ctor_sig") => {
                    return self.instantiate_ctor_full(lang, ast, &bound, ctx)
                }
                Some("type.family") => {
                    if bound.get("name").and_then(Value::as_str).unwrap_or("") == name {
                        let n = bound
                            .get("base")
                            .and_then(|b| b.get("elements"))
                            .and_then(Value::as_array)
                            .map(|e| e.len())
                            .unwrap_or(0);
                        let args: Vec<Value> = (0..n)
                            .map(|_| meta_value(ctx.fresh_meta()))
                            .collect();
                        let mut m = Map::new();
                        m.insert("name".to_string(), Value::String(name.to_string()));
                        m.insert("args".to_string(), Value::Array(args));
                        return Ok(make_tag(ctx, "type.family_app", m));
                    }
                }
                _ => return Ok(bound),
            }
        }
        // `results` bindings shadow builtins for index variables.
        if ctx.results.contains_key(name) {
            return Ok(eval_index(&var_term(name), ctx));
        }
        self.dep_builtin_or_tag(name, ctx)
    }

    /// The builtin-name table for bare type-position identifiers.
    fn dep_builtin_or_tag(&self, name: &str, ctx: &mut TypecheckContext) -> Result<Value, LangError> {
        let tag = match name {
            "Int" | "Nat" | "Number" => "type.number",
            "Bool" => "type.bool",
            "String" | "Str" => "type.string",
            "Unit" => "type.unit",
            "Type" | "Universe" | "Type0" => "type.universe",
            "List" | "Family" | "Pi" | "Sigma" | "Forall" => "type.constructor",
            _ => {
                return Err(LangError::engine(format!(
                    "dep_synth_type: unbound type variable '{name}'"
                )))
            }
        };
        let mut m = Map::new();
        Ok(make_tag(
            ctx,
            tag,
            match name {
                "Type" | "Universe" | "Type0" => {
                    m.insert("level".to_string(), Value::from(0));
                    std::mem::take(&mut m)
                }
                "List" | "Family" | "Pi" | "Sigma" | "Forall" => {
                    m.insert("name".to_string(), Value::String(name.to_string()));
                    std::mem::take(&mut m)
                }
                _ => std::mem::take(&mut m),
            },
        ))
    }

    /// `call` in type position — dependent formers (`Family`/`Pi`/`Sigma`/
    /// `Forall`/`Path`), family applications `Vec(n + 1, a)`, and
    /// fall-through unknowns.
    fn dep_synth_call_type(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("dep_synth_type: call has no children"))?;
        if kids.is_empty() {
            return Err(LangError::engine("dep_synth_type: empty call"));
        }
        let callee_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("dep_synth_type: call callee not a node index")
        })? as usize;
        let callee = ast.get(AstRef::new(callee_idx)).ok_or_else(|| {
            LangError::engine("dep_synth_type: callee out of bounds")
        })?;
        let callee_name = if callee.kind == "var" {
            callee.payload.as_str().map(str::to_string)
        } else if callee.kind == "field-access" {
            children_of(callee)
                .and_then(|k| k.get(1))
                .and_then(Value::as_u64)
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .and_then(|n| node_name(n))
        } else {
            None
        };
        let arg_indices: Vec<usize> = kids
            .iter()
            .skip(1)
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
        match callee_name.as_deref() {
            Some("Family") => {
                let base = match arg_indices.first() {
                    Some(&i) => self.dep_synth_type(lang, ast, i, ctx)?,
                    None => meta_value(ctx.fresh_meta()),
                };
                let mut m = Map::new();
                m.insert("name".to_string(), Value::Null);
                m.insert("base".to_string(), base);
                Ok(make_tag(ctx, "type.family", m))
            }
            Some(tag @ ("Pi" | "Sigma" | "Forall")) => {
                let kind = match tag {
                    "Pi" => "type.pi",
                    "Sigma" => "type.sigma",
                    _ => "type.forall",
                };
                let param = match arg_indices.first() {
                    Some(&i) => self.dep_synth_type(lang, ast, i, ctx)?,
                    None => meta_value(ctx.fresh_meta()),
                };
                // The body argument is an arrow `n ~> T`: the closure
                // records its param name and body child.
                let (var_name, body_idx) = match arg_indices.get(1) {
                    Some(&i) => self.arrow_parts(lang, ast, i).unwrap_or(("x".to_string(), i)),
                    None => ("x".to_string(), callee_idx),
                };
                let mut m = Map::new();
                let param_key = if tag == "Forall" { "kind" } else { "param" };
                m.insert(param_key.to_string(), param);
                m.insert(
                    "body_closure".to_string(),
                    serde_json::json!({"var_name": var_name, "body_child_idx": body_idx, "dep": true}),
                );
                Ok(make_tag(ctx, kind, m))
            }
            Some("Path") => {
                let frm = match arg_indices.first() {
                    Some(&i) => {
                        let t = self.node_index_or_type(lang, ast, i, ctx)?;
                        eval_index(&t, ctx)
                    }
                    None => Value::Null,
                };
                let to = match arg_indices.get(1) {
                    Some(&i) => {
                        let t = self.node_index_or_type(lang, ast, i, ctx)?;
                        eval_index(&t, ctx)
                    }
                    None => Value::Null,
                };
                Ok(path_val(frm, to))
            }
            Some(name) => {
                // A declared family's application: `Vec(i⃗)` →
                // `type.family_app{name, args}` — index arguments
                // evaluate (1+1 → 2), rigid vars stay `var{n}`.
        if let Some(bound) = ctx.lookup(name).cloned() {
                    match tag_of(&bound) {
                        Some("type.family") => {
                            let mut args = Vec::with_capacity(arg_indices.len());
                            for &i in &arg_indices {
                                let t = self.node_index_or_type(lang, ast, i, ctx)?;
                                args.push(eval_index(&t, ctx));
                            }
                            let mut m = Map::new();
                            m.insert(
                                "name".to_string(),
                                bound.get("name").cloned().unwrap_or(Value::String(name.to_string())),
                            );
                            m.insert("args".to_string(), Value::Array(args));
                            return Ok(make_tag(ctx, "type.family_app", m));
                        }
                        Some("type.ctor_sig") => {
                            return self.instantiate_ctor_full(lang, ast, &bound, ctx);
                        }
                        _ => {}
                    }
                }
                // `bound_callee`-style: a results-bound arrow applied to
                // index terms — opaque `App` term.
                if ctx.results.contains_key(name) {
                    let mut m = Map::new();
                    m.insert("callee".to_string(), var_term(name));
                    let mut args = Vec::with_capacity(arg_indices.len());
                    for &i in &arg_indices {
                        let t = self.node_index_or_type(lang, ast, i, ctx)?;
                        args.push(eval_index(&t, ctx));
                    }
                    m.insert("args".to_string(), Value::Array(args));
                    return Ok(make_tag(ctx, "App", m));
                }
                Err(LangError::engine(format!(
                    "dep_synth_type: unsupported callee '{name}'"
                )))
            }
            None => Err(LangError::engine(
                "dep_synth_type: call callee is not a name".to_string(),
            )),
        }
    }

    /// `(params, body)` parts of a `function` node for dependent formers:
    /// returns `(param_name, body_node_index)`.
    fn arrow_parts(&self, lang: &Language, ast: &Ast, idx: usize) -> Option<(String, usize)> {
        let node = ast.get(AstRef::new(idx))?;
        let inner = if node.kind == "paren" {
            ast.get(child_at(ast, node, 0)?)?
        } else {
            node
        };
        if inner.kind != "function" {
            return None;
        }
        let kids = children_of(inner)?;
        let params_idx = role_child_ref(lang, inner, "params")
            .or_else(|| child_at(ast, inner, 0))?;
        let body_idx = role_child_ref(lang, inner, "body")
            .or_else(|| child_at(ast, inner, 1))?;
        let pnode = ast.get(params_idx)?;
        let name = lang
            .param_names(ast, pnode)
            .ok()
            .and_then(|(names, _)| names.first().cloned())
            .or_else(|| node_name(pnode))
            .unwrap_or_else(|| "x".to_string());
        Some((name, body_idx.index()))
    }

    /// In `family_app` argument position a node is an *index term* when it
    /// is arithmetic or a literal, and a *type* otherwise.
    fn node_index_or_type(
        &self,
        lang: &Language,
        ast: &Ast,
        idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast.get(AstRef::new(idx)).ok_or_else(|| {
            LangError::engine(format!("dep: node {} out of bounds", idx))
        })?;
        if node.kind.starts_with("binary.")
            || node.kind.starts_with("unary.")
            || matches!(node.kind.as_str(), "value" | "lit")
        {
            return Ok(self.node_to_index_term(lang, ast, node, ctx));
        }
        self.dep_synth_type(lang, ast, idx, ctx)
    }

    /// AST node → index *term* (`var`/`Bin`/`Un`/literal/meta) — the raw
    /// algebra [`eval_index`] and [`solve_index`] compute over.
    fn node_to_index_term(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Value {
        if node.kind.starts_with("binary.") || node.kind.starts_with("unary.") {
            let kids = children_of(node).cloned().unwrap_or_default();
            if node.kind.starts_with("unary.") {
                let o = kids
                    .first()
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                    .unwrap_or(Value::Null);
                return un_term(&node.kind, o);
            }
            let l = kids
                .first()
                .and_then(Value::as_u64)
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                .unwrap_or(Value::Null);
            let r = kids
                .get(1)
                .and_then(Value::as_u64)
                .and_then(|i| ast.get(AstRef::new(i as usize)))
                .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                .unwrap_or(Value::Null);
            return bin_term(&node.kind, l, r);
        }
        match node.kind.as_str() {
            "var" => var_term(node.payload.as_str().unwrap_or("")),
            "paren" => child_at(ast, node, 0)
                .and_then(|i| ast.get(i))
                .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                .unwrap_or(Value::Null),
            "value" | "lit" => node.payload.clone(),
            _ => Value::Null,
        }
    }

    // -----------------------------------------------------------------------
    // `synth_data` / `instantiate_ctor`
    // -----------------------------------------------------------------------

    /// The `synth_data` judgment: bind `F : type.family{name, base}` (one
    /// fresh meta or the declared sort per index) and each ctor a
    /// `type.ctor_sig` telescope — with strict positivity checked here
    /// (the checker sees the ctor sigs, ADR-0152 §4).
    pub(crate) fn synth_data_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast.get(node_ref).ok_or_else(|| {
            LangError::engine("synth_data: node out of bounds".to_string())
        })?;
        // Hand-built payload (`{"name","constructors"}`) — used by the
        // desugar-native tests; no AST params to bind, return unit.
        if node.payload.get("constructors").is_some() {
            let name = node
                .payload
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                let mut m = Map::new();
                m.insert("name".to_string(), Value::String(name.clone()));
                let mut base_m = Map::new();
                base_m.insert("elements".to_string(), Value::Array(vec![]));
                m.insert("base".to_string(), make_tag(ctx, "type.array", base_m));
                ctx.extend(name, make_tag(ctx, "type.family", m));
            }
            let mut m = Map::new();
            return Ok(make_tag(ctx, "type.unit", std::mem::take(&mut m)));
        }
        let Some((name, params, ctors)) = parse_data_node(lang, node, ast) else {
            return Err(LangError::engine_node(
                "synth_data: malformed data declaration".to_string(),
                node,
            ));
        };
        // Strict positivity — the checker owns it (it sees the sigs).
        for c in &ctors {
            for (_, ty_idx) in &c.args {
                if let Some(ti) = ty_idx {
                    if let Some(n) = ast.get(AstRef::new(*ti)) {
                        if !positivity_ok(ast, n, &name) {
                            return Err(LangError::engine(format!(
                                "synth_data: strict positivity violated: constructor '{}' \
                                 mentions '{name}' in a negative position",
                                c.name
                            )));
                        }
                    }
                }
            }
        }
        // `base` = the parameter sorts (`type.array` of them).
        let mut elems = Vec::with_capacity(params.len());
        for p in &params {
            elems.push(match p.node {
                Some(ti) => self.dep_synth_type(lang, ast, ti, ctx)?,
                None => meta_value(ctx.fresh_meta()),
            });
        }
        let mut base_m = Map::new();
        base_m.insert("elements".to_string(), Value::Array(elems));
        let base = make_tag(ctx, "type.array", base_m);
        if !name.is_empty() {
            let mut m = Map::new();
            m.insert("name".to_string(), Value::String(name.clone()));
            m.insert("base".to_string(), base);
            ctx.extend(name.clone(), make_tag(ctx, "type.family", m));
        }
        // Ctor signatures — the `type.ctor_sig` spec re-elaborated per
        // use by `instantiate_ctor`.
        for c in &ctors {
            if c.name.is_empty() {
                continue;
            }
            let sig = serde_json::json!({
                "__tag__": "type.ctor_sig",
                "__registry__": ctx.registry.registry_id(),
                "ctor": true,
                "family": name,
                "params": params.iter().map(|p| {
                    serde_json::json!({"name": p.name, "node": p.node, "nat": p.nat})
                }).collect::<Vec<_>>(),
                "args": c.args.iter().map(|(n, t)| {
                    serde_json::json!({"name": n, "node": t})
                }).collect::<Vec<_>>(),
                "result": c.result,
                "bare": c.bare,
            });
            ctx.extend(c.name.clone(), sig);
        }
        let mut m = Map::new();
        Ok(make_tag(ctx, "type.unit", std::mem::take(&mut m)))
    }

    /// Instantiate a `type.ctor_sig` spec into a fresh `type.function`
    /// chain ending in `type.family_app` — one index metavariable per
    /// family parameter (bound in Γ *and* `results`, so `xs : Vec(n, a)`
    /// elaborates `n ↦ ?n` and the result index `n + 1` solves through
    /// the same meta). Constructor argument names self-bind to
    /// `var{name}` terms so later domains may mention them.
    pub(crate) fn instantiate_ctor(&self, spec: &Value, ctx: &mut TypecheckContext) -> Value {
        let mark = ctx.mark();
        let old_results = ctx.results.clone();
        let params: Vec<Value> = spec
            .get("params")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut metas = Vec::with_capacity(params.len());
        for p in &params {
            let pname = p.get("name").and_then(Value::as_str).unwrap_or("");
            let id = ctx.fresh_meta();
            let mv = meta_value(id);
            if !pname.is_empty() {
                ctx.extend(pname.to_string(), mv.clone());
                ctx.bind_result(pname.to_string(), mv.clone());
            }
            metas.push(mv);
        }
        let args: Vec<Value> = spec
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for a in &args {
            let aname = a.get("name").and_then(Value::as_str).unwrap_or("");
            if !aname.is_empty() {
                ctx.bind_result(aname.to_string(), var_term(aname));
            }
        }
        // This lang/ast-less path only mints the parameter metas and
        // returns the spec — callers that have `lang`/`ast` use
        // `instantiate_ctor_full` for the real elaboration.
        let _ = (args, metas.clone());
        ctx.restore(mark);
        ctx.results = old_results;
        spec.clone()
    }

    /// Full instantiation with AST access — the real `instantiate_ctor_`.
    pub(crate) fn instantiate_ctor_full(
        &self,
        lang: &Language,
        ast: &Ast,
        spec: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let mark = ctx.mark();
        let old_results = ctx.results.clone();
        let params: Vec<Value> = spec
            .get("params")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut metas = Vec::with_capacity(params.len());
        for p in &params {
            let pname = p.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let id = ctx.fresh_meta();
            let mv = meta_value(id);
            if !pname.is_empty() {
                ctx.extend(pname.clone(), mv.clone());
                ctx.bind_result(pname, mv.clone());
            }
            metas.push(mv);
        }
        let args: Vec<Value> = spec
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for a in &args {
            let aname = a.get("name").and_then(Value::as_str).unwrap_or("");
            if !aname.is_empty() {
                ctx.bind_result(aname.to_string(), var_term(aname));
            }
        }
        let mut domains = Vec::with_capacity(args.len());
        for a in &args {
            let ty = match a.get("node").and_then(Value::as_u64) {
                Some(ti) => self.dep_synth_type(lang, ast, ti as usize, ctx)?,
                None => {
                    let mut m = Map::new();
                    make_tag(ctx, "type.unknown", std::mem::take(&mut m))
                }
            };
            domains.push(ty);
        }
        let family = spec
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let result = match spec.get("result").and_then(Value::as_u64) {
            Some(ri) => self.dep_synth_type(lang, ast, ri as usize, ctx)?,
            None => {
                let mut m = Map::new();
                if spec.get("bare").and_then(Value::as_bool).unwrap_or(false)
                    && !params.is_empty()
                {
                    // Bare ctor convention: `0` for the first `Nat`
                    // parameter, param metas elsewhere.
                    let mut idx_args = Vec::with_capacity(params.len());
                    let mut used = false;
                    for (i, p) in params.iter().enumerate() {
                        let is_nat = p.get("nat").and_then(Value::as_bool).unwrap_or(false);
                        if is_nat && !used {
                            idx_args.push(Value::from(0));
                            used = true;
                        } else {
                            idx_args.push(metas[i].clone());
                        }
                    }
                    m.insert("args".to_string(), Value::Array(idx_args));
                } else {
                    m.insert("args".to_string(), Value::Array(metas.clone()));
                }
                m.insert("name".to_string(), Value::String(family.clone()));
                make_tag(ctx, "type.family_app", m)
            }
        };
        // Fold `type.function{param, body, param_name}` right-to-left.
        let mut ty = result;
        for (i, a) in args.iter().enumerate().rev() {
            let mut m = Map::new();
            m.insert("param".to_string(), domains[i].clone());
            m.insert("body".to_string(), ty);
            m.insert(
                "param_name".to_string(),
                a.get("name").cloned().unwrap_or(Value::Null),
            );
            ty = make_tag(ctx, "type.function", m);
        }
        ctx.restore(mark);
        ctx.results = old_results;
        Ok(ty)
    }

    /// Peel an instantiated ctor `type.function` chain into
    /// `(domains, family_app_result)`.
    fn peel_ctor(&self, ty: &Value) -> (Vec<(Value, Value)>, Value) {
        let mut domains = Vec::new();
        let mut cur = ty.clone();
        while tag_of(&cur) == Some("type.function") {
            let param = cur.get("param").cloned().unwrap_or(Value::Null);
            let pname = cur.get("param_name").cloned().unwrap_or(Value::Null);
            domains.push((pname, param));
            cur = cur.get("body").cloned().unwrap_or(Value::Null);
        }
        (domains, cur)
    }

    /// Dependent `unify`: `family_app` indices through [`unify_index`],
    /// `Path` endpoints the same, everything else the structural
    /// `unify`.
    pub(crate) fn dep_unify(
        &self,
        a: &Value,
        b: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<(), LangError> {
        if a == b {
            return Ok(());
        }
        if extract_meta_id(a).is_some() || extract_meta_id(b).is_some() {
            return self.unify(a, b, ctx);
        }
        match (tag_of(a), tag_of(b)) {
            (Some("type.family_app"), Some("type.family_app")) => {
                let na = a.get("name").and_then(Value::as_str).unwrap_or("");
                let nb = b.get("name").and_then(Value::as_str).unwrap_or("");
                if na != nb {
                    return Err(LangError::engine(format!(
                        "typecheck: cannot unify family {na} with {nb}"
                    )));
                }
                let xs = a.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
                let ys = b.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
                if xs.len() != ys.len() {
                    return Err(LangError::engine("typecheck: family arity mismatch".to_string()));
                }
                for (x, y) in xs.iter().zip(ys.iter()) {
                    let (ok, decided) = unify_index(x, y, ctx);
                    if !ok {
                        if decided {
                            return Err(LangError::engine(format!(
                                "typecheck: index mismatch in {na}({:?}) vs {na}({:?})",
                                xs, ys
                            )));
                        }
                        // Stuck indices still try the structural path
                        // (e.g. type-argument metas).
                        self.unify(x, y, ctx).map_err(|_| {
                            LangError::engine(format!(
                                "typecheck: stuck index {x:?} vs {y:?} in {na}"
                            ))
                        })?;
                    }
                }
                Ok(())
            }
            (Some("Path"), Some("Path")) => {
                let fa = a.get("frm").cloned().unwrap_or(Value::Null);
                let fb = b.get("frm").cloned().unwrap_or(Value::Null);
                let ta = a.get("to").cloned().unwrap_or(Value::Null);
                let tb = b.get("to").cloned().unwrap_or(Value::Null);
                let (ok, _) = unify_index(&fa, &fb, ctx);
                if !ok {
                    return Err(LangError::engine("typecheck: Path sources differ".to_string()));
                }
                let (ok, _) = unify_index(&ta, &tb, ctx);
                if !ok {
                    return Err(LangError::engine("typecheck: Path targets differ".to_string()));
                }
                Ok(())
            }
            (Some("type.function"), Some("type.function")) => {
                // Recurse dependently: `family_app` codomains unify by
                // index (`unify_index`), which plain `unify` cannot see
                // inside `args` arrays. `effects` fields, when both are
                // present, unify through the structural path.
                let pa = a.get("param").cloned().unwrap_or(Value::Null);
                let pb = b.get("param").cloned().unwrap_or(Value::Null);
                self.dep_unify(&pa, &pb, ctx)?;
                let ba = a.get("body").cloned().unwrap_or(Value::Null);
                let bb = b.get("body").cloned().unwrap_or(Value::Null);
                self.dep_unify(&ba, &bb, ctx)?;
                match (a.get("effects"), b.get("effects")) {
                    (Some(ea), Some(eb)) => self.unify(ea, eb, ctx),
                    _ => Ok(()),
                }
            }
            (Some("type.function"), Some("type.family_app"))
            | (Some("type.family_app"), Some("type.function")) => Err(LangError::engine(
                "typecheck: function is not a family member".to_string(),
            )),
            _ => self.unify(a, b, ctx),
        }
    }

    /// Dependent introduction — check `node` against a dependent former
    /// expected type. Returns `Some(result)` when handled, `None` to fall
    /// back to the ordinary check path.
    pub(crate) fn dep_check_intro(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Option<Result<Value, LangError>> {
        let tag = tag_of(expected)?;
        if tag == "type.family_app" {
            // Family applications check through index unification, not
            // the node's `check` scheme (which cannot see `args`).
            let ty = match self.synth_node(lang, ast, node_ref, ctx) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            };
            return Some(match self.dep_unify(&ty, expected, ctx) {
                Ok(()) => Ok(ty),
                Err(e) => Err(e),
            });
        }
        match tag {
            "type.pi" | "type.family" | "type.sigma" | "type.forall" => {
                let node = ast.get(node_ref)?;
                if node.kind == "function" {
                    Some(self.dep_intro_arrow(lang, ast, node, expected, ctx))
                } else if tag == "type.sigma" && node.kind == "array" {
                    Some(self.dep_intro_sigma(lang, ast, node, expected, ctx))
                } else {
                    None
                }
            }
            "Path" => {
                let node = ast.get(node_ref)?;
                if node.kind == "var" && node_name(node).as_deref() == Some("refl") {
                    let frm = expected.get("frm").cloned().unwrap_or(Value::Null);
                    let to = expected.get("to").cloned().unwrap_or(Value::Null);
                    let (ok, decided) = unify_index(&frm, &to, ctx);
                    if ok {
                        Some(Ok(expected.clone()))
                    } else {
                        Some(Err(LangError::engine(format!(
                            "typecheck: refl cannot prove Path({frm:?}, {to:?}){}",
                            if decided { " — decided inequality" } else { "" }
                        ))))
                    }
                } else if node.kind == "var" {
                    // A Γ/results-bound `Path` witness also inhabits Path.
                    let wit = node
                        .payload
                        .as_str()
                        .and_then(|n| {
                            ctx.results.get(n).cloned().or_else(|| ctx.lookup(n).cloned())
                        })
                        .unwrap_or(Value::Null);
                    if tag_of(&wit) == Some("Path") {
                        match self.dep_unify(&wit, expected, ctx) {
                            Ok(()) => Some(Ok(expected.clone())),
                            Err(e) => Some(Err(e)),
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// `function` checked against `type.pi`/`type.family` — bind the
    /// param at the domain, evaluate the expected body through the
    /// closure, check the body.
    fn dep_intro_arrow(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("dep_intro: function has no children"))?;
        let params_idx = role_child_ref(lang, node, "params")
            .or_else(|| child_at(ast, node, 0))
            .ok_or_else(|| LangError::engine("dep_intro: no params child"))?;
        let body_idx = role_child_ref(lang, node, "body")
            .or_else(|| child_at(ast, node, 1))
            .ok_or_else(|| LangError::engine("dep_intro: no body child"))?;
        let pnode = ast.get(params_idx).ok_or_else(|| {
            LangError::engine("dep_intro: params node out of bounds".to_string())
        })?;
        let (names, _rest) = lang
            .param_names(ast, pnode)
            .map_err(|e| LangError::engine(e.to_string()))?;
        let dom = expected
            .get("param")
            .or_else(|| expected.get("kind"))
            .or_else(|| expected.get("base"))
            .cloned()
            .unwrap_or(Value::Null);
        let mark = ctx.mark();
        let old_results = ctx.results.clone();
        let pname = names.first().cloned().unwrap_or_else(|| "x".to_string());
        ctx.extend(pname.clone(), dom.clone());
        // The closure variable resolves to the function's own binder as a
        // rigid index term — `var{n} → var{n}` when names coincide.
        let closure_var = expected
            .get("body_closure")
            .and_then(|b| b.get("var_name"))
            .and_then(Value::as_str)
            .unwrap_or(pname.as_str())
            .to_string();
        ctx.bind_result(closure_var, var_term(&pname));
        // Expected body: apply the closure, else a constant.
        let expected_body = if expected.get("body_closure").is_some() {
            self.dep_closure_apply(lang, ast, expected, &dom, ctx)?
        } else if tag_of(expected) == Some("type.family") {
            let mut m = Map::new();
            make_tag(ctx, "type.bool", std::mem::take(&mut m))
        } else {
            expected.get("body").cloned().unwrap_or(Value::Null)
        };
        let r = self.check_node(lang, ast, body_idx, &expected_body, ctx);
        ctx.restore(mark);
        ctx.results = old_results;
        r?;
        Ok(expected.clone())
    }

    /// `[a, b]` checked against `type.sigma` — componentwise: `a` against
    /// `param`, `b` against the closure applied at `a`'s *term*.
    fn dep_intro_sigma(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("dep_intro: sigma pair has no children"))?;
        let fst_idx = kids.first().and_then(Value::as_u64).ok_or_else(|| {
            LangError::engine("dep_intro: sigma pair is empty".to_string())
        })? as usize;
        let snd_idx = kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
            LangError::engine("dep_intro: sigma pair needs 2 elements".to_string())
        })? as usize;
        let param = expected.get("param").cloned().unwrap_or(Value::Null);
        self.check_node(lang, ast, AstRef::new(fst_idx), &param, ctx)?;
        let fst_node = ast.get(AstRef::new(fst_idx)).ok_or_else(|| {
            LangError::engine("dep_intro: fst node out of bounds".to_string())
        })?;
        let fst_term = self.node_to_index_term(lang, ast, fst_node, ctx);
        let snd_ty = self.dep_closure_apply_term(lang, ast, expected, &fst_term, ctx)?;
        self.check_node(lang, ast, AstRef::new(snd_idx), &snd_ty, ctx)?;
        Ok(expected.clone())
    }

    /// Apply a `body_closure` at a *type* argument (the `var_name` binds
    /// to a `var{pname}`-style term inside `results`).
    fn dep_closure_apply(
        &self,
        lang: &Language,
        ast: &Ast,
        ty: &Value,
        _arg: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let bc = ty.get("body_closure").ok_or_else(|| {
            LangError::engine("dep_closure_apply: no body_closure".to_string())
        })?;
        let var_name = bc.get("var_name").and_then(Value::as_str).unwrap_or("x");
        let body_idx = bc
            .get("body_child_idx")
            .and_then(Value::as_u64)
            .ok_or_else(|| LangError::engine("dep_closure_apply: no body idx".to_string()))?
            as usize;
        // `var_name` already binds to `var{fn_param}` in `results` (set by
        // dep_intro_arrow) — just elaborate the body.
        let _ = var_name;
        self.dep_synth_type(lang, ast, body_idx, ctx)
    }

    /// Apply a `body_closure` at an index *term* (Sigma/pi value
    /// instantiation) — `var_name ↦ term` in `results`.
    fn dep_closure_apply_term(
        &self,
        lang: &Language,
        ast: &Ast,
        ty: &Value,
        term: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let bc = ty.get("body_closure").ok_or_else(|| {
            LangError::engine("dep_closure_apply: no body_closure".to_string())
        })?;
        let var_name = bc
            .get("var_name")
            .and_then(Value::as_str)
            .unwrap_or("x")
            .to_string();
        let body_idx = bc
            .get("body_child_idx")
            .and_then(Value::as_u64)
            .ok_or_else(|| LangError::engine("dep_closure_apply: no body idx".to_string()))?
            as usize;
        let old = ctx.results.get(&var_name).cloned();
        ctx.bind_result(var_name.clone(), eval_index(term, ctx));
        let r = self.dep_synth_type(lang, ast, body_idx, ctx);
        match old {
            Some(v) => ctx.bind_result(var_name, v),
            None => {
                ctx.results.remove(&var_name);
            }
        }
        r
    }

    // -----------------------------------------------------------------------
    // Dependent `match` — `synth_match_arms`
    // -----------------------------------------------------------------------

    /// Dependent elimination (ADR-0152 §3): each ctor-pattern arm
    /// instantiates the constructor's `ctor_sig`, unifies its
    /// `family_app` result indices with the scrutinee's (rigid `var`
    /// indices rewrite through `results` + an `eq_<name>` `Path`
    /// witness), binds the pattern fields to the instantiated domains,
    /// and checks the body — against the motive applied at the refined
    /// indices when a `return … with` clause is present. `absurd` arms
    /// discharge decided-impossible index equations.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synth_match_dependent(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        arm_indices: &[usize],
        scrutinee_idx: usize,
        motive_idx: Option<usize>,
        scrutinee_type: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let mut arm_types: Vec<Value> = Vec::new();
        for &arm_idx in arm_indices {
            let arm = ast.get(AstRef::new(arm_idx)).ok_or_else(|| {
                LangError::engine(format!("match: arm {arm_idx} out of bounds"))
            })?;
            let kids = children_of(arm)
                .ok_or_else(|| LangError::engine("match: arm has no children"))?;
            if kids.len() < 2 {
                return Err(LangError::engine(
                    "match: arm needs [pattern, body]".to_string(),
                ));
            }
            let pat_idx = kids[0].as_u64().ok_or_else(|| {
                LangError::engine("match: pattern index not a u64".to_string())
            })? as usize;
            let body_idx = kids[1].as_u64().ok_or_else(|| {
                LangError::engine("match: body index not a u64".to_string())
            })? as usize;
            let pattern = ast.get(AstRef::new(pat_idx)).ok_or_else(|| {
                LangError::engine("match: pattern node out of bounds".to_string())
            })?;
            let body_node = ast.get(AstRef::new(body_idx)).ok_or_else(|| {
                LangError::engine("match: body node out of bounds".to_string())
            })?;
            let is_absurd = body_node.kind == "absurd";

            let mark = ctx.mark();
            let meta_mark = ctx.metas.len();
            let old_results = ctx.results.clone();

            // Resolve the pattern to (ctor_name, arg var names).
            let (cons_name, arg_names): (Option<String>, Vec<String>) =
                match pattern.kind.as_str() {
                    "var" => {
                        let n = node_name(pattern).unwrap_or_default();
                        if n == "_" {
                            (None, vec![])
                        } else {
                            (Some(n), vec![])
                        }
                    }
                    "call" => {
                        let pk = children_of(pattern).cloned().unwrap_or_default();
                        let cname = pk
                            .first()
                            .and_then(Value::as_u64)
                            .and_then(|i| ast.get(AstRef::new(i as usize)))
                            .and_then(|n| {
                                if n.kind == "var" {
                                    node_name(n)
                                } else {
                                    None
                                }
                            });
                        let names: Vec<String> = pk
                            .iter()
                            .skip(1)
                            .filter_map(Value::as_u64)
                            .filter_map(|i| {
                                ast.get(AstRef::new(i as usize)).and_then(|n| {
                                    (n.kind == "var").then(|| node_name(n))?
                                })
                            })
                            .collect();
                        (cname, names)
                    }
                    _ => (None, vec![]),
                };

            // Instantiate the ctor sig (if the pattern is a constructor).
            let ctor_inst: Option<Value> = cons_name
                .as_deref()
                .and_then(|n| ctx.lookup(n).cloned())
                .filter(|v| tag_of(v) == Some("type.ctor_sig"));
            let mut ctor_result = Value::Null;
            let mut domains: Vec<(String, Value)> = Vec::new();
            if let Some(sig) = &ctor_inst {
                let inst = self.instantiate_ctor_full(lang, ast, sig, ctx)?;
                let (doms, res) = self.peel_ctor(&inst);
                for (i, (_pname, dom)) in doms.into_iter().enumerate() {
                    domains.push((
                        arg_names.get(i).cloned().unwrap_or_default(),
                        dom,
                    ));
                }
                ctor_result = res;
            }

            // Refine: scrutinee family_app indices vs ctor result.
            let mut impossible = false;
            let mut idx_args: Vec<Value> = Vec::new();
            if tag_of(&ctor_result) == Some("type.family_app")
                && tag_of(scrutinee_type) == Some("type.family_app")
            {
                let s_name = scrutinee_type
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let c_name = ctor_result.get("name").and_then(Value::as_str).unwrap_or("");
                if s_name != c_name {
                    impossible = true;
                    if !is_absurd {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "error".to_string(),
                            message: format!(
                                "match arm: constructor builds {c_name}, not the scrutinee's {s_name}"
                            ),
                            span: None,
                        });
                    }
                } else {
                    let s_args = scrutinee_type
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let c_args = ctor_result
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    for (s_i, c_i) in s_args.iter().zip(c_args.iter()) {
                        if impossible {
                            break;
                        }
                        let ea = eval_index(&zonk_index(s_i, ctx), ctx);
                        let eb = eval_index(&zonk_index(c_i, ctx), ctx);
                        if tag_of(&ea) == Some("var") {
                            let pname = ea
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            if !pname.is_empty() && !var_occurs(&pname, &eb) {
                                ctx.bind_result(pname.clone(), eb.clone());
                                ctx.bind_result(
                                    format!("eq_{pname}"),
                                    path_val(ea.clone(), eb.clone()),
                                );
                            }
                            idx_args.push(eb);
                        } else {
                            let (ok, decided) = unify_index(&ea, &eb, ctx);
                            if !ok {
                                if decided
                                    || (num_val(&ea).is_some() && num_val(&eb).is_some())
                                {
                                    impossible = true;
                                    if !is_absurd {
                                        ctx.diagnostics.push(TypecheckDiagnostic {
                                            severity: "error".to_string(),
                                            message: "match arm: constructor cannot produce \
                                                      the scrutinee's index — unreachable branch"
                                                .to_string(),
                                            span: None,
                                        });
                                    }
                                }
                            }
                            idx_args.push(eb);
                        }
                    }
                }
            } else if tag_of(&ctor_result) == Some("type.family_app") {
                // Non-family scrutinee: soft refinement via dep_unify.
                let _ = self.dep_unify(scrutinee_type, &ctor_result, ctx);
                idx_args = ctor_result
                    .get("args")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
            }

            if impossible && is_absurd {
                // Discharged — the arm body is not checked.
                ctx.restore(mark);
                for id in meta_mark..ctx.metas.len() {
                    ctx.metas[id] = None;
                }
                ctx.results = old_results;
                continue;
            }
            if is_absurd && !impossible {
                ctx.restore(mark);
                for id in meta_mark..ctx.metas.len() {
                    ctx.metas[id] = None;
                }
                ctx.results = old_results;
                return Err(LangError::engine(
                    "match: absurd arm is reachable — index equations are satisfiable".to_string(),
                ));
            }
            if impossible {
                ctx.restore(mark);
                for id in meta_mark..ctx.metas.len() {
                    ctx.metas[id] = None;
                }
                ctx.results = old_results;
                return Err(LangError::engine(
                    "match: unreachable constructor arm (decided index mismatch)".to_string(),
                ));
            }

            // Bind the pattern's field vars to instantiated domains.
            for (aname, dom) in &domains {
                if !aname.is_empty() && aname != "_" {
                    ctx.extend(aname.clone(), zonk_index(dom, ctx));
                }
            }
            // Catchall var pattern: bind to the scrutinee type.
            if let Some(n) = &cons_name {
                if ctor_inst.is_none() && n != "_" {
                    ctx.extend(n.clone(), scrutinee_type.clone());
                }
            } else if pattern.kind == "var" && node_name(pattern).as_deref() == Some("_") {
                // wildcard binds nothing.
            }

            // Body: against the motive applied at the refined indices
            // when present, else synth for the join.
            if let Some(mi) = motive_idx {
                let expected = self.apply_motive(lang, ast, mi, &idx_args, scrutinee_idx, ctx)?;
                self.check_node(lang, ast, AstRef::new(body_idx), &expected, ctx)?;
                arm_types.push(expected);
            } else {
                let bt = self.synth_node(lang, ast, AstRef::new(body_idx), ctx)?;
                arm_types.push(bt);
            }

            ctx.restore(mark);
            for id in meta_mark..ctx.metas.len() {
                ctx.metas[id] = None;
            }
            ctx.results = old_results;
        }

        // Result: motive applied at the scrutinee's own indices, else the
        // arm-type join.
        if let Some(mi) = motive_idx {
            let scrut_args: Vec<Value> = match tag_of(scrutinee_type) {
                Some("type.family_app") => scrutinee_type
                    .get("args")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|a| eval_index(a, ctx))
                    .collect(),
                _ => Vec::new(),
            };
            return self.apply_motive(lang, ast, mi, &scrut_args, scrutinee_idx, ctx);
        }
        let Some(first) = arm_types.first().cloned() else {
            let mut m = Map::new();
            return Ok(make_tag(ctx, "type.unknown", std::mem::take(&mut m)));
        };
        let mut unified = first;
        for t in &arm_types[1..] {
            self.dep_unify(&unified.clone(), t, ctx)?;
            unified = zonk_index(&unified, ctx);
        }
        Ok(unified)
    }

    /// Motive application `return (i⃗, w) ~> T with`: bind the motive's
    /// params to the refined index terms (then the scrutinee's `var`
    /// term) in `results`, elaborate `T` in type position.
    fn apply_motive(
        &self,
        lang: &Language,
        ast: &Ast,
        motive_idx: usize,
        arg_vals: &[Value],
        scrutinee_idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let mnode = ast.get(AstRef::new(motive_idx)).ok_or_else(|| {
            LangError::engine("apply_motive: motive node out of bounds".to_string())
        })?;
        let inner = if mnode.kind == "paren" {
            child_at(ast, mnode, 0)
                .and_then(|i| ast.get(i))
                .unwrap_or(mnode)
        } else {
            mnode
        };
        if inner.kind != "function" {
            // A non-function motive is a constant expected type.
            return self.dep_synth_type(lang, ast, motive_idx, ctx);
        }
        let params_idx = role_child_ref(lang, inner, "params")
            .or_else(|| child_at(ast, inner, 0))
            .ok_or_else(|| LangError::engine("apply_motive: no params".to_string()))?;
        let body_idx = role_child_ref(lang, inner, "body")
            .or_else(|| child_at(ast, inner, 1))
            .ok_or_else(|| LangError::engine("apply_motive: no body".to_string()))?;
        let pnode = ast.get(params_idx).ok_or_else(|| {
            LangError::engine("apply_motive: params node out of bounds".to_string())
        })?;
        let (names, _) = lang
            .param_names(ast, pnode)
            .map_err(|e| LangError::engine(e.to_string()))?;
        let scrut_name = ast
            .get(AstRef::new(scrutinee_idx))
            .and_then(|n| {
                if n.kind == "var" {
                    node_name(n)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "_scrut".to_string());
        let old_results = ctx.results.clone();
        for (i, name) in names.iter().enumerate() {
            let v = if i < arg_vals.len() {
                arg_vals[i].clone()
            } else if i == arg_vals.len() {
                var_term(&scrut_name)
            } else {
                var_term(name)
            };
            ctx.bind_result(name.clone(), v);
        }
        let r = self.dep_synth_type(lang, ast, body_idx.index(), ctx);
        ctx.results = old_results;
        r
    }

    // -----------------------------------------------------------------------
    // `synth_transport` / `synth_j` / `synth_dim_arrow` / `synth_flow`
    // -----------------------------------------------------------------------

    /// Resolve a path operand node to a `Path{frm,to}` value: a
    /// `results`-bound `eq_<name>` witness, a Γ-bound `p : Path`, or
    /// `refl` (identity). `None` when the operand is not a path.
    fn resolve_path(&self, node: &AstNode, ctx: &TypecheckContext) -> Option<Value> {
        if node.kind == "var" {
            let name = node_name(node)?;
            if name == "refl" {
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("Refl".to_string()));
                // Identity path — `Path{refl, refl}` reads as transport
                // with frm == to; record the refl marker.
                return Some(path_val(
                    Value::Object(m.clone()),
                    Value::Object(m),
                ));
            }
            if let Some(w) = ctx.results.get(&name) {
                if tag_of(w) == Some("Path") {
                    return Some(w.clone());
                }
            }
            if let Some(w) = ctx.lookup(&name) {
                if tag_of(w) == Some("Path") {
                    return Some(w.clone());
                }
            }
        }
        None
    }

    /// `p # v` — transport `v`'s type through the path's index rewrite
    /// (`frm ↦ to` inside `family_app` args), recording the consumed
    /// witness under `via` (proof-relevant, not erased).
    pub(crate) fn synth_transport_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("transport: no children".to_string()))?;
        if kids.len() < 2 {
            return Err(LangError::engine("transport: needs [path, value]".to_string()));
        }
        let path_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("transport: path index not a u64".to_string())
        })? as usize;
        let val_idx = kids[1].as_u64().ok_or_else(|| {
            LangError::engine("transport: value index not a u64".to_string())
        })? as usize;
        let path_node = ast.get(AstRef::new(path_idx)).ok_or_else(|| {
            LangError::engine("transport: path node out of bounds".to_string())
        })?;
        let path = self.resolve_path(path_node, ctx).ok_or_else(|| {
            LangError::engine(format!(
                "transport: operand is not a Path (kind '{}')",
                path_node.kind
            ))
        })?;
        let vty = self.synth_node(lang, ast, AstRef::new(val_idx), ctx)?;
        let frm = eval_index(&path.get("frm").cloned().unwrap_or(Value::Null), ctx);
        let to = eval_index(&path.get("to").cloned().unwrap_or(Value::Null), ctx);
        // Transport: rewrite `frm` → `to` through the value's type.
        let rewritten = if frm == to {
            vty.clone()
        } else if tag_of(&vty) == Some("type.family_app") {
            let args = vty
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|a| eval_index(&subst_index(a, &frm, &to), ctx))
                .collect();
            let mut m = Map::new();
            m.insert("name".to_string(), vty.get("name").cloned().unwrap_or(Value::Null));
            m.insert("args".to_string(), Value::Array(args));
            make_tag(ctx, "type.family_app", m)
        } else {
            subst_index(&vty, &frm, &to)
        };
        // Record the consumed witness (proof-relevant provenance).
        if let Value::Object(mut m) = rewritten {
            m.insert("via".to_string(), path);
            return Ok(Value::Object(m));
        }
        Ok(rewritten)
    }

    /// `j p, motive, base` — the J eliminator: check `base` against the
    /// motive applied at `frm`, synthesize the motive applied at `to`.
    pub(crate) fn synth_j_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("j: no children".to_string()))?;
        if kids.len() < 3 {
            return Err(LangError::engine("j: needs [path, motive, base]".to_string()));
        }
        let path_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("j: path index not a u64".to_string())
        })? as usize;
        let motive_idx = kids[1].as_u64().ok_or_else(|| {
            LangError::engine("j: motive index not a u64".to_string())
        })? as usize;
        let base_idx = kids[2].as_u64().ok_or_else(|| {
            LangError::engine("j: base index not a u64".to_string())
        })? as usize;
        let path_node = ast.get(AstRef::new(path_idx)).ok_or_else(|| {
            LangError::engine("j: path node out of bounds".to_string())
        })?;
        let path = self.resolve_path(path_node, ctx).ok_or_else(|| {
            LangError::engine("j: operand is not a Path".to_string())
        })?;
        let frm = eval_index(&path.get("frm").cloned().unwrap_or(Value::Null), ctx);
        let to = eval_index(&path.get("to").cloned().unwrap_or(Value::Null), ctx);
        let expected_base = self.apply_motive(lang, ast, motive_idx, &[frm], path_idx, ctx)?;
        self.check_node(lang, ast, AstRef::new(base_idx), &expected_base, ctx)?;
        self.apply_motive(lang, ast, motive_idx, &[to], path_idx, ctx)
    }

    /// `A ~[N]~> B` — the `Arrow` record `{inputs, grade, output, dim}`.
    pub(crate) fn synth_dim_arrow_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("dim_arrow: no children".to_string()))?;
        if kids.len() < 3 {
            return Err(LangError::engine("dim_arrow: missing operands".to_string()));
        }
        let src_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("dim_arrow: src index not a u64".to_string())
        })? as usize;
        let dim_idx = kids[1].as_u64().ok_or_else(|| {
            LangError::engine("dim_arrow: dim index not a u64".to_string())
        })? as usize;
        let tgt_idx = kids[2].as_u64().ok_or_else(|| {
            LangError::engine("dim_arrow: tgt index not a u64".to_string())
        })? as usize;
        let src = self.synth_node(lang, ast, AstRef::new(src_idx), ctx)?;
        // Dimension literal (from the child payload), then its type.
        let dim_node = ast.get(AstRef::new(dim_idx)).ok_or_else(|| {
            LangError::engine("dim_arrow: dim node out of bounds".to_string())
        })?;
        let dim_inner = if dim_node.kind == "paren" {
            child_at(ast, dim_node, 0)
                .and_then(|i| ast.get(i))
                .unwrap_or(dim_node)
        } else {
            dim_node
        };
        let dim_payload = &dim_inner.payload;
        let dim_val = if dim_payload.is_object() {
            dim_payload.get("value").cloned().unwrap_or(Value::Null)
        } else {
            dim_payload.clone()
        };
        let dim_lit = if dim_inner.kind == "value" && dim_val.is_number() {
            dim_val
        } else {
            Value::Null
        };
        let grade = self.synth_node(lang, ast, AstRef::new(dim_idx), ctx)?;
        let tgt = self.synth_node(lang, ast, AstRef::new(tgt_idx), ctx)?;
        let mut m = Map::new();
        m.insert("inputs".to_string(), src);
        m.insert("grade".to_string(), grade);
        m.insert("output".to_string(), tgt);
        m.insert("dim".to_string(), dim_lit);
        Ok(make_tag(ctx, "Arrow", m))
    }

    /// `f >>> g` — composition in `type.function` coordinates
    /// (`unify(f.body, g.param)`), family reindexing `f >>> P` →
    /// `type.family{base: f.param}`, and the `Arrow`-record fallback for
    /// higher cells.
    pub(crate) fn synth_flow_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("flow: no children".to_string()))?;
        if kids.len() < 2 {
            return Err(LangError::engine("flow: needs two operands".to_string()));
        }
        let f_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("flow: lhs index not a u64".to_string())
        })? as usize;
        let g_idx = kids[1].as_u64().ok_or_else(|| {
            LangError::engine("flow: rhs index not a u64".to_string())
        })? as usize;
        let f_ty = self.synth_node(lang, ast, AstRef::new(f_idx), ctx)?;
        let g_ty = self.synth_node(lang, ast, AstRef::new(g_idx), ctx)?;
        self.flow_of(&f_ty, &g_ty, ctx)
    }

    /// The composition rule shared by `flow` and axis-0 `axis_flow`.
    fn flow_of(
        &self,
        f_ty: &Value,
        g_ty: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        // `f >>> P` — reindexing: a family over `f`'s source (pullback).
        if tag_of(g_ty) == Some("type.family") {
            let f_dom = match tag_of(f_ty) {
                Some("type.function") => f_ty.get("param").cloned().unwrap_or(Value::Null),
                _ => meta_value(ctx.fresh_meta()),
            };
            let f_cod = match tag_of(f_ty) {
                Some("type.function") => f_ty.get("body").cloned().unwrap_or(Value::Null),
                _ => meta_value(ctx.fresh_meta()),
            };
            let g_base = g_ty.get("base").cloned().unwrap_or(Value::Null);
            self.unify(&f_cod, &g_base, ctx)?;
            let mut m = Map::new();
            m.insert("name".to_string(), g_ty.get("name").cloned().unwrap_or(Value::Null));
            m.insert("base".to_string(), f_dom);
            return Ok(make_tag(ctx, "type.family", m));
        }
        match (tag_of(f_ty), tag_of(g_ty)) {
            (Some("type.function"), Some("type.function")) => {
                let f_body = f_ty.get("body").cloned().unwrap_or(Value::Null);
                let g_param = g_ty.get("param").cloned().unwrap_or(Value::Null);
                self.dep_unify(&f_body, &g_param, ctx)?;
                let mut m = Map::new();
                m.insert("param".to_string(), f_ty.get("param").cloned().unwrap_or(Value::Null));
                m.insert("body".to_string(), g_ty.get("body").cloned().unwrap_or(Value::Null));
                Ok(make_tag(ctx, "type.function", m))
            }
            (Some("type.function"), Some("type.pi"))
            | (Some("type.pi"), Some("type.function"))
            | (Some("type.function"), Some("type.forall"))
            | (Some("type.forall"), Some("type.function")) => {
                // Boundary via the dep former's domain + closure-applied
                // body; the composite flattens to `type.function`
                // (structural approximation — no AST body exists).
                let (_dom_a, cod_a) = self.dep_boundary(f_ty, ctx);
                let (dom_b, cod_b) = self.dep_boundary(g_ty, ctx);
                self.dep_unify(&cod_a, &dom_b, ctx)?;
                let mut m = Map::new();
                m.insert(
                    "param".to_string(),
                    f_ty.get("param").cloned().unwrap_or(Value::Null),
                );
                m.insert("body".to_string(), cod_b);
                Ok(make_tag(ctx, "type.function", m))
            }
            _ => {
                // Structural fallback: an opaque `Comp` record.
                let mut m = Map::new();
                m.insert("f".to_string(), f_ty.clone());
                m.insert("g".to_string(), g_ty.clone());
                Ok(make_tag(ctx, "Comp", m))
            }
        }
    }

    /// `(domain, codomain)` boundary of a `type.function`/`type.pi`/
    /// `type.forall`/`type.family` value.
    fn dep_boundary(&self, ty: &Value, ctx: &mut TypecheckContext) -> (Value, Value) {
        match tag_of(ty) {
            Some("type.function") => (
                ty.get("param").cloned().unwrap_or(Value::Null),
                ty.get("body").cloned().unwrap_or(Value::Null),
            ),
            Some("type.family") => (
                ty.get("base").cloned().unwrap_or(Value::Null),
                make_tag(ctx, "type.universe", {
                    let mut m = Map::new();
                    m.insert("level".to_string(), Value::from(0));
                    m
                }),
            ),
            Some("type.pi") | Some("type.forall") => (
                ty.get("param")
                    .or_else(|| ty.get("kind"))
                    .cloned()
                    .unwrap_or(Value::Null),
                ty.get("body_closure").cloned().unwrap_or(Value::Null),
            ),
            _ => (meta_value(ctx.fresh_meta()), meta_value(ctx.fresh_meta())),
        }
    }

    /// `a >>>[k] b` — axis-0 delegates to `flow_of`; `k ≥ 1` produces the
    /// `Arrow`-record composite (the ADR-0120 honest bound).
    pub(crate) fn synth_axis_flow_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("axis_flow: no children".to_string()))?;
        if kids.len() < 3 {
            return Err(LangError::engine("axis_flow: needs [a, k, b]".to_string()));
        }
        let a_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("axis_flow: lhs index not a u64".to_string())
        })? as usize;
        let k_idx = kids[1].as_u64().ok_or_else(|| {
            LangError::engine("axis_flow: axis index not a u64".to_string())
        })? as usize;
        let b_idx = kids[2].as_u64().ok_or_else(|| {
            LangError::engine("axis_flow: rhs index not a u64".to_string())
        })? as usize;
        let k_node = ast.get(AstRef::new(k_idx)).ok_or_else(|| {
            LangError::engine("axis_flow: axis node out of bounds".to_string())
        })?;
        let k_lit = k_node.payload.as_f64();
        let a_ty = self.synth_node(lang, ast, AstRef::new(a_idx), ctx)?;
        let b_ty = self.synth_node(lang, ast, AstRef::new(b_idx), ctx)?;
        if k_lit == Some(0.0) || k_lit.is_none() {
            return self.flow_of(&a_ty, &b_ty, ctx);
        }
        let mut m = Map::new();
        m.insert("inputs".to_string(), a_ty);
        m.insert("output".to_string(), b_ty);
        m.insert("dim".to_string(), Value::from(k_lit));
        Ok(make_tag(ctx, "Arrow", m))
    }

    /// `call` typing — curried application: the callee synthesizes a
    /// `type.function`/`type.pi`/`ctor_sig` chain; each argument child is
    /// checked against the current domain, producing the codomain for the
    /// next step (the model's ctor-application rule + ordinary calls).
    pub(crate) fn synth_call_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("call: no children".to_string()))?;
        if kids.len() < 2 {
            return Err(LangError::engine("call: needs callee and args".to_string()));
        }
        let callee_idx = kids[0].as_u64().ok_or_else(|| {
            LangError::engine("call: callee index not a u64".to_string())
        })? as usize;
        let mut cur = self.synth_node(lang, ast, AstRef::new(callee_idx), ctx)?;
        for cv in kids.iter().skip(1) {
            let arg_idx = cv.as_u64().ok_or_else(|| {
                LangError::engine("call: arg index not a u64".to_string())
            })? as usize;
            let arg_ref = AstRef::new(arg_idx);
            cur = zonk_index(&cur, ctx);
            cur = match tag_of(&cur) {
                Some("type.ctor_sig") => self.instantiate_ctor_full(lang, ast, &cur, ctx)?,
                Some("type.function") => {
                    let param = cur.get("param").cloned().unwrap_or(Value::Null);
                    self.check_node(lang, ast, arg_ref, &param, ctx)?;
                    cur.get("body").cloned().unwrap_or(Value::Null)
                }
                Some("type.pi") | Some("type.forall") => {
                    // Dependent application — check against the domain,
                    // then the closure applied at the arg's *term*.
                    let param = cur
                        .get("param")
                        .or_else(|| cur.get("kind"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    self.check_node(lang, ast, arg_ref, &param, ctx)?;
                    let arg_node = ast.get(arg_ref).ok_or_else(|| {
                        LangError::engine("call: arg node out of bounds".to_string())
                    })?;
                    let term = self.node_to_index_term(lang, ast, arg_node, ctx);
                    self.dep_closure_apply_term(lang, ast, &cur, &term, ctx)?
                }
                Some("type.family") | Some("type.family_app") => {
                    // Family application in term position → `family_app`,
                    // accumulating index terms across curried arguments
                    // (`Vec(i, Bool)` applies a `type.family`, then extends
                    // the resulting `family_app` per remaining arg).
                    let mut args: Vec<Value> = cur
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let t = self.node_index_or_type(lang, ast, arg_idx, ctx)?;
                    args.push(eval_index(&t, ctx));
                    let mut m = Map::new();
                    m.insert("name".to_string(), cur.get("name").cloned().unwrap_or(Value::Null));
                    m.insert("args".to_string(), Value::Array(args));
                    make_tag(ctx, "type.family_app", m)
                }
                Some("type.unknown") => {
                    let _ = self.synth_node(lang, ast, arg_ref, ctx)?;
                    let mut m = Map::new();
                    make_tag(ctx, "type.unknown", std::mem::take(&mut m))
                }
                _ => {
                    if extract_meta_id(&cur).is_some() {
                        // Unknown callee — mint a fresh function type.
                        let p = meta_value(ctx.fresh_meta());
                        let b = meta_value(ctx.fresh_meta());
                        let mut m = Map::new();
                        m.insert("param".to_string(), p.clone());
                        m.insert("body".to_string(), b.clone());
                        let ft = make_tag(ctx, "type.function", m);
                        self.unify(&cur, &ft, ctx)?;
                        self.check_node(lang, ast, arg_ref, &p, ctx)?;
                        b
                    } else {
                        return Err(LangError::engine(format!(
                            "call: callee is not applicable (type {cur})",
                        )));
                    }
                }
            };
        }
        Ok(zonk_index(&cur, ctx))
    }
}

impl NbeTypecheckEngine {
    /// `synth_assign` — typecheck a standalone `assign` node the way
    /// `synth_block` binds a statement: extract `(name, value, annot)` —
    /// `payload.name` + child 0, `children [var|type-annot, value]`, the
    /// 3-child `assign(target, annot, value)` desugared form, or an
    /// arrow annotation parsed as `function{type-annot(f, A), B}`
    /// (`f : A ~> B = v`) — check the value against the annotation
    /// (dependent introductions live in `dep_check_intro`), and bind
    /// the name in Γ.
    /// Returns `type.unit`. The legacy `type.assign` rule was a bare
    /// `construct type.unit`, so a last-position annotated assign was
    /// never checked — this op is the fix (ADR-0120 dependent structure
    /// depends on top-level `x : T = v` statements checking `v`).
    pub(crate) fn synth_assign_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let children = node
            .payload
            .get("children")
            .and_then(|c| c.as_array())
            .ok_or_else(|| LangError::engine("synth_assign: no children".to_string()))?;
        let (name, value_ref, ann_ref) = if let Some(n) =
            node.payload.get("name").and_then(|v| v.as_str())
        {
            if children.is_empty() {
                return Err(LangError::engine(
                    "synth_assign: assign with payload.name needs a value child".to_string(),
                ));
            }
            let vidx = children[0]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: value idx".to_string()))?
                as usize;
            (n.to_string(), AstRef::new(vidx), None)
        } else if children.len() >= 3 {
            // `assign(target, annot, value)` — the 3-child desugared
            // form: a `var`/`type-annot` target, the type expression in
            // the middle slot, the value last.
            let tidx = children[0]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: lhs idx".to_string()))?
                as usize;
            let aidx = children[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: annot idx".to_string()))?
                as usize;
            let vidx = children[2]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: value idx".to_string()))?
                as usize;
            let t_node = ast.get(AstRef::new(tidx)).ok_or_else(|| {
                LangError::engine("synth_assign: lhs out of bounds".to_string())
            })?;
            let n = if t_node.kind == "type-annot"
                || role_child_ref(lang, t_node, "annot").is_some()
            {
                let subj = lang
                    .child_role_index(&t_node.kind, "subject")
                    .ok()
                    .flatten()
                    .unwrap_or(0);
                children_of(t_node)
                    .and_then(|k| k.get(subj))
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(node_name)
                    .ok_or_else(|| LangError::engine("synth_assign: annot var".to_string()))?
            } else {
                node_name(t_node)
                    .ok_or_else(|| LangError::engine("synth_assign: var name".to_string()))?
            };
            (n, AstRef::new(vidx), Some(AstRef::new(aidx)))
        } else {
            if children.len() < 2 {
                return Err(LangError::engine(
                    "synth_assign: assign needs at least 2 children".to_string(),
                ));
            }
            let lhs_idx = children[0]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: lhs idx".to_string()))?
                as usize;
            let vidx = children[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("synth_assign: value idx".to_string()))?
                as usize;
            let lhs_node = ast.get(AstRef::new(lhs_idx)).ok_or_else(|| {
                LangError::engine("synth_assign: lhs out of bounds".to_string())
            })?;
            if role_child_ref(lang, lhs_node, "annot").is_some() || lhs_node.kind == "type-annot" {
                let ta = children_of(lhs_node)
                    .ok_or_else(|| LangError::engine("synth_assign: type-annot children".to_string()))?;
                if ta.len() < 2 {
                    return Err(LangError::engine(
                        "synth_assign: type-annot needs 2 children".to_string(),
                    ));
                }
                let iv = ta[0]
                    .as_u64()
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(node_name)
                    .ok_or_else(|| LangError::engine("synth_assign: annot var".to_string()))?;
                let tidx = ta[1]
                    .as_u64()
                    .ok_or_else(|| LangError::engine("synth_assign: annot idx".to_string()))?
                    as usize;
                (iv.to_string(), AstRef::new(vidx), Some(AstRef::new(tidx)))
            } else if lhs_node.kind == "function" {
                // `f : A ~> B` — the arrow annotation parses as a
                // `function` node whose params child is the
                // `type-annot`; the whole node is the type expression
                // `A ~> B`, and the name lives in the annot's subject.
                let params_idx = role_child_ref(lang, lhs_node, "params")
                    .or_else(|| child_at(ast, lhs_node, 0))
                    .ok_or_else(|| {
                        LangError::engine("synth_assign: function lhs has no params".to_string())
                    })?;
                let pnode = ast.get(params_idx).ok_or_else(|| {
                    LangError::engine("synth_assign: params out of bounds".to_string())
                })?;
                let pinner = if pnode.kind == "paren" {
                    child_at(ast, pnode, 0)
                        .and_then(|i| ast.get(i))
                        .unwrap_or(pnode)
                } else {
                    pnode
                };
                if pinner.kind == "type-annot"
                    || role_child_ref(lang, pinner, "annot").is_some()
                {
                    let subj = lang
                        .child_role_index(&pinner.kind, "subject")
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                    let iv = children_of(pinner)
                        .and_then(|k| k.get(subj))
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(node_name)
                        .ok_or_else(|| {
                            LangError::engine("synth_assign: annot var".to_string())
                        })?;
                    (iv, AstRef::new(vidx), Some(AstRef::new(lhs_idx)))
                } else {
                    return Err(LangError::engine(
                        "synth_assign: function lhs params is not a type-annot".to_string(),
                    ));
                }
            } else {
                let n = node_name(lhs_node)
                    .ok_or_else(|| LangError::engine("synth_assign: var name".to_string()))?;
                (n.to_string(), AstRef::new(vidx), None)
            }
        };
        let bound_type = if let Some(ann_ref) = ann_ref {
            let type_expr_node = ast.get(ann_ref).ok_or_else(|| {
                LangError::engine("synth_assign: annot out of bounds".to_string())
            })?;
            // The annotation elaborates *before* the value is judged —
            // the model's `assign` op checks `v` against `ann` directly
            // (no pre-synth): dependent introductions (`dep_intro_arrow`)
            // bind the rigid index vars and `eq_<name>` branch witnesses
            // in `results` on the way in, so a `match` in the value sees
            // the refined scrutinee type. Pre-synthesizing would check
            // the arms under an unsolved-meta scrutinee, where `results`
            // stays empty and `transport`/`j`/`absurd` cannot resolve.
            let ann = match self
                .dep_synth_type(lang, ast, ann_ref.index(), ctx)
                .or_else(|_| crate::interpret_type_expr(lang, ast, type_expr_node, &ctx.registry))
            {
                // A term-valued annotation (e.g. `call` on a results-bound
                // name — `inc(4)` lowers to `App{var inc, [4]}`) reduces
                // against the bound-value env before classification.
                Ok(a) if is_ann_term(&a) => {
                    self.eval_ann_term(lang, ast, &a, ctx, ANN_FUEL)?
                }
                Ok(a) => a,
                Err(interp_err) => {
                    // Ordered value annotation (ADR-0103): `x : E := v`
                    // evaluates `E` as a term — a value result means
                    // equality (`f : inc(4) := 5` checks `5 == inc(4)`).
                    match self.eval_ann_node(lang, ast, ann_ref.index(), ctx) {
                        Ok(v) => v,
                        Err(_) => return Err(interp_err),
                    }
                }
            };
            if matches!(
                ann.get("__tag__").and_then(Value::as_str),
                Some("type.pi" | "type.family" | "type.sigma" | "Path" | "type.forall" | "type.family_app")
            ) {
                self.check_node(lang, ast, value_ref, &ann, ctx)?;
            } else if is_plain_ann_value(&ann) {
                // The annotation evaluated to a *value*: the assigned
                // expression must evaluate to the same value.
                let v = self
                    .eval_ann_node(lang, ast, value_ref.index(), ctx)
                    .map_err(|e| {
                        LangError::engine(format!(
                            "assign: cannot evaluate the value side of '{name}'s annotation: {e}"
                        ))
                    })?;
                if !ann_value_eq(&v, &ann) {
                    return Err(LangError::engine(format!(
                        "assign: value {v} does not equal the annotation {ann} for '{name}'"
                    )));
                }
            } else if is_applicable_ann(&ann) {
                // The annotation evaluated to a closure — a predicate /
                // family-membership check: apply it to the value.
                let v = self.eval_ann_node(lang, ast, value_ref.index(), ctx);
                let applied = match v {
                    Ok(vv) => {
                        self.apply_ann_val(lang, ast, &ann, &[vv], ctx, ANN_FUEL)?
                    }
                    Err(_) => Value::Null,
                };
                if applied == Value::Bool(false) {
                    return Err(LangError::engine(format!(
                        "assign: value does not satisfy the predicate annotation for '{name}'"
                    )));
                }
                if !(applied == Value::Bool(true)) {
                    // Undecidable — symbolic predicate; accept with a
                    // runtime-assert diagnostic (the model's warning).
                    ctx.diagnostics.push(TypecheckDiagnostic {
                        severity: "runtime_assert".to_string(),
                        message: format!(
                            "assign: predicate annotation for '{name}' is symbolic — deferred to runtime"
                        ),
                        span: None,
                    });
                }
            } else {
                let value_type = self.synth_node(lang, ast, value_ref, ctx)?;
                self.dep_unify(&value_type, &ann, ctx)?;
            }
            ann
        } else {
            self.synth_node(lang, ast, value_ref, ctx)?
        };
        ctx.extend(name.clone(), bound_type);
        // ADR-0103: record the bound *term* of the value — the bound-value
        // env ordered annotations evaluate against (the model's
        // `tc_bind_result`). `var{name}` inside a call annotation resolves
        // to this term (e.g. `inc ↦ Lam{params, body}`).
        if let Some(vn) = ast.get(value_ref) {
            if let Some(t) = self.node_to_ann_term(lang, ast, vn) {
                ctx.bind_result(name.clone(), t);
            }
        }
        let mut m = Map::new();
        Ok(make_tag(ctx, "type.unit", std::mem::take(&mut m)))
    }
}

// ---------------------------------------------------------------------------
// Ordered value annotations (ADR-0103) — bound-value evaluation
// ---------------------------------------------------------------------------
//
// Port of the model's `node_term_`/`eval_term_`/`apply_val_`/`eval_node_`
// fragment: `x : E := v` treats `E` as an expression evaluated against the
// bound-value environment (`ctx.results`, populated by every `assign` —
// the model's `tc_bind_result`). A *type* result means membership
// (`5 : Number`); a *value* result means equality (`f : inc(4) := 5`
// checks `5 == inc(4)`); a closure result means predicate application.
//
// Terms are the same tagged `Value`s the index algebra uses:
// `var{name}`, `Bin{op,l,r}`, `Un{op,operand}`, plus `App{callee,args}`,
// `Lam{params,body}`, `LamVal{params,body,env}`, `Comp{l,r}`,
// `CompVal{l,r}`, `If{cond,then,else}`, `ArrLit{elements}` — everything
// else is a literal.

/// Evaluation fuel for annotation terms — the model's `tc_burn_fuel`
/// bound (guaranteed termination like every other checker recursion).
const ANN_FUEL: u32 = 4096;

/// Whether `v` is a *term* produced by the annotation lowering (not a
/// `type.*`/`Path`/family type value and not a plain value).
fn is_ann_term(v: &Value) -> bool {
    matches!(
        tag_of(v),
        Some("App" | "var" | "Lam" | "LamVal" | "Comp" | "CompVal" | "If" | "ArrLit")
    )
}

/// Whether `v` is a plain (untagged, non-meta) value — a *value* result
/// in the ordered-annotation sense.
fn is_plain_ann_value(v: &Value) -> bool {
    match v {
        Value::Object(m) => {
            !m.contains_key("__tag__")
                && !m.contains_key("__meta__")
                && !m.contains_key("__former__")
                && !m.contains_key("__sym__")
                && !m.contains_key("__neutral__")
                && !m.contains_key("__arrow__")
                && !m.contains_key("__type__")
        }
        Value::Null => false,
        _ => true,
    }
}

/// Whether `v` is an applicable annotation value — `LamVal`/`CompVal`
/// closure terms or a boundary `__arrow__`.
fn is_applicable_ann(v: &Value) -> bool {
    matches!(tag_of(v), Some("LamVal" | "CompVal"))
        || v.get("__arrow__").and_then(Value::as_bool) == Some(true)
}

/// Numeric-tolerant value equality — index evaluation folds arithmetic
/// to `f64` (`5.0`) while literal payloads stay integer (`5`).
fn ann_value_eq(a: &Value, b: &Value) -> bool {
    if a == b {
        return true;
    }
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

fn app_term(callee: Value, args: Vec<Value>) -> Value {
    let mut m = Map::new();
    m.insert("__tag__".to_string(), Value::String("App".to_string()));
    m.insert("callee".to_string(), callee);
    m.insert("args".to_string(), Value::Array(args));
    Value::Object(m)
}

impl NbeTypecheckEngine {
    /// Lower an AST node to a closed annotation *term* (the model's
    /// `node_term_`). Returns `None` for kinds outside the evaluable
    /// fragment — the annotation then keeps the type-interpretation
    /// error path.
    fn node_to_ann_term(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
    ) -> Option<Value> {
        let kids = children_of(node).cloned().unwrap_or_default();
        let at = |i: usize| -> Option<&AstNode> {
            kids.get(i)
                .and_then(Value::as_u64)
                .and_then(|j| ast.get(AstRef::new(j as usize)))
        };
        match node.kind.as_str() {
            "value" | "lit" => Some(match &node.payload {
                Value::Object(m) => m.get("value").cloned().unwrap_or(Value::Null),
                p => p.clone(),
            }),
            "var" => Some(var_term(&node_name(node)?)),
            "paren" => self.node_to_ann_term(lang, ast, at(0)?),
            "call" => {
                let callee_role = lang
                    .child_role_index(&node.kind, "callee")
                    .ok()
                    .flatten()
                    .unwrap_or(0);
                let args_role = lang
                    .child_role_index(&node.kind, "args")
                    .ok()
                    .flatten()
                    .unwrap_or(1);
                let callee = self.node_to_ann_term(lang, ast, at(callee_role)?)?;
                let mut args = Vec::new();
                for i in args_role..kids.len() {
                    args.push(self.node_to_ann_term(lang, ast, at(i)?)?);
                }
                Some(app_term(callee, args))
            }
            "function" => {
                // Rich `{params: [..], body: idx}` payload or the
                // children [params, body] layout (roles per ADR-0081).
                if let (Some(params), Some(body)) = (
                    node.payload.get("params").and_then(Value::as_array),
                    node.payload.get("body").and_then(Value::as_u64),
                ) {
                    let names: Vec<Value> = params.clone();
                    let mut m = Map::new();
                    m.insert("__tag__".to_string(), Value::String("Lam".to_string()));
                    m.insert("params".to_string(), Value::Array(names));
                    m.insert("body".to_string(), Value::from(body));
                    return Some(Value::Object(m));
                }
                let params_idx = role_child_ref(lang, node, "params").or_else(|| child_at(ast, node, 0))?;
                let body_idx = role_child_ref(lang, node, "body").or_else(|| child_at(ast, node, 1))?;
                let pnode = ast.get(params_idx)?;
                let (names, _rest) = lang.param_names(ast, pnode).ok()?;
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("Lam".to_string()));
                m.insert(
                    "params".to_string(),
                    Value::Array(names.into_iter().map(Value::String).collect()),
                );
                m.insert("body".to_string(), Value::from(body_idx.index() as u64));
                Some(Value::Object(m))
            }
            "if" => {
                let cond = role_child_ref(lang, node, "condition").or_else(|| child_at(ast, node, 0))?;
                let then = role_child_ref(lang, node, "then").or_else(|| child_at(ast, node, 1))?;
                let els = role_child_ref(lang, node, "else").or_else(|| child_at(ast, node, 2));
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("If".to_string()));
                m.insert(
                    "cond".to_string(),
                    self.node_to_ann_term(lang, ast, ast.get(cond)?)?,
                );
                m.insert(
                    "then".to_string(),
                    self.node_to_ann_term(lang, ast, ast.get(then)?)?,
                );
                if let Some(e) = els.and_then(|r| ast.get(r)) {
                    m.insert(
                        "else".to_string(),
                        self.node_to_ann_term(lang, ast, e)?,
                    );
                }
                Some(Value::Object(m))
            }
            "array" => {
                let mut els = Vec::new();
                for i in 0..kids.len() {
                    els.push(self.node_to_ann_term(lang, ast, at(i)?)?);
                }
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("ArrLit".to_string()));
                m.insert("elements".to_string(), Value::Array(els));
                Some(Value::Object(m))
            }
            "flow" => {
                let l = self.node_to_ann_term(lang, ast, at(0)?)?;
                let r = self.node_to_ann_term(lang, ast, at(1)?)?;
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("Comp".to_string()));
                m.insert("l".to_string(), l);
                m.insert("r".to_string(), r);
                Some(Value::Object(m))
            }
            _ if node.kind.starts_with("binary.") => {
                let l = role_child_ref(lang, node, "left")
                    .or_else(|| child_at(ast, node, 0))
                    .and_then(|r| ast.get(r))
                    .and_then(|n| self.node_to_ann_term(lang, ast, n))?;
                let r = role_child_ref(lang, node, "right")
                    .or_else(|| child_at(ast, node, 1))
                    .and_then(|r| ast.get(r))
                    .and_then(|n| self.node_to_ann_term(lang, ast, n))?;
                Some(bin_term(&node.kind, l, r))
            }
            _ if node.kind.starts_with("unary.") => {
                let o = role_child_ref(lang, node, "operand")
                    .or_else(|| child_at(ast, node, 0))
                    .and_then(|r| ast.get(r))
                    .and_then(|n| self.node_to_ann_term(lang, ast, n))?;
                Some(un_term(&node.kind, o))
            }
            _ => None,
        }
    }

    /// `node_to_ann_term` + `eval_ann_term` (the model's `eval_node_`).
    fn eval_ann_node(
        &self,
        lang: &Language,
        ast: &Ast,
        idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast.get(AstRef::new(idx)).ok_or_else(|| {
            LangError::engine(format!("ann eval: node {idx} out of bounds"))
        })?;
        let term = self.node_to_ann_term(lang, ast, node).ok_or_else(|| {
            LangError::engine(format!(
                "ann eval: node kind '{}' is not evaluable",
                node.kind
            ))
        })?;
        self.eval_ann_term(lang, ast, &term, ctx, ANN_FUEL)
    }

    /// Reduce an annotation term against the bound-value env
    /// (`ctx.results`) — the model's `eval_term_`. Stuck terms return
    /// their partially-evaluated form (the model returned the term).
    fn eval_ann_term(
        &self,
        lang: &Language,
        ast: &Ast,
        term: &Value,
        ctx: &mut TypecheckContext,
        fuel: u32,
    ) -> Result<Value, LangError> {
        if fuel == 0 {
            return Ok(term.clone());
        }
        match tag_of(term) {
            Some("var") => {
                let name = term.get("name").and_then(Value::as_str).unwrap_or("");
                match ctx.results.get(name) {
                    // A `var{n} ↦ var{n}` self-binding stays symbolic —
                    // recursing would burn the fuel on a fixed point.
                    Some(bound) if bound != term => {
                        self.eval_ann_term(lang, ast, &bound.clone(), ctx, fuel - 1)
                    }
                    _ => Ok(term.clone()),
                }
            }
            Some("Lam") => {
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("LamVal".to_string()));
                m.insert(
                    "params".to_string(),
                    term.get("params").cloned().unwrap_or(Value::Array(vec![])),
                );
                m.insert(
                    "body".to_string(),
                    term.get("body").cloned().unwrap_or(Value::Null),
                );
                m.insert(
                    "env".to_string(),
                    Value::Object(
                        ctx.results
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    ),
                );
                Ok(Value::Object(m))
            }
            Some("App") => {
                let callee = self.eval_ann_term(
                    lang,
                    ast,
                    &term.get("callee").cloned().unwrap_or(Value::Null),
                    ctx,
                    fuel - 1,
                )?;
                let mut args = Vec::new();
                for a in term
                    .get("args")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                {
                    args.push(self.eval_ann_term(lang, ast, &a, ctx, fuel - 1)?);
                }
                self.apply_ann_val(lang, ast, &callee, &args, ctx, fuel - 1)
            }
            Some("Comp") => {
                let l = self.eval_ann_term(
                    lang,
                    ast,
                    &term.get("l").cloned().unwrap_or(Value::Null),
                    ctx,
                    fuel - 1,
                )?;
                let r = self.eval_ann_term(
                    lang,
                    ast,
                    &term.get("r").cloned().unwrap_or(Value::Null),
                    ctx,
                    fuel - 1,
                )?;
                if is_applicable_ann(&r) && !is_applicable_ann(&l) {
                    self.apply_ann_val(lang, ast, &r, &[l], ctx, fuel - 1)
                } else if is_applicable_ann(&l) && is_applicable_ann(&r) {
                    let mut m = Map::new();
                    m.insert("__tag__".to_string(), Value::String("CompVal".to_string()));
                    m.insert("l".to_string(), l);
                    m.insert("r".to_string(), r);
                    Ok(Value::Object(m))
                } else {
                    let mut m = Map::new();
                    m.insert("__tag__".to_string(), Value::String("Comp".to_string()));
                    m.insert("l".to_string(), l);
                    m.insert("r".to_string(), r);
                    Ok(Value::Object(m))
                }
            }
            Some("Bin") | Some("Un") => {
                // Evaluate subterms first (they may contain App/Lam), then
                // fold closed arithmetic through the index evaluator.
                let evaled = if tag_of(term) == Some("Bin") {
                    let l = self.eval_ann_term(
                        lang,
                        ast,
                        &term.get("l").cloned().unwrap_or(Value::Null),
                        ctx,
                        fuel - 1,
                    )?;
                    let r = self.eval_ann_term(
                        lang,
                        ast,
                        &term.get("r").cloned().unwrap_or(Value::Null),
                        ctx,
                        fuel - 1,
                    )?;
                    bin_term(
                        term.get("op").and_then(Value::as_str).unwrap_or(""),
                        l,
                        r,
                    )
                } else {
                    let o = self.eval_ann_term(
                        lang,
                        ast,
                        &term.get("operand").cloned().unwrap_or(Value::Null),
                        ctx,
                        fuel - 1,
                    )?;
                    un_term(
                        term.get("op").and_then(Value::as_str).unwrap_or(""),
                        o,
                    )
                };
                // `binary.eq`/`neq`/comparisons on plain literals fold too —
                // `eval_index` covers arithmetic; close the scalar fragment.
                if tag_of(&evaled) == Some("Bin") {
                    let op = evaled.get("op").and_then(Value::as_str).unwrap_or("");
                    let l = eval_index(evaled.get("l").unwrap_or(&Value::Null), ctx);
                    let r = eval_index(evaled.get("r").unwrap_or(&Value::Null), ctx);
                    let (ln, rn) = (num_val(&l), num_val(&r));
                    let folded = match op {
                        "binary.eq" if l == r || (ln.is_some() && ln == rn) => {
                            Some(Value::Bool(true))
                        }
                        "binary.eq" if ln.is_some() && rn.is_some() => {
                            Some(Value::Bool(false))
                        }
                        "binary.neq" if ln.is_some() && rn.is_some() => {
                            Some(Value::Bool(ln != rn))
                        }
                        "binary.lt" => ln.zip(rn).map(|(a, b)| Value::Bool(a < b)),
                        "binary.lte" => ln.zip(rn).map(|(a, b)| Value::Bool(a <= b)),
                        "binary.gt" => ln.zip(rn).map(|(a, b)| Value::Bool(a > b)),
                        "binary.gte" => ln.zip(rn).map(|(a, b)| Value::Bool(a >= b)),
                        "binary.and" => match (&l, &r) {
                            (Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(*a && *b)),
                            _ => None,
                        },
                        "binary.or" => match (&l, &r) {
                            (Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(*a || *b)),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(v) = folded {
                        return Ok(v);
                    }
                }
                Ok(eval_index(&evaled, ctx))
            }
            Some("If") => {
                let c = self.eval_ann_term(
                    lang,
                    ast,
                    &term.get("cond").cloned().unwrap_or(Value::Null),
                    ctx,
                    fuel - 1,
                )?;
                match c {
                    Value::Bool(true) => self.eval_ann_term(
                        lang,
                        ast,
                        &term.get("then").cloned().unwrap_or(Value::Null),
                        ctx,
                        fuel - 1,
                    ),
                    Value::Bool(false) => self.eval_ann_term(
                        lang,
                        ast,
                        &term.get("else").cloned().unwrap_or(Value::Null),
                        ctx,
                        fuel - 1,
                    ),
                    _ => Ok(term.clone()),
                }
            }
            Some("ArrLit") => {
                let mut els = Vec::new();
                let mut all_plain = true;
                for e in term
                    .get("elements")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                {
                    let v = self.eval_ann_term(lang, ast, &e, ctx, fuel - 1)?;
                    all_plain &= tag_of(&v).is_none();
                    els.push(v);
                }
                if all_plain {
                    Ok(Value::Array(els))
                } else {
                    let mut m = Map::new();
                    m.insert("__tag__".to_string(), Value::String("ArrLit".to_string()));
                    m.insert("elements".to_string(), Value::Array(els));
                    Ok(Value::Object(m))
                }
            }
            _ => Ok(term.clone()),
        }
    }

    /// Apply an evaluated annotation value to evaluated arguments (the
    /// model's `apply_val_`): `LamVal` runs its body under the captured
    /// env extended with `params ↦ args`; `CompVal` composes; a boundary
    /// `__arrow__` cannot be host-called here and stays a stuck `App`.
    fn apply_ann_val(
        &self,
        lang: &Language,
        ast: &Ast,
        f: &Value,
        args: &[Value],
        ctx: &mut TypecheckContext,
        fuel: u32,
    ) -> Result<Value, LangError> {
        match tag_of(f) {
            Some("LamVal") => {
                let params: Vec<String> = f
                    .get("params")
                    .and_then(Value::as_array)
                    .map(|ps| {
                        ps.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                let body_idx = f
                    .get("body")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| LangError::engine("LamVal: no body index".to_string()))?
                    as usize;
                // bound_env = captured env + params ↦ args.
                let mut bound_env: HashMap<String, Value> = f
                    .get("env")
                    .and_then(Value::as_object)
                    .map(|e| e.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                for (p, a) in params.iter().zip(args.iter()) {
                    bound_env.insert(p.clone(), a.clone());
                }
                let old_results = ctx.results.clone();
                ctx.results = bound_env;
                let r = self.eval_ann_node(lang, ast, body_idx, ctx);
                ctx.results = old_results;
                r
            }
            Some("CompVal") => {
                let l = f.get("l").cloned().unwrap_or(Value::Null);
                let r = f.get("r").cloned().unwrap_or(Value::Null);
                let inner = self.apply_ann_val(lang, ast, &l, args, ctx, fuel - 1)?;
                self.apply_ann_val(lang, ast, &r, &[inner], ctx, fuel - 1)
            }
            _ => Ok(app_term(f.clone(), args.to_vec())),
        }
    }
}

// ---------------------------------------------------------------------------
// `stmt.import` / `stmt.export` — module signature records (phase-1 WS-B)
// ---------------------------------------------------------------------------

/// Whether `spec` is a bare module name (no path separators) — `str`,
/// `bool`; `"../lib/x"`, `"./mod.hnl"` are file specs.
fn is_bare_module_name(spec: &str) -> bool {
    !spec.is_empty()
        && !spec.contains('/')
        && !spec.contains('\\')
        && !spec.ends_with(".hnl")
        && spec
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl NbeTypecheckEngine {
    /// The `type.unknown` value — dynamic/`resolve` imports bind here
    /// (phase-1 spec WS-B: `resolve`/dynamic imports → `type.unknown`).
    fn type_unknown(&self, ctx: &TypecheckContext) -> Value {
        ctx.registry
            .construct("type.unknown", &Value::Null)
            .unwrap_or(Value::Null)
    }

    /// `Sig_m` — the module signature record for `spec` as a
    /// `type.module_sig{module, fields}` value. Fields come from the
    /// module registry's `lookup_module_exports`: each export's declared
    /// signature, or `type.unknown` when it has none. `None` from the
    /// registry (no such host/`.hnl` module) → `type.unknown`, the
    /// dynamic-import fallback.
    fn module_signature(&self, spec: &str, ctx: &mut TypecheckContext) -> Value {
        let exports = self
            .modules
            .as_ref()
            .and_then(|m| m.lookup_module_exports(spec));
        match exports {
            Some(list) => {
                let mut fields = Map::new();
                for (name, sig) in list {
                    fields.insert(
                        name,
                        sig.unwrap_or_else(|| self.type_unknown(ctx)),
                    );
                }
                let mut payload = Map::new();
                payload.insert("module".to_string(), Value::String(spec.to_string()));
                payload.insert("fields".to_string(), Value::Object(fields));
                make_tag(ctx, "type.module_sig", payload)
            }
            None => self.type_unknown(ctx),
        }
    }

    /// The type of one named field projected out of `Sig_m` — the sig's
    /// recorded field type, else a `lookup_type_signature` probe for
    /// `<spec>.<name>`, else an opaque meta-function type when the op
    /// resolves without a declared signature, else `type.unknown`.
    fn import_field_type(
        &self,
        sig: &Value,
        spec: &str,
        name: &str,
        ctx: &mut TypecheckContext,
    ) -> Value {
        if let Some(ft) = sig
            .get("fields")
            .and_then(Value::as_object)
            .and_then(|f| f.get(name))
        {
            return ft.clone();
        }
        if let Some(mods) = self.modules.as_ref() {
            let q = format!("{spec}.{name}");
            if let Some(t) = mods.lookup_type_signature(&q) {
                return t;
            }
            if mods.lookup(&q).is_some() {
                let p = meta_value(ctx.fresh_meta());
                let b = meta_value(ctx.fresh_meta());
                let mut m = Map::new();
                m.insert("param".to_string(), p);
                m.insert("body".to_string(), b);
                return make_tag(ctx, "type.function", m);
            }
        }
        self.type_unknown(ctx)
    }

    /// `synth_import` — a `stmt.import` node binds names into Γ from the
    /// imported module's signature record (phase-1 spec WS-B):
    ///
    /// - `import m` — binds `m : Sig_m` (the record).
    /// - `import "<spec>"` — splice import: binds nothing statically.
    /// - `from "<spec>" import <n>` / `{ a, b as x }` — named field
    ///   projections (`x : Sig_m.<field>`).
    /// - `from "<spec>" import *` — splices `Sig_m`'s fields into Γ; for
    ///   a bare module name also binds `spec : Sig_m` so qualified
    ///   `spec.f` access keeps working (model parity).
    /// - `from "<spec>" import * as m` — binds `m : Sig_m` (the record).
    ///
    /// Unresolvable/dynamic specs degrade each binding to `type.unknown`.
    /// Returns `type.unit`.
    pub(crate) fn synth_import_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let _ = lang;
        let unit = |ctx: &TypecheckContext| make_tag(ctx, "type.unit", Map::new());
        let path = node.payload.get("path").and_then(Value::as_str);
        let alias = node
            .payload
            .get("alias")
            .and_then(Value::as_str)
            .filter(|a| !a.is_empty());
        let names = node.payload.get("names").filter(|v| !v.is_null());
        // `import <name>` — the single child is a `var` carrying the name.
        let child_var = children_of(node)
            .and_then(|c| c.first())
            .and_then(Value::as_u64)
            .and_then(|i| ast.get(AstRef::new(i as usize)))
            .and_then(node_name);

        // `import <m>` — `m : Sig_m`, the whole record.
        if path.is_none() && names.is_none() && alias.is_none() {
            if let Some(n) = child_var {
                let sig = self.module_signature(&n, ctx);
                ctx.extend(n, sig);
            }
            return Ok(unit(ctx));
        }
        // `import "<spec>"` — splice import: nothing statically bound.
        if path.is_some() && names.is_none() && alias.is_none() && child_var.is_none() {
            return Ok(unit(ctx));
        }

        let spec = path.unwrap_or_default();
        let sig = if spec.is_empty() {
            self.type_unknown(ctx)
        } else {
            self.module_signature(spec, ctx)
        };

        // `… import * as m` / `import n as m` — the alias binds the record
        // (star) or the named field.
        if let Some(a) = alias {
            let bound = match names {
                Some(Value::String(s)) if s != "*" => self.import_field_type(&sig, spec, s, ctx),
                _ => sig.clone(),
            };
            ctx.extend(a.to_string(), bound);
            return Ok(unit(ctx));
        }

        match names {
            // `from "<spec>" import *` — splice fields into Γ.
            Some(Value::String(s)) if s == "*" => {
                if let Some(fields) = sig.get("fields").and_then(Value::as_object) {
                    for (n, t) in fields {
                        ctx.extend(n.clone(), t.clone());
                    }
                }
                // Bare module-name specs (`str`) also stay usable qualified.
                if is_bare_module_name(spec) {
                    ctx.extend(spec.to_string(), sig.clone());
                }
            }
            Some(Value::String(s)) => {
                let ty = self.import_field_type(&sig, spec, s, ctx);
                ctx.extend(s.clone(), ty);
            }
            Some(Value::Array(items)) => {
                for item in items {
                    let (bound, field) = match item {
                        Value::String(s) => (s.clone(), s.clone()),
                        Value::Object(o) => {
                            let a = o.get("as").and_then(Value::as_str).map(String::from);
                            let n = o.get("name").and_then(Value::as_str).map(String::from);
                            match (a, n) {
                                (Some(a), Some(n)) => (a, n),
                                (None, Some(n)) => (n.clone(), n),
                                (Some(a), None) => (a.clone(), a),
                                _ => continue,
                            }
                        }
                        _ => continue,
                    };
                    let ty = self.import_field_type(&sig, spec, &field, ctx);
                    ctx.extend(bound, ty);
                }
            }
            _ => {}
        }
        Ok(unit(ctx))
    }

    /// `synth_export` — a `stmt.export` records the exported name's type
    /// into the module signature (`ctx.exports` → `Sig_m`). The child is
    /// a `var` (`export x`), a binding (`export x :: expr` parses as
    /// `export(assign(x, expr))` — both declares and binds `x`), or a
    /// signature (`export x: T ~> U` parses as
    /// `export(function(type-annot{x}, T))`). Returns `type.unit`.
    pub(crate) fn synth_export_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let unit = |ctx: &TypecheckContext| make_tag(ctx, "type.unit", Map::new());
        let Some(child_idx) = children_of(node)
            .and_then(|c| c.first())
            .and_then(Value::as_u64)
            .map(|i| i as usize)
        else {
            return Ok(unit(ctx));
        };
        let child_ref = AstRef::new(child_idx);
        let Some(child) = ast.get(child_ref) else {
            return Ok(unit(ctx));
        };
        if is_assign_node(lang, child) {
            // `export x :: expr` — bind `x` in Γ, then record its type.
            self.synth_assign_judgment(lang, ast, child, ctx)?;
            if let Some(name) = assign_target_name(lang, ast, child) {
                let ty = ctx
                    .lookup(&name)
                    .cloned()
                    .unwrap_or_else(|| self.type_unknown(ctx));
                ctx.exports.push((name, ty));
            }
            return Ok(unit(ctx));
        }
        if let Some(name) = node_name(child) {
            // `export x` / `export x from "path"` — re-export an existing
            // binding; record its Γ type (unknown if unbound — the
            // `unbound-export` scope lint owns that diagnostic).
            let ty = ctx
                .lookup(&name)
                .cloned()
                .unwrap_or_else(|| self.type_unknown(ctx));
            ctx.exports.push((name, ty));
            return Ok(unit(ctx));
        }
        // `export x: T ~> U` — a type signature. The exported name is the
        // annotation's subject; the signature type is the synthesized
        // type-expression type of the child.
        if let Some(name) = signature_name(ast, child) {
            let ty = self
                .synth_node(lang, ast, child_ref, ctx)
                .unwrap_or_else(|_| self.type_unknown(ctx));
            ctx.exports.push((name, ty));
        }
        Ok(unit(ctx))
    }
}

/// The name a binding-shaped node's `target` declares (`var`, or the
/// `subject` of a `type-annot` target) — mirrors the bootstrap scope
/// pass's `binding_target_name` (ADR-0081 roles).
fn assign_target_name(lang: &Language, ast: &Ast, node: &AstNode) -> Option<String> {
    let t_idx = lang
        .child_role_index(&node.kind, "target")
        .ok()
        .flatten()?;
    let tref = children_of(node)
        .and_then(|c| c.get(t_idx))
        .and_then(Value::as_u64)?;
    let target = ast.get(AstRef::new(tref as usize))?;
    if let Some(n) = node_name(target) {
        return Some(n);
    }
    let subj = lang
        .child_role_index(&target.kind, "subject")
        .ok()
        .flatten()?;
    let sref = children_of(target)
        .and_then(|c| c.get(subj))
        .and_then(Value::as_u64)?;
    ast.get(AstRef::new(sref as usize)).and_then(node_name)
}

/// The declared name of an export signature child — the subject of a
/// `type-annot` inside the node (e.g. `x` in `function(type-annot{x}, T)`).
/// Mirrors the bootstrap `signature_name` (first `type-annot` subject).
fn signature_name(ast: &Ast, node: &AstNode) -> Option<String> {
    let subj = children_of(node)
        .and_then(|c| c.first())
        .and_then(Value::as_u64)
        .and_then(|i| ast.get(AstRef::new(i as usize)))?;
    if subj.kind == "type-annot" {
        return children_of(subj)
            .and_then(|c| c.first())
            .and_then(Value::as_u64)
            .and_then(|i| ast.get(AstRef::new(i as usize)))
            .and_then(node_name);
    }
    if let Some(kids) = children_of(node) {
        for c in kids {
            if let Some(i) = c.as_u64()
                && let Some(n) = ast.get(AstRef::new(i as usize))
                && let Some(name) = signature_name(ast, n)
            {
                return Some(name);
            }
        }
    }
    None
}
