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
//!   operators, `{"__tag__": "App", "callee", "args"}` ctor applications.
//!   Index equality is definitional via evaluation
//!   ([`eval_index`]) — `data`-declared ctor projections first
//!   (`App{Succ, x} ↦ x + 1`, `var{Zero} ↦ 0`; phase-3 WS-D), then
//!   commutative-ring normalization of `add`/`sub`/`mul`/`neg` terms to
//!   a polynomial normal form — plus the bounded solver behind the
//!   [`IndexSolver`] seam.
//! - `Path{frm, to}` — the proof-relevant index-equality witness bound
//!   as `eq_<name>` in `results` by branch refinement.

use handl_language::{Ast, AstNode, AstRef, LangError, Language};
use handl_value::TaggedValue;
use handl_value::nbe::{MetaId, Val};
use hnl_tool_typecheck::TypecheckDiagnostic;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

use crate::{
    InstanceEntry, NbeTypecheckEngine, TypecheckContext, checker, extract_meta_id, is_assign_node,
    meta_value, role_child_ref,
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

/// Insert one module export into a `type.module_sig` `fields` map.
///
/// Export names can be dotted — a namespaced op's *local* name keeps its
/// inner segments (`"ffi.rust.resolve"` in module `"ffi"` is exposed as
/// `"rust.resolve"`, P-013 T-152). Nest each `.`-separated prefix as a
/// `type.record` so `mod.rust.resolve` projects through the ordinary
/// `record_field_type` path: `mod.rust` yields the record, `.resolve`
/// its field. A prefix colliding with a non-record field is overwritten
/// by the nested record (the namespace wins — a flat `rust` export and a
/// `rust.*` op family cannot both be projected unambiguously anyway).
fn insert_module_sig_field(
    fields: &mut Map<String, Value>,
    name: &str,
    ty: Value,
    ctx: &TypecheckContext,
) {
    let Some((head, rest)) = name.split_once('.') else {
        fields.insert(name.to_string(), ty);
        return;
    };
    let entry = fields.entry(head.to_string()).or_insert_with(|| {
        let mut inner = Map::new();
        inner.insert("fields".to_string(), Value::Object(Map::new()));
        make_tag(ctx, "type.record", inner)
    });
    let is_record = entry.get("__tag__").and_then(Value::as_str) == Some("type.record");
    if !is_record {
        let mut inner = Map::new();
        inner.insert("fields".to_string(), Value::Object(Map::new()));
        *entry = make_tag(ctx, "type.record", inner);
    }
    if let Some(inner_fields) = entry
        .get_mut("fields")
        .and_then(Value::as_object_mut)
    {
        insert_module_sig_field(inner_fields, rest, ty, ctx);
    }
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

/// Internal binding name for a `data` family's declared parameter while
/// a `type.ctor_sig` telescope elaborates (`n` → `__fam_n`). The
/// `__fam_` prefix is a namespace users cannot write into, so the
/// ctor's fresh index metas never shadow — or get shadowed by — a
/// same-named rigid `var` from an enclosing annotated signature
/// (`f<n, a>` against `data Vec(n : Nat, a : Type)`).
fn fam_binding_name(name: &str) -> String {
    format!("__fam_{name}")
}

/// Resolve a source-level name through the ctor-telescope rename scope:
/// during `instantiate_ctor_full` a family parameter's occurrences
/// elaborate as its `__fam_<name>` binding; anything else (ctor args,
/// user binders, outer rigids) passes through unchanged.
fn fam_scoped(name: &str, ctx: &TypecheckContext) -> String {
    ctx.fam_scope
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
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

/// Record the consumed path witness under `via` — the model's
/// `set(ty, "via", rp.witness)` on object results (`synth_j_`'s path
/// and refl arms, `synth_transport_`). An open result over an unsolved
/// metavariable stays atomic — a `via` field would corrupt the
/// `{"__meta__"}` encoding.
fn with_path_via(ty: Value, witness: &Value) -> Value {
    if let Value::Object(mut m) = ty {
        if extract_meta_id(&Value::Object(m.clone())).is_none() {
            m.insert("via".to_string(), witness.clone());
        }
        return Value::Object(m);
    }
    ty
}

/// Endpoint pair `(frm, to)` of a path witness value: a legacy
/// `Path{frm,to}` record, a `type.eq{left,right}`, or a cubical
/// path-carrying `type.pi` (`path{x,y}` payload, ADR-0208 P1).
/// `None` for non-path carriers.
fn path_witness_endpoints(v: &Value) -> Option<(Value, Value)> {
    match tag_of(v) {
        Some("Path") => Some((
            v.get("frm").cloned().unwrap_or(Value::Null),
            v.get("to").cloned().unwrap_or(Value::Null),
        )),
        Some("type.eq") => Some((
            v.get("left").cloned().unwrap_or(Value::Null),
            v.get("right").cloned().unwrap_or(Value::Null),
        )),
        Some("type.pi") => v.get("path").map(|p| {
            (
                p.get("x").cloned().unwrap_or(Value::Null),
                p.get("y").cloned().unwrap_or(Value::Null),
            )
        }),
        _ => None,
    }
}

/// Full path-witness reading of a *type/value* carrier (ADR-0208 P2 —
/// the model's `witness_from_type_`): `(x, y, closure, self)` where
/// `x`/`y` are the endpoint terms, `closure` the carrier's
/// `body_closure` (`Null` for `Path`/`type.eq` records, which carry
/// none), and `self` the carrier itself when it is a `type.pi`
/// (`Null` otherwise — `trans`/`symm` reuse it for the identity cases).
fn witness_from_type_full(v: &Value) -> Option<(Value, Value, Value, Value)> {
    match tag_of(v) {
        Some("type.pi") => v.get("path").map(|p| {
            (
                p.get("x").cloned().unwrap_or(Value::Null),
                p.get("y").cloned().unwrap_or(Value::Null),
                v.get("body_closure").cloned().unwrap_or(Value::Null),
                v.clone(),
            )
        }),
        Some("Path") => Some((
            v.get("frm").cloned().unwrap_or(Value::Null),
            v.get("to").cloned().unwrap_or(Value::Null),
            Value::Null,
            Value::Null,
        )),
        Some("type.eq") => Some((
            v.get("left").cloned().unwrap_or(Value::Null),
            v.get("right").cloned().unwrap_or(Value::Null),
            Value::Null,
            Value::Null,
        )),
        _ => None,
    }
}

/// The `path.equiv` payload of a path-carrying `type.pi` — the
/// `{dom, cod, to, fro}` record univalence paths carry (ADR-0208 P4).
/// `Value::Null` for non-pi carriers and plain `Path`/`type.eq`
/// witnesses (the model's `ue == null` reading).
fn path_equiv_of(v: &Value) -> Value {
    if tag_of(v) == Some("type.pi") {
        v.get("path")
            .and_then(|p| p.get("equiv"))
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    }
}

/// The model's `transport_term_` (ADR-0208 P4): the *shallow* index
/// rewrite used on `type.family_app` args during transport — `v ↦ to`
/// on exact match, otherwise a structural walk that skips the
/// non-term/provenance keys (`via`/`equiv` are proof-relevant and
/// never rewritten).
fn transport_term(frm: &Value, to: &Value, t: &Value) -> Value {
    if t == frm {
        return to.clone();
    }
    match t {
        Value::Array(xs) => Value::Array(xs.iter().map(|x| transport_term(frm, to, x)).collect()),
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, x) in m {
                if matches!(
                    k.as_str(),
                    "__tag__"
                        | "__registry__"
                        | "__meta__"
                        | "name"
                        | "op"
                        | "body"
                        | "body_closure"
                        | "env"
                        | "via"
                        | "equiv"
                ) {
                    out.insert(k.clone(), x.clone());
                } else {
                    out.insert(k.clone(), transport_term(frm, to, x));
                }
            }
            Value::Object(out)
        }
        _ => t.clone(),
    }
}

/// The model's `subst_index_` (ADR-0208 P4) — the transport-time
/// dependent rewrite `frm ↦ to` through a *type* value: `type.family_app`
/// args go through [`transport_term`] (shallow — index terms are data,
/// not types); other objects walk structurally but never descend into
/// the proof-relevant/cached carriers (`type.ctor_sig`, `Path`,
/// `type.eq`, `Lam`, `LamVal`, `CompVal`) or the spelling/provenance
/// keys. This is deliberately *not* [`subst_index`], which serves
/// index-term substitution elsewhere and rewrites more aggressively.
fn subst_transport(frm: &Value, to: &Value, v: &Value) -> Value {
    if frm.is_null() {
        return v.clone();
    }
    if v == frm {
        return to.clone();
    }
    if tag_of(v) == Some("type.family_app") {
        let args = v
            .get("args")
            .and_then(Value::as_array)
            .map(|xs| Value::Array(xs.iter().map(|a| transport_term(frm, to, a)).collect()))
            .unwrap_or(Value::Array(Vec::new()));
        let mut m = v.as_object().cloned().unwrap_or_default();
        m.insert("args".to_string(), args);
        return Value::Object(m);
    }
    match v {
        Value::Array(xs) => Value::Array(xs.iter().map(|x| subst_transport(frm, to, x)).collect()),
        Value::Object(m) => {
            let carrier = matches!(
                tag_of(v),
                Some("type.ctor_sig" | "Path" | "type.eq" | "Lam" | "LamVal" | "CompVal")
            );
            if carrier {
                return v.clone();
            }
            let mut out = Map::new();
            for (k, x) in m {
                if matches!(
                    k.as_str(),
                    "__tag__"
                        | "__registry__"
                        | "__meta__"
                        | "name"
                        | "op"
                        | "body"
                        | "body_closure"
                        | "env"
                        | "via"
                        | "equiv"
                        | "param_name"
                        | "ctor"
                ) {
                    out.insert(k.clone(), x.clone());
                } else {
                    out.insert(k.clone(), subst_transport(frm, to, x));
                }
            }
            Value::Object(out)
        }
        _ => v.clone(),
    }
}

/// A cubical face formula Φ (ADR-0208 P2) — the Rust mirror of the
/// model's `{"__tag__":"face","op","args"}` records: `Tt`/`Ff` are the
/// full/empty faces, `Eq0`/`Eq1` the endpoint equations `i = 0`/`i = 1`
/// (the payload is the evaluated dimension *term*), and
/// `Not`/`Or`/`And` the connectives `~φ`, `φ ∨ ψ`, `φ ∧ ψ`.
#[derive(Debug, Clone, PartialEq)]
enum CubFace {
    Tt,
    Ff,
    Eq0(Value),
    Eq1(Value),
    Not(Box<CubFace>),
    Or(Box<CubFace>, Box<CubFace>),
    And(Box<CubFace>, Box<CubFace>),
}

/// Interpret an evaluated term as a face Φ — the model's
/// `face_of_val_`. The surface constructors `f0(i)`/`f1(i)`/`fnot φ`/
/// `f_or φ ψ`/`f_and φ ψ` evaluate to symbolic `App{var{f*}, args}`
/// terms (the names are unbound); `fT`/`fF`/`true`/`false` stay `var`
/// atoms; `0`/`1`/`true`/`false` literals read as the empty/full
/// faces. `None` for anything else (a non-face operand).
fn cub_face_of_val(v: &Value) -> Option<CubFace> {
    // Literal payloads can arrive wrapped as `{"value": x[, …]}`
    // (the `value`/`lit` node payload shape) — unwrap before tagging.
    if let Value::Object(m) = v
        && m.get("__tag__").is_none()
        && let Some(inner) = m.get("value")
    {
        return cub_face_of_val(inner);
    }
    match tag_of(v) {
        Some("App") => {
            let callee = v.get("callee").cloned().unwrap_or(Value::Null);
            let cname = if tag_of(&callee) == Some("var") {
                callee.get("name").and_then(Value::as_str).unwrap_or("")
            } else {
                ""
            };
            let args = v.get("args").and_then(Value::as_array);
            match cname {
                "f0" | "f1" => args.and_then(|a| a.first()).map(|d| {
                    if cname == "f0" {
                        CubFace::Eq0(d.clone())
                    } else {
                        CubFace::Eq1(d.clone())
                    }
                }),
                "fnot" => args
                    .and_then(|a| a.first())
                    .and_then(|a0| cub_face_of_val(a0))
                    .map(|a| CubFace::Not(Box::new(a))),
                "f_or" | "f_and" => {
                    let (Some(a0), Some(a1)) =
                        (args.and_then(|a| a.first()), args.and_then(|a| a.get(1)))
                    else {
                        return None;
                    };
                    let fa = cub_face_of_val(a0)?;
                    let fb = cub_face_of_val(a1)?;
                    Some(if cname == "f_or" {
                        CubFace::Or(Box::new(fa), Box::new(fb))
                    } else {
                        CubFace::And(Box::new(fa), Box::new(fb))
                    })
                }
                _ => None,
            }
        }
        Some("var") => match v.get("name").and_then(Value::as_str).unwrap_or("") {
            "fT" | "true" => Some(CubFace::Tt),
            "fF" | "false" => Some(CubFace::Ff),
            _ => None,
        },
        _ => {
            if let Some(b) = v.as_bool() {
                Some(if b { CubFace::Tt } else { CubFace::Ff })
            } else {
                match num_val(v) {
                    Some(n) if n == 1.0 => Some(CubFace::Tt),
                    Some(n) if n == 0.0 => Some(CubFace::Ff),
                    _ => None,
                }
            }
        }
    }
}

/// Endpoint-equation normalization shared by `face_norm`'s `eq0`/`eq1`
/// arms: `i = 0` holds at `i0` and fails at `i1` (and dually).
fn face_norm_eq(d: Value, is_eq0: bool) -> CubFace {
    let (holds, fails) = if is_eq0 {
        (d == var_term("i0"), d == var_term("i1"))
    } else {
        (d == var_term("i1"), d == var_term("i0"))
    };
    if holds {
        CubFace::Tt
    } else if fails {
        CubFace::Ff
    } else if is_eq0 {
        CubFace::Eq0(d)
    } else {
        CubFace::Eq1(d)
    }
}

/// Normalize a face — the model's `face_norm_`: endpoint equations
/// against the `i0`/`i1` terms (`i0 = 0 → tt`, `i1 = 0 → ff`), `not`
/// over constants, and the `∧`/`∨` unit/annihilator, idempotence,
/// `φ ∧ ¬φ → ff` / `φ ∨ ¬φ → tt`, and `i=0 ∧ i=1 → ff` laws — enough
/// for the `i ∉ φ` side conditions. Endpoint *substitution* is
/// deferred, as in the model.
fn face_norm(f: CubFace) -> CubFace {
    let is_or = matches!(f, CubFace::Or(_, _));
    match f {
        CubFace::Tt | CubFace::Ff => f,
        CubFace::Eq0(d) => face_norm_eq(d, true),
        CubFace::Eq1(d) => face_norm_eq(d, false),
        CubFace::Not(a) => {
            let a = face_norm(*a);
            match a {
                CubFace::Tt => CubFace::Ff,
                CubFace::Ff => CubFace::Tt,
                a => CubFace::Not(Box::new(a)),
            }
        }
        CubFace::Or(a, b) | CubFace::And(a, b) => {
            let a = face_norm(*a);
            let b = face_norm(*b);
            let (unit, annih) = if is_or {
                (CubFace::Ff, CubFace::Tt)
            } else {
                (CubFace::Tt, CubFace::Ff)
            };
            if a == annih || b == annih {
                if is_or { CubFace::Tt } else { CubFace::Ff }
            } else if a == unit {
                b
            } else if b == unit {
                a
            } else {
                let a_inner = match &a {
                    CubFace::Not(x) => Some(&**x),
                    _ => None,
                };
                let b_inner = match &b {
                    CubFace::Not(x) => Some(&**x),
                    _ => None,
                };
                let complemented =
                    a_inner.is_some_and(|x| *x == b) || b_inner.is_some_and(|x| *x == a);
                if a == b {
                    a
                } else if complemented {
                    if is_or { CubFace::Tt } else { CubFace::Ff }
                } else {
                    // `i=0 ∧ i=1 → ff` — contradictory endpoint equations
                    // over the *same* dimension term.
                    let contra = !is_or
                        && matches!(
                            (&a, &b),
                            (CubFace::Eq0(d1), CubFace::Eq1(d2))
                                | (CubFace::Eq1(d1), CubFace::Eq0(d2))
                                if d1 == d2
                        );
                    if contra {
                        CubFace::Ff
                    } else if is_or {
                        CubFace::Or(Box::new(a), Box::new(b))
                    } else {
                        CubFace::And(Box::new(a), Box::new(b))
                    }
                }
            }
        }
    }
}

/// The verdict of [`NbeTypecheckEngine::path_witness`] — the model's
/// `path`/`refl`/`bad` classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CubPathVerdict {
    Path,
    Refl,
    Bad,
}

/// A resolved path operand for `trans`/`symm` — the model's
/// `path_witness_` result: endpoints `x`/`y`, the witness's
/// `body_closure` (`Null` for `Path`/`type.eq` carriers), and `self`
/// the path-carrying `type.pi` itself (`Null` otherwise).
struct CubPathWitness {
    verdict: CubPathVerdict,
    x: Value,
    y: Value,
    closure: Value,
    self_ty: Value,
}

/// A resolved path operand for `#`/`j` — the model's `resolve_path_`
/// result (ADR-0208 P4): `verdict` is `path`/`refl` (`bad` never
/// reaches here — it is the `None` case), `frm`/`to` the endpoints,
/// `equiv` the `path.equiv` payload when the witness is a univalence
/// path (`Null` otherwise), and `witness` the carrier recorded under
/// the transport result's `via` (the bound type/value itself).
struct ResolvedPath {
    verdict: CubPathVerdict,
    frm: Value,
    to: Value,
    equiv: Value,
    witness: Value,
}

/// Does index term `t` mention `var{name}`? Walks record/array
/// structure, skipping the same non-term fields the model's
/// `term_has_var_` skips (spelling, identity, provenance, arena
/// positions).
fn term_has_var(v: &Value, name: &str) -> bool {
    if tag_of(v) == Some("var") {
        return v.get("name").and_then(Value::as_str) == Some(name);
    }
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| {
            !matches!(
                k.as_str(),
                "__tag__"
                    | "__registry__"
                    | "__meta__"
                    | "name"
                    | "op"
                    | "body"
                    | "body_closure"
                    | "env"
                    | "via"
                    | "equiv"
            ) && term_has_var(x, name)
        }),
        Value::Array(xs) => xs.iter().any(|x| term_has_var(x, name)),
        _ => false,
    }
}

/// Does dimension term `dim` occur in face `f`? `Eq0`/`Eq1` arguments
/// are dimension terms — occurrence is structural equality, or a
/// `var{name}` mention inside a compound term when `dim` is itself a
/// variable (the model's `face_has_dim_`).
fn face_has_dim(dim: &Value, f: &CubFace) -> bool {
    match f {
        CubFace::Eq0(d) | CubFace::Eq1(d) => {
            if d == dim {
                return true;
            }
            let dname = if tag_of(dim) == Some("var") {
                dim.get("name").and_then(Value::as_str).unwrap_or("")
            } else {
                ""
            };
            !dname.is_empty() && term_has_var(d, dname)
        }
        CubFace::Not(a) => face_has_dim(dim, a),
        CubFace::Or(a, b) | CubFace::And(a, b) => face_has_dim(dim, a) || face_has_dim(dim, b),
        _ => false,
    }
}

/// Serialize a normalized [`CubFace`] back to the model's
/// `{"__tag__":"face","op","args"}` record shape (ADR-0208 P3) — the
/// `face` field of a `type.glue` value. `op` is `tt`/`ff`/`eq0`/`eq1`/
/// `not`/`or`/`and`; `args` carries the dimension term for the
/// endpoint equations and the sub-faces for the connectives.
fn cub_face_to_val(ctx: &TypecheckContext, f: &CubFace) -> Value {
    fn ser_args(ctx: &TypecheckContext, f: &CubFace) -> Vec<Value> {
        match f {
            CubFace::Eq0(d) | CubFace::Eq1(d) => vec![d.clone()],
            CubFace::Not(a) => vec![cub_face_to_val(ctx, a)],
            CubFace::Or(a, b) | CubFace::And(a, b) => {
                vec![cub_face_to_val(ctx, a), cub_face_to_val(ctx, b)]
            }
            _ => vec![],
        }
    }
    let op = match f {
        CubFace::Tt => "tt",
        CubFace::Ff => "ff",
        CubFace::Eq0(_) => "eq0",
        CubFace::Eq1(_) => "eq1",
        CubFace::Not(_) => "not",
        CubFace::Or(_, _) => "or",
        CubFace::And(_, _) => "and",
    };
    let mut m = Map::new();
    m.insert("op".to_string(), Value::String(op.to_string()));
    m.insert("args".to_string(), Value::Array(ser_args(ctx, f)));
    make_tag(ctx, "face", m)
}

/// The structural fibrancy verdict (ADR-0208 P3) — the model's
/// `fibrant_verdict_` tri-state: `Yes` derives the fibrancy type,
/// `No` is a hard rejection, `Unknown` warns but never fails silently
/// (indexed families, metas, and unhandled formers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FibrantVerdict {
    Yes,
    No,
    Unknown,
}

/// The body a `body_closure` stands for when no AST elaboration is
/// available (dependent unification, ADR-0208 P1): `const_body` /
/// `apply_type` path closures carry their body as data; a plain
/// closure's `body` field (implicit pis) reads directly; AST-backed
/// closures (`body_child_idx`) have no data proxy — `Null`.
fn closure_body_proxy(ty: &Value) -> Value {
    if let Some(bc) = ty.get("body_closure") {
        if let Some(cb) = bc.get("const_body") {
            return cb.clone();
        }
        if let Some(fam) = bc.get("apply_type") {
            return fam.clone();
        }
    }
    ty.get("body").cloned().unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// The `Val`-domain bridge (`veval` → `v*` — phase-1 spec WS-B)
// ---------------------------------------------------------------------------
//
// The judgment ops below speak the *legacy* tagged encoding
// (`{"__tag__": …}` records, `{"__meta__": n}` metas), while the
// canonical cell data is `Val` domain data — `Former{type.hom}`/
// `Former{cell.comp}` applications that reify to the `quote`
// spellings (`__former__`/`__sym__`/`__neutral__`/`__type__`).
// `domain_lift`/`domain_quote` are the two crossings; the tagged
// `__meta__` store and the domain `MetaStore` are disjoint id spaces,
// so metas remap onto fresh domain metas at the boundary (the stores
// never alias).

/// Whether `v` carries a `Val`-domain encoding at its head — the
/// `quote` spellings checker-produced types arrive in.
pub(crate) fn is_domain_encoded(v: &Value) -> bool {
    match v {
        Value::Object(m) => {
            m.contains_key("__former__")
                || m.contains_key("__sym__")
                || m.contains_key("__neutral__")
                || m.contains_key("__type__")
        }
        _ => false,
    }
}

/// Rewrite tagged `{"__meta__": n}` leaves into `{"__neutral__":
/// {"head":"meta","id": m}}` carrying *fresh* domain metas — the two
/// meta stores never alias: a tagged meta crossing the domain bridge
/// mints a real `ctx.vcheck` meta (memoized within one lift so
/// co-occurring metas stay shared).
pub(crate) fn remap_tagged_metas(
    v: &Value,
    map: &mut HashMap<u64, MetaId>,
    ctx: &mut TypecheckContext,
) -> Value {
    if let Some(id) = extract_meta_id(v) {
        let m = *map
            .entry(id)
            .or_insert_with(|| ctx.vcheck.fresh_meta(Val::Universe(0)));
        return serde_json::json!({"__neutral__": {"head": "meta", "id": m, "spine": []}});
    }
    match v {
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, x)| (k.clone(), remap_tagged_metas(x, map, ctx)))
                .collect(),
        ),
        Value::Array(xs) => {
            Value::Array(xs.iter().map(|x| remap_tagged_metas(x, map, ctx)).collect())
        }
        _ => v.clone(),
    }
}

/// Lift a tagged type value into the `Val` domain — `checker::lift`
/// (the `__tag__`/`__former__`/`__sym__`/`__meta__` bridge) after
/// remapping tagged metas onto fresh domain metas.
pub(crate) fn domain_lift(v: &Value, ctx: &mut TypecheckContext) -> Val {
    let mut map = HashMap::new();
    let v = remap_tagged_metas(v, &mut map, ctx);
    checker::lift(ctx.vcheck.lvl(), &handl_value::TaggedValue::from(v))
}

/// Reify a `Val` back to boundary JSON — the `__former__`/`__sym__`/
/// `__neutral__`/`__type__` spellings of `quote`.
pub(crate) fn domain_quote(v: &Val, ctx: &TypecheckContext) -> Value {
    handl_value::tagged_to_value(&handl_value::nbe::quote(
        &ctx.vcheck.metas,
        ctx.vcheck.lvl(),
        v,
    ))
}

/// Whether `v` is a `Former{type.io}` application — either `__former__`
/// spelling (quote's `{"sym", "args"}` or the surface `{"indices"}`).
/// Identity is the seeded-sym *handle* (`Sym::seeded("type.io")` spells
/// its short display name `"io"`, never the full seed — the quoted
/// `__sym__` payload carries `handle` for exactly this compare); the
/// name check is a fallback for spellings that drop the handle.
fn io_headed(v: &Value) -> bool {
    fn sym_matches(s: &Value) -> bool {
        let handle = s
            .get("__sym__")
            .and_then(|x| x.get("handle"))
            .or_else(|| s.get("handle"))
            .and_then(Value::as_u64);
        if let Some(h) = handle {
            // A spelled handle is authoritative — a different sym that
            // happens to display as "io" must not false-positive.
            return h == checker::type_sym("type.io").handle();
        }
        let name = s
            .get("__sym__")
            .and_then(|x| x.get("name"))
            .or_else(|| s.get("name"))
            .and_then(Value::as_str);
        matches!(name, Some("type.io" | "io"))
    }
    v.get("__former__")
        .map(|f| sym_matches(f) || f.get("sym").is_some_and(sym_matches))
        .unwrap_or(false)
}

/// Whether `v` is `type.unknown` — the dynamic top — in any boundary
/// spelling: the tagged `{"__tag__": "type.unknown"}` value, the
/// `__sym__` reification `quote` emits for `Nominal{type.unknown}`, or
/// a `__former__` application headed by the `type.unknown` sym. The
/// handle compare is authoritative (mirroring `io_headed`); the name
/// check is a fallback for spellings that drop the handle.
pub(crate) fn is_unknown_ty(v: &Value) -> bool {
    if tag_of(v) == Some("type.unknown") {
        return true;
    }
    fn sym_matches(s: &Value) -> bool {
        let inner = s.get("__sym__").unwrap_or(s);
        if let Some(h) = inner.get("handle").and_then(Value::as_u64) {
            return h == checker::type_sym("type.unknown").handle();
        }
        matches!(
            inner.get("name").and_then(Value::as_str),
            Some("type.unknown" | "unknown")
        )
    }
    // The bare-sym spelling only counts when `v` actually reifies a
    // sym (`{"__sym__": {…}}`); a tagged record whose `name` field
    // happens to read "unknown" is not the dynamic top.
    if v.get("__sym__").is_some() && sym_matches(v) {
        return true;
    }
    v.get("__former__")
        .map(|f| sym_matches(f) || f.get("sym").is_some_and(sym_matches))
        .unwrap_or(false)
}

/// A call domain a zero-argument `f()` may peel: literally `Unit`
/// (`type.unit`) or the dynamic top (`type.unknown`). Meta domains are
/// *not* unit — `f := (x ~> x); f()` must still report a missing
/// argument rather than silently discarding the binder. `type.null` is
/// deliberately *not* peelable even though it unifies with `type.unit`
/// (T-153): a `Null ~> T` domain asks for the `null` value itself, and
/// `f()` at eval passes zero args — only the explicit `f(null)` form
/// supplies it.
fn nullary_domain(param: &Value) -> bool {
    tag_of(param) == Some("type.unit") || is_unknown_ty(param)
}

/// The canonical `Former{type.hom}` application `Hom(d, a, b)` emitted
/// as boundary JSON (the `__former__` encoding).
fn hom_former(d: Val, a: Val, b: Val, ctx: &TypecheckContext) -> Value {
    domain_quote(
        &Val::Former {
            head: checker::type_sym("type.hom"),
            args: vec![d, a, b],
        },
        ctx,
    )
}

/// A `Dim`-typed domain meta — the symbolic dimension index of an
/// `f<d>`-style dim/axis that is not a literal.
fn dim_meta(ctx: &mut TypecheckContext) -> Val {
    let m = ctx
        .vcheck
        .fresh_meta(Val::Nominal(checker::type_sym("type.Dim")));
    ctx.vcheck.meta_val(m)
}

/// The literal dimension of a lifted `Former{type.hom}` head —
/// `Some(m)` when `v` is `Hom(m, ·, ·)` with a literal `m`.
fn hom_dim_lit(v: &Val) -> Option<u64> {
    if let Val::Former { head, args } = v
        && head.handle() == checker::type_sym("type.hom").handle()
        && args.len() == 3
        && let Val::Const(TaggedValue::Int(m)) = &args[0]
    {
        return u64::try_from(*m).ok();
    }
    None
}

/// Whether a lifted value is a `Hom(m, ·, ·)` cell at dimension `m ≥ 3`.
/// The boundary *law* is checked at every literal `m` since phase-3
/// WS-B (the `n ≤ 3`/`n ≥ 4` caps are gone); this marker keeps the
/// *unification* escape: a conversion failure between higher-cell
/// types is stuck, never a decided failure — higher-cell *terms* are
/// proof-relevant data the law deliberately under-decides.
pub(crate) fn beyond_checked_dim(v: &Val) -> bool {
    hom_dim_lit(v).is_some_and(|m| m >= 3)
}

fn num_val(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn children_of(node: &AstNode) -> Option<&Vec<Value>> {
    node.payload.get("children").and_then(|c| c.as_array())
}

fn child_at(_ast: &Ast, node: &AstNode, i: usize) -> Option<AstRef> {
    children_of(node)?
        .get(i)?
        .as_u64()
        .map(|n| AstRef::new(n as usize))
}

fn node_name(node: &AstNode) -> Option<String> {
    node.payload.as_str().map(str::to_string).or_else(|| {
        node.payload
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

/// Whether `name` is a builtin type-position identifier — the name table
/// `dep_builtin_or_tag` resolves (`Bool`/`Nat`/`IO`/`Type`/`Pi`/…).
/// Injected `with_global` seeds must not shadow these (they are
/// `type.unknown` placeholders for wire names, not real bindings).
fn is_builtin_type_name(name: &str) -> bool {
    matches!(
        name,
        "Int"
            | "Nat"
            | "Number"
            | "Bool"
            | "String"
            | "Str"
            | "Unit"
            | "Null"
            | "IO"
            | "Type"
            | "Universe"
            | "Type0"
            | "List"
            | "Family"
            | "Sig"
            | "Pi"
            | "Sigma"
            | "Forall"
            | "I"
            | "Interval"
    )
}

// ---------------------------------------------------------------------------
// Index-term algebra
// ---------------------------------------------------------------------------

/// Replace solved metas with their solutions (deep). Meta-solution
/// cycles (`?x := ?y`, `?y := ?x` — reachable whenever two unsolved
/// metas unify in both orders, e.g. shared callee metas in nested
/// applications) break to the bare meta rather than recursing — the
/// same `visited` discipline as `Checker::zonk`.
pub(crate) fn zonk_index(v: &Value, ctx: &TypecheckContext) -> Value {
    let mut visited = std::collections::HashSet::new();
    zonk_index_inner(v, ctx, &mut visited)
}

fn zonk_index_inner(
    v: &Value,
    ctx: &TypecheckContext,
    visited: &mut std::collections::HashSet<u64>,
) -> Value {
    if let Some(id) = extract_meta_id(v) {
        if visited.contains(&id) {
            return v.clone();
        }
        if let Some(sol) = ctx.metas.get(id as usize).and_then(|o| o.as_ref()) {
            visited.insert(id);
            let out = zonk_index_inner(sol, ctx, visited);
            visited.remove(&id);
            return out;
        }
        return v.clone();
    }
    if let Value::Object(m) = v {
        if m.len() == 1 {
            return v.clone();
        }
        let mut out = Map::new();
        for (k, x) in m {
            out.insert(k.clone(), zonk_index_inner(x, ctx, visited));
        }
        return Value::Object(out);
    }
    if let Value::Array(xs) = v {
        return Value::Array(
            xs.iter()
                .map(|x| zonk_index_inner(x, ctx, visited))
                .collect(),
        );
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

// ---------------------------------------------------------------------------
// Diagnostic rendering (phase-3 WS-G)
// ---------------------------------------------------------------------------

/// Render a type value for user-facing diagnostics — surface spellings
/// like `Number ~> Bool`, `Vec(n + 1, a)`, `Type_0`, `?m4` instead of
/// the raw `{"__tag__": …}` / `__former__` JSON payloads. Solved tagged
/// metas are zonked first so messages show solutions, not `?m` handles.
pub(crate) fn display_type(v: &Value, ctx: &TypecheckContext) -> String {
    display_inner(&zonk_index(v, ctx), 0)
}

/// Render a type value like [`display_type`], but without a
/// [`TypecheckContext`] — unsolved metas show as `?m{id}` and solved metas
/// are *not* zonked. For tooling hosts (the LSP's hover) that hold a
/// `TypecheckResult`'s `node_types` after the check has finished and have
/// no context to zonk against.
pub fn display_type_unzonked(v: &Value) -> String {
    display_inner(v, 0)
}

/// Zonk one *reported* type value against the final meta stores — the
/// pass applied to `TypecheckResult`'s `typ`/`node_types`/`signature`
/// at the end of `typecheck` (`hnl lsp` v2 hover).
///
/// Two stores, two passes:
///
/// 1. [`zonk_index`] — tagged `{"__meta__": id}` leaves resolve through
///    the tagged `ctx.metas` table.
/// 2. A lift→quote round-trip on every *domain-encoded* subtree —
///    `domain_quote` forces solved `Val::Flex` heads through
///    `ctx.vcheck.metas`, covering metas that solved after the value
///    was quoted. Tagged structure outside domain subtrees keeps its
///    spelling (a `__tag__` value is not lifted just because a leaf
///    is domain-encoded).
///
/// Unsolved metas survive unchanged in both encodings, so `?m` keeps
/// displaying as `?m`.
pub(crate) fn zonk_reported(v: &Value, ctx: &mut TypecheckContext) -> Value {
    let v = zonk_index(v, ctx);
    zonk_domain_walk(&v, ctx)
}

fn zonk_domain_walk(v: &Value, ctx: &mut TypecheckContext) -> Value {
    if is_domain_encoded(v) {
        let lifted = domain_lift(v, ctx);
        return domain_quote(&lifted, ctx);
    }
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), zonk_domain_walk(x, ctx)))
                .collect(),
        ),
        Value::Array(xs) => Value::Array(xs.iter().map(|x| zonk_domain_walk(x, ctx)).collect()),
        _ => v.clone(),
    }
}

/// Recursion guard — a pathological payload (deep nesting) gets a
/// bounded render rather than an unbounded walk.
const DISPLAY_DEPTH: usize = 40;

fn display_list(xs: &[Value], depth: usize) -> String {
    xs.iter()
        .map(|x| display_inner(x, depth + 1))
        .collect::<Vec<_>>()
        .join(", ")
}

fn display_inner(v: &Value, depth: usize) -> String {
    if depth > DISPLAY_DEPTH {
        return "…".to_string();
    }
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        // Integral floats render without the `.0` (index arithmetic is
        // computed in f64 — `n + 1`, not `n + 1.0`).
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.is_finite() && f.fract() == 0.0 => format!("{}", f as i64),
            _ => n.to_string(),
        },
        Value::String(s) => s.clone(),
        Value::Array(xs) => format!("[{}]", display_list(xs, depth)),
        Value::Object(o) => {
            if let Some(id) = extract_meta_id(v) {
                return format!("?m{id}");
            }
            // `quote` spellings — domain `Val`s reified to boundary JSON.
            if let Some(sym) = o.get("__sym__") {
                return sym
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
            }
            if let Some(f) = o.get("__former__") {
                let sym = f.get("sym").unwrap_or(f);
                let name = sym
                    .get("__sym__")
                    .and_then(|s| s.get("name"))
                    .or_else(|| sym.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("?");
                return match f
                    .get("args")
                    .or_else(|| f.get("indices"))
                    .and_then(Value::as_array)
                {
                    Some(xs) if !xs.is_empty() => {
                        format!("{name}({})", display_list(xs, depth))
                    }
                    _ => name.to_string(),
                };
            }
            if let Some(n) = o.get("__neutral__") {
                let base = match n.get("head").and_then(Value::as_str) {
                    Some("rigid") => {
                        format!("#{}", n.get("index").and_then(Value::as_u64).unwrap_or(0))
                    }
                    Some("meta") => {
                        format!("?m{}", n.get("id").and_then(Value::as_u64).unwrap_or(0))
                    }
                    Some("unfold") => n
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unfold")
                        .to_string(),
                    other => other.unwrap_or("?").to_string(),
                };
                return match n.get("spine").and_then(Value::as_array) {
                    Some(xs) if !xs.is_empty() => format!("{base}({})", display_list(xs, depth)),
                    _ => base,
                };
            }
            if let Some(i) = o.get("__type__").and_then(Value::as_u64) {
                return format!("Type_{i}");
            }
            // Legacy tagged type values (`{"__tag__": …}`).
            if let Some(tag) = tag_of(v) {
                return display_tagged(tag, o, depth);
            }
            // A plain record — render fields compactly.
            let fields = o
                .iter()
                .map(|(k, x)| format!("{k}: {}", display_inner(x, depth + 1)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{fields}}}")
        }
    }
}

/// `type.*` tag → its user-facing name via the canonical nominal table
/// (`type.number` → `type.Number` → `Number`); tags the table doesn't
/// know pass through capitalized (`Path` → `Path`).
fn display_tag_name(tag: &str) -> String {
    let seed = checker::canonical_tag_seed(tag);
    let name = seed.strip_prefix("type.").unwrap_or(seed);
    let mut chars = name.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => name.to_string(),
    }
}

/// Render a `{"__tag__": tag, …}` value — the checker-internal index
/// terms (`var`/`Bin`/`Un`/`App`) get infix/prefix spellings; unknown
/// tags fall back to `Name(field: v, …)`.
fn display_tagged(tag: &str, o: &Map<String, Value>, depth: usize) -> String {
    let field = |k: &str| {
        o.get(k)
            .map(|x| display_inner(x, depth + 1))
            .unwrap_or_else(|| "?".to_string())
    };
    match tag {
        "type.function" => {
            let base = format!("{} ~> {}", field("param"), field("body"));
            match o.get("effects") {
                Some(e) => format!("{base} | {}", display_inner(e, depth + 1)),
                None => base,
            }
        }
        "type.pi" | "type.sigma" | "type.forall" | "type.vau" => {
            let name = display_tag_name(tag);
            let var = o
                .get("body_closure")
                .and_then(|b| b.get("var_name"))
                .and_then(Value::as_str)
                .unwrap_or("x");
            let param_field = if tag == "type.forall" {
                "kind"
            } else {
                "param"
            };
            format!("{name}({var}: {})", field(param_field))
        }
        "type.family_app" | "type.family" => {
            let name = o.get("name").and_then(Value::as_str).unwrap_or("?");
            match o.get("args").and_then(Value::as_array) {
                Some(xs) if !xs.is_empty() => format!("{name}({})", display_list(xs, depth)),
                _ => name.to_string(),
            }
        }
        "Path" => format!("Path({}, {})", field("frm"), field("to")),
        "type.eq" => format!("Eq({}, {})", field("left"), field("right")),
        "type.record" => {
            let fields = o
                .get("fields")
                .and_then(Value::as_object)
                .map(|fs| {
                    fs.iter()
                        .map(|(k, x)| format!("{k}: {}", display_inner(x, depth + 1)))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            match o.get("rest") {
                Some(r) => format!("{{{fields}, …|{}}}", display_inner(r, depth + 1)),
                None => format!("{{{fields}}}"),
            }
        }
        "type.array" => match o.get("elements").and_then(Value::as_array) {
            Some(xs) => format!("[{}]", display_list(xs, depth)),
            None => "Array".to_string(),
        },
        "type.effects" => match o.get("ops").and_then(Value::as_array) {
            Some(xs) => format!("effects({})", display_list(xs, depth)),
            None => "effects".to_string(),
        },
        "type.universe" => format!(
            "Type_{}",
            o.get("level").and_then(Value::as_u64).unwrap_or(0)
        ),
        // Index terms.
        "var" => o
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string(),
        "Bin" => {
            let sym = match o.get("op").and_then(Value::as_str) {
                Some("binary.add") => "+",
                Some("binary.sub") => "-",
                Some("binary.mul") => "*",
                Some("binary.div") => "/",
                Some("binary.mod") => "%",
                _ => "?",
            };
            format!("{} {sym} {}", field("l"), field("r"))
        }
        "Un" => format!("-{}", field("operand")),
        "App" => {
            let callee = field("callee");
            match o.get("args").and_then(Value::as_array) {
                Some(xs) if !xs.is_empty() => format!("{callee}({})", display_list(xs, depth)),
                _ => callee,
            }
        }
        _ => {
            let name = display_tag_name(tag);
            let fields = o
                .iter()
                .filter(|(k, _)| !k.starts_with("__") && k.as_str() != "body_closure")
                .map(|(k, x)| format!("{k}: {}", display_inner(x, depth + 1)))
                .collect::<Vec<_>>()
                .join(", ");
            if fields.is_empty() {
                name
            } else {
                format!("{name}({fields})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `data`-declared ctor index projections (phase-3 WS-D)
// ---------------------------------------------------------------------------
//
// A `data` declaration contributes per-ctor index projections: index-level
// `App{C, a⃗}`/`var{C}` rewrites to a declared index-term template inside
// `eval_index`, *before* ring normalization — so `Succ x` reads as
// `x + 1` and `Zero` as `0`, and `Vec(Succ n, a) ≡ Vec(n + 1, a)` holds
// definitionally. The table lives on `TypecheckContext` keyed by ctor
// sym and is populated by `synth_data` from the declaration's *shape*
// (the Nat-former shape: a unary ctor `S(arg : F)` building `F` declares
// successor arithmetic; a unique bare ctor `Z` of such a family declares
// `0`) — never hardcoded in the solver. Ctors without the shape get no
// projection and stay honest `App` atoms: the decided-vs-stuck contract
// is unchanged — `App{F, x}` for unregistered `F` still can't equal
// `x + 1` (it is `Unknown`, not silently accepted).

/// Hole prefix inside a [`CtorProj`] template: `var{__ctor_proj_<i>}`
/// stands for the `i`-th index argument.
const CTOR_PROJ_HOLE: &str = "__ctor_proj_";

/// A declared index projection for a term-level ctor: index-level
/// `C(a₀,…)` rewrites to `template` with `var{__ctor_proj_<i>} ↦ aᵢ`.
#[derive(Debug, Clone)]
pub(crate) struct CtorProj {
    /// Number of index arguments the projection consumes (`Zero` → 0,
    /// `Succ` → 1).
    pub(crate) arity: usize,
    /// Index-term template; `var{__ctor_proj_<i>}` is argument `i`.
    pub(crate) template: Value,
}

/// Instantiate a [`CtorProj`] template: `var{__ctor_proj_<i>} ↦ args[i]`.
fn proj_instantiate(template: &Value, args: &[Value]) -> Value {
    if tag_of(template) == Some("var") {
        if let Some(i) = template
            .get("name")
            .and_then(Value::as_str)
            .and_then(|n| n.strip_prefix(CTOR_PROJ_HOLE))
            .and_then(|s| s.parse::<usize>().ok())
        {
            return args.get(i).cloned().unwrap_or(Value::Null);
        }
        return template.clone();
    }
    match template {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), proj_instantiate(x, args)))
                .collect(),
        ),
        Value::Array(xs) => Value::Array(xs.iter().map(|x| proj_instantiate(x, args)).collect()),
        _ => template.clone(),
    }
}

/// Evaluate an index term: resolve `var{name}` through `results`
/// (self-bindings `var{n} → var{n}` are the rigid fixed point), zonk
/// metas, apply declared ctor projections (`App{C, a⃗}`/`var{C}` for
/// registered `C`), fold closed `Bin`/`Un` arithmetic, canonicalize the
/// rest.
pub(crate) fn eval_index(v: &Value, ctx: &TypecheckContext) -> Value {
    let _t = crate::stats::timer(&crate::stats::T_EVAL_INDEX);
    crate::stats::eval_index();
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
            // `var{n} ↦ t` where `var{n}` occurs inside `t` — a
            // dependent rewrite (`n ↦ var{n} + 1` minted when a
            // `body_closure`'s `var_name` collides with the bound
            // name in the witness term). The inner `var{n}` is the
            // *pre-binding* atom: resolve it opaquely, never re-apply
            // the rewrite — otherwise every `eval` iteration wraps a
            // fresh `+1` (`var{n} + 64` at the fuel bound).
            if var_occurs(name, bound) {
                let opaque = var_term(&format!("__bound_{name}"));
                let shifted = subst_index(bound, &var_term(name), &opaque);
                let evaluated = eval_index_fuel(&shifted, ctx, fuel - 1);
                return subst_index(&evaluated, &opaque, &var_term(name));
            }
            return eval_index_fuel(bound, ctx, fuel - 1);
        }
        // Declared arity-0 ctor projection: a bare index-position `Z`
        // naming a registered nullary ctor (`Zero`) reads as its
        // template (`0`). A `results` binding shadows — checked above.
        if let Some(p) = ctx.ctor_proj.get(name)
            && p.arity == 0
        {
            let t = p.template.clone();
            return eval_index_fuel(&t, ctx, fuel - 1);
        }
        return v.clone();
    }
    if tag_of(v) == Some("App") {
        let callee = v.get("callee").cloned().unwrap_or(Value::Null);
        let args: Vec<Value> = v
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|a| eval_index_fuel(&a, ctx, fuel - 1))
            .collect();
        // Declared ctor projection: `App{C, a⃗}` where `C` registered an
        // `arity`-matching projection rewrites to the instantiated
        // template — `App{Succ, x} ↦ x + 1` — then keeps normalizing.
        // Unregistered heads stay honest `App` atoms (args still
        // evaluate pointwise, so `App{F, 0+1}` canonicalizes to
        // `App{F, 1}`).
        if tag_of(&callee) == Some("var")
            && let Some(cname) = callee.get("name").and_then(Value::as_str)
            && let Some(p) = ctx.ctor_proj.get(cname)
            && p.arity == args.len()
        {
            let t = proj_instantiate(&p.template, &args);
            return eval_index_fuel(&t, ctx, fuel - 1);
        }
        return app_term(callee, args);
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
        return ring_canon(canon_bin(op, l, r));
    }
    if tag_of(v) == Some("Un") {
        let op = v.get("op").and_then(Value::as_str).unwrap_or("");
        let o = eval_index_fuel(v.get("operand").unwrap_or(&Value::Null), ctx, fuel - 1);
        if op == "unary.neg"
            && let Some(n) = num_val(&o)
        {
            return Value::from(-n);
        }
        return ring_canon(un_term(op, o));
    }
    v.clone()
}

// ---------------------------------------------------------------------------
// Ring normal form (phase-2 WS-D)
// ---------------------------------------------------------------------------
//
// Commutative-ring normalization for `Nat`-index terms over
// `add`/`sub`/`mul`/`neg`: every such term rewrites to a *polynomial
// normal form* — a constant plus a sorted sum of monomials, each a
// coefficient times a sorted product of atoms. Atoms are the
// non-arithmetic evaluated subterms (`var{name}`, metas, `App` ctor
// terms, opaque `div`/`mod`/`pow`/`eq` nodes), keyed by a canonical
// string so like terms collect: `a+b+a → a*2+b`, `n+m ≡ m+n`,
// `2*n ≡ n*2`, `n*(m+k) → n*m+n*k`, `n*0 → 0`, `(n-1)+1 → n`.
//
// The rewrite is *decidable, not a rewrite rule*: it runs inside
// `eval_index`, so every consumer (`unify_index`, `dep_unify`, match
// refinement, `closed_num`, `eval_ann_term`) sees normal forms before
// meta-solving. Distribution is bounded (`POLY_*` caps) — a product too
// large to expand keeps the factored `canon_bin` shape instead of
// blowing up; that case is not silent, it just stays unnormalized.

/// Maximum monomials produced by one `mul` distribution.
const POLY_MAX_TERMS: usize = 64;
/// Maximum factors kept inside one monomial.
const POLY_MAX_FACTORS: usize = 16;
/// Maximum `poly_of` recursion depth.
const POLY_MAX_DEPTH: u32 = 24;

/// A monomial: `coeff × Π atoms`, atoms sorted by canonical key.
struct IndexMono {
    coeff: f64,
    atoms: Vec<(String, Value)>,
}

/// A polynomial normal form: canonical-key → monomial (like terms
/// already merged), plus a separate constant summand.
#[derive(Default)]
struct IndexPoly {
    terms: std::collections::BTreeMap<String, IndexMono>,
    constant: f64,
}

/// Canonical atom key — `var`s key by name, unsolved metas by id, and
/// every other evaluated subterm (`App`, `div`/`mod`/`pow`/`eq` nodes)
/// by its deterministic JSON serialization (serde_json `Map` is
/// `BTreeMap` — sorted keys).
fn index_atom_key(v: &Value) -> String {
    if let Some(id) = extract_meta_id(v) {
        return format!("m:{id}");
    }
    if tag_of(v) == Some("var") {
        return format!("v:{}", v.get("name").and_then(Value::as_str).unwrap_or(""));
    }
    format!("j:{}", serde_json::to_string(v).unwrap_or_default())
}

fn poly_atom(v: &Value) -> IndexPoly {
    let key = index_atom_key(v);
    let mut terms = std::collections::BTreeMap::new();
    terms.insert(
        key.clone(),
        IndexMono {
            coeff: 1.0,
            atoms: vec![(key, v.clone())],
        },
    );
    IndexPoly {
        terms,
        constant: 0.0,
    }
}

fn poly_scale(p: &mut IndexPoly, k: f64) {
    if k == 0.0 {
        p.terms.clear();
        p.constant = 0.0;
        return;
    }
    p.constant *= k;
    for m in p.terms.values_mut() {
        m.coeff *= k;
    }
}

fn poly_merge(dst: &mut IndexPoly, src: IndexPoly) {
    dst.constant += src.constant;
    for (k, m) in src.terms {
        dst.terms
            .entry(k)
            .or_insert(IndexMono {
                coeff: 0.0,
                atoms: m.atoms.clone(),
            })
            .coeff += m.coeff;
    }
    dst.terms.retain(|_, m| m.coeff != 0.0);
}

/// Distribute `a * b` over the sums — `(x+y)*(u+v) → xu+xv+yu+yv`.
/// Returns `None` when expansion would exceed the bound; the caller
/// keeps the factored form.
fn poly_mul(a: &IndexPoly, b: &IndexPoly) -> Option<IndexPoly> {
    let a_terms = a.terms.len().max(1);
    let b_terms = b.terms.len().max(1);
    if a_terms * b_terms > POLY_MAX_TERMS {
        return None;
    }
    let mut out = IndexPoly {
        constant: a.constant * b.constant,
        ..Default::default()
    };
    // a.terms * b.constant and a.constant * b.terms
    for (k, m) in &a.terms {
        out.terms.insert(
            k.clone(),
            IndexMono {
                coeff: m.coeff * b.constant,
                atoms: m.atoms.clone(),
            },
        );
    }
    for (k, m) in &b.terms {
        out.terms
            .entry(k.clone())
            .or_insert(IndexMono {
                coeff: 0.0,
                atoms: m.atoms.clone(),
            })
            .coeff += m.coeff * a.constant;
    }
    // a.terms * b.terms — merge atom bags (sorted by key).
    for ma in a.terms.values() {
        for mb in b.terms.values() {
            let mut atoms: Vec<(String, Value)> =
                ma.atoms.iter().chain(mb.atoms.iter()).cloned().collect();
            atoms.sort_by(|x, y| x.0.cmp(&y.0));
            if atoms.len() > POLY_MAX_FACTORS {
                return None;
            }
            let key = atoms
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>()
                .join("*");
            out.terms
                .entry(key)
                .or_insert(IndexMono { coeff: 0.0, atoms })
                .coeff += ma.coeff * mb.coeff;
        }
    }
    out.terms.retain(|_, m| m.coeff != 0.0);
    if out.terms.len() > POLY_MAX_TERMS {
        return None;
    }
    Some(out)
}

/// Parse an *evaluated* index term into its polynomial normal form.
/// `None` = outside the bounded fragment (the term keeps its `canon_bin`
/// shape — still canonical, just not expanded).
fn poly_of(t: &Value, depth: u32) -> Option<IndexPoly> {
    if depth > POLY_MAX_DEPTH {
        return None;
    }
    if let Some(n) = num_val(t) {
        return Some(IndexPoly {
            constant: n,
            ..Default::default()
        });
    }
    // An unsolved meta is an honest atom — `{"__meta__": id}` carries no
    // `__tag__`, so the `tag_of` gate below would drop it from the
    // polynomial entirely and `m + ?k ≡ ?k + m` never normalized.
    if extract_meta_id(t).is_some() {
        return Some(poly_atom(t));
    }
    let tag = tag_of(t)?;
    if tag == "Bin" {
        let op = t.get("op").and_then(Value::as_str).unwrap_or("");
        if !matches!(op, "binary.add" | "binary.sub" | "binary.mul") {
            return Some(poly_atom(t));
        }
        let mut l = poly_of(t.get("l").unwrap_or(&Value::Null), depth + 1)?;
        let mut r = poly_of(t.get("r").unwrap_or(&Value::Null), depth + 1)?;
        return match op {
            "binary.add" => {
                poly_merge(&mut l, r);
                Some(l)
            }
            "binary.sub" => {
                poly_scale(&mut r, -1.0);
                poly_merge(&mut l, r);
                Some(l)
            }
            _ => poly_mul(&l, &r),
        };
    }
    if tag == "Un" {
        let op = t.get("op").and_then(Value::as_str).unwrap_or("");
        if op != "unary.neg" {
            return Some(poly_atom(t));
        }
        let mut o = poly_of(t.get("operand").unwrap_or(&Value::Null), depth + 1)?;
        poly_scale(&mut o, -1.0);
        return Some(o);
    }
    Some(poly_atom(t))
}

/// Render one monomial `coeff × Π atoms` — atoms left-associated
/// `a*b*c`, the coefficient (when ≠ ±1) appended on the right,
/// preserving the `open × closed` convention.
fn mono_term(m: &IndexMono) -> Value {
    let mut term: Option<Value> = None;
    for (_, a) in &m.atoms {
        term = Some(match term {
            None => a.clone(),
            Some(t) => bin_term("binary.mul", t, a.clone()),
        });
    }
    let term = term.unwrap_or(Value::from(1));
    let c = m.coeff.abs();
    if c == 1.0 {
        term
    } else {
        bin_term("binary.mul", term, Value::from(c))
    }
}

fn add_fold(terms: &[Value]) -> Option<Value> {
    terms
        .iter()
        .cloned()
        .reduce(|a, b| bin_term("binary.add", a, b))
}

/// Rebuild a canonical `Bin`/`Un`/numeral term from a polynomial:
/// positive monomials folded with `add`, then the constant summand, then
/// negative monomials subtracted — so `x - y` keeps a `binary.sub`
/// spelling and `2 + x` writes `x + 2`.
fn poly_to_term(p: &IndexPoly) -> Value {
    let mut pos: Vec<Value> = Vec::new();
    let mut neg: Vec<Value> = Vec::new();
    for m in p.terms.values() {
        if m.coeff > 0.0 {
            pos.push(mono_term(m));
        } else if m.coeff < 0.0 {
            neg.push(mono_term(m));
        }
    }
    if p.constant > 0.0 {
        pos.push(Value::from(p.constant));
    } else if p.constant < 0.0 {
        neg.push(Value::from(-p.constant));
    }
    let ps = add_fold(&pos);
    let ns = add_fold(&neg);
    match (ps, ns) {
        (Some(p), Some(n)) => bin_term("binary.sub", p, n),
        (Some(p), None) => p,
        (None, Some(n)) => {
            if neg.len() == 1 {
                un_term("unary.neg", n)
            } else {
                bin_term("binary.sub", Value::from(0), n)
            }
        }
        (None, None) => Value::from(0),
    }
}

/// Rewrite an evaluated `Bin`/`Un` index term into ring normal form.
/// `None` when the term is outside the bounded fragment — the caller
/// keeps the `canon_bin` shape (not silent-accepted, just not expanded).
fn ring_normalize(v: &Value) -> Option<Value> {
    match tag_of(v) {
        Some("Bin") => {
            let op = v.get("op").and_then(Value::as_str).unwrap_or("");
            if !matches!(op, "binary.add" | "binary.sub" | "binary.mul") {
                return None;
            }
        }
        Some("Un") => {
            if v.get("op").and_then(Value::as_str) != Some("unary.neg") {
                return None;
            }
        }
        _ => return None,
    }
    poly_of(v, 0).map(|p| poly_to_term(&p))
}

/// `canon_bin` first (unit laws, folding, commutation), then the ring
/// normal form when the term is in the bounded fragment.
fn ring_canon(v: Value) -> Value {
    ring_normalize(&v).unwrap_or(v)
}

/// The closed numeric value of a term, if it evaluates to a numeral.
fn closed_num(v: &Value, ctx: &TypecheckContext) -> Option<f64> {
    num_val(&eval_index(v, ctx))
}

/// Whether the `{"__meta__": id}` term `needle` occurs anywhere in `v`
/// (guards `instantiate_ctor_full`'s unrecoverable-index check: an
/// argument domain mentioning `?i` pins the family index to it).
fn meta_value_occurs(needle: &Value, v: &Value) -> bool {
    let Some(id) = extract_meta_id(needle) else {
        return false;
    };
    if extract_meta_id(v) == Some(id) {
        return true;
    }
    if let Value::Array(xs) = v {
        return xs.iter().any(|x| meta_value_occurs(needle, x));
    }
    if let Value::Object(m) = v {
        return m.values().any(|x| meta_value_occurs(needle, x));
    }
    false
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
/// (occurs-guarded); anything else is stuck — unless the
/// nonnegativity gap decides it (see [`nonneg_gap_decides`]).
fn solve_index(term: &Value, target: i64, ctx: &mut TypecheckContext) -> Solve {
    match solve_index_cases(term, target, ctx) {
        Solve::Stuck => {
            let t = eval_index(term, ctx);
            if nonneg_gap_decides(&t, target, ctx) {
                Solve::Decided
            } else {
                Solve::Stuck
            }
        }
        r => r,
    }
}

/// The case-analysis body of [`solve_index`] — `Stuck` returns funnel
/// through the wrapper's nonnegativity-gap check.
fn solve_index_cases(term: &Value, target: i64, ctx: &mut TypecheckContext) -> Solve {
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
        // A `Nat`-sorted metavariable can never solve to a negative
        // numeral — decided-unsatisfiable, not `?k := -1` (phase-3
        // WS-C; this is what discharges `VCons: absurd` on `Vec(0)`).
        if ctx.nonneg_metas.contains(&id) && target < 0 {
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
                    return if target == 0 {
                        Solve::Ok
                    } else {
                        Solve::Decided
                    };
                }
                if c.fract() == 0.0 && target % c as i64 == 0 {
                    solve_index(&l, target / c as i64, ctx)
                } else {
                    Solve::Decided
                }
            } else if let Some(c) = ln {
                if c == 0.0 {
                    return if target == 0 {
                        Solve::Ok
                    } else {
                        Solve::Decided
                    };
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

/// Whether an evaluated index *atom* is known nonnegative — a literal
/// `≥ 0`, a `Nat`-sorted metavariable (its family parameter was
/// declared `n : Nat`), or a rigid `var` marked `Nat`-sorted by the
/// match coverage machinery (split vars inherit their ctor-arg sort).
/// Opaque `App`s and unknown-sorted names are honestly *not* nonneg.
fn index_atom_nonneg(v: &Value, ctx: &TypecheckContext) -> bool {
    if let Some(n) = num_val(v) {
        return n >= 0.0;
    }
    if let Some(id) = extract_meta_id(v) {
        return ctx.nonneg_metas.contains(&id);
    }
    if tag_of(v) == Some("var") {
        let name = v.get("name").and_then(Value::as_str).unwrap_or("");
        return ctx.nonneg_index_vars.contains(name);
    }
    false
}

/// `term ≡ target` decided-unsatisfiable by the nonnegativity gap: the
/// term's ring normal form is `c + Σ monomials` where every monomial
/// atom is known-nonneg — all-positive coefficients bound it below by
/// `c` (`term ≥ c > target`), all-negative bound it above
/// (`term ≤ c < target`). This is what decides `m + 1 ≡ 0` for a
/// `Nat`-sorted `m` — the `n ↦ Succ m` split leaf's `VNil` proof —
/// where the pointwise linear fragment stays stuck.
fn nonneg_gap_decides(t: &Value, target: i64, ctx: &TypecheckContext) -> bool {
    let Some(p) = poly_of(t, 0) else {
        return false;
    };
    if p.terms.is_empty() {
        return false;
    }
    let all_nonneg = p
        .terms
        .values()
        .all(|m| m.atoms.iter().all(|(_, a)| index_atom_nonneg(a, ctx)));
    if !all_nonneg {
        return false;
    }
    let pos = p.terms.values().all(|m| m.coeff > 0.0);
    let neg = p.terms.values().all(|m| m.coeff < 0.0);
    let t_f = target as f64;
    (pos && p.constant > t_f) || (neg && p.constant < t_f)
}

/// The `var` head of an `App` index term when the head names a
/// `data`-declared constructor — the index-level ctor-disjointness
/// table key. Unregistered heads (`f` a bound function) stay `None`:
/// `App{f, x}` is an honest atom, never decided against a numeral.
fn app_ctor_name<'a>(v: &'a Value, ctx: &TypecheckContext) -> Option<&'a str> {
    if tag_of(v) != Some("App") {
        return None;
    }
    let callee = v.get("callee")?;
    if tag_of(callee) != Some("var") {
        return None;
    }
    let name = callee.get("name").and_then(Value::as_str)?;
    ctx.ctor_family.contains_key(name).then_some(name)
}

// ---------------------------------------------------------------------------
// `IndexSolver` — bounded index-equality seam (phase-2 WS-D)
// ---------------------------------------------------------------------------

/// Verdict of a bounded `a ≡ b` index-equality query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexVerdict {
    /// Convertible (possibly after solving metas).
    Proven,
    /// Provably unsatisfiable within the decidable fragment — a real
    /// error, never "stuck".
    Disproven,
    /// Outside the decidable fragment — stuck/postpone. `Unknown` is
    /// never silently accepted as equality.
    Unknown,
}

/// Bounded solver interface the `conv` path calls for index equality
/// (phase-2 spec WS-D). Contract for every implementation:
///
/// - **Decidable fragments first**: ring-normalization, then linear
///   integer solving, exact division, integer roots.
/// - **Bounded**: fuel + term-size caps (`eval_index` fuel, `POLY_*`),
///   deterministic — same inputs, same verdict.
/// - **Honest**: decided-unsatisfiable → [`IndexVerdict::Disproven`];
///   undecidable → [`IndexVerdict::Unknown`] (stuck/postpone), never
///   silent-accept. Two symbolic factors (`?n * ?m`) stay `Unknown`.
///
/// The seam is sized so an SMT backend (Z3/CVC class) can slot in later
/// behind the same signature — e.g. as an `OpHandler`/service
/// implementation swapped at this one static, with no call-site changes
/// in `unify_index`/`dep_unify`. Not built here.
trait IndexSolver {
    /// Prove `a ≡ b` on index terms, else decide/stay unknown.
    fn prove(&self, a: &Value, b: &Value, ctx: &mut TypecheckContext) -> IndexVerdict;
}

/// The built-in solver: ring-normalized terms first (identity through
/// [`eval_index`]), metas solve, numerals compare, open terms go through
/// the linear/division/root fragment in [`solve_index`].
struct BuiltinIndexSolver;

impl IndexSolver for BuiltinIndexSolver {
    fn prove(&self, a: &Value, b: &Value, ctx: &mut TypecheckContext) -> IndexVerdict {
        let ea = eval_index(a, ctx);
        let eb = eval_index(b, ctx);
        // Ring normal forms: `n+m ≡ m+n`, `2*n ≡ n*2`, `n*(m+k) ≡ n*m+n*k`
        // hold by normalization *before* any meta-solving.
        if ea == eb {
            return IndexVerdict::Proven;
        }
        if let Some(id) = extract_meta_id(&ea) {
            ctx.solve_meta(id, eb);
            return IndexVerdict::Proven;
        }
        if let Some(id) = extract_meta_id(&eb) {
            ctx.solve_meta(id, ea);
            return IndexVerdict::Proven;
        }
        for (target, term) in [(num_val(&eb), &ea), (num_val(&ea), &eb)] {
            if let Some(t) = target
                && t.fract() == 0.0
            {
                return match solve_index(term, t as i64, ctx) {
                    Solve::Ok => IndexVerdict::Proven,
                    Solve::Decided => IndexVerdict::Disproven,
                    Solve::Stuck => IndexVerdict::Unknown,
                };
            }
        }
        IndexVerdict::Unknown
    }
}

/// The solver the `conv` path uses. Swap this static for an SMT-backed
/// `IndexSolver` when one lands — call sites stay unchanged.
static INDEX_SOLVER: BuiltinIndexSolver = BuiltinIndexSolver;

/// Index-term unification: `a ≡ b` — metas solve, rigids bind through
/// `results`, closed terms compare, open `Bin`s go through the bounded
/// [`IndexSolver`]. Returns `(ok, decided_failure)`.
fn unify_index(a: &Value, b: &Value, ctx: &mut TypecheckContext) -> (bool, bool) {
    let _t = crate::stats::timer(&crate::stats::T_UNIFY_INDEX);
    crate::stats::unify_index(a, b);
    let ea = eval_index(a, ctx);
    let eb = eval_index(b, ctx);
    if ea == eb {
        return (true, false);
    }
    if let Some(id) = extract_meta_id(&ea) {
        // A `Nat`-sorted meta never solves to a negative numeral.
        if ctx.nonneg_metas.contains(&id) && num_val(&eb).is_some_and(|n| n < 0.0) {
            return (false, true);
        }
        ctx.solve_meta(id, eb);
        return (true, false);
    }
    if let Some(id) = extract_meta_id(&eb) {
        if ctx.nonneg_metas.contains(&id) && num_val(&ea).is_some_and(|n| n < 0.0) {
            return (false, true);
        }
        ctx.solve_meta(id, ea);
        return (true, false);
    }
    if tag_of(&ea) == Some("var") {
        let name = ea
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if !name.is_empty() && !var_occurs(&name, &eb) {
            // A `Nat`-sorted rigid var cannot rewrite to a negative
            // numeral — decided-unsatisfiable, not bound.
            if ctx.nonneg_index_vars.contains(name.as_str())
                && num_val(&eb).is_some_and(|n| n < 0.0)
            {
                return (false, true);
            }
            ctx.bind_result(name.clone(), eb.clone());
            ctx.bind_result(format!("eq_{name}"), path_val(ea.clone(), eb));
        }
        return (true, false);
    }
    if tag_of(&eb) == Some("var") {
        let name = eb
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if !name.is_empty() && !var_occurs(&name, &ea) {
            if ctx.nonneg_index_vars.contains(name.as_str())
                && num_val(&ea).is_some_and(|n| n < 0.0)
            {
                return (false, true);
            }
            ctx.bind_result(name.clone(), ea.clone());
            ctx.bind_result(format!("eq_{name}"), path_val(eb.clone(), ea));
        }
        return (true, false);
    }
    // Constructor-injective `App` heads: `C(a⃗) ≡ C(b⃗)` unifies
    // pointwise (`Succ p ≡ Succ ?k` solves `?k := p`). Two *declared*
    // ctor heads that differ — or a declared ctor against a numeral —
    // are decided-impossible (a `__ctor__` index value is never a
    // numeral and two ctors never coincide); unknown heads stay
    // honestly stuck. `Succ`/`Zero`-projected heads never reach here —
    // `eval_index` already rewrote them to arithmetic.
    if tag_of(&ea) == Some("App") && tag_of(&eb) == Some("App") {
        if ea.get("callee") == eb.get("callee") {
            let xs = ea
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let ys = eb
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if xs.len() == ys.len() {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    let (ok, _) = unify_index(x, y, ctx);
                    if !ok {
                        return (false, false);
                    }
                }
                return (true, false);
            }
            // Same head, different arity — decided when the head is a
            // declared ctor (ctor arity is fixed), stuck otherwise.
            if app_ctor_name(&ea, ctx).is_some() {
                return (false, true);
            }
            return (false, false);
        }
        if app_ctor_name(&ea, ctx).is_some() && app_ctor_name(&eb, ctx).is_some() {
            return (false, true);
        }
        return (false, false);
    }
    if app_ctor_name(&ea, ctx).is_some() && num_val(&eb).is_some()
        || app_ctor_name(&eb, ctx).is_some() && num_val(&ea).is_some()
    {
        return (false, true);
    }
    // Indexed-family applications (phase-3 WS-A): `F(i⃗) ≡ F(j⃗)`
    // unifies pointwise — `type.family_app` rides in index position
    // through instance constraints (`Eq (List a)` vs `Eq (List n)`
    // binds `a := n`). Different nominal heads or arities are a
    // decided mismatch — families are nominal, never accidentally
    // equal.
    if tag_of(&ea) == Some("type.family_app") && tag_of(&eb) == Some("type.family_app") {
        if ea.get("name") != eb.get("name") {
            return (false, true);
        }
        let xs = ea
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let ys = eb
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if xs.len() != ys.len() {
            return (false, true);
        }
        for (x, y) in xs.iter().zip(ys.iter()) {
            let (ok, _) = unify_index(x, y, ctx);
            if !ok {
                return (false, false);
            }
        }
        return (true, false);
    }
    // Same-op `Bin` congruence: `?k + 1 ≡ var{m} + 1` — the Sigma
    // destructure's repack (`VCons(x, ys)` at `Vec(var{m} + 1, a)`
    // against `Vec(?k + 1, a)`) — solves pointwise: `?k := var{m}`,
    // `1 ≡ 1`. The operand walk is *speculative*: it commits a
    // unifier, not the only one (`?x + 1 ≡ var{n} + 2` has solutions
    // the pointwise split misses), so it runs under a store mark and
    // rolls back on any failed pair, falling through to the solver.
    if tag_of(&ea) == Some("Bin") && tag_of(&eb) == Some("Bin") {
        let same_op = ea.get("op") == eb.get("op");
        let op = ea.get("op").and_then(Value::as_str).unwrap_or("");
        if same_op && matches!(op, "binary.add" | "binary.sub" | "binary.mul") {
            let la = ea.get("l").cloned().unwrap_or(Value::Null);
            let ra = ea.get("r").cloned().unwrap_or(Value::Null);
            let lb = eb.get("l").cloned().unwrap_or(Value::Null);
            let rb = eb.get("r").cloned().unwrap_or(Value::Null);
            let mark = ctx.store_mark();
            let (okl, _) = unify_index(&la, &lb, ctx);
            let (okr, _) = if okl {
                unify_index(&ra, &rb, ctx)
            } else {
                (false, false)
            };
            if okl && okr {
                return (true, false);
            }
            ctx.store_restore(mark);
        }
    }
    match INDEX_SOLVER.prove(&ea, &eb, ctx) {
        IndexVerdict::Proven => (true, false),
        IndexVerdict::Disproven => (false, true),
        IndexVerdict::Unknown => (false, false),
    }
}

/// A dependent `match` motive: either the explicit `return … with` AST
/// clause, or an inferred `{vars, template}` built from the expected
/// result type (phase-2 WS-C — `match` may drop `return` when an expected
/// type exists).
pub(crate) enum MatchMotive {
    /// The `return … with` clause's AST node index.
    Ast(usize),
    /// Inferred motive — `vars` are the rigid index binders the refined
    /// arm indices substitute into `template` (a type value).
    Inferred {
        /// Motive binder names (`im_<pos>_<name>`), applied pointwise.
        vars: Vec<String>,
        /// The expected type with abstracted positions replaced by
        /// `var{vars[i]}` holes.
        template: Value,
    },
}

// ---------------------------------------------------------------------------
// Split-tree coverage for indexed `match` (phase-3 WS-C) — data types
// ---------------------------------------------------------------------------

/// Split-tree bounds: at most this many nested index-var case-splits
/// deep, and this many leaf probes total, per `match`. Over the bound
/// the undecided obligation degrades to a `coverage unknown` warning —
/// never a silent accept.
const SPLIT_MAX_DEPTH: usize = 3;
const SPLIT_MAX_NODES: usize = 64;

/// The sort of an index term's home position — where a splittable
/// `var` case-splits. `Nat` (declared `Nat`-shaped family or a
/// `nat`-flagged builtin index position) splits `v ↦ 0` /
/// `v ↦ m + 1`; `Family(G)` splits `v ↦ App{K, f⃗}` per ctor of `G`.
#[derive(Debug, Clone)]
enum IndexSort {
    /// Natural-number sort — `n : Nat`.
    Nat,
    /// A `data`-declared family's name — its ctors are the cases.
    Family(String),
}

/// How one match arm participates in coverage.
#[derive(Debug)]
enum ArmCover {
    /// `C(p⃗)`/`C` where `C` is a ctor of the scrutinee's family.
    Ctor(String),
    /// `_` or a binding pattern — covers every ctor.
    Catchall,
    /// `absurd` — discharges a decided-impossible ctor; covers none.
    Absurd,
    /// Any other shape (literal, sequence, foreign ctor) — covers
    /// no ctor of this family.
    Other,
}

/// A ctor's reachability at one split-tree leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtorStatus {
    /// Index equations unify under the leaf substitution.
    Possible,
    /// Decided-unsatisfiable — cannot occur here.
    Impossible,
    /// Stuck — a split deeper may still decide it.
    Undecided,
}

/// Mutable state for one match's bounded split-tree traversal.
struct CoverRun {
    /// Leaf nodes visited (bounded by [`SPLIT_MAX_NODES`]).
    nodes: usize,
    /// Fresh split-var counter (`__split_<n>` names).
    fresh: usize,
    /// `var` → declared index sort, including split-introduced vars.
    sorts: HashMap<String, IndexSort>,
    /// `(ctor, witness path)` — reachable-but-uncovered, first leaf.
    uncovered: Vec<(String, String)>,
    /// Ctors that stayed undecided-uncovered at a final leaf.
    warned: HashSet<String>,
    /// A leaf hit a split bound (depth or nodes).
    bound_hit: bool,
}

/// The `var` head of a match arm pattern and its bound field names —
/// `var`/`call` pattern shapes only; `None` head for `_` and
/// non-ctor-shaped patterns.
fn match_pattern_head(ast: &Ast, pattern: &AstNode) -> (Option<String>, Vec<String>) {
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
                .and_then(|n| if n.kind == "var" { node_name(n) } else { None });
            let names: Vec<String> = pk
                .iter()
                .skip(1)
                .filter_map(Value::as_u64)
                .filter_map(|i| {
                    ast.get(AstRef::new(i as usize))
                        .and_then(|n| (n.kind == "var").then(|| node_name(n))?)
                })
                .collect();
            (cname, names)
        }
        _ => (None, vec![]),
    }
}

/// Collect the `var` names occurring in an evaluated index term.
fn collect_index_vars(v: &Value, out: &mut std::collections::BTreeSet<String>) {
    if tag_of(v) == Some("var") {
        if let Some(n) = v.get("name").and_then(Value::as_str) {
            out.insert(n.to_string());
        }
        return;
    }
    match v {
        Value::Object(m) => m.values().for_each(|x| collect_index_vars(x, out)),
        Value::Array(xs) => xs.iter().for_each(|x| collect_index_vars(x, out)),
        _ => {}
    }
}

/// Whether an index-projection template is the successor shape —
/// `hole + k` or `k + hole` with `k > 0` — as minted by `synth_data`
/// for a unary same-family ctor (`Succ ↦ λx. x + 1`).
fn proj_is_successor(t: &Value) -> bool {
    if tag_of(t) != Some("Bin") || t.get("op").and_then(Value::as_str) != Some("binary.add") {
        return false;
    }
    let is_hole = |v: &Value| {
        tag_of(v) == Some("var")
            && v.get("name")
                .and_then(Value::as_str)
                .is_some_and(|n| n.starts_with(CTOR_PROJ_HOLE))
    };
    let l = t.get("l").unwrap_or(&Value::Null);
    let r = t.get("r").unwrap_or(&Value::Null);
    (is_hole(l) && num_val(r).is_some_and(|n| n > 0.0))
        || (is_hole(r) && num_val(l).is_some_and(|n| n > 0.0))
}

/// Whether `g`'s ctor set is fully index-projected as naturals — every
/// ctor carries a `CtorProj` that is either a nonneg numeral (the bare
/// base) or a successor step. `data Nat :: Zero | Succ(pred : Nat)` is
/// the exemplar; the test reads declared metadata, never the name.
fn nat_shaped_family(g: &str, ctx: &TypecheckContext) -> bool {
    let Some(ctors) = ctx.family_ctors.get(g) else {
        return false;
    };
    !ctors.is_empty()
        && ctors.iter().all(|c| {
            ctx.ctor_proj.get(c).is_some_and(|p| match p.arity {
                0 => num_val(&p.template).is_some_and(|n| n >= 0.0),
                _ => proj_is_successor(&p.template),
            })
        })
}

/// Whether a sort marks its vars nonnegative (`Nat` outright, or a
/// declared family whose ctors all project as naturals).
fn index_sort_nonneg(sort: &IndexSort, ctx: &TypecheckContext) -> bool {
    match sort {
        IndexSort::Nat => true,
        IndexSort::Family(g) => nat_shaped_family(g, ctx),
    }
}

/// The numeric payload of a `value`/`lit` node — unwraps the
/// `{"value": x}` literal-payload envelope literal nodes can carry.
fn literal_payload_num(node: &AstNode) -> Option<f64> {
    let raw = match &node.payload {
        Value::Object(m) if m.get("__tag__").is_none() => m.get("value")?,
        p => p,
    };
    raw.as_f64()
}

/// `true` when `node` is a non-negative *integer* literal — a `value`/`lit`
/// node with an integral numeric payload (`paren`-transparent). This is
/// the surface spelling of a `Succ^k(Zero)` index term: `idn(2)`
/// elaborates exactly like `idn(Succ(Succ(Zero)))`. Fractionals and
/// negatives keep the ordinary `Number` typing.
pub(crate) fn nat_literal_node(node: &AstNode, ast: &Ast) -> bool {
    match node.kind.as_str() {
        "value" | "lit" => literal_payload_num(node).is_some_and(|n| n >= 0.0 && n.fract() == 0.0),
        "paren" => child_at(ast, node, 0)
            .and_then(|i| ast.get(i))
            .is_some_and(|inner| nat_literal_node(inner, ast)),
        _ => false,
    }
}

/// `expected` (already zonked) is a *parameterless* Nat-shaped family —
/// `type.family_app`/`type.family` naming a family whose ctors all carry
/// numeric index projections (`nat_shaped_family`) and which takes no
/// index arguments. Only there is the literal's elaborated type
/// index-independent (`Zero` and every `Succ^k(Zero)` share the same
/// `F`), so `2` reads as `Succ²(Zero)` with no index-unification
/// obligation. An indexed Nat-shaped family would need the literal's
/// own index computed — out of scope, stays `Number`.
pub(crate) fn nat_family_expected(expected: &Value, ctx: &TypecheckContext) -> bool {
    let (name, n_args) = match tag_of(expected) {
        Some("type.family_app") => (
            expected.get("name").and_then(Value::as_str),
            expected
                .get("args")
                .and_then(Value::as_array)
                .map(|a| a.len())
                .unwrap_or(0),
        ),
        Some("type.family") => (
            expected.get("name").and_then(Value::as_str),
            expected
                .get("base")
                .and_then(|b| b.get("elements"))
                .and_then(Value::as_array)
                .map(|e| e.len())
                .unwrap_or(0),
        ),
        _ => (None, usize::MAX),
    };
    n_args == 0 && name.is_some_and(|n| nat_shaped_family(n, ctx))
}

/// The declared-index-sort descriptor for one `type.family` position:
/// the position's evaluated sort value + the family's `nat` flag (the
/// `n : Nat` annotation convention — needed because `Nat` evaluates to
/// `type.number` when no `data Nat` shadows it).
fn position_index_sort(
    elem: Option<&Value>,
    nat_flag: bool,
    ctx: &TypecheckContext,
) -> Option<IndexSort> {
    let e = elem?;
    match tag_of(e) {
        Some("type.family_app") | Some("type.family") => {
            let name = e.get("name").and_then(Value::as_str)?;
            ctx.family_ctors
                .contains_key(name)
                .then(|| IndexSort::Family(name.to_string()))
        }
        Some("type.number") => nat_flag.then_some(IndexSort::Nat),
        _ => None,
    }
}

/// The sort of a fresh split var for ctor arg `i` — the arg's
/// instantiated domain when it names a declared family, else the
/// `arg : Nat` annotation convention on the raw `type.number` sort.
fn ctor_arg_index_sort(
    domain: &Value,
    arg_node: Option<&AstNode>,
    ctx: &TypecheckContext,
) -> Option<IndexSort> {
    match tag_of(domain) {
        Some("type.family_app") | Some("type.family") => {
            let name = domain.get("name").and_then(Value::as_str)?;
            ctx.family_ctors
                .contains_key(name)
                .then(|| IndexSort::Family(name.to_string()))
        }
        Some("type.number") => arg_node
            .and_then(node_name)
            .is_some_and(|n| n == "Nat")
            .then_some(IndexSort::Nat),
        _ => None,
    }
}

/// Whether the arms cover `ctor` — a `C`-headed non-absurd arm, or any
/// catchall.
fn arm_covers(arms: &[ArmCover], ctor: &str) -> bool {
    arms.iter().any(|a| match a {
        ArmCover::Ctor(c) => c == ctor,
        ArmCover::Catchall => true,
        _ => false,
    })
}

/// Pick the split var: a `var` occurring in an undecided scrutinee
/// index position whose sort the census knows how to case-split.
/// Deterministic (sorted by name) for reproducible diagnostics.
fn pick_split_var(
    eargs: &[Value],
    undecided_positions: &[usize],
    sorts: &HashMap<String, IndexSort>,
) -> Option<(String, IndexSort)> {
    let mut vars = std::collections::BTreeSet::new();
    for &i in undecided_positions {
        if let Some(t) = eargs.get(i) {
            collect_index_vars(t, &mut vars);
        }
    }
    vars.into_iter()
        .find_map(|v| sorts.get(&v).map(|s| (v, s.clone())))
}

/// Render a leaf's accumulated substitution for diagnostics —
/// `n ↦ 0, m ↦ k + 1`-style context on the non-exhaustive error.
fn render_split_path(path: &[(String, String)]) -> String {
    if path.is_empty() {
        return String::new();
    }
    let s = path
        .iter()
        .map(|(v, t)| format!("{v} ↦ {t}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(" under {s}")
}

/// The name a `Former{var, [n]}` value carries — the domain reification
/// of `var_term(n)` (`domain_lift` of an index var). `IO(s)` sigs quoted
/// into `Former{type.io}.args` carry `s` that way.
fn former_var_name(v: &Value) -> Option<&str> {
    let f = v.get("__former__")?;
    let sym = f.get("sym")?;
    let name = sym
        .get("__sym__")
        .and_then(|s| s.get("name"))
        .or_else(|| sym.get("name"))
        .and_then(Value::as_str)?;
    if name != "var" {
        return None;
    }
    f.get("args")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
}

/// Deep-substitute `to` for `frm` inside an index term (transport).
fn subst_index(term: &Value, frm: &Value, to: &Value) -> Value {
    if term == frm {
        return to.clone();
    }
    // `var{n}` inside a domain-encoded former — the `Former{var,[n]}`
    // spelling the `var_term` shape lifts/round-trips to (e.g. an
    // `IO(s)` sig var inside `Former{type.io}.args`).
    if tag_of(frm) == Some("var")
        && former_var_name(term) == frm.get("name").and_then(Value::as_str)
    {
        return to.clone();
    }
    match term {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                // Proof-relevant provenance keys (`via`/`equiv`, ADR-0208
                // P4) are never rewritten — the model's `subst_index_`
                // skip list.
                if matches!(k.as_str(), "via" | "equiv") {
                    out.insert(k.clone(), v.clone());
                    continue;
                }
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
    // A codomain spelled as the bare family name (`C : (a⃗) ~> F`) names
    // the family with no index vector — the record-spelling equivalent of
    // the retired `data F :: C(a⃗)` result-less ctor. Normalize it to
    // `result: None` so the unrecoverable-index `lit null` convention
    // applies instead of synthesizing the bare `var F` (which reads as
    // the family *type*, not an indexed application, and would unify
    // with any `F(i⃗)`).
    for c in &mut ctors {
        if let Some(ri) = c.result
            && let Some(n) = ast.get(AstRef::new(ri))
            && n.kind == "var"
            && node_name(n).as_deref() == Some(name.as_str())
        {
            c.result = None;
        }
    }
    Some((name, params, ctors))
}

fn parse_family_head(
    lang: &Language,
    node: &AstNode,
    ast: &Ast,
) -> Option<(String, Vec<DepParam>)> {
    match node.kind.as_str() {
        "var" => Some((node.payload.as_str()?.to_string(), Vec::new())),
        // `dimension` shares the `[target, args…]` layout (index-position
        // `<>` — a dimension IS an index): `data Vec<n, a>` parameterizes
        // the family head exactly like the retired `Vec{n, a}` brace form.
        "call" | "index_app" | "dimension" => {
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
                    if let Some(n) = ast.get(AstRef::new(c.as_u64().unwrap_or(u64::MAX) as usize)) {
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
            if let Some(ci) = cons_idx
                && let Some(cn) = ast.get(AstRef::new(ci))
            {
                // Peel nested `type-annot` wrappers down to the ctor
                // leaf: the record-lowered `C : F(i⃗)` spelling emits
                // `type-annot(type-annot(C, R), R, name)` — the outer
                // `annot` is the declared result, the innermost
                // `subject` is the `var`/`call` leaf consumers read
                // (same descent `tool-typecheck`'s
                // `collect_data_ctors` performs).
                let mut leaf = cn;
                while leaf.kind == "type-annot" {
                    let subj = lang
                        .child_role_index(&leaf.kind, "subject")
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                    let inner = children_of(leaf)
                        .and_then(|k| k.get(subj).and_then(Value::as_u64))
                        .map(|v| v as usize);
                    match inner.and_then(|i| ast.get(AstRef::new(i))) {
                        Some(n) => leaf = n,
                        None => break,
                    }
                }
                collect_ctor_leaf(leaf, ast, out, result_idx, false);
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
            let Some(kids) = children_of(node) else {
                return;
            };
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
    if let Some(b) = node.payload.get("body").and_then(Value::as_u64)
        && let Some(n) = ast.get(AstRef::new(b as usize))
    {
        return occurs_name(ast, n, name);
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
                    n.kind == "var"
                        && (n.payload.as_str() == Some("->") || n.payload.as_str() == Some("~>"))
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
        let _t = crate::stats::timer(&crate::stats::T_DEP_SYNTH_TYPE);
        crate::stats::dep_synth_type(idx);
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            let node = ast.get(AstRef::new(idx)).ok_or_else(|| {
                LangError::engine(format!("dep_synth_type: node {} out of bounds", idx))
            })?;
            match node.kind.as_str() {
                "paren" => {
                    let kids = children_of(node)
                        .ok_or_else(|| LangError::engine("dep_synth_type: empty paren"))?;
                    if kids.len() > 1 {
                        // Dependent pair `(x : A, y : B(x), …)` — every
                        // child a typed name binder (`type-annot(var, T)`)
                        // and some later annotation mentioning an earlier
                        // binder name mints a right-nested `type.sigma`
                        // chain (the same value `Sigma(A, x ~> Sigma(B,
                        // y ~> …))` produces; `[w, v]`/`[w, [v, u]]`
                        // literals check through `dep_intro_sigma`).
                        // Non-dependent binder lists and mixed/untyped
                        // children keep the `type.array{elements}`
                        // product convention below.
                        let binder_nodes: Vec<usize> = kids
                            .iter()
                            .filter_map(Value::as_u64)
                            .map(|v| v as usize)
                            .collect();
                        let binders: Option<Vec<(String, usize)>> = binder_nodes
                            .iter()
                            .map(|&i| Self::typed_binder(lang, ast, i))
                            .collect();
                        if let Some(bs) = binders {
                            // Dependence: binder `j`'s annotation mentions
                            // an earlier binder name (conservative —
                            // `occurs_name` ignores binder shadowing,
                            // which only over-approximates toward sigma,
                            // and a constant-body sigma still accepts the
                            // same `[w, v]` intro).
                            let dependent = bs.iter().enumerate().any(|(j, (_, a))| {
                                j > 0
                                    && bs[..j].iter().any(|(n, _)| {
                                        ast.get(AstRef::new(*a))
                                            .is_some_and(|an| occurs_name(ast, an, n))
                                    })
                            });
                            if dependent {
                                let param =
                                    self.dep_synth_type(lang, ast, bs[0].1, ctx)?;
                                let rest: Vec<Value> = bs[1..]
                                    .iter()
                                    .map(|(n, a)| {
                                        serde_json::json!({"name": n, "annot": a})
                                    })
                                    .collect();
                                let last_annot = bs.last().map(|(_, a)| *a).unwrap_or(0);
                                let mut m = Map::new();
                                m.insert("param".to_string(), param);
                                m.insert(
                                    "body_closure".to_string(),
                                    serde_json::json!({
                                        "var_name": bs[0].0,
                                        "rest": rest,
                                        "chain": "sigma",
                                        // Compat: readers that only know
                                        // `body_child_idx` see the last
                                        // component's annotation.
                                        "body_child_idx": last_annot,
                                        "dep": true,
                                    }),
                                );
                                return Ok(make_tag(ctx, "type.sigma", m));
                            }
                        }
                        // Tuple position `(A, B)` — the `type.array{elements}`
                        // convention `synth_function` mints for `(x, y) ~>`
                        // domains.
                        let mut els = Vec::with_capacity(kids.len());
                        for k in kids {
                            let i = k.as_u64().ok_or_else(|| {
                                LangError::engine("dep_synth_type: paren child idx")
                            })? as usize;
                            els.push(self.dep_synth_type(lang, ast, i, ctx)?);
                        }
                        let mut m = Map::new();
                        m.insert("elements".to_string(), Value::Array(els));
                        return Ok(make_tag(ctx, "type.array", m));
                    }
                    let inner = child_at(ast, node, 0)
                        .ok_or_else(|| LangError::engine("dep_synth_type: empty paren"))?;
                    self.dep_synth_type(lang, ast, inner.index(), ctx)
                }
                "type-annot" => {
                    let annot = role_child_ref(lang, node, "annot")
                        .or_else(|| child_at(ast, node, 1))
                        .ok_or_else(|| {
                            LangError::engine("dep_synth_type: type-annot missing annot")
                        })?;
                    self.dep_synth_type(lang, ast, annot.index(), ctx)
                }
                "value" | "lit" => Ok(node.payload.clone()),
                "var" => self.dep_synth_var_type(lang, ast, node, ctx),
                "field-access" => {
                    let kids = children_of(node).ok_or_else(|| {
                        LangError::engine("dep_synth_type: field-access no children")
                    })?;
                    let field_idx = kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
                        LangError::engine("dep_synth_type: field-access missing field child")
                    })? as usize;
                    let field = node_name(ast.get(AstRef::new(field_idx)).ok_or_else(|| {
                        LangError::engine("dep_synth_type: field node out of bounds")
                    })?)
                    .ok_or_else(|| LangError::engine("dep_synth_type: field name missing"))?;
                    // `type.Name` — a declared alias spelled through the
                    // type namespace (`type.Name := body`); expands
                    // transparently when Γ-bound, else the builtin table.
                    if let Some(bound) = ctx.lookup(&field).cloned()
                        && tag_of(&bound) == Some("type.alias")
                    {
                        return self.expand_alias(lang, ast, &bound, &[], ctx);
                    }
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
                    let _kids = children_of(node)
                        .ok_or_else(|| LangError::engine("dep_synth_type: function no children"))?;
                    let params_idx = role_child_ref(lang, node, "params")
                        .or_else(|| child_at(ast, node, 0))
                        .ok_or_else(|| {
                            LangError::engine("dep_synth_type: function missing params")
                        })?;
                    let body_idx = role_child_ref(lang, node, "body")
                        .or_else(|| child_at(ast, node, 1))
                        .ok_or_else(|| {
                            LangError::engine("dep_synth_type: function missing body")
                        })?;
                    // Typed-param arrow `(x : A, …) ~> B` in *type*
                    // position is a dependent arrow — the same
                    // `type.pi` `Pi(A, x ~> B)` mints: the domain reads
                    // each binder's `type-annot`, the names bind the
                    // lazy `body_closure` (`rest` chain for multiple
                    // binders). Untyped params keep the `type.function`
                    // reading below. Term position is unaffected —
                    // `synth_node` handles lambdas there.
                    let param_nodes: Vec<usize> = if let Some(p) = ast.get(params_idx) {
                        if p.kind == "paren" {
                            children_of(p)
                                .map(|k| {
                                    k.iter()
                                        .filter_map(Value::as_u64)
                                        .map(|v| v as usize)
                                        .collect()
                                })
                                .unwrap_or_default()
                        } else {
                            vec![params_idx.index()]
                        }
                    } else {
                        Vec::new()
                    };
                    let binders: Option<Vec<(String, usize)>> = if param_nodes.is_empty() {
                        None
                    } else {
                        param_nodes
                            .iter()
                            .map(|&i| Self::typed_binder(lang, ast, i))
                            .collect()
                    };
                    if let Some(bs) = binders {
                        if !bs.is_empty() {
                            let param = self.dep_synth_type(lang, ast, bs[0].1, ctx)?;
                            let rest: Vec<Value> = bs[1..]
                                .iter()
                                .map(|(n, a)| serde_json::json!({"name": n, "annot": a}))
                                .collect();
                            let mut m = Map::new();
                            m.insert("param".to_string(), param);
                            m.insert(
                                "body_closure".to_string(),
                                serde_json::json!({
                                    "var_name": bs[0].0,
                                    "rest": rest,
                                    "chain": "pi",
                                    "body_child_idx": body_idx.index(),
                                    "dep": true,
                                }),
                            );
                            return Ok(make_tag(ctx, "type.pi", m));
                        }
                    }
                    // The params child is a type expression — a `paren`-wrapped
                    // list takes its last child as the domain (the `A -> B`
                    // convention); any other node (`call` former
                    // applications like `Vec(n, a)`/`IO(sig)`, `var`, …) IS
                    // the domain.
                    let dom_idx = if let Some(p) = ast.get(params_idx) {
                        if p.kind == "paren" {
                            children_of(p)
                                .and_then(|k| k.last())
                                .and_then(Value::as_u64)
                                .map(|i| AstRef::new(i as usize))
                                .unwrap_or(params_idx)
                        } else {
                            params_idx
                        }
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
                // `dimension` (`F<i⃗>`, the `<>` index-application spelling)
                // elaborates through the same call-type path: `Vec<n, a>` is
                // the former application `Vec{n, a}` used to spell.
                "call" | "index_app" | "dimension" => {
                    self.dep_synth_call_type(lang, ast, node, ctx)
                }
                // `glue A, φ, T̂` in annotation position (ADR-0208 P3):
                // the glue type former — the same formation check the
                // `synth_glue` judgment runs in term position, yielding
                // the `type.glue{family, face, section}` it forms.
                "glue" => {
                    let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
                    if kids_len < 3 {
                        self.cub_diag(
                            ctx,
                            "error",
                            "glue: needs family, face, and section operands".to_string(),
                            node.span,
                        );
                        return Ok(self.cub_open_meta(ctx));
                    }
                    let (Some(fam_ref), Some(face_ref), Some(sec_ref)) = (
                        Self::cub_child(lang, ast, node, "family", 0),
                        Self::cub_child(lang, ast, node, "face", 1),
                        Self::cub_child(lang, ast, node, "section", 2),
                    ) else {
                        self.cub_diag(
                            ctx,
                            "error",
                            "glue: needs family, face, and section operands".to_string(),
                            node.span,
                        );
                        return Ok(self.cub_open_meta(ctx));
                    };
                    let v = self.dep_synth_glue_type(
                        lang,
                        ast,
                        fam_ref.index(),
                        face_ref.index(),
                        sec_ref.index(),
                        node.span,
                        ctx,
                    )?;
                    // A degraded formation (`type.unknown`) reads as an
                    // open meta in type position — see `cub_open_meta`.
                    Ok(if tag_of(&v) == Some("type.unknown") {
                        self.cub_open_meta(ctx)
                    } else {
                        v
                    })
                }
                // `univalence(e)` in annotation position (ADR-0208 P4):
                // the same path-carrying `type.pi` the keyword's term
                // judgment mints — `ua(e)`'s type-position spelling.
                "univalence" => {
                    let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
                    let e_ref = Self::cub_child(lang, ast, node, "equiv", 0);
                    if kids_len < 1 || e_ref.is_none() {
                        self.cub_diag(
                            ctx,
                            "error",
                            "univalence: needs an equivalence operand".to_string(),
                            node.span,
                        );
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        let v = self.synth_univalence_core(
                            lang,
                            ast,
                            e_ref.unwrap().index(),
                            node.span,
                            ctx,
                        )?;
                        Ok(if tag_of(&v) == Some("type.unknown") {
                            self.cub_open_meta(ctx)
                        } else {
                            v
                        })
                    }
                }
                // `fibrant A` in annotation position: the fibrancy
                // *type* `isFibrant(A)` — `type.fibrant{of}` on a `yes`
                // verdict, `type.unknown` + warning otherwise (the
                // model's `synth_fibrant_former_`).
                "fibrant" => {
                    let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
                    let t_ref = Self::cub_child(lang, ast, node, "target", 0);
                    if kids_len < 1 || t_ref.is_none() {
                        self.cub_diag(
                            ctx,
                            "warning",
                            "synth_type: fibrant needs a type operand".to_string(),
                            node.span,
                        );
                        return Ok(self.cub_open_meta(ctx));
                    }
                    self.dep_synth_fibrant_former(lang, ast, t_ref.unwrap().index(), node.span, ctx)
                }
                // `forall b, T` / `exists b, T` — parse-level `Pi`/`Sigma`
                // aliases (phase-2 WS-C sugar; `exists` is the existential
                // twin). The binder child is `type-annot(var, A)` or a bare
                // `var` (unsorted — a fresh meta domain), the body child the
                // type expression. Mints the *same* `type.pi{param,
                // body_closure}` `Pi(A, b ~> T)` / `type.sigma{param,
                // body_closure}` `Sigma(A, b ~> T)` produces — no new
                // semantics.
                kind @ ("forall" | "exists") => {
                    let (tag, kw) = if kind == "exists" {
                        ("type.sigma", "exists")
                    } else {
                        ("type.pi", "forall")
                    };
                    let kids = children_of(node).ok_or_else(|| {
                        LangError::engine(format!("dep_synth_type: {kw} has no children"))
                    })?;
                    if kids.len() < 2 {
                        return Err(LangError::engine(format!(
                            "dep_synth_type: {kw} needs [binder, body]"
                        )));
                    }
                    let binder_idx = kids[0].as_u64().ok_or_else(|| {
                        LangError::engine(format!("dep_synth_type: {kw} binder idx"))
                    })? as usize;
                    let body_idx = kids[1].as_u64().ok_or_else(|| {
                        LangError::engine(format!("dep_synth_type: {kw} body idx"))
                    })? as usize;
                    let bnode = ast.get(AstRef::new(binder_idx)).ok_or_else(|| {
                        LangError::engine(format!("dep_synth_type: {kw} binder out of bounds"))
                    })?;
                    let (var_name, param) = if bnode.kind == "type-annot"
                        || role_child_ref(lang, bnode, "annot").is_some()
                    {
                        let bk = children_of(bnode).cloned().unwrap_or_default();
                        let subj = lang
                            .child_role_index(&bnode.kind, "subject")
                            .ok()
                            .flatten()
                            .unwrap_or(0);
                        let annot = lang
                            .child_role_index(&bnode.kind, "annot")
                            .ok()
                            .flatten()
                            .unwrap_or(1);
                        let vn = bk
                            .get(subj)
                            .and_then(Value::as_u64)
                            .and_then(|i| ast.get(AstRef::new(i as usize)))
                            .and_then(node_name)
                            .ok_or_else(|| {
                                LangError::engine(format!(
                                    "dep_synth_type: {kw} binder is not a name"
                                ))
                            })?;
                        let ty_idx = bk.get(annot).and_then(Value::as_u64).ok_or_else(|| {
                            LangError::engine(format!("dep_synth_type: {kw} binder missing sort"))
                        })? as usize;
                        (vn, self.dep_synth_type(lang, ast, ty_idx, ctx)?)
                    } else {
                        let vn = node_name(bnode).ok_or_else(|| {
                            LangError::engine(format!("dep_synth_type: {kw} binder is not a name"))
                        })?;
                        (vn, meta_value(ctx.fresh_meta()))
                    };
                    let mut m = Map::new();
                    m.insert("param".to_string(), param);
                    m.insert(
                    "body_closure".to_string(),
                    serde_json::json!({"var_name": var_name, "body_child_idx": body_idx, "dep": true}),
                );
                    Ok(make_tag(ctx, tag, m))
                }
                "dim_arrow" => {
                    // `A ~[k]~> B` in annotation position — `Hom(k, A, B)`
                    // (ADR-0153). The endpoints are *type expressions* —
                    // unlike `synth_dim_arrow` (term position, `synth_node`
                    // on operands) — and a src of `type-annot(f, A)` (the
                    // `f : A ~[k]~> B` assign-annot shape) unwraps to `A`
                    // through the `type-annot` arm above, so the bound
                    // name never enters the type.
                    let kids = children_of(node).ok_or_else(|| {
                        LangError::engine("dep_synth_type: dim_arrow no children")
                    })?;
                    if kids.len() < 3 {
                        return Err(LangError::engine(
                            "dep_synth_type: dim_arrow missing operands",
                        ));
                    }
                    let src_idx = kids[0]
                        .as_u64()
                        .ok_or_else(|| LangError::engine("dep_synth_type: dim_arrow src index"))?
                        as usize;
                    let dim_idx = kids[1]
                        .as_u64()
                        .ok_or_else(|| LangError::engine("dep_synth_type: dim_arrow dim index"))?
                        as usize;
                    let tgt_idx = kids[2]
                        .as_u64()
                        .ok_or_else(|| LangError::engine("dep_synth_type: dim_arrow tgt index"))?
                        as usize;
                    let src = self.dep_synth_type(lang, ast, src_idx, ctx)?;
                    let tgt = self.dep_synth_type(lang, ast, tgt_idx, ctx)?;
                    let dim_node = ast.get(AstRef::new(dim_idx)).ok_or_else(|| {
                        LangError::engine("dep_synth_type: dim node out of bounds")
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
                    let dim_v = if dim_inner.kind == "value" {
                        match dim_val.as_i64() {
                            Some(n) => Val::Const(TaggedValue::Int(n)),
                            None => dim_meta(ctx),
                        }
                    } else {
                        // A scope-bound dimension variable (`f<N> : T`
                        // binder, ADR-0106 §12) elaborates as its rigid
                        // index term — same rule `synth_dim_arrow` applies
                        // in term position.
                        let resolved = node_name(dim_inner)
                            .filter(|_| dim_inner.kind == "var")
                            .and_then(|n| ctx.results.get(n.as_str()).cloned())
                            .map(|b| eval_index(&b, ctx))
                            .filter(|v| tag_of(v) == Some("var") || num_val(v).is_some());
                        match resolved {
                            Some(v) => domain_lift(&v, ctx),
                            None => dim_meta(ctx),
                        }
                    };
                    let va = domain_lift(&src, ctx);
                    let vb = domain_lift(&tgt, ctx);
                    Ok(hom_former(dim_v, va, vb, ctx))
                }
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
        })
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
        // Ctor-telescope scope: a family parameter's own occurrences
        // resolve to the `__fam_<name>` binding, never the user's
        // same-named rigid (instantiate_ctor_full).
        let scoped = fam_scoped(name, ctx);
        let name = scoped.as_str();
        // `results` bindings (index vars, ctor args, alias params,
        // assign-bound terms — all lexically inner) shadow Γ and
        // injected globals alike: a `results`-bound name is the nearest
        // scope, so `m` bound `var{m}` by a Sigma destructure or a
        // match refinement reads as the index atom in `Vec(m, a)`,
        // never the `Nat` sort Γ would hand back for a bare name.
        if ctx.results.contains_key(name) {
            // …unless Γ carries a *type declaration* for the name —
            // `type.alias`/`type.ctor_sig`/`type.family`, the bindings
            // `dep_bound_var_type` interprets structurally. Every
            // `assign` also records the bound *term* in `results`
            // (ADR-0103's bound-value env — `type.Score := Nat` leaves
            // `Score ↦ var{Nat}`), and that term is for *value*
            // position only: `s : Score` must expand the alias, not
            // evaluate `var{Nat}` as an index atom.
            let declared = ctx.lookup_gamma(name).is_some_and(|bound| {
                matches!(
                    tag_of(bound),
                    Some("type.alias" | "type.ctor_sig" | "type.family")
                )
            });
            if declared {
                let bound = ctx.lookup_gamma(name).cloned().unwrap();
                if let Some(ty) = self.dep_bound_var_type(lang, ast, &bound, name, ctx)? {
                    return Ok(ty);
                }
                // `type.family` naming a *different* family — the
                // decl does not claim `name`; the results binding
                // stands.
            }
            return Ok(eval_index(&var_term(name), ctx));
        }
        // Γ bindings — source-level declarations (`data`, `type` aliases,
        // `assign`s, pattern binders) — legitimately shadow builtin names:
        // `data Bool :: …` must rebind `Bool` away from `type.bool`.
        if let Some(bound) = ctx.lookup_gamma(name).cloned()
            && let Some(ty) = self.dep_bound_var_type(lang, ast, &bound, name, ctx)?
        {
            return Ok(ty);
        }
        // Builtin type names resolve before *injected* globals: `hnl
        // typecheck` seeds the whole wire-prelude name set as
        // `type.unknown` placeholders, which would otherwise report
        // `Bool`/`Nat`/`IO`/… as `type.unknown` instead of their real
        // types — and would shadow results-bound index variables the
        // same way.
        if is_builtin_type_name(name) {
            return self.dep_builtin_or_tag(name, ctx);
        }
        if let Some(bound) = ctx.lookup_global(name).cloned()
            && let Some(ty) = self.dep_bound_var_type(lang, ast, &bound, name, ctx)?
        {
            return Ok(ty);
        }
        self.dep_builtin_or_tag(name, ctx)
    }

    /// Interpret a Γ/global binding in type position: `type.ctor_sig`
    /// specs instantiate, `type.alias` records expand, `type.family`
    /// self-references mint fresh `type.family_app` indices. Returns
    /// `Ok(None)` only for a `type.family` binding that names a
    /// *different* family (falls through to the next scope).
    fn dep_bound_var_type(
        &self,
        lang: &Language,
        ast: &Ast,
        bound: &Value,
        name: &str,
        ctx: &mut TypecheckContext,
    ) -> Result<Option<Value>, LangError> {
        match tag_of(bound) {
            Some("type.ctor_sig") => self.instantiate_ctor_full(lang, ast, bound, ctx).map(Some),
            Some("type.alias") => self.expand_alias(lang, ast, bound, &[], ctx).map(Some),
            Some("type.family") => {
                if bound.get("name").and_then(Value::as_str).unwrap_or("") == name {
                    let n = bound
                        .get("base")
                        .and_then(|b| b.get("elements"))
                        .and_then(Value::as_array)
                        .map(|e| e.len())
                        .unwrap_or(0);
                    let nat_flags = ctx.family_param_nat.get(name).cloned().unwrap_or_default();
                    let args: Vec<Value> = (0..n)
                        .map(|i| {
                            let id = ctx.fresh_meta();
                            if nat_flags.get(i).copied().unwrap_or(false) {
                                ctx.nonneg_metas.insert(id);
                            }
                            meta_value(id)
                        })
                        .collect();
                    let mut m = Map::new();
                    m.insert("name".to_string(), Value::String(name.to_string()));
                    m.insert("args".to_string(), Value::Array(args));
                    Ok(Some(make_tag(ctx, "type.family_app", m)))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(Some(bound.clone())),
        }
    }

    /// The builtin-name table for bare type-position identifiers.
    fn dep_builtin_or_tag(
        &self,
        name: &str,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let tag = match name {
            "Int" | "Nat" | "Number" => "type.number",
            "Bool" => "type.bool",
            "String" | "Str" => "type.string",
            "Unit" => "type.unit",
            // `Null` is the null sort (T-153): `type.unit` ≡ `type.null`,
            // so both spellings resolve — `Unit` stays canonical for the
            // statement/zero-arg peel, `Null` for explicit null domains.
            "Null" => "type.null",
            // Bare `IO` = `IO(⊤)` — the arity-0 `Former{type.io}`
            // (any signature) for backward compat (ADR-0153 §4).
            "IO" => {
                return Ok(domain_quote(
                    &Val::Former {
                        head: checker::type_sym("type.io"),
                        args: vec![],
                    },
                    ctx,
                ));
            }
            "Type" | "Universe" | "Type0" => "type.universe",
            // The cubical interval sort (ADR-0208 P1): `I` is the type of
            // dimension variables and endpoints `i0`/`i1`.
            "I" | "Interval" => "type.interval",
            // `Sig` — the `IO(sig)` signature-record sort (a row
            // binder's declared sort in `f<s : Sig> : … IO(s)`): an
            // op-name → signature record, i.e. the `Obj` nominal.
            "Sig" => "type.obj",
            "List" | "Family" | "Pi" | "Sigma" | "Forall" => "type.constructor",
            _ => {
                return Err(LangError::engine(format!(
                    "dep_synth_type: unbound type variable '{name}'"
                )));
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
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            let kids = children_of(node)
                .ok_or_else(|| LangError::engine("dep_synth_type: call has no children"))?;
            if kids.is_empty() {
                return Err(LangError::engine("dep_synth_type: empty call"));
            }
            let callee_idx = kids[0]
                .as_u64()
                .ok_or_else(|| LangError::engine("dep_synth_type: call callee not a node index"))?
                as usize;
            let callee = ast
                .get(AstRef::new(callee_idx))
                .ok_or_else(|| LangError::engine("dep_synth_type: callee out of bounds"))?;
            let callee_name = if callee.kind == "var" {
                callee.payload.as_str().map(str::to_string)
            } else if callee.kind == "field-access" {
                children_of(callee)
                    .and_then(|k| k.get(1))
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(node_name)
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
                        Some(&i) => self
                            .arrow_parts(lang, ast, i)
                            .unwrap_or(("x".to_string(), i)),
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
                Some(former @ ("Path" | "PathP")) => {
                    // `Path`/`PathP` — cubical path types (ADR-0208 P1):
                    // `Path(A, x, y)` is the homogeneous interval function
                    // `(i:I) → A`, `PathP(F, x, y)` the heterogeneous
                    // `(i:I) → F i` for `F : I → Type`; `Path(a, b)` with
                    // two arguments stays the index path. All mint
                    // `type.pi` over `type.interval` with a `path`
                    // endpoint payload — see `dep_synth_path_type`.
                    self.dep_synth_path_type(lang, ast, node, &arg_indices, former, ctx)
                }
                Some("Glue") => {
                    // `Glue(A, φ, T̂)` — cubical glue type former
                    // (ADR-0208 P3): ordinary call-form identifier, no
                    // keyword (the `Path`/`PathP` precedent). Exactly
                    // three arguments — the `glue` keyword's
                    // `family`/`face`/`section`.
                    if arg_indices.len() != 3 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message:
                                "synth_type: Glue needs a family, a face, and a section — Glue(A, φ, T̂)"
                                    .to_string(),
                            span: node.span,
                        });
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        let v = self.dep_synth_glue_type(
                            lang,
                            ast,
                            arg_indices[0],
                            arg_indices[1],
                            arg_indices[2],
                            node.span,
                            ctx,
                        )?;
                        // Degraded formation (`type.unknown`) reads as an
                        // open meta in type position — `cub_open_meta`.
                        Ok(if tag_of(&v) == Some("type.unknown") {
                            self.cub_open_meta(ctx)
                        } else {
                            v
                        })
                    }
                }
                Some("isFibrant") => {
                    // `isFibrant(A)` — the fibrancy *type*
                    // `type.fibrant{of}` (ADR-0208 P3).
                    if arg_indices.len() != 1 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message: "synth_type: isFibrant needs exactly one type argument"
                                .to_string(),
                            span: node.span,
                        });
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        self.dep_synth_fibrant_former(lang, ast, arg_indices[0], node.span, ctx)
                    }
                }
                Some("unglue") => {
                    // `unglue(G)` — the type-level projection back to
                    // the underlying family `A` of `Glue A φ T̂`
                    // (ADR-0208 P3).
                    if arg_indices.len() != 1 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message: "synth_type: unglue needs exactly one Glue-type argument"
                                .to_string(),
                            span: node.span,
                        });
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        let u = self.dep_synth_type(lang, ast, arg_indices[0], ctx)?;
                        if tag_of(&u) == Some("type.glue") {
                            Ok(u.get("family")
                                .cloned()
                                .unwrap_or_else(|| self.cub_open_meta(ctx)))
                        } else {
                            ctx.diagnostics.push(TypecheckDiagnostic {
                                severity: "error".to_string(),
                                message: "unglue: operand is not a Glue type — expected glue(A, φ, T̂) or Glue(A, φ, T̂)"
                                    .to_string(),
                                span: node.span,
                            });
                            Ok(self.cub_open_meta(ctx))
                        }
                    }
                }
                Some("Equiv") => {
                    // `Equiv(A, B)` — the call-form equivalence type
                    // former (ADR-0208 P4): mints `type.equiv{dom,cod}`.
                    // Inhabitants are `{to : A ~> B, fro : B ~> A}`
                    // records (checked by the `dep_unify`
                    // `type.equiv`/`type.record` arm).
                    if arg_indices.len() != 2 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message:
                                "synth_type: Equiv needs a domain and a codomain — Equiv(A, B)"
                                    .to_string(),
                            span: node.span,
                        });
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        let d = self.dep_synth_type(lang, ast, arg_indices[0], ctx)?;
                        let c = self.dep_synth_type(lang, ast, arg_indices[1], ctx)?;
                        let mut m = Map::new();
                        m.insert("dom".to_string(), d);
                        m.insert("cod".to_string(), c);
                        Ok(make_tag(ctx, "type.equiv", m))
                    }
                }
                Some("ua") | Some("Univalence") => {
                    // `ua(e)`/`Univalence(e)` in *type* position — the
                    // same path-carrying `type.pi` the term-position
                    // call judgment mints (ADR-0208 P4).
                    if arg_indices.len() != 1 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message:
                                "synth_type: ua/Univalence needs exactly one equivalence operand"
                                    .to_string(),
                            span: node.span,
                        });
                        Ok(self.cub_open_meta(ctx))
                    } else {
                        let v =
                            self.synth_univalence_core(lang, ast, arg_indices[0], node.span, ctx)?;
                        // Degraded formation (`type.unknown`) reads as
                        // an open meta in type position.
                        Ok(if tag_of(&v) == Some("type.unknown") {
                            self.cub_open_meta(ctx)
                        } else {
                            v
                        })
                    }
                }
                Some("transport") => {
                    // `transport(p, v)` in *type* position — `p # v`
                    // is a term-level eliminator; the `transport` node
                    // reads as `type.unknown` in annotations, so the
                    // call form does the same (an open meta — the
                    // degraded-formation reading).
                    if arg_indices.len() != 2 {
                        ctx.diagnostics.push(TypecheckDiagnostic {
                            severity: "warning".to_string(),
                            message: "synth_type: transport(p, v) needs a path and a value operand"
                                .to_string(),
                            span: node.span,
                        });
                    }
                    Ok(self.cub_open_meta(ctx))
                }
                Some("IO") => {
                    // `IO(sig)` — the typed-effect former (ADR-0153 §4):
                    // canonical `Former{type.io}` data. The sig index is
                    // an op-signature record — `{op: T, …}` entries
                    // synthesize their value types; `IO()` is `IO(⊤)`.
                    // The brace-free `IO<puts, fs>` / `IO<puts: T, fs>`
                    // spelling — the `dimension` node's argument list,
                    // unambiguous there: each arg is an expression, a
                    // bare `var` declares an op with an unstated
                    // request and `name : T` its payload — is the same
                    // sig-record sugar, single op included
                    // (`IO<puts>` ≡ `IO<{ puts }>`,
                    // `IO<puts: T>` ≡ `IO<{ puts: T }>`). Call-form
                    // `IO(s)`/`IO(sig)` is untouched — a bare variable
                    // there is a sig-typed *index*, not an op list —
                    // and the sugar fires only when EVERY argument is a
                    // bare name or typed binder; anything else keeps
                    // the single-sig-argument reading.
                    let sig = if node.kind == "dimension"
                        && !arg_indices.is_empty()
                        && arg_indices.iter().all(|&i| {
                            ast.get(AstRef::new(i)).is_some_and(|n| {
                                n.kind == "var" || Self::typed_binder(lang, ast, i).is_some()
                            })
                        }) {
                        let mut m = Map::new();
                        for &i in &arg_indices {
                            if let Some((k, ty_idx)) = Self::typed_binder(lang, ast, i) {
                                m.insert(k, self.dep_synth_type(lang, ast, ty_idx, ctx)?);
                            } else if let Some(n) = ast.get(AstRef::new(i))
                                && let Some(k) = node_name(n)
                            {
                                let mut u = Map::new();
                                m.insert(
                                    k,
                                    make_tag(ctx, "type.unknown", std::mem::take(&mut u)),
                                );
                            }
                        }
                        Value::Object(m)
                    } else {
                        match arg_indices.first() {
                            Some(&i) => self.io_sig(lang, ast, i, ctx)?,
                            None => Value::Null,
                        }
                    };
                    let vs = domain_lift(&sig, ctx);
                    Ok(domain_quote(
                        &Val::Former {
                            head: checker::type_sym("type.io"),
                            args: if sig == Value::Null { vec![] } else { vec![vs] },
                        },
                        ctx,
                    ))
                }
                Some(name) => {
                    // A declared family's application: `Vec(i⃗)` →
                    // `type.family_app{name, args}` — index arguments
                    // evaluate (1+1 → 2), rigid vars stay `var{n}`.
                    // Inside a ctor telescope a family-param callee means
                    // its `__fam_<name>` binding.
                    let scoped = fam_scoped(name, ctx);
                    let name = scoped.as_str();
                    if let Some(bound) = ctx.lookup(name).cloned() {
                        match tag_of(&bound) {
                            Some("type.alias") => {
                                // A declared `type.Name(p…) := T` alias —
                                // transparent substitution: bind the params
                                // to the argument types and re-synthesize
                                // the body (ADR-0154 §3, no former mint).
                                let mut args = Vec::with_capacity(arg_indices.len());
                                for &i in &arg_indices {
                                    let t = self.node_index_or_type(lang, ast, i, ctx)?;
                                    args.push(eval_index(&t, ctx));
                                }
                                return self.expand_alias(lang, ast, &bound, &args, ctx);
                            }
                            Some("type.family") => {
                                let mut args = Vec::with_capacity(arg_indices.len());
                                for &i in &arg_indices {
                                    let t = self.node_index_or_type(lang, ast, i, ctx)?;
                                    args.push(eval_index(&t, ctx));
                                }
                                let mut m = Map::new();
                                m.insert(
                                    "name".to_string(),
                                    bound
                                        .get("name")
                                        .cloned()
                                        .unwrap_or(Value::String(name.to_string())),
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
        })
    }

    /// `Path`/`PathP` in type position (ADR-0208 P1, port of the model's
    /// `synth_path_type_`): mints a `type.pi` over the interval sort
    /// `type.interval` whose `body_closure` is a *type-level* closure —
    /// `const_body` for the homogeneous `Path` (`(i:I) → A`, `Path` is
    /// the constant family special case of `PathP`) and `apply_type`
    /// for the heterogeneous `PathP` (`(i:I) → F i` for `F : I → Type`,
    /// typically written `Pi(I, i ~> …)`). The `path` payload records
    /// the endpoint terms `x`/`y` for introduction (`refl`, path
    /// witnesses) and elimination (`#`, `j`) — proof-relevant, never
    /// erased.
    ///
    /// `Path(a, b)` with exactly two arguments is the legacy *index*
    /// path: both arguments are index terms and the interval body is
    /// `type.unknown` (index sorts are not tracked). Its `eq_*`/unify
    /// witnesses stay `Path{frm,to}` records — only the user-facing
    /// type former changed meaning.
    fn dep_synth_path_type(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        arg_indices: &[usize],
        former_name: &str,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let mut unknown_m = Map::new();
        let unknown = make_tag(ctx, "type.unknown", std::mem::take(&mut unknown_m));
        let warn = |ctx: &mut TypecheckContext, msg: String| {
            ctx.diagnostics.push(TypecheckDiagnostic {
                severity: "warning".to_string(),
                message: msg,
                span: node.span,
            });
        };
        if arg_indices.len() < 2 {
            warn(
                ctx,
                format!("synth_type: {former_name} needs endpoint arguments"),
            );
            return Ok(unknown);
        }
        // Endpoint arguments elaborate as index *terms* —
        // `node_to_index_term` + `eval_index`, the same reading
        // `Path(a, b)`'s legacy arm used for its endpoints through
        // `node_index_or_type`. The difference matters for term
        // endpoints that are not types: `true`, `Zero`, `Succ(Zero)`
        // are `var`/`call` nodes whose `dep_synth_type` reading would
        // look for a *type* binding (`unbound type variable 'true'`)
        // — as index atoms they are `var{true}`/`App{Succ, …}` terms
        // `eval_index` canonicalizes (`Zero ↦ 0`, `Succ(Zero) ↦ 1`
        // through the declared ctor projections).
        let endpoint = |i: usize, ctx: &mut TypecheckContext| -> Result<Value, LangError> {
            let node = ast.get(AstRef::new(i)).ok_or_else(|| {
                LangError::engine(format!(
                    "dep_synth_path_type: endpoint node {i} out of bounds"
                ))
            })?;
            Ok(eval_index(
                &self.node_to_index_term(lang, ast, node, ctx),
                ctx,
            ))
        };
        if arg_indices.len() == 2 && former_name == "Path" {
            // `Path(a, b)` — index path: `a`/`b` are index *terms*.
            let x = endpoint(arg_indices[0], ctx)?;
            let y = endpoint(arg_indices[1], ctx)?;
            let mut m = Map::new();
            m.insert("param".to_string(), interval);
            m.insert(
                "body_closure".to_string(),
                serde_json::json!({
                    "var_name": "",
                    "body_child_idx": Value::Null,
                    "dep": true,
                    "const_body": unknown,
                }),
            );
            m.insert("path".to_string(), serde_json::json!({"x": x, "y": y}));
            return Ok(make_tag(ctx, "type.pi", m));
        }
        if arg_indices.len() < 3 {
            warn(
                ctx,
                format!("synth_type: {former_name} needs a type/family argument and two endpoints"),
            );
            return Ok(unknown);
        }
        // `Path(A, x, y)` / `PathP(F, x, y)`: the first argument is a
        // *type* — synthesized through `dep_synth_type` — and `x`, `y`
        // are endpoint terms.
        let fam = self.dep_synth_type(lang, ast, arg_indices[0], ctx)?;
        let x = endpoint(arg_indices[1], ctx)?;
        let y = endpoint(arg_indices[2], ctx)?;
        let fam_is_interval_pi = tag_of(&fam) == Some("type.pi")
            && tag_of(fam.get("param").unwrap_or(&Value::Null)) == Some("type.interval");
        let closure = if former_name == "PathP" && fam_is_interval_pi {
            serde_json::json!({
                "var_name": "i",
                "body_child_idx": Value::Null,
                "dep": true,
                "apply_type": fam,
            })
        } else {
            // `Path` is always the constant family; a `PathP` whose
            // family did not synthesize to an interval-Pi degrades to
            // the same constant reading.
            serde_json::json!({
                "var_name": "",
                "body_child_idx": Value::Null,
                "dep": true,
                "const_body": fam,
            })
        };
        let mut m = Map::new();
        m.insert("param".to_string(), interval);
        m.insert("body_closure".to_string(), closure);
        m.insert("path".to_string(), serde_json::json!({"x": x, "y": y}));
        Ok(make_tag(ctx, "type.pi", m))
    }

    /// `type.Name` (a `field-access` on the `var "type"` namespace) or
    /// `type.Name(p, …)` (a `call` thereof) as an `assign` lhs — the
    /// transparent alias form (`type` is a keyword, so `type.` never
    /// names a real object). Returns `(name, params)`.
    fn type_alias_head(&self, ast: &Ast, node: &AstNode) -> Option<(String, Vec<String>)> {
        let kids = children_of(node)?;
        let (fa, params) = match node.kind.as_str() {
            "field-access" => (node, Vec::new()),
            "call" => {
                let fa = ast.get(AstRef::new(kids.first()?.as_u64()? as usize))?;
                if fa.kind != "field-access" {
                    return None;
                }
                let mut params = Vec::new();
                for arg in kids.iter().skip(1) {
                    let n = ast.get(AstRef::new(arg.as_u64()? as usize))?;
                    if n.kind != "var" {
                        return None;
                    }
                    params.push(node_name(n)?);
                }
                (fa, params)
            }
            _ => return None,
        };
        let fa_kids = children_of(fa)?;
        let subj = ast.get(AstRef::new(fa_kids.first()?.as_u64()? as usize))?;
        if subj.kind != "var" || node_name(subj).as_deref() != Some("type") {
            return None;
        }
        let name = fa
            .payload
            .get("field")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                fa_kids
                    .get(1)
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(node_name)
            })?;
        Some((name, params))
    }

    /// Expand a `type.alias{params, body}` record (ADR-0154 §3 —
    /// aliases are transparent: plain substitution, no former mint).
    /// Binds each param name to its argument type in `results` (fresh
    /// metas when the alias is referenced bare) and re-synthesizes the
    /// body type expression.
    fn expand_alias(
        &self,
        lang: &Language,
        ast: &Ast,
        bound: &Value,
        args: &[Value],
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            let params: Vec<String> = bound
                .get("params")
                .and_then(Value::as_array)
                .map(|ps| {
                    ps.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            if !args.is_empty() && args.len() != params.len() {
                return Err(LangError::engine(format!(
                    "type.alias: {} applied to {} args (expects {})",
                    bound.get("name").and_then(Value::as_str).unwrap_or("?"),
                    args.len(),
                    params.len()
                )));
            }
            let body_idx =
                bound.get("body").and_then(Value::as_u64).ok_or_else(|| {
                    LangError::engine("type.alias: missing body index".to_string())
                })? as usize;
            let old_results = ctx.results.clone();
            for (i, p) in params.iter().enumerate() {
                let v = args
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| meta_value(ctx.fresh_meta()));
                ctx.results.insert(p.clone(), v);
            }
            let r = self.dep_synth_type(lang, ast, body_idx, ctx);
            ctx.results = old_results;
            r
        })
    }

    /// The sig argument of `IO(sig)` — an `object` literal becomes a
    /// record of op-name → synthesized value type; anything else is an
    /// index term / type expression.
    fn io_sig(
        &self,
        lang: &Language,
        ast: &Ast,
        idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast
            .get(AstRef::new(idx))
            .ok_or_else(|| LangError::engine("io: sig node out of bounds".to_string()))?;
        if node.kind == "object" {
            let mut m = Map::new();
            if let Some(entries) = node.payload.get("entries").and_then(Value::as_array) {
                for e in entries {
                    if let Some(k) = e.get("key").and_then(Value::as_str) {
                        match e.get("value").and_then(Value::as_u64) {
                            Some(vi) => {
                                m.insert(
                                    k.to_string(),
                                    self.dep_synth_type(lang, ast, vi as usize, ctx)?,
                                );
                            }
                            None => {
                                if let Some(v) = e.get("value") {
                                    m.insert(k.to_string(), v.clone());
                                }
                            }
                        }
                    }
                }
            }
            if m.is_empty() {
                for k in children_of(node).cloned().unwrap_or_default() {
                    let Some(en) = k.as_u64().and_then(|i| ast.get(AstRef::new(i as usize))) else {
                        continue;
                    };
                    // `IO<{ puts }>` — a bare-name child declares the op
                    // with an unstated request (`type.unknown`).
                    if en.kind == "var" {
                        if let Some(k) = node_name(en) {
                            let mut u = Map::new();
                            m.insert(k, make_tag(ctx, "type.unknown", std::mem::take(&mut u)));
                        }
                        continue;
                    }
                    if en.kind != "entry" {
                        continue;
                    }
                    let ek = children_of(en)
                        .and_then(|c| c.first())
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(|n| {
                            node_name(n).or_else(|| {
                                n.payload
                                    .get("value")
                                    .and_then(Value::as_str)
                                    .map(str::to_string)
                            })
                        });
                    let v_idx = children_of(en)
                        .and_then(|c| c.get(1))
                        .and_then(Value::as_u64)
                        .map(|i| i as usize);
                    if let (Some(k), Some(vi)) = (ek, v_idx) {
                        m.insert(k, self.dep_synth_type(lang, ast, vi, ctx)?);
                    }
                }
            }
            return Ok(Value::Object(m));
        }
        if node.kind == "block" {
            // Object-literal blocks (ADR-0079): `{ op: T, … }` in a
            // block position parses as `type-annot` children — each
            // reads as an `entry` (subject → op name, annot → sig type).
            let mut m = Map::new();
            let mut entries = 0usize;
            for k in children_of(node).cloned().unwrap_or_default() {
                let Some(tn) = k.as_u64().and_then(|i| ast.get(AstRef::new(i as usize))) else {
                    continue;
                };
                if tn.kind != "type-annot" {
                    continue;
                }
                let tk = children_of(tn)
                    .and_then(|c| c.first())
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(|n| {
                        node_name(n).or_else(|| {
                            n.payload
                                .get("value")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                    });
                let v_idx = children_of(tn)
                    .and_then(|c| c.get(1))
                    .and_then(Value::as_u64)
                    .map(|i| i as usize);
                if let (Some(k), Some(vi)) = (tk, v_idx) {
                    m.insert(k, self.dep_synth_type(lang, ast, vi, ctx)?);
                    entries += 1;
                }
            }
            if entries > 0 {
                return Ok(Value::Object(m));
            }
        }
        self.node_index_or_type(lang, ast, idx, ctx)
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
        let _kids = children_of(inner)?;
        let params_idx =
            role_child_ref(lang, inner, "params").or_else(|| child_at(ast, inner, 0))?;
        let body_idx = role_child_ref(lang, inner, "body").or_else(|| child_at(ast, inner, 1))?;
        let pnode = ast.get(params_idx)?;
        let name = lang
            .param_names(ast, pnode)
            .ok()
            .and_then(|(names, _)| names.first().cloned())
            .or_else(|| node_name(pnode))
            .unwrap_or_else(|| "x".to_string());
        Some((name, body_idx.index()))
    }

    /// A *typed name binder* `x : T` — a `type-annot` node whose subject
    /// is a bare `var`. Returns `(name, annot type-expr node idx)`; `None`
    /// for anything else (untyped `var`, computed subject, non-annot).
    /// Shared by the dependent-pair `paren` arm and the typed-param
    /// `function` arm of [`Self::dep_synth_type`].
    fn typed_binder(
        lang: &Language,
        ast: &Ast,
        idx: usize,
    ) -> Option<(String, usize)> {
        let n = ast.get(AstRef::new(idx))?;
        if n.kind != "type-annot" && role_child_ref(lang, n, "annot").is_none() {
            return None;
        }
        let kids = children_of(n)?;
        let subj_i = lang
            .child_role_index(&n.kind, "subject")
            .ok()
            .flatten()
            .unwrap_or(0);
        let annot_i = lang
            .child_role_index(&n.kind, "annot")
            .ok()
            .flatten()
            .unwrap_or(1);
        let subj = ast.get(AstRef::new(kids.get(subj_i)?.as_u64()? as usize))?;
        if subj.kind != "var" {
            return None;
        }
        let name = node_name(subj)?;
        let ty = kids.get(annot_i)?.as_u64()? as usize;
        Some((name, ty))
    }

    /// Expand one step of a `body_closure` `rest` chain — the lazy
    /// binder tail minted by `dep_synth_type` for dependent pairs
    /// `(x : A, y : B(…), …)` (`"chain": "sigma"`) and typed-param
    /// arrows `(x : A, …) ~> C` (`"chain": "pi"`). Each `rest` entry is
    /// `{name, annot}` — the next binder's name and its type-expression
    /// node, elaborated *now* so earlier binders are in scope.
    ///
    /// - `sigma`: yields `type.sigma{param, body_closure}` per remaining
    ///   component; the last entry's annotation IS the final component
    ///   type (returned directly).
    /// - `pi`: yields `type.pi{param, body_closure}` per remaining
    ///   binder; `body_child_idx` on the parent closure is the shared
    ///   terminal body (`~>` right-hand side).
    fn dep_closure_chain(
        &self,
        lang: &Language,
        ast: &Ast,
        bc: &Value,
        rest: &[Value],
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let head = &rest[0];
        let hname = head
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("x")
            .to_string();
        let hannot = head
            .get("annot")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                LangError::engine("dep_closure_chain: rest entry missing annot idx".to_string())
            })? as usize;
        let param = self.dep_synth_type(lang, ast, hannot, ctx)?;
        let chain = bc.get("chain").and_then(Value::as_str).unwrap_or("sigma");
        if chain == "pi" {
            let body_idx = bc
                .get("body_child_idx")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    LangError::engine("dep_closure_chain: pi chain missing body idx".to_string())
                })?;
            let mut m = Map::new();
            m.insert("param".to_string(), param);
            m.insert(
                "body_closure".to_string(),
                serde_json::json!({
                    "var_name": hname,
                    "rest": rest[1..],
                    "chain": "pi",
                    "body_child_idx": body_idx,
                    "dep": true,
                }),
            );
            return Ok(make_tag(ctx, "type.pi", m));
        }
        if rest.len() == 1 {
            // Last sigma component: its annotation is the body type.
            return Ok(param);
        }
        let mut m = Map::new();
        m.insert("param".to_string(), param);
        m.insert(
            "body_closure".to_string(),
            serde_json::json!({
                "var_name": hname,
                "rest": rest[1..],
                "chain": "sigma",
                "dep": true,
            }),
        );
        Ok(make_tag(ctx, "type.sigma", m))
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
        let _t = crate::stats::timer(&crate::stats::T_NODE_INDEX_OR_TYPE);
        crate::stats::node_index_or_type(idx);
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            let node = ast
                .get(AstRef::new(idx))
                .ok_or_else(|| LangError::engine(format!("dep: node {} out of bounds", idx)))?;
            if node.kind.starts_with("binary.")
                || node.kind.starts_with("unary.")
                || matches!(node.kind.as_str(), "value" | "lit")
            {
                return Ok(self.node_to_index_term(lang, ast, node, ctx));
            }
            // `C(i⃗)`/`C<i⃗>` where `C` is a *term-level* constructor (a
            // `type.ctor_sig` Γ binding, e.g. `Succ`) is an index *term*
            // (`App{var C, i⃗}`), not an instantiated type — `Vec<Succ n, a>`
            // carries `Succ n` as the index it refines against. Family
            // callees (`Vec`, `Pi`, …) still synthesize types. The ctor's
            // arguments elaborate through `node_to_index_term` — the index
            // reading: a `var` arg is the rigid index atom `var{n}`, not the
            // sort `Nat` a bare var resolves to in type position.
            if matches!(node.kind.as_str(), "call" | "index_app" | "dimension") {
                let kids = children_of(node).cloned().unwrap_or_default();
                let callee_name = kids
                    .first()
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .filter(|n| n.kind == "var")
                    .and_then(node_name);
                if let Some(cname) = callee_name {
                    // Ctor-telescope scope: a family-param callee means its
                    // `__fam_<name>` binding — and must never hit an outer
                    // ctor_sig of the same name.
                    let cname = fam_scoped(&cname, ctx);
                    if ctx
                        .lookup(&cname)
                        .is_some_and(|b| tag_of(b) == Some("type.ctor_sig"))
                    {
                        let mut args = Vec::with_capacity(kids.len().saturating_sub(1));
                        for cv in kids.iter().skip(1) {
                            let Some(ai) = cv.as_u64() else { continue };
                            let Some(an) = ast.get(AstRef::new(ai as usize)) else {
                                continue;
                            };
                            let t = self.node_to_index_term(lang, ast, an, ctx);
                            args.push(eval_index(&t, ctx));
                        }
                        return Ok(app_term(var_term(&cname), args));
                    }
                }
            }
            // A bare `var` naming a declared ctor (`Zero` in `Vec(Zero, a)`)
            // is an index atom `var{C}` — a registered arity-0 projection
            // (`Zero ↦ 0`) then applies in `eval_index`. Without this arm the
            // var would instantiate the ctor *signature* (its type), which
            // is not an index term.
            if node.kind == "var"
                && let Some(cname) = node_name(node)
            {
                let cname = fam_scoped(&cname, ctx);
                if ctx
                    .lookup(&cname)
                    .is_some_and(|b| tag_of(b) == Some("type.ctor_sig"))
                {
                    return Ok(var_term(&cname));
                }
            }
            self.dep_synth_type(lang, ast, idx, ctx)
        })
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
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
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
                // A `var` leaf mints the rigid index atom — inside a ctor
                // telescope the family params mint their `__fam_<name>`
                // atoms so the meta bindings, not the user's rigids, solve.
                "var" => var_term(
                    &node
                        .payload
                        .as_str()
                        .map(|n| fam_scoped(n, ctx))
                        .unwrap_or_default(),
                ),
                "paren" => child_at(ast, node, 0)
                    .and_then(|i| ast.get(i))
                    .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                    .unwrap_or(Value::Null),
                "value" | "lit" => node.payload.clone(),
                // `f(i⃗)`/`f<i⃗>` as an index term — an opaque `App` on the
                // callee's `var` (constructor or results-bound function).
                "call" | "index_app" | "dimension" => {
                    let kids = children_of(node).cloned().unwrap_or_default();
                    let callee = kids
                        .first()
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(node_name)
                        .map(|c| fam_scoped(&c, ctx))
                        .unwrap_or_default();
                    let args: Vec<Value> = kids
                        .iter()
                        .skip(1)
                        .filter_map(Value::as_u64)
                        .filter_map(|i| ast.get(AstRef::new(i as usize)))
                        .map(|n| self.node_to_index_term(lang, ast, n, ctx))
                        .collect();
                    app_term(var_term(&callee), args)
                }
                _ => Value::Null,
            }
        })
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
        crate::stats::synth_data(node_ref.index());
        let node = ast
            .get(node_ref)
            .ok_or_else(|| LangError::engine("synth_data: node out of bounds".to_string()))?;
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
                ctx.extend(name.clone(), make_tag(ctx, "type.family", m));
                // WS-C census: the ctor names are declared even though
                // no `ctor_sig` telescopes exist on this path — a probe
                // that can't instantiate stays honestly undecided.
                let names: Vec<String> = node
                    .payload
                    .get("constructors")
                    .and_then(Value::as_array)
                    .map(|cs| {
                        cs.iter()
                            .filter_map(|c| c.get("name").and_then(Value::as_str))
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                for c in &names {
                    ctx.ctor_family.insert(c.clone(), name.clone());
                }
                ctx.family_ctors.insert(name.clone(), names);
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
                if let Some(ti) = ty_idx
                    && let Some(n) = ast.get(AstRef::new(*ti))
                    && !positivity_ok(ast, n, &name)
                {
                    return Err(LangError::engine(format!(
                        "synth_data: strict positivity violated: constructor '{}' \
                                 mentions '{name}' in a negative position",
                        c.name
                    )));
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
            m.insert("base".to_string(), base.clone());
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
        // WS-C coverage census: the family's ctor list + the reverse
        // ctor → family table (index-level ctor disjointness reads it),
        // plus the per-parameter `nat` flags the `type.family` value's
        // evaluated `base.elements` can no longer see (`Nat` lands as
        // `type.number`).
        ctx.family_ctors.insert(
            name.clone(),
            ctors
                .iter()
                .map(|c| c.name.clone())
                .filter(|n| !n.is_empty())
                .collect(),
        );
        ctx.family_param_nat
            .insert(name.clone(), params.iter().map(|p| p.nat).collect());
        for c in &ctors {
            if !c.name.is_empty() {
                ctx.ctor_family.insert(c.name.clone(), name.clone());
            }
        }
        // `F.rec` — the family's dependent eliminator (ADR-0152). The
        // desugar binds a var literally named `F.rec` in the `__splice__`
        // block's scope; the checker mirrors it as a Γ binding so
        // `field_access(F, "rec")` resolves (see `record_field_type`'s
        // `type.family` arm). The eliminator's type is a nested
        // `type.function` — `motive ~> c0 ~> … ~> cN ~> i0 ~> … ~> iP
        // ~> F(?u) ~> ?mo`. Each case is the ctor's instantiated domain
        // chain, then one `?mo` per *recursive* (self-family) argument
        // for the induction hypothesis, then `?mo` — `c_VCons :
        // a ~> Vec(?k, a) ~> ?mo ~> ?mo`. `?mm` stays a plain meta (the
        // fold machine's `null`-motive call is legitimate); `?mo` is
        // shared across cases and the final body — the non-dependent
        // fold contract that every case and every `ih` agrees on one
        // result type.
        {
            let mo = meta_value(ctx.fresh_meta());
            let mut case_tys = Vec::with_capacity(ctors.len());
            for c in &ctors {
                if c.name.is_empty() {
                    continue;
                }
                let spec = serde_json::json!({
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
                let inst = self.instantiate_ctor_full(lang, ast, &spec, ctx)?;
                // Unfold `dom0 ~> … ~> domk ~> result`: the domains are
                // the case's leading params; each domain that is the
                // family itself contributes one `?mo` hypothesis arg.
                let mut domains = Vec::new();
                let mut cur = inst;
                while tag_of(&cur) == Some("type.function") {
                    domains.push(cur.get("param").cloned().unwrap_or(Value::Null));
                    cur = cur.get("body").cloned().unwrap_or(Value::Null);
                }
                let n_rec = domains
                    .iter()
                    .filter(|d| {
                        tag_of(d) == Some("type.family_app")
                            && d.get("name").and_then(Value::as_str) == Some(name.as_str())
                    })
                    .count();
                let mut case = mo.clone();
                for _ in 0..n_rec {
                    let mut m = Map::new();
                    m.insert("param".to_string(), mo.clone());
                    m.insert("body".to_string(), case);
                    case = make_tag(ctx, "type.function", m);
                }
                for d in domains.into_iter().rev() {
                    let mut m = Map::new();
                    m.insert("param".to_string(), d);
                    m.insert("body".to_string(), case);
                    case = make_tag(ctx, "type.function", m);
                }
                case_tys.push(case);
            }
            // The second-round call takes the family's index arguments
            // then the scrutinee `F(?u)`; the result is `?mo`. A
            // `nat`-flagged parameter's sort is the arithmetic
            // `type.number` sort — `Nat` in `data …(n : Nat)` names the
            // index sort, and once `data Nat` itself is declared the
            // bare name resolves to *that* family instead.
            let elems: Vec<Value> = base
                .get("elements")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let sorts: Vec<Value> = params
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if p.nat {
                        let mut m = Map::new();
                        make_tag(ctx, "type.number", std::mem::take(&mut m))
                    } else {
                        elems.get(i).cloned().unwrap_or(Value::Null)
                    }
                })
                .collect();
            let scrut_args: Vec<Value> = (0..sorts.len())
                .map(|_| meta_value(ctx.fresh_meta()))
                .collect();
            let mut sm = Map::new();
            sm.insert("name".to_string(), Value::String(name.clone()));
            sm.insert("args".to_string(), Value::Array(scrut_args));
            let scrut = make_tag(ctx, "type.family_app", sm);
            let mut ty = {
                let mut m = Map::new();
                m.insert("param".to_string(), scrut);
                m.insert("body".to_string(), mo.clone());
                make_tag(ctx, "type.function", m)
            };
            for s in sorts.iter().rev() {
                let mut m = Map::new();
                m.insert("param".to_string(), s.clone());
                m.insert("body".to_string(), ty);
                ty = make_tag(ctx, "type.function", m);
            }
            for c in case_tys.into_iter().rev() {
                let mut m = Map::new();
                m.insert("param".to_string(), c);
                m.insert("body".to_string(), ty);
                ty = make_tag(ctx, "type.function", m);
            }
            let mut m = Map::new();
            m.insert("param".to_string(), meta_value(ctx.fresh_meta()));
            m.insert("body".to_string(), ty);
            ty = make_tag(ctx, "type.function", m);
            ctx.extend(format!("{name}.rec"), ty);
        }
        // Index projections (phase-3 WS-D): a unary ctor `S(arg : F)`
        // building `F` declares successor arithmetic — index-level
        // `S x` reads as `x + 1` — and when the family declares exactly
        // one bare ctor `Z`, `Z` reads as `0`. This is the Nat-former
        // shape: `data Nat :: Zero | Succ(pred : Nat)` registers
        // `Succ ↦ λx.x+1` and `Zero ↦ 0`. Declared, not hardcoded — any
        // `data` family with the shape contributes its own ctors;
        // ctors without the shape stay honest `App` atoms.
        let mut succ: Vec<&str> = Vec::new();
        let mut bare: Vec<&str> = Vec::new();
        for c in &ctors {
            if c.name.is_empty() {
                continue;
            }
            // `result` names this family when absent or spelled `F` /
            // `F(i⃗)` — a ctor building some *other* family contributes
            // no projection for `F`'s index discipline.
            let result_is_self = match c.result {
                None => true,
                Some(ri) => {
                    let head = ast.get(AstRef::new(ri)).and_then(|n| {
                        if n.kind == "var" {
                            node_name(n)
                        } else {
                            children_of(n)
                                .and_then(|k| k.first())
                                .and_then(Value::as_u64)
                                .and_then(|i| ast.get(AstRef::new(i as usize)))
                                .and_then(node_name)
                        }
                    });
                    head.as_deref() == Some(name.as_str())
                }
            };
            if c.args.len() == 1
                && result_is_self
                && c.args[0]
                    .1
                    .and_then(|ti| ast.get(AstRef::new(ti)))
                    .and_then(node_name)
                    .as_deref()
                    == Some(name.as_str())
            {
                succ.push(c.name.as_str());
            } else if c.args.is_empty() && result_is_self {
                bare.push(c.name.as_str());
            }
        }
        for s in &succ {
            ctx.ctor_proj.insert(
                (*s).to_string(),
                CtorProj {
                    arity: 1,
                    template: bin_term(
                        "binary.add",
                        var_term(&format!("{CTOR_PROJ_HOLE}0")),
                        Value::from(1),
                    ),
                },
            );
        }
        // The `0` projection only registers when the base is unique —
        // a `data Bool :: True | False` (two bare ctors, no successor)
        // must never equate `True ≡ False` at index level.
        if !succ.is_empty() && bare.len() == 1 {
            ctx.ctor_proj.insert(
                bare[0].to_string(),
                CtorProj {
                    arity: 0,
                    template: Value::from(0),
                },
            );
        }
        let mut m = Map::new();
        Ok(make_tag(ctx, "type.unit", std::mem::take(&mut m)))
    }

    /// Instantiate a `type.ctor_sig` spec into a fresh `type.function`
    /// chain ending in `type.family_app` — one index metavariable per
    /// family parameter (bound in Γ *and* `results` under the internal
    /// `__fam_<name>` spelling, so `xs : Vec(n, a)` elaborates
    /// `__fam_n ↦ ?n` and the result index `n + 1` solves through the
    /// same meta while a user's rigid `var{n}` stays untouched).
    /// Constructor argument names self-bind to `var{name}` terms so
    /// later domains may mention them.
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
            if p.get("nat").and_then(Value::as_bool).unwrap_or(false) {
                ctx.nonneg_metas.insert(id);
            }
            let mv = meta_value(id);
            if !pname.is_empty() {
                // `__fam_<name>` — family param names are internal to
                // the telescope, never a user's same-named rigid.
                let bname = fam_binding_name(pname);
                ctx.extend(bname.clone(), mv.clone());
                ctx.bind_result(bname, mv.clone());
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
                ctx.extend(aname.to_string(), var_term(aname));
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
        let _t = crate::stats::timer(&crate::stats::T_INSTANTIATE_CTOR);
        crate::stats::instantiate_ctor(
            spec.get("ctor")
                .or_else(|| spec.get("name"))
                .and_then(Value::as_str)
                .unwrap_or(""),
        );
        let mark = ctx.mark();
        let old_results = ctx.results.clone();
        let old_fam_scope = ctx.fam_scope.clone();
        let params: Vec<Value> = spec
            .get("params")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut metas = Vec::with_capacity(params.len());
        for p in &params {
            let pname = p
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let id = ctx.fresh_meta();
            if p.get("nat").and_then(Value::as_bool).unwrap_or(false) {
                ctx.nonneg_metas.insert(id);
            }
            let mv = meta_value(id);
            if !pname.is_empty() {
                // The family's parameter names bind under `__fam_<name>`
                // — internal names no user binder can collide with — and
                // `fam_scope` redirects the telescope's own `var{name}`
                // occurrences to them. Binding `n`/`a` directly rewrote
                // the user's same-named rigid vars through `results`
                // (the annotated-arm `VCons(x, xs)` reconstruction bug:
                // `Vec(n+1, ?k+1)` vs `Vec(?k+1, a)`).
                let bname = fam_binding_name(&pname);
                ctx.extend(bname.clone(), mv.clone());
                ctx.bind_result(bname.clone(), mv.clone());
                ctx.fam_scope.insert(pname, bname);
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
                // Arg names self-bind in Γ *and* `results` as the rigid
                // `var{name}` index atom. The Γ entry is what stops a
                // same-named outer binding — e.g. a match-arm pattern
                // field `k : Nat` — from shadowing the telescope's
                // `var{k}` so `xs : Vec(k, a)` resolves to `Number`.
                ctx.extend(aname.to_string(), var_term(aname));
                ctx.bind_result(aname.to_string(), var_term(aname));
            }
        }
        // The fallible telescope elaboration runs under a scope guard so
        // `fam_scope`/`results`/Γ restore on every exit — a failed
        // instantiation must not leak the internal `__fam_` bindings
        // into the caller's scope.
        let out = (|ctx: &mut TypecheckContext| -> Result<Value, LangError> {
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
                        // Result-less ctor (`VCons(k : Nat, x : a,
                        // xs : Vec(k, a))` — no `: F(i⃗)` annotation): an
                        // index meta an argument domain can pin stays a
                        // meta (`?a` solves through `x : a` /
                        // `xs : Vec(k, a)`); a `nat` index parameter no
                        // domain mentions is *unrecoverable* — the ctor
                        // cannot compute it — so it mints the rigid
                        // `var{name}` atom instead of a solvable meta.
                        // `VCons(0, true, VNil) : Vec(5, Bool)` then
                        // *fails* (`var{n} ≢ 5`) instead of letting
                        // `?n := 5` — the F1 soundness hole.
                        let idx_args: Vec<Value> = params
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                let is_nat = p.get("nat").and_then(Value::as_bool).unwrap_or(false);
                                let recoverable =
                                    domains.iter().any(|d| meta_value_occurs(&metas[i], d));
                                if is_nat && !recoverable {
                                    // `lit null` — the desugar's
                                    // unrecoverable-index convention: a
                                    // `nat` index the ctor cannot compute
                                    // never unifies with a numeral
                                    // (`null ≢ 5`), so the result index is
                                    // honestly absent, not a lie.
                                    Value::Null
                                } else {
                                    metas[i].clone()
                                }
                            })
                            .collect();
                        m.insert("args".to_string(), Value::Array(idx_args));
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
            Ok(ty)
        })(ctx);
        ctx.restore(mark);
        ctx.results = old_results;
        ctx.fam_scope = old_fam_scope;
        out
    }

    // -----------------------------------------------------------------------
    // Split-tree coverage for indexed `match` (phase-3 WS-C)
    // -----------------------------------------------------------------------
    //
    // Per-arm index unification decides *this arm is impossible*; it says
    // nothing about the cases no arm mentions. Coverage computes the
    // Kőszegi-style split tree: at each leaf, every ctor of the
    // scrutinee's family is {possible | impossible | undecided} against
    // the leaf's refined index substitution — an undecided ctor with no
    // covering arm triggers a case-split on a splittable index `var`
    // (`n ↦ 0` / `n ↦ m + 1` for a `Nat` sort, `n ↦ App{K, f⃗}` per ctor
    // for a declared index family), recursing until every leaf decides
    // or the bound runs out. A decided-reachable ctor with no arm is a
    // non-exhaustive error; a ctor still undecided at a final leaf is a
    // `coverage unknown` warning — bounded, never silent (decision 4b).
    //
    // Scope: single-scrutinee matches on `type.family_app` scrutinees
    // whose family was `data`-declared (the `family_ctors` census).
    // Nested/multi-scrutinee patterns are not split — the bound is the
    // honest cut.

    /// Probe one ctor at the current leaf: instantiate its signature
    /// under a full ctx snapshot and unify each result index against
    /// the leaf's scrutinee indices. Returns the status plus the
    /// scrutinee positions whose equations stayed undecided (the split
    /// candidates live there).
    fn ctor_leaf_status(
        &self,
        lang: &Language,
        ast: &Ast,
        ctor_name: &str,
        family: &str,
        eargs: &[Value],
        ctx: &mut TypecheckContext,
    ) -> (CtorStatus, Vec<usize>) {
        let Some(sig) = ctx
            .lookup(ctor_name)
            .filter(|v| tag_of(v) == Some("type.ctor_sig"))
            .cloned()
        else {
            // Declared in the census but no telescope to instantiate —
            // honestly undecidable.
            return (CtorStatus::Undecided, Vec::new());
        };
        // A probe is pure speculation: snapshot every mutable channel
        // the instantiation/unification can touch.
        let mark = ctx.mark();
        let saved_results = ctx.results.clone();
        let saved_metas = ctx.metas.clone();
        let diag_len = ctx.diagnostics.len();
        let mut status = CtorStatus::Possible;
        let mut undecided: Vec<usize> = Vec::new();
        match self.instantiate_ctor_full(lang, ast, &sig, ctx) {
            Ok(inst) => {
                let (_doms, res) = self.peel_ctor(&inst);
                if tag_of(&res) == Some("type.family_app")
                    && res.get("name").and_then(Value::as_str) == Some(family)
                {
                    let c_args = res
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    for (i, (s, c)) in eargs.iter().zip(c_args.iter()).enumerate() {
                        let (ok, decided) = unify_index(s, c, ctx);
                        if !ok {
                            if decided {
                                status = CtorStatus::Impossible;
                                break;
                            }
                            status = CtorStatus::Undecided;
                            undecided.push(i);
                        }
                    }
                } else {
                    // The ctor's result names another family — it cannot
                    // produce this scrutinee.
                    status = CtorStatus::Impossible;
                }
            }
            Err(_) => status = CtorStatus::Undecided,
        }
        ctx.restore(mark);
        ctx.results = saved_results;
        ctx.metas = saved_metas;
        ctx.diagnostics.truncate(diag_len);
        (status, undecided)
    }

    /// The split children of index var `v`'s sort: `Nat` gives `v ↦ 0`
    /// and `v ↦ m + 1`; a declared family gives `v ↦ App{K, f⃗}` per
    /// ctor (the index term `eval_index` then projects, e.g.
    /// `App{Succ, m} ↦ m + 1`). Each child carries the fresh vars it
    /// introduced with their own index sorts for deeper splits.
    fn split_branches(
        &self,
        lang: &Language,
        ast: &Ast,
        sort: &IndexSort,
        run: &mut CoverRun,
        ctx: &mut TypecheckContext,
    ) -> Vec<(Value, String, Vec<(String, IndexSort)>)> {
        match sort {
            IndexSort::Nat => {
                let m = format!("__split_{}", run.fresh);
                run.fresh += 1;
                vec![
                    (Value::from(0), "0".to_string(), Vec::new()),
                    (
                        bin_term("binary.add", var_term(&m), Value::from(1)),
                        format!("{m} + 1"),
                        vec![(m, IndexSort::Nat)],
                    ),
                ]
            }
            IndexSort::Family(g) => {
                let Some(ctors) = ctx.family_ctors.get(g).cloned() else {
                    return Vec::new();
                };
                let mut out = Vec::new();
                for k in &ctors {
                    let Some(sig) = ctx
                        .lookup(k)
                        .cloned()
                        .filter(|v| tag_of(v) == Some("type.ctor_sig"))
                    else {
                        continue;
                    };
                    let Ok(inst) = self.instantiate_ctor_full(lang, ast, &sig, ctx) else {
                        continue;
                    };
                    let (doms, _res) = self.peel_ctor(&inst);
                    let arg_nodes: Vec<Option<usize>> = sig
                        .get("args")
                        .and_then(Value::as_array)
                        .map(|xs| {
                            xs.iter()
                                .map(|a| a.get("node").and_then(Value::as_u64).map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut fresh: Vec<(String, IndexSort)> = Vec::new();
                    let mut args = Vec::with_capacity(doms.len());
                    for (i, (_pname, dom)) in doms.iter().enumerate() {
                        let fv = format!("__split_{}", run.fresh);
                        run.fresh += 1;
                        let arg_node = arg_nodes
                            .get(i)
                            .and_then(|o| *o)
                            .and_then(|ix| ast.get(AstRef::new(ix)));
                        if let Some(s) = ctor_arg_index_sort(dom, arg_node, ctx) {
                            fresh.push((fv.clone(), s));
                        }
                        args.push(var_term(&fv));
                    }
                    out.push((app_term(var_term(k), args), k.clone(), fresh));
                }
                out
            }
        }
    }

    /// One leaf of the split tree: probe every ctor against the
    /// scrutinee indices as the accumulated substitution evaluates
    /// them, record reachable-but-uncovered errors, and case-split a
    /// splittable index var while an undecided-uncovered ctor remains
    /// and the bounds allow it. Over-bound leaves warn `coverage
    /// unknown` — never a silent accept (decision 4b).
    #[allow(clippy::too_many_arguments)]
    fn cover_split_tree(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        family: &str,
        s_args: &[Value],
        ctors: &[String],
        arms: &[ArmCover],
        ctx: &mut TypecheckContext,
        run: &mut CoverRun,
        path: &mut Vec<(String, String)>,
        depth: usize,
    ) {
        run.nodes += 1;
        let eargs: Vec<Value> = s_args.iter().map(|a| eval_index(a, ctx)).collect();
        let mut undecided_positions: Vec<usize> = Vec::new();
        let mut open_ctors: Vec<String> = Vec::new();
        for cname in ctors {
            let (status, upos) = self.ctor_leaf_status(lang, ast, cname, family, &eargs, ctx);
            match status {
                CtorStatus::Possible => {
                    if !arm_covers(arms, cname) && !run.uncovered.iter().any(|(c, _)| c == cname) {
                        run.uncovered.push((cname.clone(), render_split_path(path)));
                    }
                }
                CtorStatus::Undecided => {
                    undecided_positions.extend(upos);
                    if !arm_covers(arms, cname) {
                        open_ctors.push(cname.clone());
                    }
                }
                CtorStatus::Impossible => {}
            }
        }
        if open_ctors.is_empty() {
            return;
        }
        let split = pick_split_var(&eargs, &undecided_positions, &run.sorts);
        let Some((vname, sort)) = split else {
            for c in open_ctors {
                run.warned.insert(c);
            }
            return;
        };
        if depth >= SPLIT_MAX_DEPTH || run.nodes + 1 > SPLIT_MAX_NODES {
            run.bound_hit = true;
            for c in open_ctors {
                run.warned.insert(c);
            }
            return;
        }
        let branches = self.split_branches(lang, ast, &sort, run, ctx);
        if branches.is_empty() || run.nodes + branches.len() > SPLIT_MAX_NODES {
            run.bound_hit = true;
            for c in open_ctors {
                run.warned.insert(c);
            }
            return;
        }
        for (term, desc, fresh) in branches {
            let saved_results = ctx.results.clone();
            let saved_metas = ctx.metas.clone();
            ctx.bind_result(vname.clone(), term);
            path.push((vname.clone(), desc));
            for (fv, fs) in &fresh {
                run.sorts.insert(fv.clone(), fs.clone());
                if index_sort_nonneg(fs, ctx) {
                    ctx.nonneg_index_vars.insert(fv.clone());
                }
            }
            self.cover_split_tree(
                lang,
                ast,
                node,
                family,
                s_args,
                ctors,
                arms,
                ctx,
                run,
                path,
                depth + 1,
            );
            path.pop();
            ctx.results = saved_results;
            ctx.metas = saved_metas;
            for (fv, _) in &fresh {
                run.sorts.remove(fv);
                ctx.nonneg_index_vars.remove(fv);
            }
        }
    }

    /// Drive split-tree coverage for a `match` on a `data`-declared
    /// family scrutinee. Errors on a decided-reachable ctor with no
    /// covering arm (`non-exhaustive`); warns `coverage unknown` per
    /// ctor still undecided at a final leaf.
    #[allow(clippy::too_many_arguments)]
    fn match_coverage(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        family: &str,
        s_args: &[Value],
        arms: &[ArmCover],
        index_sorts: HashMap<String, IndexSort>,
        ctx: &mut TypecheckContext,
    ) -> Result<(), LangError> {
        let Some(ctors) = ctx.family_ctors.get(family).cloned() else {
            return Ok(());
        };
        if ctors.is_empty() {
            return Ok(());
        }
        let mut run = CoverRun {
            nodes: 0,
            fresh: 0,
            sorts: index_sorts,
            uncovered: Vec::new(),
            warned: HashSet::new(),
            bound_hit: false,
        };
        let mut path = Vec::new();
        self.cover_split_tree(
            lang, ast, node, family, s_args, &ctors, arms, ctx, &mut run, &mut path, 0,
        );
        if !run.uncovered.is_empty() {
            let names = run
                .uncovered
                .iter()
                .map(|(c, _)| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let witness = run
                .uncovered
                .first()
                .map(|(_, w)| w.clone())
                .unwrap_or_default();
            let (plural, pronoun) = if run.uncovered.len() == 1 {
                ("", "it")
            } else {
                ("s", "them")
            };
            return Err(LangError::engine_node(
                format!(
                    "match: non-exhaustive — constructor{plural} {names} can occur{witness} \
                     but no arm covers {pronoun}"
                ),
                node,
            ));
        }
        for c in &run.warned {
            ctx.diagnostics.push(TypecheckDiagnostic {
                severity: "warning".to_string(),
                message: format!(
                    "match: coverage unknown — cannot decide whether constructor `{c}` \
                     is reachable{}",
                    if run.bound_hit {
                        " (split bound reached)"
                    } else {
                        ""
                    }
                ),
                span: node.span,
            });
        }
        Ok(())
    }

    /// Resolve the scrutinee's declared family and raw index args for
    /// coverage: `type.family_app{name, args}` direct, `type.family`
    /// as the arity-0 spelling. Returns `None` for non-family
    /// scrutinees — they get no coverage pass.
    fn match_scrutinee_family(&self, scrutinee_type: &Value) -> Option<(String, Vec<Value>)> {
        match tag_of(scrutinee_type) {
            Some("type.family_app") => Some((
                scrutinee_type
                    .get("name")
                    .and_then(Value::as_str)?
                    .to_string(),
                scrutinee_type
                    .get("args")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            )),
            Some("type.family") => Some((
                scrutinee_type
                    .get("name")
                    .and_then(Value::as_str)?
                    .to_string(),
                Vec::new(),
            )),
            _ => None,
        }
    }

    /// Per-index-position split sorts for `family`: the family's
    /// `base.elements` sorts crossed with its `nat` flags.
    fn family_index_sorts(
        &self,
        family: &str,
        scrutinee_type: &Value,
        ctx: &TypecheckContext,
    ) -> Vec<Option<IndexSort>> {
        let fam = if tag_of(scrutinee_type) == Some("type.family") {
            scrutinee_type.clone()
        } else {
            ctx.lookup(family).cloned().unwrap_or(Value::Null)
        };
        let Some(elems) = fam
            .get("base")
            .and_then(|b| b.get("elements"))
            .and_then(Value::as_array)
            .cloned()
        else {
            return Vec::new();
        };
        let flags = ctx
            .family_param_nat
            .get(family)
            .cloned()
            .unwrap_or_default();
        elems
            .iter()
            .enumerate()
            .map(|(i, e)| position_index_sort(Some(e), flags.get(i).copied().unwrap_or(false), ctx))
            .collect()
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

    // -----------------------------------------------------------------------
    // Instance resolution (phase-3 WS-A, ADR-0106 §11)
    // -----------------------------------------------------------------------
    //
    // `instance <C(i⃗)> := <dict>` declares `dict` as the dictionary
    // value for the constraint former application `C(i⃗)` — an algebra
    // is an ordinary object whose `type` link is the former app, and a
    // "trait instance" is a value you hold and pass. Implicit
    // `type.pi` domains `C(i⃗)` whose head has registered instances are
    // *evidence* binders: discharged by dictionary search (this table),
    // not by a fresh index meta — the second erasure class the
    // implicit-binder slot carries (runtime-relevant, NOT index-erased).

    /// `synth_instance` — elaborate the constraint to a
    /// `type.family_app{head, args}`, enforce coherence, check the
    /// dictionary, bind it in Γ *at the constraint type* under a
    /// generated name, and register `(head, args) ↦ dict-name`.
    ///
    /// Index variables in the constraint that name nothing bound
    /// implicitly generalize (`instance Eq (List a)` — the generic
    /// instance): they bind as rigid `var` terms for the elaboration,
    /// so the registered index shape carries `var{a}` holes that
    /// per-lookup unification instantiates. A bare `instance Eq` under
    /// a parameterized former mints `Eq ?` — unsolved metas in the
    /// shape are rewritten to per-instance rigid vars so a lookup can
    /// never *solve* the registered shape itself.
    ///
    /// Coherence: one instance per `(head, index-shape)` — a second
    /// registration whose index shape unifies with an existing one is
    /// a declaration-time error (no overlap/priority — decided
    /// policy; ordered/first-match is the deferred alternative).
    pub(crate) fn synth_instance_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node).ok_or_else(|| {
            LangError::engine_node(
                "synth_instance: instance needs [constraint, dict] children".to_string(),
                node,
            )
        })?;
        let constraint_idx = kids.first().and_then(Value::as_u64).ok_or_else(|| {
            LangError::engine_node("synth_instance: missing constraint child".to_string(), node)
        })? as usize;
        let dict_idx = kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
            LangError::engine_node("synth_instance: missing dict child".to_string(), node)
        })? as usize;
        // Implicit generalization — bind the constraint's unbound
        // index vars rigidly for the elaboration.
        let mut gen_vars = Vec::new();
        collect_unbound_index_vars(ast, AstRef::new(constraint_idx), ctx, &mut gen_vars);
        let old_results = ctx.results.clone();
        for n in &gen_vars {
            ctx.bind_result(n.clone(), var_term(n));
        }
        let constraint = self
            .dep_synth_type(lang, ast, constraint_idx, ctx)
            .map(|c| zonk_index(&c, ctx));
        ctx.results = old_results;
        let constraint = constraint?;
        let Some((head, args)) = instance_goal_parts(&constraint) else {
            return Err(LangError::engine_node(
                format!(
                    "synth_instance: constraint must be a former application `C(i⃗)` — got {}",
                    display_type(&constraint, ctx)
                ),
                node,
            ));
        };
        // The head must name a declared former — `type.family` is how
        // `data` registers its head; an instance for an unknown or
        // non-former head is a declaration error, not a silently
        // registered candidate.
        if !ctx
            .lookup(&head)
            .is_some_and(|b| tag_of(b) == Some("type.family"))
        {
            return Err(LangError::engine_node(
                format!(
                    "synth_instance: `{head}` is not a declared family — an instance constraint's head must be a `data` former"
                ),
                node,
            ));
        }
        // Arity: the constraint must apply the family's full index
        // telescope (a bare `instance Eq` mints `Eq ?` through the
        // family self-application — that still counts).
        let declared_arity = ctx
            .lookup(&head)
            .and_then(|b| b.get("base").cloned())
            .and_then(|b| b.get("elements").cloned())
            .and_then(|e| e.as_array().map(|a| a.len()));
        if let Some(n) = declared_arity
            && n != args.len()
        {
            return Err(LangError::engine_node(
                format!(
                    "synth_instance: instance constraint `{head}` takes {n} index argument(s), got {}",
                    args.len()
                ),
                node,
            ));
        }
        // Unsolved metas in the shape (the bare `instance Eq` catch-all)
        // become per-instance rigid vars — a lookup must instantiate the
        // *shape*, never solve the shared meta cell.
        let inst_no = ctx.instance_count;
        let mut arg_pos = 0usize;
        let args: Vec<Value> = args
            .iter()
            .map(|a| freeze_metas(a, inst_no, &mut arg_pos))
            .collect();
        // Coherence — no two instances may declare unifiable index
        // shapes for one head (decision: error at declaration time,
        // not ordered first-match at search). The candidate list is
        // cloned out — `unify_index`/`store_restore` borrow `ctx`
        // mutably inside the scan.
        if let Some(existing) = ctx.instances.get(&head).cloned() {
            for cand in &existing {
                if cand.args.len() != args.len() {
                    continue;
                }
                let mark = ctx.store_mark();
                let mut unifiable = true;
                for (ca, na) in cand.args.iter().zip(args.iter()) {
                    let (ok, _) = unify_index(ca, na, ctx);
                    if !ok {
                        unifiable = false;
                        break;
                    }
                }
                ctx.store_restore(mark);
                if unifiable {
                    return Err(LangError::engine_node(
                        format!(
                            "synth_instance: overlapping instance for `{}` — an instance with a unifiable index shape is already declared",
                            display_type(&constraint, ctx)
                        ),
                        node,
                    ));
                }
            }
        }
        // The dictionary is ordinary data — it must synth (a record,
        // a ctor application, an arrow). Its *declared* type is the
        // constraint: the instance assertion is that this value's
        // `type` link is the former application.
        self.synth_node(lang, ast, AstRef::new(dict_idx), ctx)?;
        let inst_name = format!("$instance:{head}:{inst_no}");
        ctx.instance_count += 1;
        ctx.extend(inst_name.clone(), constraint.clone());
        ctx.bind_result(inst_name.clone(), var_term(&inst_name));
        ctx.instances.entry(head).or_default().push(InstanceEntry {
            name: inst_name,
            args,
        });
        let mut m = Map::new();
        Ok(make_tag(ctx, "type.unit", std::mem::take(&mut m)))
    }

    /// Dictionary search: the declared candidates for `head` tried in
    /// declaration order — each candidate's index shape unifies with
    /// the goal's through [`unify_index`] under a store mark, and a
    /// failed candidate's meta/result side effects restore before the
    /// next try. The first unifiable candidate wins; declaration-time
    /// coherence makes the order observably irrelevant. Returns the
    /// elaboration term `var{dict-name}`.
    fn resolve_instance(
        &self,
        head: &str,
        goal_args: &[Value],
        ctx: &mut TypecheckContext,
    ) -> Option<Value> {
        let candidates = ctx.instances.get(head).cloned().unwrap_or_default();
        for cand in &candidates {
            if cand.args.len() != goal_args.len() {
                continue;
            }
            let mark = ctx.store_mark();
            let mut ok = true;
            for (ca, ga) in cand.args.iter().zip(goal_args.iter()) {
                let (u, _) = unify_index(ca, ga, ctx);
                if !u {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(var_term(&cand.name));
            }
            ctx.store_restore(mark);
        }
        None
    }

    /// The term filling one implicit `type.pi` binder at a call site:
    /// an implicit domain `C(i⃗)` whose head has registered instances
    /// is *evidence* — discharged by dictionary search before the
    /// index-meta fallback. A goal whose index args still hold
    /// unsolved metas (the drain reaches `{eq : Eq a}` before the
    /// explicit args solve `?a`) becomes a deferred obligation —
    /// retried at end-of-call and at end-of-module — while a plain
    /// meta fills the binder meanwhile.
    fn implicit_pi_fill(
        &self,
        node: &AstNode,
        param: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let goal = zonk_index(param, ctx);
        let Some((head, args)) = instance_goal_parts(&goal) else {
            return Ok(meta_value(ctx.fresh_meta()));
        };
        if !ctx.instances.contains_key(&head) {
            return Ok(meta_value(ctx.fresh_meta()));
        }
        if args.iter().any(|a| has_unsolved_meta(a, ctx)) {
            let m = ctx.fresh_meta();
            ctx.pending_instances.push(crate::PendingInstance {
                meta: m,
                goal,
                span: node.span,
            });
            return Ok(meta_value(m));
        }
        let guarded = guard_rigid_vars(&args);
        match self.resolve_instance(&head, &guarded, ctx) {
            Some(dict) => Ok(dict),
            None => Err(LangError::engine_node(
                format!("typecheck: no instance for `{}`", display_type(&goal, ctx)),
                node,
            )),
        }
    }

    /// Retry deferred evidence obligations — at the end of a call
    /// (`final_ = false`: determined goals resolve or requeue;
    /// underdetermined ones stay pending) and at the end of the
    /// module (`final_ = true`: every still-open obligation is an
    /// honest `no instance` error, never a silently unsolved
    /// dictionary meta).
    pub(crate) fn retry_pending_instances(
        &self,
        ctx: &mut TypecheckContext,
        final_: bool,
    ) -> Result<(), LangError> {
        if ctx.pending_instances.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut ctx.pending_instances);
        let mut still = Vec::new();
        for p in pending {
            let goal = zonk_index(&p.goal, ctx);
            let Some((head, args)) = instance_goal_parts(&goal) else {
                continue;
            };
            if args.iter().any(|a| has_unsolved_meta(a, ctx)) {
                if final_ {
                    let mut e = LangError::engine(format!(
                        "typecheck: no instance for `{}` — the constraint's index is underdetermined",
                        display_type(&goal, ctx)
                    ));
                    if let Some(span) = p.span {
                        e = e.with_span(span);
                    }
                    return Err(e);
                }
                still.push(crate::PendingInstance {
                    meta: p.meta,
                    goal,
                    span: p.span,
                });
                continue;
            }
            let guarded = guard_rigid_vars(&args);
            match self.resolve_instance(&head, &guarded, ctx) {
                Some(dict) => ctx.solve_meta(p.meta, dict),
                None if final_ => {
                    let mut e = LangError::engine(format!(
                        "typecheck: no instance for `{}`",
                        display_type(&goal, ctx)
                    ));
                    if let Some(span) = p.span {
                        e = e.with_span(span);
                    }
                    return Err(e);
                }
                // Determined but unresolvable for now — a later
                // `instance` declaration may still satisfy it; the
                // final drain owns the error.
                None => still.push(crate::PendingInstance {
                    meta: p.meta,
                    goal,
                    span: p.span,
                }),
            }
        }
        ctx.pending_instances = still;
        Ok(())
    }
    pub(crate) fn dep_unify(
        &self,
        a: &Value,
        b: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<(), LangError> {
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            if a == b {
                return Ok(());
            }
            // A solved meta stands for its solution: dereference it before
            // the meta-solve shortcut so a second constraint on a shared
            // meta checks against the *solution* — re-solving the cell
            // would silently overwrite it. This is what rejects the IO
            // widen: `x ~> x` synths `type.function{?m, ?m}` (param and
            // body share the meta), so checking it against
            // `IO(sa) ~> IO(sb)` solves `?m := IO(sa)` at the domain and
            // leaves `dep_unify(IO(sa), IO(sb))` — signature subtyping
            // through the IO arm below — at the codomain.
            let za;
            let zb;
            let a = if extract_meta_id(a).is_some() {
                za = self.zonk(a, ctx);
                &za
            } else {
                a
            };
            let b = if extract_meta_id(b).is_some() {
                zb = self.zonk(b, ctx);
                &zb
            } else {
                b
            };
            let _t = crate::stats::timer(&crate::stats::T_DEP_UNIFY);
            crate::stats::dep_unify(a, b);
            if a == b {
                return Ok(());
            }
            if extract_meta_id(a).is_some() || extract_meta_id(b).is_some() {
                return self.unify(a, b, ctx);
            }
            // `type.unknown` is the dynamic top (see `unify`): consistent
            // with every type on either side, in both the tagged and the
            // domain-encoded spellings. Checked here — before the tag match
            // — because the `io_headed` signature-subtyping arm below would
            // otherwise reject `IO` vs `unknown` before `self.unify`'s
            // accept could fire.
            if is_unknown_ty(a) || is_unknown_ty(b) {
                return Ok(());
            }
            match (tag_of(a), tag_of(b)) {
                (Some("type.family_app"), Some("type.family_app")) => {
                    let na = a.get("name").and_then(Value::as_str).unwrap_or("");
                    let nb = b.get("name").and_then(Value::as_str).unwrap_or("");
                    if na != nb {
                        return Err(LangError::engine(format!(
                            "typecheck: cannot unify — expected {}, got {}",
                            display_type(b, ctx),
                            display_type(a, ctx)
                        )));
                    }
                    let xs = a
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let ys = b
                        .get("args")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    if xs.len() != ys.len() {
                        return Err(LangError::engine(format!(
                            "typecheck: cannot unify — expected {}, got {}",
                            display_type(b, ctx),
                            display_type(a, ctx)
                        )));
                    }
                    for (x, y) in xs.iter().zip(ys.iter()) {
                        let (ok, decided) = unify_index(x, y, ctx);
                        if !ok {
                            if decided {
                                return Err(LangError::engine(format!(
                                    "typecheck: cannot unify — expected {}, got {}",
                                    display_type(b, ctx),
                                    display_type(a, ctx)
                                )));
                            }
                            // Stuck indices still try the structural path
                            // (e.g. type-argument metas).
                            self.unify(x, y, ctx).map_err(|_| {
                                LangError::engine(format!(
                                    "typecheck: cannot unify — expected {}, got {}",
                                    display_type(b, ctx),
                                    display_type(a, ctx)
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
                        return Err(LangError::engine(format!(
                            "typecheck: cannot unify Path sources — expected {}, got {}",
                            display_type(&fb, ctx),
                            display_type(&fa, ctx)
                        )));
                    }
                    let (ok, _) = unify_index(&ta, &tb, ctx);
                    if !ok {
                        return Err(LangError::engine(format!(
                            "typecheck: cannot unify Path targets — expected {}, got {}",
                            display_type(&tb, ctx),
                            display_type(&ta, ctx)
                        )));
                    }
                    Ok(())
                }
                (Some("type.pi"), Some("type.pi")) => {
                    // Cubical path types (ADR-0208 P1): a `path` payload
                    // marks the Pi as `Path`/`PathP` — endpoints `x`/`y`
                    // are part of the type's identity and unify as index
                    // terms alongside the param; a path type never
                    // unifies with a plain `type.pi`. Plain pis keep the
                    // structural `unify` reading (param + payload fields;
                    // `body_closure` is skipped there as before).
                    match (a.get("path"), b.get("path")) {
                        (Some(_), None) | (None, Some(_)) => {
                            return Err(LangError::engine(
                                "typecheck: cannot unify — path type vs plain Pi".to_string(),
                            ));
                        }
                        (Some(pa), Some(pb)) => {
                            for key in ["x", "y"] {
                                let (ok, _) = unify_index(
                                    pa.get(key).unwrap_or(&Value::Null),
                                    pb.get(key).unwrap_or(&Value::Null),
                                    ctx,
                                );
                                if !ok {
                                    return Err(LangError::engine(
                                        "typecheck: cannot unify — path endpoints differ"
                                            .to_string(),
                                    ));
                                }
                            }
                            let pa_ty = a.get("param").cloned().unwrap_or(Value::Null);
                            let pb_ty = b.get("param").cloned().unwrap_or(Value::Null);
                            self.dep_unify(&pa_ty, &pb_ty, ctx)?;
                            // Bodies compare through their data proxy —
                            // `const_body` stores the family type
                            // verbatim, `apply_type` the `I → Type`
                            // family (full applied-body unification
                            // needs the AST this judgment does not
                            // carry; the family's own `type.pi` still
                            // compares by param).
                            let ba = closure_body_proxy(a);
                            let bb = closure_body_proxy(b);
                            self.dep_unify(&ba, &bb, ctx)?;
                            Ok(())
                        }
                        (None, None) => self.unify(a, b, ctx),
                    }
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
                | (Some("type.family_app"), Some("type.function")) => {
                    Err(LangError::engine(format!(
                        "typecheck: cannot unify — expected {}, got {}",
                        display_type(b, ctx),
                        display_type(a, ctx)
                    )))
                }
                (Some("type.fibrant"), Some("type.fibrant")) => {
                    // `isFibrant(A)` types (ADR-0208 P3): equality is
                    // equality of the fibrant target, routed through
                    // `dep_unify` so a `Pi`/`Glue` target compares
                    // bodies, not just domains (the model's
                    // `unify_dependent_` arm).
                    let oa = a.get("of").cloned().unwrap_or(Value::Null);
                    let ob = b.get("of").cloned().unwrap_or(Value::Null);
                    self.dep_unify(&oa, &ob, ctx)
                }
                (Some("type.glue"), Some("type.glue")) => {
                    // `Glue A φ T̂` types (ADR-0208 P3): `family`/
                    // `section` unify as types; the face `φ` is a
                    // `face` record compared structurally (the model's
                    // `==` — unification cannot see inside it).
                    let fa = a.get("family").cloned().unwrap_or(Value::Null);
                    let fb = b.get("family").cloned().unwrap_or(Value::Null);
                    self.dep_unify(&fa, &fb, ctx)?;
                    if a.get("face") != b.get("face") {
                        return Err(LangError::engine(
                            "typecheck: cannot unify — glue faces differ".to_string(),
                        ));
                    }
                    let sa = a.get("section").cloned().unwrap_or(Value::Null);
                    let sb = b.get("section").cloned().unwrap_or(Value::Null);
                    self.dep_unify(&sa, &sb, ctx)
                }
                (Some("type.equiv"), Some("type.equiv")) => {
                    // `Equiv(A, B)` types (ADR-0208 P4): `dom`/`cod`
                    // unify as types — the model's
                    // `unify_dependent_` arm.
                    let da = a.get("dom").cloned().unwrap_or(Value::Null);
                    let db = b.get("dom").cloned().unwrap_or(Value::Null);
                    self.dep_unify(&da, &db, ctx)?;
                    let ca = a.get("cod").cloned().unwrap_or(Value::Null);
                    let cb = b.get("cod").cloned().unwrap_or(Value::Null);
                    self.dep_unify(&ca, &cb, ctx)
                }
                (Some("type.equiv"), Some("type.record"))
                | (Some("type.record"), Some("type.equiv")) => {
                    // `Equiv(A, B)` vs a record type — the record must
                    // carry `to : dom ~> cod` and `fro : cod ~> dom`
                    // (ADR-0208 P4, `unify_equiv_record_`).
                    let (equiv_t, record_t) = if tag_of(a) == Some("type.equiv") {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    self.unify_equiv_record(equiv_t, record_t, ctx)
                }
                _ => {
                    // `Former{type.io}` — signature subtyping (record-width
                    // on the sig index): `IO(s) <: IO(s')` iff `s'` names
                    // at least the ops of `s` — directed, unlike index conv.
                    if io_headed(a) || io_headed(b) {
                        let mark = ctx.vcheck.store_mark();
                        let va = domain_lift(a, ctx);
                        let vb = domain_lift(b, ctx);
                        if ctx.vcheck.subtype(&va, &vb) {
                            return Ok(());
                        }
                        ctx.vcheck.store_restore(mark);
                        return Err(LangError::engine(format!(
                            "typecheck: cannot unify — expected {}, got {}",
                            display_type(b, ctx),
                            display_type(a, ctx)
                        )));
                    }
                    self.unify(a, b, ctx)
                }
            }
        })
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
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            // `match` checked against an expected type: motive inference —
            // a `return … with`-less match takes its motive from the expected
            // type's family-app index pattern (or the constant expected
            // type), checking each arm body against it rather than joining
            // synthesized body types.
            if let Some(node) = ast.get(node_ref)
                && node.kind == "match"
            {
                match self.dep_check_match(lang, ast, node, expected, ctx) {
                    Ok(Some(ty)) => return Some(Ok(ty)),
                    Ok(None) => {}
                    Err(e) => return Some(Err(e)),
                }
            }
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
                    if tag == "type.pi" && expected.get("implicit") == Some(&Value::Bool(true)) {
                        // Implicit index binder (phase-2 WS-C `f<a, …}`): the
                        // binder does NOT consume a lambda — bind its name
                        // rigidly (Γ + results) and check the same node
                        // against the already-elaborated `body`.
                        Some(self.dep_intro_implicit(lang, ast, node_ref, expected, ctx))
                    } else if tag == "type.pi"
                        && expected.get("path").is_some()
                        && node.kind != "function"
                    {
                        // A `type.pi` carrying a `path` payload is a cubical
                        // path type (ADR-0208 P1): `(i:I) → A`/`(i:I) → F i`
                        // with endpoints `x`,`y`. `refl` and path witnesses
                        // check through `dep_check_path_intro`; a `function`
                        // node is still a genuine Π-introduction (`λi. body`
                        // IS a path) and falls to `dep_intro_arrow`.
                        Some(self.dep_check_path_intro(lang, ast, node_ref, expected, ctx))
                    } else if node.kind == "function" {
                        Some(self.dep_intro_arrow(lang, ast, node, expected, ctx))
                    } else if tag == "type.sigma" && node.kind == "array" {
                        Some(self.dep_intro_sigma(lang, ast, node, expected, ctx))
                    } else {
                        None
                    }
                }
                // `type.array{elements}` (the tuple/product convention):
                // a `paren`/`array` node checks positionally — element `i`
                // against `elements[i]`.
                "type.array" => {
                    let node = ast.get(node_ref)?;
                    let els = expected.get("elements").and_then(Value::as_array).cloned();
                    match (els, node.kind.as_str()) {
                        (Some(els), "paren" | "array") => {
                            let kids = children_of(node).cloned().unwrap_or_default();
                            if kids.len() != els.len() {
                                return Some(Err(LangError::engine(format!(
                                    "typecheck: tuple arity {} does not match expected {}",
                                    kids.len(),
                                    els.len()
                                ))));
                            }
                            for (k, el) in kids.iter().zip(els.iter()) {
                                let idx = match k.as_u64() {
                                    Some(i) => AstRef::new(i as usize),
                                    None => {
                                        return Some(Err(LangError::engine(
                                            "typecheck: bad tuple child",
                                        )));
                                    }
                                };
                                if let Err(e) = self.check_node(lang, ast, idx, el, ctx) {
                                    return Some(Err(e));
                                }
                            }
                            Some(Ok(expected.clone()))
                        }
                        _ => None,
                    }
                }
                // `type.function` is the *structural* arrow: its `check`
                // scheme checks params/body positionally, which is the
                // only path that handles non-binder params (synthetic
                // `value` params in unit tests). Route to the dependent
                // intro — bind the name at the domain, check the body
                // against the codomain — only when the params side spells
                // an actual binder: `param_names` accepts any
                // string-payload leaf, so also require the shape data —
                // `pattern_role` (`var`/`arr`/`rest`/`sequence`), a
                // `subject` role (`type-annot`), or a variadic params
                // collection (`paren`/`tuple`/`array`). A bare `value`/`lit`
                // satisfies none of these.
                "type.function" => {
                    let node = ast.get(node_ref)?;
                    if node.kind != "function" {
                        None
                    } else {
                        let binder = children_of(node)
                            .and_then(|kids| kids.first().and_then(Value::as_u64))
                            .and_then(|i| ast.get(AstRef::new(i as usize)))
                            .map(|p| {
                                let shaped = lang.pattern_role(&p.kind).is_some()
                                    || lang.has_variadic_role(&p.kind)
                                    || lang
                                        .child_role_index(&p.kind, "subject")
                                        .ok()
                                        .flatten()
                                        .is_some();
                                shaped
                                    && lang
                                        .param_names(ast, p)
                                        .map(|(names, rest)| !names.is_empty() || rest.is_some())
                                        .unwrap_or(false)
                            })
                            .unwrap_or(false);
                        if binder {
                            Some(self.dep_intro_arrow(lang, ast, node, expected, ctx))
                        } else {
                            None
                        }
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
                                if decided {
                                    " — decided inequality"
                                } else {
                                    ""
                                }
                            ))))
                        }
                    } else if node.kind == "var" {
                        // A Γ/results-bound `Path` witness also inhabits Path.
                        let wit = node
                            .payload
                            .as_str()
                            .and_then(|n| {
                                ctx.results
                                    .get(n)
                                    .cloned()
                                    .or_else(|| ctx.lookup(n).cloned())
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
                // `Equiv(A, B)` (ADR-0208 P4): the record literal
                // checks through the `type.equiv`/`type.record`
                // `dep_unify` arm — synthesize then unify, the same
                // fallback the model's `check_dependent_fallback_`
                // applies.
                "type.equiv" => {
                    let ty = match self.synth_node(lang, ast, node_ref, ctx) {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e)),
                    };
                    return Some(match self.dep_unify(&ty, expected, ctx) {
                        Ok(()) => Ok(ty),
                        Err(e) => Err(e),
                    });
                }
                _ => None,
            }
        })
    }

    /// Path introduction against a cubical path type (ADR-0208 P1 — port
    /// of the model's `check_path_intro_`): `expected` is a `type.pi`
    /// over `type.interval` carrying a `path{x,y}` endpoint payload.
    /// `refl` inhabits the reflexive case — the expected endpoints must
    /// unify — and a path witness (a `Path{frm,to}`/`type.eq` value, a
    /// path-carrying `type.pi`, bound in `results` as an `eq_<name>`
    /// witness, in Γ as a path-typed hypothesis, or synthesized as the
    /// node's own type) inhabits the expected path when its endpoints
    /// unify with the expected ones. Anything else is not a path.
    fn dep_check_path_intro(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast.get(node_ref).ok_or_else(|| {
            LangError::engine("dep_check_path_intro: node out of bounds".to_string())
        })?;
        let epath = expected.get("path").cloned().unwrap_or(Value::Null);
        let efrm = zonk_index(epath.get("x").unwrap_or(&Value::Null), ctx);
        let eto = zonk_index(epath.get("y").unwrap_or(&Value::Null), ctx);
        if node.kind == "var" && node_name(node).as_deref() == Some("refl") {
            let (ok, decided) = unify_index(&efrm, &eto, ctx);
            return if ok {
                Ok(expected.clone())
            } else {
                Err(LangError::engine(format!(
                    "typecheck: refl cannot prove Path({efrm:?}, {eto:?}){}",
                    if decided {
                        " — decided inequality"
                    } else {
                        ""
                    }
                )))
            };
        }
        // Witness carriers, in the model's pv→gt order: a var's bound
        // values (`results` `eq_<name>` Path records, then its Γ/global
        // type — a path-typed hypothesis), then the node's synthesized
        // type. The first carrier that *is* a path decides; its
        // endpoints unify once against the expected ones.
        let mut carriers: Vec<Value> = Vec::new();
        if node.kind == "var"
            && let Some(n) = node_name(node)
        {
            if let Some(w) = ctx.results.get(&n) {
                carriers.push(w.clone());
            }
            if let Some(w) = ctx.lookup(&n) {
                carriers.push(w.clone());
            }
        }
        if let Ok(wt) = self.synth_node(lang, ast, node_ref, ctx) {
            carriers.push(wt);
        }
        for cand in &carriers {
            if let Some((wfrm, wto)) = path_witness_endpoints(cand) {
                let (uf, _) = unify_index(&wfrm, &efrm, ctx);
                let (ut, _) = if uf {
                    unify_index(&wto, &eto, ctx)
                } else {
                    (false, false)
                };
                return if uf && ut {
                    Ok(expected.clone())
                } else {
                    Err(LangError::engine(
                        "typecheck: path endpoints do not match the expected Path".to_string(),
                    ))
                };
            }
        }
        Err(LangError::engine(
            "typecheck: term is not a Path witness".to_string(),
        ))
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
        let _kids = children_of(node)
            .ok_or_else(|| LangError::engine("dep_intro: function has no children"))?;
        let params_idx = role_child_ref(lang, node, "params")
            .or_else(|| child_at(ast, node, 0))
            .ok_or_else(|| LangError::engine("dep_intro: no params child"))?;
        let body_idx = role_child_ref(lang, node, "body")
            .or_else(|| child_at(ast, node, 1))
            .ok_or_else(|| LangError::engine("dep_intro: no body child"))?;
        let pnode = ast
            .get(params_idx)
            .ok_or_else(|| LangError::engine("dep_intro: params node out of bounds".to_string()))?;
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
    /// Any node checked against an *implicit* `type.pi` — the `f<a, …>`
    /// implicit index binder (phase-2 WS-C). The binder never consumes a
    /// lambda: it is bound rigidly (`var{b}` in `results`, the declared
    /// `param` sort in Γ) and the *same* node is checked against `body`.
    /// Nested implicit binders recurse through `check_node`/`dep_check_intro`.
    fn dep_intro_implicit(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let binder = expected
            .get("binder")
            .and_then(Value::as_str)
            .unwrap_or("_i")
            .to_string();
        let param = expected.get("param").cloned().unwrap_or(Value::Null);
        let mark = ctx.mark();
        let old_results = ctx.results.clone();
        // The binder scopes into the body as a *rigid index variable* —
        // the same `var{b}` in Γ and `results` (the `synth_assign`
        // convention), so a name used inside a body annotation —
        // `v : Vec(n, a)` under `f<n, a>` — resolves to the index
        // atom, not the binder's sort. Binding the `param` sort in Γ
        // made `<n, a>`-style binders read as their unsolved sort meta
        // (`Vec(n, a)` ≡ `Vec(?m1, ?m0)`) — and even a declared
        // `{n : Nat}` sort is not `n`, it is `n`'s type. Only an
        // *evidence* binder (`<ev : Eq(a)>`, `evidence: true` on the
        // pi) keeps its former-app type in Γ — `ev` is a runtime
        // dictionary its eliminations match on, not an index atom.
        let evidence = expected.get("evidence") == Some(&Value::Bool(true));
        ctx.extend(
            binder.clone(),
            if evidence { param } else { var_term(&binder) },
        );
        ctx.bind_result(binder.clone(), var_term(&binder));
        let body = expected.get("body").cloned().unwrap_or(Value::Null);
        let r = self.check_node(lang, ast, node_ref, &body, ctx);
        ctx.restore(mark);
        ctx.results = old_results;
        r?;
        Ok(expected.clone())
    }

    /// `[a, b]` checked against `type.sigma` — componentwise: `a` against
    /// `param`, `b` against the closure applied at `a`'s *term`.
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
        let fst_idx = kids
            .first()
            .and_then(Value::as_u64)
            .ok_or_else(|| LangError::engine("dep_intro: sigma pair is empty".to_string()))?
            as usize;
        let snd_idx = kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
            LangError::engine("dep_intro: sigma pair needs 2 elements".to_string())
        })? as usize;
        let param = expected.get("param").cloned().unwrap_or(Value::Null);
        self.check_node(lang, ast, AstRef::new(fst_idx), &param, ctx)?;
        let fst_node = ast
            .get(AstRef::new(fst_idx))
            .ok_or_else(|| LangError::engine("dep_intro: fst node out of bounds".to_string()))?;
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
        let bc = ty
            .get("body_closure")
            .ok_or_else(|| LangError::engine("dep_closure_apply: no body_closure".to_string()))?;
        // Cubical path closures (ADR-0208 P1) carry no `body_child_idx`:
        // `const_body` is a stored type value returned verbatim (the
        // homogeneous `(i:I) → A` ignores the dimension), and
        // `apply_type` is a stored `I → Type` family value applied at
        // the term this closure's `var_name` is bound to in `results`
        // (set by `dep_intro_arrow` — `(i:I) → F i` has no AST node for
        // `F i`, so the body is computed by re-entering the term
        // applicator on the family).
        if let Some(cb) = bc.get("const_body") {
            return Ok(cb.clone());
        }
        if let Some(fam) = bc.get("apply_type") {
            let var_name = bc.get("var_name").and_then(Value::as_str).unwrap_or("i");
            let term = ctx
                .results
                .get(var_name)
                .cloned()
                .unwrap_or_else(|| var_term(var_name));
            return self.dep_closure_apply_term(lang, ast, fam, &term, ctx);
        }
        // Dependent binder chains (`rest` — dependent pairs and
        // typed-param arrows): expand one `type.sigma`/`type.pi` level
        // per remaining binder. `var_name` is already bound by the
        // caller (`dep_intro_arrow` sets `results[var_name]`).
        if let Some(rest) = bc.get("rest").and_then(Value::as_array)
            && !rest.is_empty()
        {
            return self.dep_closure_chain(lang, ast, bc, rest, ctx);
        }
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

    /// The eliminator half of a `type.sigma` pair pattern
    /// `[witness, payload]`: the witness binds *rigidly* — `var{name}`
    /// in both Γ and `results`, the same convention as implicit index
    /// binders ([`Self::dep_intro_implicit`]) — so a destructure `m`
    /// used as a family index elaborates to `var{m}` rather than the
    /// `Nat` sort Γ would otherwise hand back for a bare name. The
    /// payload binds against the sigma's `body_closure` instantiated
    /// at that atom, which is what lets `Vec(m, a)` and the repack
    /// `Vec(m + 1, a)` unify.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_sigma_destructure(
        &self,
        lang: &Language,
        ast: &Ast,
        witness_node: &AstNode,
        witness_ref: AstRef,
        payload_ref: AstRef,
        scrutinee_type: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<(), LangError> {
        let param = scrutinee_type.get("param").cloned().unwrap_or(Value::Null);
        let wname = node_name(witness_node).unwrap_or_default();
        let witness_term = if !wname.is_empty() && wname != "_" {
            // Γ keeps the declared `param` sort for *term* positions
            // (`Succ(m)` checks `m` against `Nat`); `results` carries
            // `var{m}` so the name reads as a rigid index atom in
            // `dep_synth_var_type` — the implicit-binder convention.
            ctx.extend(wname.clone(), param);
            ctx.bind_result(wname.clone(), var_term(&wname));
            var_term(&wname)
        } else {
            // A non-`var` witness pattern (literal/compound) binds
            // normally and applies the closure at its own index term.
            self.bind_pattern_vars(lang, ast, witness_ref, &param, ctx)?;
            self.node_to_index_term(lang, ast, witness_node, ctx)
        };
        let payload_ty =
            self.dep_closure_apply_term(lang, ast, scrutinee_type, &witness_term, ctx)?;
        self.bind_pattern_vars(lang, ast, payload_ref, &payload_ty, ctx)
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
        let bc = ty
            .get("body_closure")
            .ok_or_else(|| LangError::engine("dep_closure_apply: no body_closure".to_string()))?;
        // Cubical path closures (ADR-0208 P1): `const_body` returns the
        // stored type verbatim; `apply_type` re-enters this applicator
        // on the stored `I → Type` family at the same dimension term —
        // `(i:I) → F i` has no AST body of its own.
        if let Some(cb) = bc.get("const_body") {
            return Ok(cb.clone());
        }
        if let Some(fam) = bc.get("apply_type") {
            return self.dep_closure_apply_term(lang, ast, fam, &term, ctx);
        }
        let var_name = bc
            .get("var_name")
            .and_then(Value::as_str)
            .unwrap_or("x")
            .to_string();
        // `rest`-chain closures may carry no `body_child_idx` (sigma
        // chains terminate on the last binder's annotation), so the
        // index is only required on the non-chain path below.
        let body_idx = bc.get("body_child_idx").and_then(Value::as_u64).map(|v| v as usize);
        let bound = eval_index(term, ctx);
        // Occurs guard: `var_name` occurring inside its own binding
        // term (`Sigma(Nat, m ~> Vec(m, a))` applied at `var{m}+1` —
        // the user's `m` and the closure's `m` collide). Binding under
        // `var_name` directly mints a self-referential `results` entry
        // (`eval_index(var{m})` re-applies the rewrite — `var{m} + 64`
        // at the fuel bound) *and* poisons outer-scope `var{m}` atoms
        // (`ys`'s `Vec(var{m}, a)`). Alpha-rename the *binding key*
        // through `fam_scope` instead — the body's `var{m}` resolves
        // `__bound_m` → `var{m} + 1` while the outer `var{m}` atom
        // stays untouched (`__bound_m` then unifies with `var{m}` at
        // comparison time — a rigid-var rewrite, the `n ↦ ?k + 1`
        // convention).
        let scoped_key = if var_occurs(&var_name, &bound) {
            let internal = format!("__bound_{var_name}");
            ctx.fam_scope.insert(var_name.clone(), internal.clone());
            internal
        } else {
            var_name.clone()
        };
        let old = ctx.results.get(&scoped_key).cloned();
        ctx.bind_result(scoped_key.clone(), bound);
        // Dependent binder chains (`rest` — dependent pairs and
        // typed-param arrows): expand one `type.sigma`/`type.pi` level
        // per remaining binder, inside this binding's scope.
        let r = if let Some(rest) = bc.get("rest").and_then(Value::as_array)
            && !rest.is_empty()
        {
            self.dep_closure_chain(lang, ast, bc, rest, ctx)
        } else {
            let body_idx = body_idx.ok_or_else(|| {
                LangError::engine("dep_closure_apply: no body idx".to_string())
            })?;
            self.dep_synth_type(lang, ast, body_idx, ctx)
        };
        match old {
            Some(v) => ctx.bind_result(scoped_key.clone(), v),
            None => {
                ctx.results.remove(&scoped_key);
            }
        }
        if scoped_key != var_name {
            ctx.fam_scope.remove(&var_name);
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
    /// indices when a `return … with` clause is present. A decided
    /// index-unsatisfiable arm is *auto-discharged*: coverage doesn't
    /// require it, its body is unchecked but allowed, and `absurd`
    /// stays legal purely as explicit documentation of the dead
    /// branch. `absurd` on a *reachable* arm remains an error.
    #[allow(clippy::too_many_arguments)]
    /// How the per-arm expected type of a dependent `match` is produced:
    /// an explicit `return … with` AST motive, or an *inferred* motive
    /// built from the expected result type (phase-2 WS-C) — `{vars,
    /// template}` where `template` is a type value carrying `var{vars[i]}`
    /// holes substituted with the refined index terms per arm.
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
        self.synth_match_with_motive(
            lang,
            ast,
            node,
            arm_indices,
            scrutinee_idx,
            motive_idx.map(MatchMotive::Ast).as_ref(),
            scrutinee_type,
            ctx,
        )
    }

    /// The motive-general form of [`Self::synth_match_dependent`]: same
    /// arm machinery, but the motive may be an inferred
    /// `{vars, template}` instead of an AST clause.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synth_match_with_motive(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        arm_indices: &[usize],
        scrutinee_idx: usize,
        motive: Option<&MatchMotive>,
        scrutinee_type: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        // --- WS-C pre-pass: scrutinee index-var sorts + nonneg marks ---
        // Rigid `var`s in a `Nat`-sorted index position are known
        // nonnegative for the duration of this match — that's what lets
        // `0 ≡ n + 1` decide `VNil` impossible on a `Vec(Succ n)`
        // scrutinee. Scoped: restored before every return so a
        // same-named binder elsewhere can't inherit the fact.
        let saved_nonneg = ctx.nonneg_index_vars.clone();
        let scrut = self.match_scrutinee_family(scrutinee_type);
        let mut index_sorts: HashMap<String, IndexSort> = HashMap::new();
        if let Some((fname, sargs)) = &scrut {
            let pos_sorts = self.family_index_sorts(fname, scrutinee_type, ctx);
            for (i, sa) in sargs.iter().enumerate() {
                let Some(sort) = pos_sorts.get(i).cloned().flatten() else {
                    continue;
                };
                let ea = eval_index(sa, ctx);
                let mut vars = std::collections::BTreeSet::new();
                collect_index_vars(&ea, &mut vars);
                for v in vars {
                    index_sorts.entry(v.clone()).or_insert_with(|| sort.clone());
                    if index_sort_nonneg(&sort, ctx) {
                        ctx.nonneg_index_vars.insert(v);
                    }
                }
            }
        }
        let mut arm_types: Vec<Value> = Vec::new();
        let mut arm_covers: Vec<ArmCover> = Vec::new();
        for &arm_idx in arm_indices {
            let arm = ast
                .get(AstRef::new(arm_idx))
                .ok_or_else(|| LangError::engine(format!("match: arm {arm_idx} out of bounds")))?;
            let kids =
                children_of(arm).ok_or_else(|| LangError::engine("match: arm has no children"))?;
            if kids.len() < 2 {
                return Err(LangError::engine(
                    "match: arm needs [pattern, body]".to_string(),
                ));
            }
            let pat_idx = kids[0]
                .as_u64()
                .ok_or_else(|| LangError::engine("match: pattern index not a u64".to_string()))?
                as usize;
            let body_idx = kids[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("match: body index not a u64".to_string()))?
                as usize;
            let pattern = ast.get(AstRef::new(pat_idx)).ok_or_else(|| {
                LangError::engine("match: pattern node out of bounds".to_string())
            })?;
            let body_node = ast
                .get(AstRef::new(body_idx))
                .ok_or_else(|| LangError::engine("match: body node out of bounds".to_string()))?;
            let is_absurd = body_node.kind == "absurd";

            let mark = ctx.mark();
            let meta_mark = ctx.metas.len();
            let old_results = ctx.results.clone();

            // Resolve the pattern to (ctor_name, arg var names).
            let (cons_name, arg_names) = match_pattern_head(ast, pattern);
            // WS-C: classify the arm for the coverage pass — `absurd`
            // covers nothing; a `var`/`call` head naming a ctor of this
            // family covers that ctor; any other binding shape catches
            // everything.
            arm_covers.push(if is_absurd {
                ArmCover::Absurd
            } else {
                match &cons_name {
                    Some(n) => {
                        let family = scrut.as_ref().map(|(f, _)| f.as_str()).unwrap_or("");
                        let is_own_ctor = ctx
                            .lookup(n)
                            .filter(|v| tag_of(v) == Some("type.ctor_sig"))
                            .and_then(|s| s.get("family").and_then(Value::as_str))
                            .is_some_and(|f| f == family);
                        if is_own_ctor {
                            ArmCover::Ctor(n.clone())
                        } else if ctx.ctor_family.contains_key(n.as_str()) {
                            ArmCover::Other
                        } else {
                            ArmCover::Catchall
                        }
                    }
                    None => {
                        if pattern.kind == "var" {
                            ArmCover::Catchall
                        } else {
                            ArmCover::Other
                        }
                    }
                }
            });

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
                    domains.push((arg_names.get(i).cloned().unwrap_or_default(), dom));
                }
                ctor_result = res;
            }

            // Refine: scrutinee family_app indices vs ctor result.
            // `impossible` = the ctor cannot produce this scrutinee;
            // `index_decided` = proven so by *index* unification (the
            // auto-discharge case — see below). A ctor result naming a
            // different family is nominally impossible too, but that
            // names a foreign family rather than a dead branch of this
            // one, so it stays an error.
            let mut impossible = false;
            let mut index_decided = false;
            let mut idx_args: Vec<Value> = Vec::new();
            if tag_of(&ctor_result) == Some("type.family_app")
                && tag_of(scrutinee_type) == Some("type.family_app")
            {
                let s_name = scrutinee_type
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let c_name = ctor_result
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
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
                            if !ok
                                && (decided || (num_val(&ea).is_some() && num_val(&eb).is_some()))
                            {
                                // Decided-unsatisfiable index
                                // equation: the arm can never fire.
                                // No diagnostic — the arm is
                                // auto-discharged below.
                                impossible = true;
                                index_decided = true;
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
                if index_decided {
                    // Auto-discharge: the ctor's index unification is
                    // decided-unsatisfiable, so the arm can never fire —
                    // coverage doesn't require it and `absurd` isn't
                    // needed (it stays legal purely as documentation).
                    // The body is unchecked but allowed: it contributes
                    // no arm type and binds no fields.
                    continue;
                }
                // A ctor that builds a *different* family is impossible
                // nominally, not by index unification — almost always a
                // typo, so it keeps its error.
                return Err(LangError::engine(
                    "match: unreachable constructor arm (foreign family)".to_string(),
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
            // Destructuring patterns — `sequence`/`entries`/`entry`/`rest`
            // `pattern_role`s (ADR-0081) — bind element/field vars against
            // the scrutinee's `element`/`fields` (fresh metas when the
            // scrutinee is opaque). `leaf` (`var`/`value`/`arr`) and
            // role-less patterns (`call` ctor patterns, literals) are
            // handled by the ctor/catchall path above.
            if let Some("sequence" | "entries" | "entry" | "rest") = lang
                .ast
                .get_by_identity(&pattern.kind)
                .and_then(|s| s.pattern_role.as_deref())
            {
                self.bind_pattern_vars(lang, ast, AstRef::new(pat_idx), scrutinee_type, ctx)?;
            }

            // Body: against the motive applied at the refined indices
            // when present, else synth for the join.
            if let Some(m) = motive {
                let expected =
                    self.apply_motive_repr(lang, ast, m, &idx_args, scrutinee_idx, ctx)?;
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

        // WS-C split-tree coverage: every ctor the scrutinee's indices
        // leave possible must have a covering arm — decided-reachable
        // and uncovered is a non-exhaustive error; still-undecided at a
        // final leaf is a `coverage unknown` warning (bounded split).
        if let Some((fname, sargs)) = &scrut {
            self.match_coverage(lang, ast, node, fname, sargs, &arm_covers, index_sorts, ctx)?;
        }
        ctx.nonneg_index_vars = saved_nonneg;

        // Result: motive applied at the scrutinee's own indices, else the
        // arm-type join.
        if let Some(m) = motive {
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
            return self.apply_motive_repr(lang, ast, m, &scrut_args, scrutinee_idx, ctx);
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

    /// Apply a [`MatchMotive`] at refined index terms — the AST-clause
    /// path delegates to [`Self::apply_motive`]; an inferred motive
    /// substitutes its `vars` holes inside `template` directly.
    fn apply_motive_repr(
        &self,
        lang: &Language,
        ast: &Ast,
        motive: &MatchMotive,
        arg_vals: &[Value],
        scrutinee_idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        match motive {
            MatchMotive::Ast(mi) => self.apply_motive(lang, ast, *mi, arg_vals, scrutinee_idx, ctx),
            MatchMotive::Inferred { vars, template } => {
                let mut out = template.clone();
                for (i, v) in vars.iter().enumerate() {
                    let Some(arg) = arg_vals.get(i) else { break };
                    out = subst_index(&out, &var_term(v), arg);
                }
                Ok(zonk_index(&out, ctx))
            }
        }
    }

    /// `match` in *check* position with no `return … with` clause —
    /// motive inference (phase-2 WS-C): the expected type gives the
    /// motive. When both the scrutinee and the expected type are
    /// applications of the *same* family, the scrutinee's rigid index
    /// `var`s that occur in the expected args become motive binders
    /// (`Vec<var n, …>` scrutinee vs `Vec<var n, b>` expected ⇒ motive
    /// `(i₀) ~> Vec<i₀, b>`); otherwise the motive is the constant
    /// expected type. Returns `None` when the match carries an explicit
    /// motive (the existing synth path handles it) — the caller keeps the
    /// default synth-then-unify path in that case.
    pub(crate) fn dep_check_match(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        expected: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Option<Value>, LangError> {
        if extract_meta_id(expected).is_some() {
            return Ok(None);
        }
        let children = children_of(node)
            .ok_or_else(|| LangError::engine("match: no children".to_string()))?
            .clone();
        if children.len() != 2 {
            // An explicit `return … with` motive (3 children) stays on the
            // existing synth path — inference only fills an *absent* one.
            return Ok(None);
        }
        let scrutinee_idx = children[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("match: scrutinee index not a u64".to_string()))?
            as usize;
        let block_idx = children[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("match: block index not a u64".to_string()))?
            as usize;
        let arm_indices: Vec<usize> = ast
            .get(AstRef::new(block_idx))
            .and_then(children_of)
            .map(|bs| {
                bs.iter()
                    .filter_map(Value::as_u64)
                    .map(|i| i as usize)
                    .collect()
            })
            .unwrap_or_default();
        let scrutinee_type = self.synth_node(lang, ast, AstRef::new(scrutinee_idx), ctx)?;
        let motive = Self::inferred_motive(expected, &scrutinee_type, ctx);
        self.synth_match_with_motive(
            lang,
            ast,
            node,
            &arm_indices,
            scrutinee_idx,
            Some(&motive),
            &scrutinee_type,
            ctx,
        )
        .map(Some)
    }

    /// Build the inferred motive `{vars, template}` for `dep_check_match`:
    /// for each scrutinee index arg that is a rigid `var` occurring in the
    /// expected family args, mint a motive binder `im_i` and substitute it
    /// into the expected type.
    fn inferred_motive(
        expected: &Value,
        scrutinee_type: &Value,
        ctx: &mut TypecheckContext,
    ) -> MatchMotive {
        let mut vars: Vec<String> = Vec::new();
        let mut template = expected.clone();
        if tag_of(expected) == Some("type.family_app")
            && tag_of(scrutinee_type) == Some("type.family_app")
            && expected.get("name").and_then(Value::as_str)
                == scrutinee_type.get("name").and_then(Value::as_str)
        {
            let sargs = scrutinee_type
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for (j, sa) in sargs.iter().enumerate() {
                let ea = eval_index(sa, ctx);
                if tag_of(&ea) == Some("var") {
                    let sname = ea
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if !sname.is_empty() && var_occurs(&sname, &template) {
                        let iv = format!("im_{j}_{sname}");
                        template = subst_index(&template, &ea, &var_term(&iv));
                        vars.push(iv);
                    }
                }
            }
        }
        MatchMotive::Inferred { vars, template }
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
        if inner.kind == "var" {
            // A named motive: `_motive := (i, w) ~> Vec(i, Bool)` binds
            // the arrow's `Lam` term in `results` (synth_assign's
            // `bind_result`) — resolve the var to it and apply that
            // body's type expression as the motive. Any other var
            // binding falls through to the constant-type reading.
            if let Some(lam) = node_name(inner)
                .and_then(|n| ctx.results.get(&n).cloned())
                .filter(|v| matches!(tag_of(v), Some("Lam" | "LamVal")))
            {
                let names: Vec<String> = lam
                    .get("params")
                    .and_then(Value::as_array)
                    .map(|ps| {
                        ps.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                let body = lam.get("body").and_then(Value::as_u64).unwrap_or(0) as usize;
                let scrut_name = ast
                    .get(AstRef::new(scrutinee_idx))
                    .and_then(|n| if n.kind == "var" { node_name(n) } else { None })
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
                let r = self.dep_synth_type(lang, ast, body, ctx);
                ctx.results = old_results;
                return r;
            }
        }
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
            .and_then(|n| if n.kind == "var" { node_name(n) } else { None })
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

    /// The model's `is_univalence_form_` (ADR-0208 P4): a `univalence`
    /// keyword node, or a `call` whose callee is the bare identifier
    /// `ua`/`Univalence` — the univalence path former. `ua` stays an
    /// ordinary identifier everywhere else; only the call *form* is
    /// intercepted.
    fn is_univalence_form(&self, lang: &Language, ast: &Ast, node_ref: AstRef) -> bool {
        let Some(node) = ast.get(node_ref) else {
            return false;
        };
        if node.kind == "univalence" {
            return true;
        }
        if node.kind != "call" {
            return false;
        }
        let Some(callee_ref) = Self::cub_child(lang, ast, node, "callee", 0) else {
            return false;
        };
        let Some(callee) = ast.get(callee_ref) else {
            return false;
        };
        callee.kind == "var" && matches!(node_name(callee).as_deref(), Some("ua" | "Univalence"))
    }

    /// Resolve a path operand node — the model's `resolve_path_`
    /// (ADR-0208 P4): `trans`/`symm`/univalence forms synthesize
    /// through `synth_node` (their types carry the `path` payload);
    /// a `var` consults its bound witnesses/declared types first
    /// (`results` then Γ — a `p := ua(e)` binding has no eval rule,
    /// so the *declared* `type.pi` is the only carrier of the
    /// endpoints and the `path.equiv` payload), then falls back to
    /// an open `Path(?a, ?b)` when the binding is an unsolved
    /// metavariable (stuck, not decided-not-a-path). `refl` is the
    /// identity witness. `None` only when the operand is
    /// decided-not-a-path.
    fn resolve_path_full(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        ctx: &mut TypecheckContext,
    ) -> Option<ResolvedPath> {
        let node = ast.get(node_ref)?;
        if node.kind == "trans"
            || node.kind == "symm"
            || self.is_univalence_form(lang, ast, node_ref)
        {
            let ty = self.synth_node(lang, ast, node_ref, ctx).ok()?;
            let (x, y, _, _) = witness_from_type_full(&ty)?;
            return Some(ResolvedPath {
                verdict: CubPathVerdict::Path,
                frm: eval_index(&x, ctx),
                to: eval_index(&y, ctx),
                equiv: path_equiv_of(&ty),
                witness: ty,
            });
        }
        if node.kind == "var" {
            let name = node_name(node)?;
            if name == "refl" {
                let mut m = Map::new();
                m.insert("__tag__".to_string(), Value::String("Refl".to_string()));
                // Identity path — `frm == to` reads as transport with
                // equal endpoints; record the refl marker.
                let w = Value::Object(m);
                return Some(ResolvedPath {
                    verdict: CubPathVerdict::Refl,
                    frm: w.clone(),
                    to: w.clone(),
                    equiv: Value::Null,
                    witness: w,
                });
            }
            // Bound witnesses/declared types — `Path{frm,to}` records,
            // `type.eq`, and path-carrying `type.pi`s (plain `Path`/
            // `PathP` and univalence paths alike) all resolve through
            // `witness_from_type_full`.
            for bound in [ctx.results.get(&name), ctx.lookup(&name)]
                .into_iter()
                .flatten()
            {
                if let Some((x, y, _, _)) = witness_from_type_full(bound) {
                    return Some(ResolvedPath {
                        verdict: CubPathVerdict::Path,
                        frm: eval_index(&x, ctx),
                        to: eval_index(&y, ctx),
                        equiv: path_equiv_of(bound),
                        witness: bound.clone(),
                    });
                }
            }
            let bound = ctx
                .results
                .get(&name)
                .or_else(|| ctx.lookup(&name))
                .cloned();
            if bound.as_ref().and_then(extract_meta_id).is_some() {
                // Stuck path operand — an open `Path(?a, ?b)` whose
                // endpoints checking resolves against the use site.
                let frm = meta_value(ctx.fresh_meta());
                let to = meta_value(ctx.fresh_meta());
                return Some(ResolvedPath {
                    verdict: CubPathVerdict::Path,
                    frm: frm.clone(),
                    to: to.clone(),
                    equiv: Value::Null,
                    witness: path_val(frm, to),
                });
            }
        }
        None
    }

    /// `p # v` — transport `v`'s type along the path (the model's
    /// `synth_transport_`): a *univalence* path (`path.equiv`
    /// present) first tries a shallow unify of `v`'s type against the
    /// domain — success yields the codomain; failure falls back to
    /// the dependent `subst_index_` rewrite and warns when the type
    /// does not mention the equivalence domain at all. A plain path
    /// rewrites `frm ↦ to` through the value's type
    /// ([`subst_transport`]). The consumed witness is recorded under
    /// `via` (proof-relevant, not erased).
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
            return Err(LangError::engine(
                "transport: needs [path, value]".to_string(),
            ));
        }
        let path_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("transport: path index not a u64".to_string()))?
            as usize;
        let val_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("transport: value index not a u64".to_string()))?
            as usize;
        self.transport_judgment_at(lang, ast, path_idx, val_idx, node.span, ctx)
    }

    /// The transport judgment over explicit operand indices — shared by
    /// the `p # v` node ([`Self::synth_transport_judgment`]) and the
    /// `transport(p, v)` call form intercepted in `synth_call_judgment`
    /// /`check_call_judgment` (the `ua`/`Univalence` precedent: the same
    /// judgment under a plain identifier, no new node kind).
    fn transport_judgment_at(
        &self,
        lang: &Language,
        ast: &Ast,
        path_idx: usize,
        val_idx: usize,
        span: Option<handl_language::Span>,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let path_node = ast
            .get(AstRef::new(path_idx))
            .ok_or_else(|| LangError::engine("transport: path node out of bounds".to_string()))?;
        let path = self
            .resolve_path_full(lang, ast, AstRef::new(path_idx), ctx)
            .ok_or_else(|| {
                LangError::engine(format!(
                    "transport: operand is not a Path (kind '{}')",
                    path_node.kind
                ))
            })?;
        let vty = self.synth_node(lang, ast, AstRef::new(val_idx), ctx)?;
        let transported = if path.verdict == CubPathVerdict::Path {
            if !path.equiv.is_null() {
                // Univalence transport (ADR-0208 P4): `p # v` where
                // `p : ua(e)` for `e : A ≃ B` reads `v : A` as `v : B`.
                // The shallow `unify` is speculative — a failed attempt
                // restores the meta store before the subst fallback.
                let mark = ctx.store_mark();
                match self.unify(&vty, &path.frm, ctx) {
                    Ok(()) => zonk_index(&path.to, ctx),
                    Err(_) => {
                        ctx.store_restore(mark);
                        let t1 = subst_transport(&path.frm, &path.to, &vty);
                        if t1 != vty {
                            t1
                        } else {
                            self.cub_diag(
                                ctx,
                                "warning",
                                "transport along univalence: value type does not mention the equivalence domain — passing through unchanged".to_string(),
                                span,
                            );
                            vty.clone()
                        }
                    }
                }
            } else {
                subst_transport(&path.frm, &path.to, &vty)
            }
        } else {
            // `refl` — the identity transport.
            vty.clone()
        };
        // Record the consumed witness (proof-relevant provenance). An
        // open transport over an unsolved meta stays atomic — a `via`
        // field would corrupt the `{"__meta__"}` encoding.
        if let Value::Object(mut m) = transported {
            if extract_meta_id(&Value::Object(m.clone())).is_none() {
                m.insert("via".to_string(), path.witness);
            }
            return Ok(Value::Object(m));
        }
        Ok(transported)
    }

    /// `j p, motive, base` — the J eliminator: at `refl` the motive is
    /// applied at the base's own family indices plus the base term and
    /// the base is checked against it; on a decided path `base` is
    /// checked against the motive applied at `frm` (with the path
    /// witness bound to the motive's `w`) and the motive applied at
    /// `to` is synthesized — the model's `synth_j_` (ADR-0208 P5).
    pub(crate) fn synth_j_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids =
            children_of(node).ok_or_else(|| LangError::engine("j: no children".to_string()))?;
        if kids.len() < 3 {
            return Err(LangError::engine(
                "j: needs [path, motive, base]".to_string(),
            ));
        }
        let path_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("j: path index not a u64".to_string()))?
            as usize;
        let motive_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("j: motive index not a u64".to_string()))?
            as usize;
        let base_idx = kids[2]
            .as_u64()
            .ok_or_else(|| LangError::engine("j: base index not a u64".to_string()))?
            as usize;
        let path = self
            .resolve_path_full(lang, ast, AstRef::new(path_idx), ctx)
            .ok_or_else(|| LangError::engine("j: operand is not a Path".to_string()))?;
        // `refl` iota (the model's `synth_j_` refl arm, ADR-0208 P5): at
        // the identity path the endpoints are unidentified, so J
        // computes on the base's own fibre — the motive is applied at
        // the base's family indices plus the base *term* (the same
        // convention `synth_match_arms` uses for the scrutinee) and the
        // base is checked against it.
        if path.verdict == CubPathVerdict::Refl {
            let base_ref = AstRef::new(base_idx);
            let base_ty = self.synth_node(lang, ast, base_ref, ctx)?;
            let idx_args: Vec<Value> = match tag_of(&base_ty) {
                Some("type.family_app") => base_ty
                    .get("args")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|a| eval_index(a, ctx))
                    .collect(),
                _ => Vec::new(),
            };
            let base_node = ast
                .get(base_ref)
                .ok_or_else(|| LangError::engine("j: base node out of bounds".to_string()))?;
            let bterm = self.node_to_index_term(lang, ast, base_node, ctx);
            let mut arg_vals = idx_args;
            arg_vals.push(bterm);
            let motive_ty = self.apply_motive(lang, ast, motive_idx, &arg_vals, base_idx, ctx)?;
            self.check_node(lang, ast, base_ref, &motive_ty, ctx)?;
            return Ok(with_path_via(motive_ty, &path.witness));
        }
        let frm = eval_index(&path.frm, ctx);
        let to = eval_index(&path.to, ctx);
        // The motive's path binder `w` receives the resolved witness
        // (the model's `apply_motive_ … [endpoint, witness]` — the
        // `j_motive` closure's `witness` field), not a `var` of the
        // operand's name.
        let witness = path.witness.clone();
        let expected_base = self.apply_motive(
            lang,
            ast,
            motive_idx,
            &[frm, witness.clone()],
            path_idx,
            ctx,
        )?;
        self.check_node(lang, ast, AstRef::new(base_idx), &expected_base, ctx)?;
        let result = self.apply_motive(lang, ast, motive_idx, &[to, witness], path_idx, ctx)?;
        Ok(with_path_via(result, &path.witness))
    }

    // -----------------------------------------------------------------------
    // Cubical judgments (ADR-0208 P2): transp / hcomp / comp / trans / symm
    // -----------------------------------------------------------------------
    //
    // CCHM-style cubical operations over the interval sort `I` (P1).
    // This is a *typechecker*, not a cubical evaluator: faces and
    // partiality are represented and checked structurally, and the
    // interval-computational content is deferred. What IS checked:
    //
    //   * `transp A, i, φ, u` — transport in the family `A : I → Type`
    //     along dimension `i`, partial on face `φ`. CCHM direction:
    //     `transp^i_φ A u : A(i1) → A(i0)`. `u` is checked against the
    //     full fibre `A(i)` — genuine `Partial φ A` checking is
    //     approximated (mismatch is a warning, not an error).
    //   * `hcomp A, i, φ, u, cap` — homogeneous composition over a
    //     single type `A` (NOT an `I → Type` family — `comp` is the
    //     heterogeneous form), `cap : A` the `i = 0` face.
    //   * `comp F, i, φ, u, cap` — heterogeneous composition over
    //     `F : I → Type`: `cap : F(i0)`, result `: F(i1)`.
    //   * All three require `i : I` and `i ∉ φ` (`face_has_dim`).
    //   * `trans p, q` / `symm p` — derived path algebra on cubical
    //     path witnesses (P1 `type.pi` + `path` payloads; legacy
    //     `Path{frm,to}`/`type.eq` records).

    /// Record a judgment diagnostic — the model threads a diagnostic
    /// state rather than aborting, so failures here push onto
    /// `ctx.diagnostics` and return `type.unknown` (the same
    /// error-diagnostic-and-continue discipline `synth_match_arms`
    /// uses).
    fn cub_diag(
        &self,
        ctx: &mut TypecheckContext,
        severity: &str,
        msg: String,
        span: Option<handl_language::Span>,
    ) {
        ctx.diagnostics.push(TypecheckDiagnostic {
            severity: severity.to_string(),
            message: msg,
            span,
        });
    }

    /// `type.unknown` — the failure/degenerate result the model's
    /// judgments return alongside their diagnostics.
    fn cub_unknown(&self, ctx: &mut TypecheckContext) -> Value {
        let mut m = Map::new();
        make_tag(ctx, "type.unknown", std::mem::take(&mut m))
    }

    /// The *type-position* spelling of a degraded cubical former
    /// (ADR-0208 P3): the model returns `type.unknown` ("left open"),
    /// but NBE's assign layer re-evaluates a `type.unknown` annotation
    /// as a *term* (`eval_ann_node` — the ordered value-annotation
    /// fallback), which would replace the honest unknown with a stuck
    /// application term and fail unification. A fresh metavariable is
    /// the same "accept anything" reading without tripping that
    /// fallback: unification solves it to the checked type.
    fn cub_open_meta(&self, ctx: &mut TypecheckContext) -> Value {
        meta_value(ctx.fresh_meta())
    }

    /// Resolve a cubical operand child by declared role with the
    /// positional fallback (same discipline as the model's
    /// `child_role_index` + `at(children, n)`).
    fn cub_child(
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        role: &str,
        pos: usize,
    ) -> Option<AstRef> {
        role_child_ref(lang, node, role).or_else(|| child_at(ast, node, pos))
    }

    /// Evaluate a face-operand node to a [`CubFace`] — the model's
    /// `eval_node_` + `face_of_val_` reading. `value`/`lit` payloads
    /// unwrap their literal; every other node goes through
    /// `node_to_index_term` + `eval_index`, which leaves the unbound
    /// `f0`/`f1`/`fnot`/`f_or`/`f_and` constructor applications as
    /// symbolic `App` terms and `fT`/`fF`/`true`/`false`/`i0`/`i1` as
    /// `var` atoms.
    fn cub_face_of_node(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Option<CubFace> {
        let term = if matches!(node.kind.as_str(), "value" | "lit") {
            match &node.payload {
                Value::Object(m) => m.get("value").cloned().unwrap_or(Value::Null),
                p => p.clone(),
            }
        } else {
            eval_index(&self.node_to_index_term(lang, ast, node, ctx), ctx)
        };
        cub_face_of_val(&term)
    }

    /// `A(point)` — the model's `fam_at_` (ADR-0208 P3/P4): a
    /// `type.glue` reads its *family* `A` off-face, its *section*
    /// `T̂` on the `φ ≡ 1` face (`tt`), and `A` on every other face —
    /// including undecided `eq0`/`eq1`/`not`/`or`/`and` formulas, whose
    /// on-face agreement is only up to unchecked equivalence data.
    /// A path-carrying `type.pi` whose `path` payload has an `equiv`
    /// record is a *univalence* family `ua(e)` (ADR-0208 P4):
    /// `fam(i0) = dom`, `fam(i1) = cod`, and an interior point reads
    /// the path-Pi Glue `Glue(cod, i=0, dom)` — the `fam_at_`
    /// path-Pi arm. Any other interval-param `type.pi` (`A : I →
    /// Type`) applies its `body_closure` at the interval *term*
    /// `point` (`var{i0}`/`var{i1}`/the dimension `var`); a closure
    /// that computes to a `type.glue` re-enters `fam_at` on it (the
    /// applied-glue recursion). Any other type is a constant family,
    /// `A(point) = A`.
    fn fam_at(
        &self,
        lang: &Language,
        ast: &Ast,
        fam: &Value,
        point: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        if tag_of(fam) == Some("type.glue") {
            // On-face reading: `tt` selects the section, everything
            // else (including undecided face formulas) the family —
            // the model's `fam_at_` `type.glue` arm.
            let fop = fam
                .get("face")
                .and_then(|f| f.get("op"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let inner = match fop {
                "tt" => fam.get("section").cloned().unwrap_or_else(|| fam.clone()),
                _ => fam.get("family").cloned().unwrap_or_else(|| fam.clone()),
            };
            return self.fam_at(lang, ast, &inner, point, ctx);
        }
        let pld = if tag_of(fam) == Some("type.pi") {
            fam.get("path").cloned()
        } else {
            None
        };
        if let Some(pld) = pld
            && !pld.get("equiv").map(Value::is_null).unwrap_or(true)
        {
            // Univalence family `ua(e)` — `fam(i0) = dom`,
            // `fam(i1) = cod`, interior `Glue(cod, i=0, dom)`.
            let i0t = var_term("i0");
            let i1t = var_term("i1");
            if *point == i0t {
                return Ok(pld.get("x").cloned().unwrap_or(Value::Null));
            }
            if *point == i1t {
                return Ok(pld.get("y").cloned().unwrap_or(Value::Null));
            }
            let mut m = Map::new();
            m.insert(
                "family".to_string(),
                pld.get("y").cloned().unwrap_or(Value::Null),
            );
            let mut face_m = Map::new();
            face_m.insert("__tag__".to_string(), Value::String("face".to_string()));
            face_m.insert("op".to_string(), Value::String("eq0".to_string()));
            face_m.insert("args".to_string(), Value::Array(vec![point.clone()]));
            m.insert("face".to_string(), Value::Object(face_m));
            m.insert(
                "section".to_string(),
                pld.get("x").cloned().unwrap_or(Value::Null),
            );
            return Ok(make_tag(ctx, "type.glue", m));
        }
        let is_family = tag_of(fam) == Some("type.pi")
            && tag_of(fam.get("param").unwrap_or(&Value::Null)) == Some("type.interval")
            && fam.get("body_closure").is_some();
        if is_family {
            let applied = self.dep_closure_apply_term(lang, ast, fam, point, ctx)?;
            // An `I → Type` closure may compute to a `type.glue` —
            // `fam_at` recurses on it (the model's applied-glue arm).
            if tag_of(&applied) == Some("type.glue") {
                return self.fam_at(lang, ast, &applied, point, ctx);
            }
            Ok(applied)
        } else {
            Ok(fam.clone())
        }
    }

    /// Shared prologue for `hcomp`/`comp` — the model's
    /// `cubical_prologue_`: synthesize the family/type argument,
    /// check the dimension against `I`, evaluate it, read and
    /// normalize the face, and enforce `i ∉ φ`. `Ok(None)` means the
    /// prologue already recorded its error diagnostic.
    fn cubical_prologue(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        kind: &str,
        ctx: &mut TypecheckContext,
    ) -> Result<Option<(Value, Value)>, LangError> {
        let span = node.span;
        let (Some(fam_ref), Some(dim_ref), Some(face_ref)) = (
            Self::cub_child(lang, ast, node, "family", 0),
            Self::cub_child(lang, ast, node, "dim", 1),
            Self::cub_child(lang, ast, node, "face", 2),
        ) else {
            return Ok(None);
        };
        let fam = self.dep_synth_type(lang, ast, fam_ref.index(), ctx)?;
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let dim_ty = self.synth_node(lang, ast, dim_ref, ctx)?;
        let (dim_ok, _) = unify_index(&dim_ty, &interval, ctx);
        if !dim_ok {
            self.cub_diag(
                ctx,
                "error",
                format!("{kind}: dimension operand must have type I"),
                span,
            );
            return Ok(None);
        }
        let dim_node = ast
            .get(dim_ref)
            .ok_or_else(|| LangError::engine(format!("{kind}: dim node out of bounds")))?;
        let dim_val = eval_index(&self.node_to_index_term(lang, ast, dim_node, ctx), ctx);
        let face_node = ast
            .get(face_ref)
            .ok_or_else(|| LangError::engine(format!("{kind}: face node out of bounds")))?;
        let Some(phi_raw) = self.cub_face_of_node(lang, ast, face_node, ctx) else {
            self.cub_diag(
                ctx,
                "error",
                format!(
                    "{kind}: face operand is not a face formula — expected f0(i), f1(i), fnot φ, f_or φ ψ, f_and φ ψ, fT, or fF"
                ),
                span,
            );
            return Ok(None);
        };
        let phi = face_norm(phi_raw);
        if face_has_dim(&dim_val, &phi) {
            self.cub_diag(
                ctx,
                "error",
                format!("{kind}: the dimension must not occur in the face φ"),
                span,
            );
            return Ok(None);
        }
        Ok(Some((fam, dim_val)))
    }

    /// `transp A, i, φ, u` — see the section comment. The partial
    /// element `u` is checked against the full fibre `A(i)` (the
    /// `Partial φ A` approximation — mismatch warns, never errors);
    /// the result is the non-dependent arrow `A(i1) → A(i0)`.
    pub(crate) fn synth_transp_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        if kids_len < 4 {
            self.cub_diag(
                ctx,
                "error",
                "transp: needs family, dimension, face, and partial operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let (Some(fam_ref), Some(dim_ref), Some(face_ref), Some(part_ref)) = (
            Self::cub_child(lang, ast, node, "family", 0),
            Self::cub_child(lang, ast, node, "dim", 1),
            Self::cub_child(lang, ast, node, "face", 2),
            Self::cub_child(lang, ast, node, "partial", 3),
        ) else {
            self.cub_diag(
                ctx,
                "error",
                "transp: needs family, dimension, face, and partial operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        let fam = self.dep_synth_type(lang, ast, fam_ref.index(), ctx)?;
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let dim_ty = self.synth_node(lang, ast, dim_ref, ctx)?;
        let (dim_ok, _) = unify_index(&dim_ty, &interval, ctx);
        if !dim_ok {
            self.cub_diag(
                ctx,
                "error",
                "transp: dimension operand must have type I".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let dim_node = ast
            .get(dim_ref)
            .ok_or_else(|| LangError::engine("transp: dim node out of bounds".to_string()))?;
        let dim_val = eval_index(&self.node_to_index_term(lang, ast, dim_node, ctx), ctx);
        let face_node = ast
            .get(face_ref)
            .ok_or_else(|| LangError::engine("transp: face node out of bounds".to_string()))?;
        let Some(phi_raw) = self.cub_face_of_node(lang, ast, face_node, ctx) else {
            self.cub_diag(
                ctx,
                "error",
                "transp: face operand is not a face formula — expected f0(i), f1(i), fnot φ, f_or φ ψ, f_and φ ψ, fT, or fF".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        let phi = face_norm(phi_raw);
        if face_has_dim(&dim_val, &phi) {
            self.cub_diag(
                ctx,
                "error",
                "transp: the dimension must not occur in the face φ".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let u_ty = self.synth_node(lang, ast, part_ref, ctx)?;
        let fam_dim = self.fam_at(lang, ast, &fam, &dim_val, ctx)?;
        // `Partial φ A` is approximated by the full fibre: a failed
        // unify warns and its meta bindings are rolled back (the
        // model discards `uu.metas` on the warning path).
        let mark = ctx.store_mark();
        let (uu_ok, _) = unify_index(&u_ty, &fam_dim, ctx);
        if !uu_ok {
            ctx.store_restore(mark);
            self.cub_diag(
                ctx,
                "warning",
                "transp: partial element does not obviously inhabit the family on φ — `Partial φ A` is approximated by the full fibre".to_string(),
                span,
            );
        }
        let i1 = var_term("i1");
        let i0 = var_term("i0");
        let a1 = self.fam_at(lang, ast, &fam, &i1, ctx)?;
        let a0 = self.fam_at(lang, ast, &fam, &i0, ctx)?;
        let mut m = Map::new();
        m.insert("param".to_string(), a1);
        m.insert("body".to_string(), a0);
        m.insert("param_name".to_string(), Value::String(String::new()));
        Ok(make_tag(ctx, "type.function", m))
    }

    /// `hcomp A, i, φ, u, cap` — homogeneous composition (see the
    /// section comment): `A` is read as a *single type*, `cap : A` is
    /// the `i = 0` face (checked strictly), `u` the partial on `φ`
    /// (approximated by the full type — mismatch warns); the result
    /// is `: A`.
    pub(crate) fn synth_hcomp_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        self.synth_hcomp_comp_judgment(lang, ast, node, "hcomp", ctx)
    }

    /// `comp F, i, φ, u, cap` — heterogeneous composition over
    /// `F : I → Type` (see the section comment): `cap : F(i0)`
    /// (checked strictly), `u` the partial on `φ` (approximated by the
    /// fibre `F(i)` — mismatch warns); the result is `: F(i1)`.
    pub(crate) fn synth_comp_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        self.synth_hcomp_comp_judgment(lang, ast, node, "comp", ctx)
    }

    /// Shared body of `hcomp`/`comp` — the model's `synth_hcomp_` /
    /// `synth_comp_` over `cubical_prologue_`.
    fn synth_hcomp_comp_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        kind: &str,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        if kids_len < 5 {
            self.cub_diag(
                ctx,
                "error",
                format!("{kind}: needs family, dimension, face, partial, and cap operands"),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let Some((fam, dim_val)) = self.cubical_prologue(lang, ast, node, kind, ctx)? else {
            return Ok(self.cub_unknown(ctx));
        };
        let (Some(part_ref), Some(cap_ref)) = (
            Self::cub_child(lang, ast, node, "partial", 3),
            Self::cub_child(lang, ast, node, "cap", 4),
        ) else {
            self.cub_diag(
                ctx,
                "error",
                format!("{kind}: needs family, dimension, face, partial, and cap operands"),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        let u_ty = self.synth_node(lang, ast, part_ref, ctx)?;
        if kind == "hcomp" {
            // Homogeneous: `u` inhabits the single type `A` on `φ`.
            let mark = ctx.store_mark();
            let (uu_ok, _) = unify_index(&u_ty, &fam, ctx);
            if !uu_ok {
                ctx.store_restore(mark);
                self.cub_diag(
                    ctx,
                    "warning",
                    "hcomp: partial element does not obviously inhabit A on φ — `Partial φ A` is approximated by the full type".to_string(),
                    span,
                );
            }
            let cap_ty = self.synth_node(lang, ast, cap_ref, ctx)?;
            let (cu_ok, _) = unify_index(&cap_ty, &fam, ctx);
            if !cu_ok {
                self.cub_diag(
                    ctx,
                    "error",
                    "hcomp: cap does not inhabit A — the i = 0 face must have type A".to_string(),
                    span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            // A family-looking `A` is almost certainly a `comp` mistake.
            if tag_of(&fam) == Some("type.pi")
                && tag_of(fam.get("param").unwrap_or(&Value::Null)) == Some("type.interval")
            {
                self.cub_diag(
                    ctx,
                    "warning",
                    "hcomp: A looks like an I → Type family — `comp` is the heterogeneous composition form".to_string(),
                    span,
                );
            }
            Ok(fam)
        } else {
            // Heterogeneous: `u` inhabits the fibre `F(i)` on `φ`,
            // `cap : F(i0)`, result `: F(i1)`.
            let fam_dim = self.fam_at(lang, ast, &fam, &dim_val, ctx)?;
            let mark = ctx.store_mark();
            let (uu_ok, _) = unify_index(&u_ty, &fam_dim, ctx);
            if !uu_ok {
                ctx.store_restore(mark);
                self.cub_diag(
                    ctx,
                    "warning",
                    "comp: partial element does not obviously inhabit the family on φ — `Partial φ F` is approximated by the full fibre".to_string(),
                    span,
                );
            }
            let cap_ty = self.synth_node(lang, ast, cap_ref, ctx)?;
            let i0 = var_term("i0");
            let i1 = var_term("i1");
            let fam_i0 = self.fam_at(lang, ast, &fam, &i0, ctx)?;
            let (cu_ok, _) = unify_index(&cap_ty, &fam_i0, ctx);
            if !cu_ok {
                self.cub_diag(
                    ctx,
                    "error",
                    "comp: cap does not inhabit F(i0) — the i = 0 face must have type F(i0)"
                        .to_string(),
                    span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            self.fam_at(lang, ast, &fam, &i1, ctx)
        }
    }

    /// Resolve a path *operand* (the `p`/`q` in `trans p, q`, the `p`
    /// in `symm p`) to its endpoints — the model's `path_witness_`:
    /// a `var`'s declared type (Γ) and bound witness values
    /// (`results`) are consulted before synthesis, which is what
    /// admits `trans`/`symm` nodes and any other path-typed
    /// expression as operands. `refl` is the identity witness.
    fn path_witness(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        ctx: &mut TypecheckContext,
    ) -> Result<CubPathWitness, LangError> {
        let node = ast
            .get(node_ref)
            .ok_or_else(|| LangError::engine("path_witness: node out of bounds".to_string()))?;
        if node.kind == "var"
            && let Some(name) = node_name(node)
        {
            if name == "refl" {
                return Ok(CubPathWitness {
                    verdict: CubPathVerdict::Refl,
                    x: Value::Null,
                    y: Value::Null,
                    closure: Value::Null,
                    self_ty: Value::Null,
                });
            }
            // A `var` operand's *declared* type is consulted first: it
            // carries the path endpoints even when the bound value is
            // `refl` (`p : Path(2, 2) := refl` is a path 2 ~ 2, not
            // the undetermined identity). Bound witness *values*
            // (`Path{frm,to}` records, `eq_*` refinements) read
            // through the same `witness_from_type_` lens.
            for bound in [ctx.lookup(&name).cloned(), ctx.results.get(&name).cloned()]
                .into_iter()
                .flatten()
            {
                if let Some((x, y, closure, self_ty)) = witness_from_type_full(&bound) {
                    return Ok(CubPathWitness {
                        verdict: CubPathVerdict::Path,
                        x,
                        y,
                        closure,
                        self_ty,
                    });
                }
            }
        }
        let ty = self.synth_node(lang, ast, node_ref, ctx)?;
        // The `Refl` marker type is the identity witness — `refl`
        // synthesizes to it (the model reads `type.refl`/`var{refl}`
        // off the evaluated term).
        if matches!(tag_of(&ty), Some("Refl" | "type.refl")) {
            return Ok(CubPathWitness {
                verdict: CubPathVerdict::Refl,
                x: Value::Null,
                y: Value::Null,
                closure: Value::Null,
                self_ty: Value::Null,
            });
        }
        if let Some((x, y, closure, self_ty)) = witness_from_type_full(&ty) {
            return Ok(CubPathWitness {
                verdict: CubPathVerdict::Path,
                x,
                y,
                closure,
                self_ty,
            });
        }
        Ok(CubPathWitness {
            verdict: CubPathVerdict::Bad,
            x: Value::Null,
            y: Value::Null,
            closure: Value::Null,
            self_ty: Value::Null,
        })
    }

    /// The path *type* denoted by a [`CubPathWitness`] — the model's
    /// `witness_path_type_`: the witness's own `type.pi` when it is
    /// one (identity cases — `trans(refl, q)`), else a fresh interval
    /// `type.pi` over `type.unknown` with the `path{x,y}` payload.
    fn witness_path_type(&self, ctx: &mut TypecheckContext, w: &CubPathWitness) -> Value {
        if tag_of(&w.self_ty) == Some("type.pi") {
            return w.self_ty.clone();
        }
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let mut unknown_m = Map::new();
        let unknown = make_tag(ctx, "type.unknown", std::mem::take(&mut unknown_m));
        let closure = if !w.closure.is_null() {
            w.closure.clone()
        } else {
            serde_json::json!({
                "var_name": "",
                "body_child_idx": Value::Null,
                "dep": true,
                "const_body": unknown,
            })
        };
        let mut m = Map::new();
        m.insert("param".to_string(), interval);
        m.insert("body_closure".to_string(), closure);
        m.insert("path".to_string(), serde_json::json!({"x": w.x, "y": w.y}));
        make_tag(ctx, "type.pi", m)
    }

    /// Mint the composed/symmetric path type — the model's
    /// `type.pi` construction shared by `synth_trans_`/`synth_symm_`:
    /// interval param, the first available `body_closure` (or a
    /// constant `type.unknown` body), and the `path{x,y}` payload.
    /// `equiv` is the composed `path.equiv` record for univalence
    /// paths (ADR-0208 P4) — `Value::Null` leaves the payload
    /// `{x, y}`-only.
    fn mint_path_type(
        &self,
        ctx: &mut TypecheckContext,
        x: Value,
        y: Value,
        closure: Value,
        equiv: Value,
    ) -> Value {
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let mut unknown_m = Map::new();
        let unknown = make_tag(ctx, "type.unknown", std::mem::take(&mut unknown_m));
        let closure = if !closure.is_null() {
            closure
        } else {
            serde_json::json!({
                "var_name": "",
                "body_child_idx": Value::Null,
                "dep": true,
                "const_body": unknown,
            })
        };
        let mut path = serde_json::json!({"x": x, "y": y});
        if let Value::Object(ref mut pm) = path {
            if !equiv.is_null() {
                pm.insert("equiv".to_string(), equiv);
            }
        }
        let mut m = Map::new();
        m.insert("param".to_string(), interval);
        m.insert("body_closure".to_string(), closure);
        m.insert("path".to_string(), path);
        make_tag(ctx, "type.pi", m)
    }

    /// `trans p, q` — derived path composition (see the section
    /// comment): unify `p.y` with `q.x`, yield the path type
    /// `x ~ z` over the first operand's family. `refl` is the
    /// identity — `trans(refl, q) = q`, `trans(p, refl) = p`.
    pub(crate) fn synth_trans_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        if kids_len < 2 {
            self.cub_diag(
                ctx,
                "error",
                "trans: needs two path operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let (Some(f_ref), Some(s_ref)) = (
            Self::cub_child(lang, ast, node, "first", 0),
            Self::cub_child(lang, ast, node, "second", 1),
        ) else {
            self.cub_diag(
                ctx,
                "error",
                "trans: needs two path operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        let w1 = self.path_witness(lang, ast, f_ref, ctx)?;
        let w2 = self.path_witness(lang, ast, s_ref, ctx)?;
        if w1.verdict == CubPathVerdict::Bad || w2.verdict == CubPathVerdict::Bad {
            self.cub_diag(
                ctx,
                "error",
                "trans: operand is not a Path".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        if w1.verdict == CubPathVerdict::Refl && w2.verdict == CubPathVerdict::Refl {
            self.cub_diag(
                ctx,
                "warning",
                "trans: refl · refl — endpoints undetermined".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        if w1.verdict == CubPathVerdict::Refl {
            return Ok(self.witness_path_type(ctx, &w2));
        }
        if w2.verdict == CubPathVerdict::Refl {
            return Ok(self.witness_path_type(ctx, &w1));
        }
        let (u_ok, _) = unify_index(&w1.y, &w2.x, ctx);
        if !u_ok {
            self.cub_diag(
                ctx,
                "error",
                "trans: path endpoints do not compose — the first path's target is not the second path's source".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        // The composed path lives over the first operand's family —
        // `w1.closure`, falling back to `w2.closure`, then a constant
        // `type.unknown` body (the model's closure selection).
        let closure = if !w1.closure.is_null() {
            w1.closure.clone()
        } else {
            w2.closure.clone()
        };
        // Univalence composition (ADR-0208 P4): when *both* operands
        // carry `path.equiv` records the composed path keeps the
        // equivalence — `dom`/`cod` span the outer endpoints while
        // `to`/`fro` stay deferred (`null`, proof-irrelevant).
        let w1e = path_equiv_of(&w1.self_ty);
        let w2e = path_equiv_of(&w2.self_ty);
        let equiv = if !w1e.is_null() && !w2e.is_null() {
            let field_or = |e: &Value, k: &str, fallback: &Value| {
                e.get(k)
                    .filter(|v| !v.is_null())
                    .cloned()
                    .unwrap_or_else(|| fallback.clone())
            };
            serde_json::json!({
                "dom": field_or(&w1e, "dom", &w1.x),
                "cod": field_or(&w2e, "cod", &w2.y),
                "to": Value::Null,
                "fro": Value::Null,
            })
        } else {
            Value::Null
        };
        Ok(self.mint_path_type(ctx, w1.x.clone(), w2.y.clone(), closure, equiv))
    }

    /// `symm p` — derived path symmetry (see the section comment):
    /// swaps the endpoints of `p : x ~ y` into `y ~ x` over the same
    /// family. `refl` has no determined endpoints — a warning, not a
    /// silent `Path(?, ?)`.
    pub(crate) fn synth_symm_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        if kids_len < 1 {
            self.cub_diag(ctx, "error", "symm: needs a path operand".to_string(), span);
            return Ok(self.cub_unknown(ctx));
        }
        let Some(p_ref) = Self::cub_child(lang, ast, node, "path", 0) else {
            self.cub_diag(ctx, "error", "symm: needs a path operand".to_string(), span);
            return Ok(self.cub_unknown(ctx));
        };
        let w = self.path_witness(lang, ast, p_ref, ctx)?;
        match w.verdict {
            CubPathVerdict::Bad => {
                self.cub_diag(
                    ctx,
                    "error",
                    "symm: operand is not a Path".to_string(),
                    span,
                );
                Ok(self.cub_unknown(ctx))
            }
            CubPathVerdict::Refl => {
                self.cub_diag(
                    ctx,
                    "warning",
                    "symm: refl has no determined endpoints".to_string(),
                    span,
                );
                Ok(self.cub_unknown(ctx))
            }
            CubPathVerdict::Path => {
                // Univalence symmetry (ADR-0208 P4): `symm` on a
                // `ua(e)` path swaps the equivalence direction —
                // `dom ↔ cod`, `to ↔ fro`.
                let we = path_equiv_of(&w.self_ty);
                let equiv = if !we.is_null() {
                    let field_or = |k: &str, fallback: &Value| {
                        we.get(k)
                            .filter(|v| !v.is_null())
                            .cloned()
                            .unwrap_or_else(|| fallback.clone())
                    };
                    serde_json::json!({
                        "dom": field_or("cod", &w.y),
                        "cod": field_or("dom", &w.x),
                        "to": we.get("fro").cloned().unwrap_or(Value::Null),
                        "fro": we.get("to").cloned().unwrap_or(Value::Null),
                    })
                } else {
                    Value::Null
                };
                Ok(self.mint_path_type(ctx, w.y.clone(), w.x.clone(), w.closure.clone(), equiv))
            }
        }
    }

    // -------------------------------------------------------------------
    // Cubical univalence (ADR-0208 P4): ua / Univalence / univalence /
    // Equiv
    // -------------------------------------------------------------------
    //
    // `ua(e)` / `univalence(e)` for an equivalence `e : A ≃ B`
    // synthesizes a path-carrying `type.pi` over `I` whose `path`
    // payload is `{x: A, y: B, equiv: {dom, cod, to, fro}}` —
    // proof-relevant provenance that `p # v` (transport), `symm`,
    // `trans`, and `fam_at` read. `Equiv(A, B)` mints the
    // `type.equiv{dom, cod}` former; `{to : A ~> B, fro : B ~> A}`
    // record literals check against it through `unify_equiv_record`.
    // `sect`/`retr` are deferred proof-irrelevant.

    /// `equiv_field_value_` — the provenance value of `field` (`to`/
    /// `fro`) when the equivalence operand is an *object literal*:
    /// the entry's value node evaluated through the annotation
    /// evaluator (`eval_ann_node`, the model's `node_term_` +
    /// `eval_term_`). `Null` for non-object operands or absent
    /// fields — proof-irrelevant provenance, best-effort.
    fn equiv_field_value(
        &self,
        lang: &Language,
        ast: &Ast,
        equiv_idx: usize,
        field: &str,
        ctx: &mut TypecheckContext,
    ) -> Value {
        let Some(node) = ast.get(AstRef::new(equiv_idx)) else {
            return Value::Null;
        };
        if node.kind != "object" {
            return Value::Null;
        }
        let Some(entries) = children_of(node) else {
            return Value::Null;
        };
        for e in entries {
            let Some(eidx) = e.as_u64() else {
                continue;
            };
            let Some(entry) = ast.get(AstRef::new(eidx as usize)) else {
                continue;
            };
            let Some(ekids) = children_of(entry) else {
                continue;
            };
            if ekids.len() < 2 {
                continue;
            }
            let (Some(kidx), Some(vidx)) = (ekids[0].as_u64(), ekids[1].as_u64()) else {
                continue;
            };
            let Some(key_node) = ast.get(AstRef::new(kidx as usize)) else {
                continue;
            };
            let key = key_node.payload.as_str().map(str::to_string).or_else(|| {
                key_node
                    .payload
                    .get("name")
                    .or_else(|| key_node.payload.get("value"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
            if key.as_deref() == Some(field)
                && let Ok(v) = self.eval_ann_node(lang, ast, vidx as usize, ctx)
                && !v.is_null()
            {
                return v;
            }
        }
        Value::Null
    }

    /// `arrow_dom_cod_` — `(domain, codomain)` of an arrow *type*:
    /// `type.function` reads `param`/`body` directly; a `type.pi`
    /// applies its `body_closure` at its own parameter (the
    /// dependent-arrow reading). `None` for non-arrows.
    fn arrow_dom_cod(
        &self,
        lang: &Language,
        ast: &Ast,
        t: &Value,
        ctx: &mut TypecheckContext,
    ) -> Option<(Value, Value)> {
        match tag_of(t) {
            Some("type.function") => Some((
                t.get("param").cloned().unwrap_or(Value::Null),
                t.get("body").cloned().unwrap_or(Value::Null),
            )),
            Some("type.pi") => {
                let p = t.get("param").cloned().unwrap_or(Value::Null);
                let cod = self.dep_closure_apply(lang, ast, t, &p, ctx).ok()?;
                if cod.is_null() { None } else { Some((p, cod)) }
            }
            _ => None,
        }
    }

    /// `equiv_parts_` — read the equivalence operand of `ua`/`univalence`
    /// as `(dom, cod, to, fro)`: a `type.equiv{dom,cod}` type reads its
    /// former fields directly; a `type.record` with `to`/`fro` arrow
    /// fields infers the domain/codomain from them, cross-checking
    /// that `fro` inverts `to` (`fro.dom ~ to.cod`, `fro.cod ~ to.dom`).
    /// `to`/`fro` are best-effort eval'd provenance (`Null` when
    /// unavailable). `Ok(None)` means the operand is not an
    /// equivalence — the diagnostic is already recorded.
    fn equiv_parts(
        &self,
        lang: &Language,
        ast: &Ast,
        equiv_idx: usize,
        span: Option<handl_language::Span>,
        ctx: &mut TypecheckContext,
    ) -> Result<Option<(Value, Value, Value, Value)>, LangError> {
        let et = self.synth_node(lang, ast, AstRef::new(equiv_idx), ctx)?;
        if tag_of(&et) == Some("type.equiv") {
            let tv = self.equiv_field_value(lang, ast, equiv_idx, "to", ctx);
            let fv = self.equiv_field_value(lang, ast, equiv_idx, "fro", ctx);
            return Ok(Some((
                et.get("dom").cloned().unwrap_or(Value::Null),
                et.get("cod").cloned().unwrap_or(Value::Null),
                tv,
                fv,
            )));
        }
        let fields = et.get("fields").cloned().unwrap_or(Value::Null);
        let to_t = fields.get("to").cloned().unwrap_or(Value::Null);
        let fro_t = fields.get("fro").cloned().unwrap_or(Value::Null);
        if to_t.is_null() || fro_t.is_null() {
            self.cub_diag(
                ctx,
                "error",
                "univalence: equivalence operand must be an Equiv(A, B) record — {to : A ~> B, fro : B ~> A}"
                    .to_string(),
                span,
            );
            return Ok(None);
        }
        let Some(ad) = self.arrow_dom_cod(lang, ast, &to_t, ctx) else {
            self.cub_diag(
                ctx,
                "error",
                "univalence: equivalence `to` field is not an arrow type".to_string(),
                span,
            );
            return Ok(None);
        };
        let Some(af) = self.arrow_dom_cod(lang, ast, &fro_t, ctx) else {
            self.cub_diag(
                ctx,
                "error",
                "univalence: equivalence `fro` field is not an arrow type".to_string(),
                span,
            );
            return Ok(None);
        };
        // Cross-check the inverse direction: `fro : B ~> A`.
        let mark = ctx.store_mark();
        let ok =
            self.dep_unify(&af.0, &ad.1, ctx).is_ok() && self.dep_unify(&af.1, &ad.0, ctx).is_ok();
        if !ok {
            ctx.store_restore(mark);
            self.cub_diag(
                ctx,
                "error",
                "univalence: equivalence `fro` does not invert `to` — expected B ~> A".to_string(),
                span,
            );
            return Ok(None);
        }
        let tv = self.equiv_field_value(lang, ast, equiv_idx, "to", ctx);
        let fv = self.equiv_field_value(lang, ast, equiv_idx, "fro", ctx);
        Ok(Some((ad.0, ad.1, tv, fv)))
    }

    /// `synth_univalence_core_` — the shared univalence introduction:
    /// given the equivalence operand's AST index, mint the
    /// path-carrying `type.pi` — `(i:I) → Type` (universe-level: the
    /// endpoints are *type values*) with `path{x: A, y: B, equiv:
    /// {dom, cod, to, fro}}`. Used by the `univalence` keyword node,
    /// the `ua`/`Univalence` call form (term and type position), and
    /// `Equiv`-checked annotations.
    fn synth_univalence_core(
        &self,
        lang: &Language,
        ast: &Ast,
        equiv_idx: usize,
        span: Option<handl_language::Span>,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let Some((dom, cod, to, fro)) = self.equiv_parts(lang, ast, equiv_idx, span, ctx)? else {
            return Ok(self.cub_unknown(ctx));
        };
        if dom.is_null() || cod.is_null() {
            return Ok(self.cub_unknown(ctx));
        }
        let mut interval_m = Map::new();
        let interval = make_tag(ctx, "type.interval", std::mem::take(&mut interval_m));
        let mut universe_m = Map::new();
        universe_m.insert("level".to_string(), Value::from(0));
        let universe = make_tag(ctx, "type.universe", universe_m);
        let mut equiv_m = Map::new();
        equiv_m.insert("dom".to_string(), dom.clone());
        equiv_m.insert("cod".to_string(), cod.clone());
        equiv_m.insert("to".to_string(), to);
        equiv_m.insert("fro".to_string(), fro);
        let mut path_m = Map::new();
        path_m.insert("x".to_string(), dom);
        path_m.insert("y".to_string(), cod);
        path_m.insert("equiv".to_string(), Value::Object(equiv_m));
        let mut m = Map::new();
        m.insert("param".to_string(), interval);
        m.insert(
            "body_closure".to_string(),
            serde_json::json!({
                "var_name": "",
                "body_child_idx": Value::Null,
                "dep": true,
                "const_body": universe,
            }),
        );
        m.insert("path".to_string(), Value::Object(path_m));
        Ok(make_tag(ctx, "type.pi", m))
    }

    /// `univalence(e)` — the keyword node's judgment (term position):
    /// child role `equiv` is the equivalence operand.
    pub(crate) fn synth_univalence_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        let e_ref = Self::cub_child(lang, ast, node, "equiv", 0);
        if kids_len < 1 || e_ref.is_none() {
            self.cub_diag(
                ctx,
                "error",
                "univalence: needs an equivalence operand".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        self.synth_univalence_core(lang, ast, e_ref.unwrap().index(), span, ctx)
    }

    /// `unify_equiv_record_` — unify a `type.equiv{dom,cod}` against a
    /// `type.record{fields}` (either order): the record must carry
    /// `to : dom ~> cod` and `fro : cod ~> dom` arrow fields; extra
    /// fields are ignored. Returns the model's diagnostics as
    /// `LangError`s (the `dep_unify` failure channel).
    fn unify_equiv_record(
        &self,
        equiv_t: &Value,
        record_t: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<(), LangError> {
        let dom = equiv_t.get("dom").cloned().unwrap_or(Value::Null);
        let cod = equiv_t.get("cod").cloned().unwrap_or(Value::Null);
        let fields = record_t.get("fields").cloned().unwrap_or(Value::Null);
        let to_f = fields.get("to").cloned().unwrap_or(Value::Null);
        let fro_f = fields.get("fro").cloned().unwrap_or(Value::Null);
        if to_f.is_null() || fro_f.is_null() {
            return Err(LangError::engine(
                "type mismatch: Equiv record needs `to` and `fro` fields".to_string(),
            ));
        }
        let mut to_m = Map::new();
        to_m.insert("param".to_string(), dom.clone());
        to_m.insert("body".to_string(), cod.clone());
        to_m.insert("param_name".to_string(), Value::String(String::new()));
        let to_arr = make_tag(ctx, "type.function", to_m);
        let mut fro_m = Map::new();
        fro_m.insert("param".to_string(), cod.clone());
        fro_m.insert("body".to_string(), dom.clone());
        fro_m.insert("param_name".to_string(), Value::String(String::new()));
        let fro_arr = make_tag(ctx, "type.function", fro_m);
        let dom_tag = tag_of(&dom).unwrap_or("?");
        let cod_tag = tag_of(&cod).unwrap_or("?");
        self.dep_unify(&to_f, &to_arr, ctx).map_err(|_| {
            LangError::engine(format!(
                "type mismatch: Equiv `to` field is not {dom_tag} ~> {cod_tag}"
            ))
        })?;
        self.dep_unify(&fro_f, &fro_arr, ctx).map_err(|_| {
            LangError::engine(format!(
                "type mismatch: Equiv `fro` field is not {cod_tag} ~> {dom_tag}"
            ))
        })
    }

    // -------------------------------------------------------------------
    // Cubical Glue & fibrancy (ADR-0208 P3): glue / Glue / unglue /
    // isFibrant / fibrant
    // -------------------------------------------------------------------
    //
    // CCHM-style glue types over the P1 interval and P2 faces — a
    // *typechecker*, not a cubical kernel: `Glue A φ T̂` is represented
    // and checked structurally as `type.glue{family, face, section}`;
    // the interval-computational content (transport computing through
    // the glued type, `ua`'s `transp (ua e) ≡ e`) is deferred to P4.
    // What IS checked: the face operand must normalize to a face
    // formula (`cub_face_of_node`/`face_norm`), the section should be
    // an `I → Type` family (a non-family section warns — the
    // `Partial φ (Σ T:Type, T ≃ A i)` obligation is approximated), and
    // degenerate faces (`φ ≡ 0`/`φ ≡ 1`) warn. `unglue(G)` projects a
    // `type.glue` back to its `family`. `isFibrant(A)`/`fibrant A`
    // form/inhabit `type.fibrant{of}` via the structural
    // `fibrant_verdict` below.

    /// `Glue A φ T̂` formation — the model's `synth_glue_type_`, shared
    /// by the `glue` keyword node and the `Glue(...)` call form.
    /// `A` synthesizes as a type (a plain type is the constant
    /// family); `φ` evaluates through the P2 face machinery; `T̂`
    /// should synthesize as an interval `type.pi`. Mints
    /// `type.glue{family, face, section}` with the face stored as the
    /// model's `face{op, args}` record.
    fn dep_synth_glue_type(
        &self,
        lang: &Language,
        ast: &Ast,
        fam_idx: usize,
        face_idx: usize,
        sec_idx: usize,
        span: Option<handl_language::Span>,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let fam = self.dep_synth_type(lang, ast, fam_idx, ctx)?;
        let face_node = ast.get(AstRef::new(face_idx)).ok_or_else(|| {
            LangError::engine("dep_synth_glue_type: face node out of bounds".to_string())
        })?;
        let Some(phi_raw) = self.cub_face_of_node(lang, ast, face_node, ctx) else {
            self.cub_diag(
                ctx,
                "error",
                "glue: face operand is not a face formula — expected f0(i), f1(i), fnot φ, f_or φ ψ, f_and φ ψ, fT, or fF".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        let phi = face_norm(phi_raw);
        let sec = self.dep_synth_type(lang, ast, sec_idx, ctx)?;
        let sec_is_pi = tag_of(&sec) == Some("type.pi")
            && tag_of(sec.get("param").unwrap_or(&Value::Null)) == Some("type.interval");
        if !sec_is_pi {
            self.cub_diag(
                ctx,
                "warning",
                "glue: section is not an I → Type family — expected `Pi(I, i ~> T i)`; the `Partial φ (Σ T:Type, T ≃ A i)` obligation is approximated, treating the section as constant".to_string(),
                span,
            );
        }
        match phi {
            CubFace::Tt => self.cub_diag(
                ctx,
                "warning",
                "glue: face φ ≡ 1 — Glue agrees with the section on every dimension (degenerate)"
                    .to_string(),
                span,
            ),
            CubFace::Ff => self.cub_diag(
                ctx,
                "warning",
                "glue: face φ ≡ 0 — Glue agrees with the family on every dimension (degenerate)"
                    .to_string(),
                span,
            ),
            _ => {}
        }
        let mut m = Map::new();
        m.insert("family".to_string(), fam);
        m.insert("face".to_string(), cub_face_to_val(ctx, &phi));
        m.insert("section".to_string(), sec);
        Ok(make_tag(ctx, "type.glue", m))
    }

    /// `glue A, φ, T̂` — the keyword node's judgment (term position):
    /// runs the same formation checks and returns the `type.glue` it
    /// forms as the node's type (type formers are their own type
    /// here — the `Nat : Family` reading — which keeps `G := glue …`
    /// aliases working).
    pub(crate) fn synth_glue_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        if kids_len < 3 {
            self.cub_diag(
                ctx,
                "error",
                "glue: needs family, face, and section operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let (Some(fam_ref), Some(face_ref), Some(sec_ref)) = (
            Self::cub_child(lang, ast, node, "family", 0),
            Self::cub_child(lang, ast, node, "face", 1),
            Self::cub_child(lang, ast, node, "section", 2),
        ) else {
            self.cub_diag(
                ctx,
                "error",
                "glue: needs family, face, and section operands".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        };
        self.dep_synth_glue_type(
            lang,
            ast,
            fam_ref.index(),
            face_ref.index(),
            sec_ref.index(),
            span,
            ctx,
        )
    }

    /// The codomain type of a dependent `type.pi`/`type.sigma`/
    /// `type.forall` at its own bound variable — the model's
    /// `apply_dep_closure_` reading inside `fibrant_verdict_`: the
    /// closure applies at `var{var_name}` so the body reads the binder
    /// rigidly. A closure with no computable body (`body_closure`
    /// absent, or neither `const_body`/`apply_type`/`body_child_idx`)
    /// yields `type.unknown`, whose verdict is `Unknown`.
    fn fibrant_codomain(
        &self,
        lang: &Language,
        ast: &Ast,
        ty: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let unknown = |ctx: &mut TypecheckContext| {
            let mut m = Map::new();
            make_tag(ctx, "type.unknown", std::mem::take(&mut m))
        };
        let Some(bc) = ty.get("body_closure") else {
            return Ok(unknown(ctx));
        };
        if bc.get("const_body").is_none()
            && bc.get("apply_type").is_none()
            && bc.get("body_child_idx").and_then(Value::as_u64).is_none()
        {
            return Ok(unknown(ctx));
        }
        let var_name = bc
            .get("var_name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // The model binds `var{var_name}` as the argument term (or no
        // term at all for an unnamed closure); `dep_closure_apply_term`
        // records `results[var_name] = eval_index(term)` — pass the
        // rigid var atom so the body sees the binder.
        let term = if var_name.is_empty() {
            Value::Null
        } else {
            var_term(&var_name)
        };
        self.dep_closure_apply_term(lang, ast, ty, &term, ctx)
    }

    /// Structural fibrancy verdict — the model's `fibrant_verdict_`:
    /// base sorts (`type.number`/`type.bool`/`type.string`/
    /// `type.unit`/`type.interval`/`type.null`), `type.fibrant`,
    /// dependent `type.pi`/`type.sigma`/`type.forall`/`type.glue`
    /// whose parts are fibrant (codomains computed through
    /// `fibrant_codomain`), `type.function`, and `type.array`/
    /// `type.record` with all-fibrant elements/fields are `Yes`;
    /// `type.universe` (`Type` is not fibrant) and non-types are `No`;
    /// `type.unknown`, metas, `type.family`/`type.family_app`
    /// (indexed data — fibrancy depends on the indices),
    /// `type.constructor`, legacy `Path`/`type.eq`, and anything else
    /// are `Unknown`.
    fn fibrant_verdict(
        &self,
        lang: &Language,
        ast: &Ast,
        t: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<FibrantVerdict, LangError> {
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            if extract_meta_id(t).is_some() || tag_of(t) == Some("type.unknown") {
                return Ok(FibrantVerdict::Unknown);
            }
            let Some(tag) = tag_of(t) else {
                // Non-types (untagged values, `tag == ""` in the model)
                // are not fibrant.
                return Ok(FibrantVerdict::No);
            };
            match tag {
                "type.universe" => Ok(FibrantVerdict::No),
                "type.number" | "type.bool" | "type.string" | "type.unit" | "type.interval"
                | "type.null" | "type.fibrant" => Ok(FibrantVerdict::Yes),
                "type.pi" | "type.sigma" | "type.forall" => {
                    let dom_field = if tag == "type.forall" {
                        "kind"
                    } else {
                        "param"
                    };
                    let dom = t.get(dom_field).cloned().unwrap_or(Value::Null);
                    let dv = self.fibrant_verdict(lang, ast, &dom, ctx)?;
                    if dv != FibrantVerdict::Yes {
                        return Ok(dv);
                    }
                    let cod = self.fibrant_codomain(lang, ast, t, ctx)?;
                    self.fibrant_verdict(lang, ast, &cod, ctx)
                }
                "type.function" => {
                    let pv = self.fibrant_verdict(
                        lang,
                        ast,
                        t.get("param").unwrap_or(&Value::Null),
                        ctx,
                    )?;
                    if pv != FibrantVerdict::Yes {
                        return Ok(pv);
                    }
                    self.fibrant_verdict(lang, ast, t.get("body").unwrap_or(&Value::Null), ctx)
                }
                "type.glue" => {
                    // `Glue A φ T̂` is fibrant when the family `A` and
                    // the section `T̂` are (CCHM: Glue of fibrant data
                    // is fibrant). The face `φ` carries no fibrancy
                    // content.
                    let fv = self.fibrant_verdict(
                        lang,
                        ast,
                        t.get("family").unwrap_or(&Value::Null),
                        ctx,
                    )?;
                    if fv != FibrantVerdict::Yes {
                        return Ok(fv);
                    }
                    self.fibrant_verdict(lang, ast, t.get("section").unwrap_or(&Value::Null), ctx)
                }
                "type.array" => {
                    // NBE mints both spellings: `element` (the `array`
                    // kind) and `elements` (the `paren` tuple) — fold
                    // over whichever is present.
                    let mut verdict = FibrantVerdict::Yes;
                    let mut els: Vec<Value> = t
                        .get("elements")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    if let Some(el) = t.get("element") {
                        els.push(el.clone());
                    }
                    for el in els {
                        let ev = self.fibrant_verdict(lang, ast, &el, ctx)?;
                        verdict = match (verdict, ev) {
                            (FibrantVerdict::No, _) | (_, FibrantVerdict::No) => FibrantVerdict::No,
                            (FibrantVerdict::Unknown, _) | (_, FibrantVerdict::Unknown) => {
                                FibrantVerdict::Unknown
                            }
                            _ => FibrantVerdict::Yes,
                        };
                        if verdict == FibrantVerdict::No {
                            break;
                        }
                    }
                    Ok(verdict)
                }
                "type.record" => {
                    let mut verdict = FibrantVerdict::Yes;
                    if let Some(fields) = t.get("fields").and_then(Value::as_object) {
                        for (_, fv) in fields {
                            let ev = self.fibrant_verdict(lang, ast, fv, ctx)?;
                            verdict = match (verdict, ev) {
                                (FibrantVerdict::No, _) | (_, FibrantVerdict::No) => {
                                    FibrantVerdict::No
                                }
                                (FibrantVerdict::Unknown, _) | (_, FibrantVerdict::Unknown) => {
                                    FibrantVerdict::Unknown
                                }
                                _ => FibrantVerdict::Yes,
                            };
                            if verdict == FibrantVerdict::No {
                                break;
                            }
                        }
                    }
                    Ok(verdict)
                }
                _ => Ok(FibrantVerdict::Unknown),
            }
        })
    }

    /// `isFibrant(A)` / `fibrant A` in *type* position — the model's
    /// `synth_fibrant_former_`: mints `type.fibrant{of}` on a `Yes`
    /// verdict; `No`/`Unknown` degrade to `type.unknown` + warning so
    /// unsupported formers are never silently accepted.
    fn dep_synth_fibrant_former(
        &self,
        lang: &Language,
        ast: &Ast,
        arg_idx: usize,
        span: Option<handl_language::Span>,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let a = self.dep_synth_type(lang, ast, arg_idx, ctx)?;
        let fv = self.fibrant_verdict(lang, ast, &a, ctx)?;
        if fv == FibrantVerdict::Yes {
            let mut m = Map::new();
            m.insert("of".to_string(), a);
            return Ok(make_tag(ctx, "type.fibrant", m));
        }
        let tag = tag_of(&a).unwrap_or("");
        let msg = if fv == FibrantVerdict::No {
            format!("isFibrant: {tag} is not fibrant")
        } else {
            format!(
                "isFibrant: fibrancy of {tag} is undecided — the fibrancy type is left open (type.unknown)"
            )
        };
        self.cub_diag(ctx, "warning", msg, span);
        Ok(self.cub_open_meta(ctx))
    }

    /// `fibrant A` — the keyword node's judgment (term position): the
    /// fibrancy *witness*. Its type is `type.fibrant{of}` on `Yes` and
    /// on `Unknown` (optimistic — the P2 `Partial` approximation); a
    /// `No` verdict is a hard error.
    pub(crate) fn synth_fibrant_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let span = node.span;
        let kids_len = children_of(node).map(|c| c.len()).unwrap_or(0);
        let t_ref = Self::cub_child(lang, ast, node, "target", 0);
        if kids_len < 1 || t_ref.is_none() {
            self.cub_diag(
                ctx,
                "error",
                "fibrant: needs a type operand".to_string(),
                span,
            );
            return Ok(self.cub_unknown(ctx));
        }
        let a = self.dep_synth_type(lang, ast, t_ref.unwrap().index(), ctx)?;
        let fv = self.fibrant_verdict(lang, ast, &a, ctx)?;
        let mut m = Map::new();
        m.insert("of".to_string(), a.clone());
        let fibrant_t = make_tag(ctx, "type.fibrant", m);
        match fv {
            FibrantVerdict::Yes => Ok(fibrant_t),
            FibrantVerdict::No => {
                let tag = tag_of(&a).unwrap_or("");
                self.cub_diag(
                    ctx,
                    "error",
                    format!(
                        "fibrant: {tag} is not fibrant — no homogeneous-composition witness exists"
                    ),
                    span,
                );
                Ok(self.cub_unknown(ctx))
            }
            FibrantVerdict::Unknown => {
                let tag = tag_of(&a).unwrap_or("");
                self.cub_diag(
                    ctx,
                    "warning",
                    format!(
                        "fibrant: fibrancy of {tag} is undecided — accepting the witness optimistically"
                    ),
                    span,
                );
                Ok(fibrant_t)
            }
        }
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
        let src_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("dim_arrow: src index not a u64".to_string()))?
            as usize;
        let dim_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("dim_arrow: dim index not a u64".to_string()))?
            as usize;
        let tgt_idx = kids[2]
            .as_u64()
            .ok_or_else(|| LangError::engine("dim_arrow: tgt index not a u64".to_string()))?
            as usize;
        // The `source`/`target` endpoints are quoted *law terms*
        // (ADR-0119): the bootstrap scope pass checks them in pattern
        // mode — a free `var` is a pattern variable (`g` in
        // `dedupe<N> := (g >>> g) ~[N]~> g`), not a wire reference —
        // so each unresolvable name binds a fresh meta for the
        // duration of the endpoint synth.
        let law_mark = ctx.mark();
        for idx in [src_idx, tgt_idx] {
            if let Some(n) = ast.get(AstRef::new(idx)) {
                self.bind_law_pattern_vars(lang, ast, n, ctx);
            }
        }
        let judged = (|ctx: &mut TypecheckContext| -> Result<(Value, Value), LangError> {
            let src = self.synth_node(lang, ast, AstRef::new(src_idx), ctx)?;
            // The dim expression still synthesizes (node_types + meta
            // solving); the emitted former no longer carries a `grade`
            // field — the dimension index IS the first `Hom` arg.
            let _grade = self.synth_node(lang, ast, AstRef::new(dim_idx), ctx)?;
            let tgt = self.synth_node(lang, ast, AstRef::new(tgt_idx), ctx)?;
            Ok((src, tgt))
        })(ctx);
        ctx.restore(law_mark);
        let (src, tgt) = judged?;
        // Dimension literal (from the child payload), then its type.
        let dim_node = ast
            .get(AstRef::new(dim_idx))
            .ok_or_else(|| LangError::engine("dim_arrow: dim node out of bounds".to_string()))?;
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
            Some(dim_val)
        } else {
            None
        };
        // Canonical `Former{type.hom}` data (ADR-0153): `~[n]~>` is
        // `Hom(n, src, tgt)` — a literal dim is a `Const` natural; a
        // scope-bound dimension variable (`f<N> := … ~[N]~> …`,
        // ADR-0106 §12) elaborates as its rigid index term; any other
        // symbolic dim mints a `Dim`-typed meta. Endpoints lift through
        // the `__tag__` bridge into the `Val` domain.
        let dim_v = match dim_lit.and_then(|d| d.as_i64()) {
            Some(n) => Val::Const(TaggedValue::Int(n)),
            None => {
                let resolved = node_name(dim_inner)
                    .filter(|_| dim_inner.kind == "var")
                    .and_then(|n| ctx.results.get(n.as_str()).cloned())
                    .map(|b| eval_index(&b, ctx))
                    .filter(|v| tag_of(v) == Some("var") || num_val(v).is_some());
                match resolved {
                    Some(v) => domain_lift(&v, ctx),
                    None => dim_meta(ctx),
                }
            }
        };
        let va = domain_lift(&src, ctx);
        let vb = domain_lift(&tgt, ctx);
        Ok(hom_former(dim_v, va, vb, ctx))
    }

    /// Bind the free `var`s of a `~[N]~>` law endpoint to fresh metas —
    /// the pattern-mode counterpart of the scope pass's rule-term
    /// check. A name already resolvable (Γ, results, injected or
    /// ambient globals) is a real reference and is left alone; `_`
    /// binds nothing. Callers scope the bindings with
    /// `ctx.mark`/`ctx.restore`.
    fn bind_law_pattern_vars(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) {
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
            if node.kind == "var" {
                if let Some(name) = node_name(node) {
                    let resolvable = name == "_"
                        || ctx.lookup(&name).is_some()
                        || ctx.results.contains_key(name.as_str())
                        || lang.globals.contains_key(name.as_str());
                    if !resolvable {
                        let meta = ctx.fresh_meta();
                        ctx.extend(name, meta_value(meta));
                    }
                }
                return;
            }
            if let Some(kids) = children_of(node) {
                for cv in kids {
                    if let Some(i) = cv.as_u64()
                        && let Some(child) = ast.get(AstRef::new(i as usize))
                    {
                        self.bind_law_pattern_vars(lang, ast, child, ctx);
                    }
                }
            }
        })
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
        let kids =
            children_of(node).ok_or_else(|| LangError::engine("flow: no children".to_string()))?;
        if kids.len() < 2 {
            return Err(LangError::engine("flow: needs two operands".to_string()));
        }
        let f_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("flow: lhs index not a u64".to_string()))?
            as usize;
        let g_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("flow: rhs index not a u64".to_string()))?
            as usize;
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
        // Domain-encoded operands (a `Former{type.hom}` produced by
        // `~[n]~>` or a prior composition) route through the domain
        // boundary law — `f >>> g` on `Hom`-typed cells is the
        // axis-0 paste.
        if is_domain_encoded(f_ty) || is_domain_encoded(g_ty) {
            return self.comp_axis_typed(&Val::Const(TaggedValue::Int(0)), f_ty, g_ty, ctx);
        }
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
            m.insert(
                "name".to_string(),
                g_ty.get("name").cloned().unwrap_or(Value::Null),
            );
            m.insert("base".to_string(), f_dom);
            return Ok(make_tag(ctx, "type.family", m));
        }
        match (tag_of(f_ty), tag_of(g_ty)) {
            (Some("type.function"), Some("type.function")) => {
                let f_body = f_ty.get("body").cloned().unwrap_or(Value::Null);
                let g_param = g_ty.get("param").cloned().unwrap_or(Value::Null);
                self.dep_unify(&f_body, &g_param, ctx)?;
                let mut m = Map::new();
                m.insert(
                    "param".to_string(),
                    f_ty.get("param").cloned().unwrap_or(Value::Null),
                );
                m.insert(
                    "body".to_string(),
                    g_ty.get("body").cloned().unwrap_or(Value::Null),
                );
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
        let a_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("axis_flow: lhs index not a u64".to_string()))?
            as usize;
        let k_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("axis_flow: axis index not a u64".to_string()))?
            as usize;
        let b_idx = kids[2]
            .as_u64()
            .ok_or_else(|| LangError::engine("axis_flow: rhs index not a u64".to_string()))?
            as usize;
        let k_node = ast
            .get(AstRef::new(k_idx))
            .ok_or_else(|| LangError::engine("axis_flow: axis node out of bounds".to_string()))?;
        let k_lit = k_node.payload.as_f64();
        let a_ty = self.synth_node(lang, ast, AstRef::new(a_idx), ctx)?;
        let b_ty = self.synth_node(lang, ast, AstRef::new(b_idx), ctx)?;
        if k_lit == Some(0.0) || k_lit.is_none() {
            return self.flow_of(&a_ty, &b_ty, ctx);
        }
        let kv = Val::Const(TaggedValue::Int(k_lit.unwrap_or(0.0) as i64));
        self.comp_axis_typed(&kv, &a_ty, &b_ty, ctx)
    }

    /// `a >>>[k] b` (and the derived `=>>`/`=>|` cell ops) over lifted
    /// operand *types*: the globular boundary law is decided in the
    /// `Val` domain through [`checker::Checker::infer_comp_axis_tys`].
    /// Decided errors (dimension, boundary mismatch) propagate; stuck
    /// operands (metas, non-cell types, dim ≥ 3) store the composite
    /// `Hom(d, a_ty, b_ty)` unchecked — `d` the greatest operand
    /// dimension or a `Dim` meta — the ADR-0153 honest bound.
    fn comp_axis_typed(
        &self,
        kv: &Val,
        a_ty: &Value,
        b_ty: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let va = domain_lift(a_ty, ctx);
        let vb = domain_lift(b_ty, ctx);
        match ctx.vcheck.infer_comp_axis_tys(kv, &va, &vb) {
            Ok(t) => Ok(domain_quote(&t, ctx)),
            Err(e @ (checker::TcError::Dim { .. } | checker::TcError::Mismatch { .. })) => {
                Err(LangError::engine(format!("typecheck: axis_flow: {e}")))
            }
            Err(_) => {
                let d = hom_dim_lit(&va)
                    .into_iter()
                    .chain(hom_dim_lit(&vb))
                    .max()
                    .map(|m| Val::Const(TaggedValue::Int(m as i64)))
                    .unwrap_or_else(|| dim_meta(ctx));
                Ok(hom_former(d, va, vb, ctx))
            }
        }
    }

    /// `id2 x` — the identity 2-cell on `x`: `Hom(2, x_ty, x_ty)` in
    /// the tagged approximation (operand types stand in for cell
    /// endpoints — the same convention `comp_axis_typed` keeps).
    pub(crate) fn synth_cell_id_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("cell_id: no children".to_string()))?;
        let x_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("cell_id: operand index not a u64".to_string()))?
            as usize;
        let x_ty = self.synth_node(lang, ast, AstRef::new(x_idx), ctx)?;
        let vx = domain_lift(&x_ty, ctx);
        Ok(hom_former(
            Val::Const(TaggedValue::Int(2)),
            vx.clone(),
            vx,
            ctx,
        ))
    }

    /// `a =>> b` (vertical, axis 2) / `a =>| b` (horizontal, axis 0) —
    /// the derived cell compositions share the `>>>[k]` boundary law.
    pub(crate) fn synth_cell_comp_axis_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        axis: i64,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("cell_comp: no children".to_string()))?;
        if kids.len() < 2 {
            return Err(LangError::engine(
                "cell_comp: needs two operands".to_string(),
            ));
        }
        let a_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("cell_comp: lhs index not a u64".to_string()))?
            as usize;
        let b_idx = kids[1]
            .as_u64()
            .ok_or_else(|| LangError::engine("cell_comp: rhs index not a u64".to_string()))?
            as usize;
        let a_ty = self.synth_node(lang, ast, AstRef::new(a_idx), ctx)?;
        let b_ty = self.synth_node(lang, ast, AstRef::new(b_idx), ctx)?;
        self.comp_axis_typed(&Val::Const(TaggedValue::Int(axis)), &a_ty, &b_ty, ctx)
    }

    /// Whether `node` (a `call`/`index_app`/`dimension`) invokes the
    /// `perform` op-emission builtin: the callee is `var("perform")`, or
    /// a `field-access` selecting `.perform` off `var("io")`
    /// (`io.perform`). A Γ/results binding for `perform` shadows the
    /// builtin — the same source-level-shadows-builtin convention
    /// `dep_synth_var_type` applies (injected `type.unknown` wire
    /// placeholders do not shadow: `hnl typecheck`'s prelude seeds the
    /// `io`/`perform` module wires as placeholders, which is exactly
    /// what lets the builtin name reach `synth_call` unannotated).
    fn is_perform_call(&self, ast: &Ast, node: &AstNode, ctx: &TypecheckContext) -> bool {
        let Some(kids) = children_of(node) else {
            return false;
        };
        let Some(callee) = kids
            .first()
            .and_then(Value::as_u64)
            .and_then(|i| ast.get(AstRef::new(i as usize)))
        else {
            return false;
        };
        let is_perform = match callee.kind.as_str() {
            "var" => node_name(callee).as_deref() == Some("perform"),
            "field-access" => {
                // `io.perform` — the receiver must be the module wire
                // `var("io")`; the field is child 1 (`payload.field` as
                // the fallback for hand-built nodes).
                let receiver_is_io = children_of(callee)
                    .and_then(|k| k.first())
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .is_some_and(|r| r.kind == "var" && node_name(r).as_deref() == Some("io"))
                    && ctx.lookup_gamma("io").is_none();
                let field_is_perform = children_of(callee)
                    .and_then(|k| k.get(1))
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(node_name)
                    .or_else(|| {
                        callee
                            .payload
                            .get("field")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .as_deref()
                    == Some("perform");
                receiver_is_io && field_is_perform
            }
            _ => false,
        };
        is_perform && ctx.lookup_gamma("perform").is_none() && !ctx.results.contains_key("perform")
    }

    /// `perform(op, arg)` — synthesize `IO{op: T}` (ADR-0153 §4). `op` is
    /// a *named* op: a `var` reads as the op *name* (at runtime an
    /// unbound var evaluates to its name string; `puts` is deliberately
    /// never checked as a value), a string literal reads directly, and
    /// `sym("name")`/`sym.seeded("name")` reads its seed. `T` is `arg`'s
    /// synthesized type (`type.unit` when the arg is omitted). The
    /// result is canonical `Former{type.io}` data — the same spelling
    /// `dep_synth_call_type`'s `IO(sig)` arm emits — so `handle`'s
    /// `handle_sig_check` and sig record-width subtyping read it
    /// unchanged.
    fn synth_perform(
        &self,
        lang: &Language,
        ast: &Ast,
        _node: &AstNode,
        kids: &[Value],
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let op_idx =
            kids.get(1).and_then(Value::as_u64).ok_or_else(|| {
                LangError::engine("perform: expected perform(op, arg)".to_string())
            })? as usize;
        let op_node = ast
            .get(AstRef::new(op_idx))
            .ok_or_else(|| LangError::engine("perform: op operand out of bounds".to_string()))?;
        let string_payload = |n: &AstNode| {
            n.payload
                .get("value")
                .and_then(Value::as_str)
                .or_else(|| n.payload.as_str())
                .map(str::to_string)
        };
        let op_name = match op_node.kind.as_str() {
            "var" => node_name(op_node),
            "value" | "lit" => string_payload(op_node),
            "call" => {
                // `sym("puts")` / `sym.seeded("puts")` — the op name is
                // the sym's seed string.
                let ck = children_of(op_node);
                let cname = ck
                    .and_then(|k| k.first())
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(|c| {
                        if c.kind == "var" {
                            node_name(c)
                        } else {
                            children_of(c)
                                .and_then(|k| k.get(1))
                                .and_then(Value::as_u64)
                                .and_then(|i| ast.get(AstRef::new(i as usize)))
                                .and_then(node_name)
                        }
                    });
                match cname.as_deref() {
                    Some("sym") | Some("seeded") => ck
                        .and_then(|k| k.get(1))
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(string_payload),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(op_name) = op_name else {
            return Err(LangError::engine(
                "perform: op operand must name an op — a bare identifier, \
                 a string literal, or sym(\"name\")"
                    .to_string(),
            ));
        };
        let arg_ty = match kids.get(2).and_then(Value::as_u64) {
            Some(i) => self.synth_node(lang, ast, AstRef::new(i as usize), ctx)?,
            None => {
                let mut m = Map::new();
                make_tag(ctx, "type.unit", std::mem::take(&mut m))
            }
        };
        let mut sig = Map::new();
        sig.insert(op_name, arg_ty);
        let vs = domain_lift(&Value::Object(sig), ctx);
        Ok(domain_quote(
            &Val::Former {
                head: checker::type_sym("type.io"),
                args: vec![vs],
            },
            ctx,
        ))
    }

    /// `check_call` — the directed `call` check (the `type.call` `check`
    /// scheme, lifted to a judgment so `perform` can intercept). For a
    /// `perform(op, arg)` call the generic scheme cannot see that `op`
    /// is a *name* — checking it as a value reports an unbound `puts` —
    /// so the call is synthesized (`synth_call`'s `synth_perform` branch
    /// mints `IO{op: T}`) and unified against the expected type — record
    /// -width sig subtyping included (`IO{puts: Str}` accepts an
    /// `IO{puts, fs}` expectation). Every other call reproduces the
    /// former scheme verbatim: synth the callee, check the first
    /// argument against its `param`, yield `body`.
    pub(crate) fn check_call_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node_ref: AstRef,
        node: &AstNode,
        incoming: Option<Value>,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        if self.is_perform_call(ast, node, ctx) {
            let expected = incoming.unwrap_or(Value::Null);
            let ty = self.synth_node(lang, ast, node_ref, ctx)?;
            self.dep_unify(&ty, &expected, ctx)?;
            return Ok(ty);
        }
        // `ua(e)`/`Univalence(e)` in check position (ADR-0208 P4):
        // synthesize the univalence path type and unify it against
        // the expected type — the model's `call` judgment intercept.
        let ua_call = children_of(node)
            .and_then(|k| k.first())
            .and_then(Value::as_u64)
            .and_then(|i| ast.get(AstRef::new(i as usize)))
            .is_some_and(|c| {
                c.kind == "var" && matches!(node_name(c).as_deref(), Some("ua" | "Univalence"))
            });
        if ua_call {
            let kids = children_of(node)
                .ok_or_else(|| LangError::engine("check_call: no children".to_string()))?;
            if kids.len() != 2 {
                self.cub_diag(
                    ctx,
                    "warning",
                    "ua/Univalence needs exactly one equivalence operand".to_string(),
                    node.span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            let arg_idx = kids[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("check_call: arg index not a u64".to_string()))?
                as usize;
            let ty = self.synth_univalence_core(lang, ast, arg_idx, node.span, ctx)?;
            if let Some(exp) = incoming {
                self.dep_unify(&ty, &exp, ctx)?;
            }
            return Ok(ty);
        }
        // `transport(p, v)` call form in check position — synthesize
        // the transport type and unify it against the expected type
        // (the model's `call` judgment intercept, `ua` precedent).
        let transport_call = children_of(node)
            .and_then(|k| k.first())
            .and_then(Value::as_u64)
            .and_then(|i| ast.get(AstRef::new(i as usize)))
            .is_some_and(|c| c.kind == "var" && node_name(c).as_deref() == Some("transport"));
        if transport_call {
            let kids = children_of(node)
                .ok_or_else(|| LangError::engine("check_call: no children".to_string()))?;
            if kids.len() != 3 {
                self.cub_diag(
                    ctx,
                    "warning",
                    "transport(p, v) needs exactly two operands — a path and a value".to_string(),
                    node.span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            let path_idx = kids[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("check_call: path index not a u64".to_string()))?
                as usize;
            let val_idx = kids[2]
                .as_u64()
                .ok_or_else(|| LangError::engine("check_call: value index not a u64".to_string()))?
                as usize;
            let ty = self.transport_judgment_at(lang, ast, path_idx, val_idx, node.span, ctx)?;
            if let Some(exp) = incoming {
                self.dep_unify(&ty, &exp, ctx)?;
            }
            return Ok(ty);
        }
        let kids = children_of(node)
            .ok_or_else(|| LangError::engine("check_call: no children".to_string()))?;
        let callee_idx =
            kids.first().and_then(Value::as_u64).ok_or_else(|| {
                LangError::engine("check_call: callee index not a u64".to_string())
            })? as usize;
        let funtype = self.synth_node(lang, ast, AstRef::new(callee_idx), ctx)?;
        // `f()` in check position — delegate to the zero-argument peel
        // (`Unit`/unknown domain only; anything else is a missing
        // argument, same as `synth_call`).
        let Some(arg_raw) = kids.get(1) else {
            return self.synth_call_no_args(lang, ast, &funtype, ctx);
        };
        let arg_idx = arg_raw
            .as_u64()
            .ok_or_else(|| LangError::engine("check_call: arg index not a u64".to_string()))?
            as usize;
        let param = funtype.get("param").cloned().unwrap_or(Value::Null);
        self.check_node(lang, ast, AstRef::new(arg_idx), &param, ctx)?;
        Ok(funtype.get("body").cloned().unwrap_or(Value::Null))
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
        let kids =
            children_of(node).ok_or_else(|| LangError::engine("call: no children".to_string()))?;
        if kids.is_empty() {
            return Err(LangError::engine("call: needs callee".to_string()));
        }
        let callee_idx = kids[0]
            .as_u64()
            .ok_or_else(|| LangError::engine("call: callee index not a u64".to_string()))?
            as usize;
        // `perform` — the `IO<sig>` op-emission builtin (ADR-0153 §4):
        // `perform(op, arg)` synthesizes `IO{op: T}` — `op` is a *named*
        // op (a name operand, not a checked value) and `T` the arg's
        // type. The name-keyed callee dispatch mirrors
        // `dep_synth_call_type`'s `IO`/`Pi`/`Family` table — the scheme
        // vocabulary cannot compute a sig record from an op name.
        if self.is_perform_call(ast, node, ctx) {
            return self.synth_perform(lang, ast, node, kids, ctx);
        }
        // Cubical univalence call form (ADR-0208 P4): `ua(e)` /
        // `Univalence(e)` mint the same path-carrying `type.pi` the
        // `univalence` keyword produces — intercepted by callee name
        // before ordinary application (`ua` stays an ordinary
        // identifier everywhere else).
        if let Some(callee_node) = ast.get(AstRef::new(callee_idx))
            && callee_node.kind == "var"
            && matches!(node_name(callee_node).as_deref(), Some("ua" | "Univalence"))
        {
            if kids.len() != 2 {
                self.cub_diag(
                    ctx,
                    "warning",
                    "ua/Univalence needs exactly one equivalence operand".to_string(),
                    node.span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            let arg_idx = kids[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("call: arg index not a u64".to_string()))?
                as usize;
            return self.synth_univalence_core(lang, ast, arg_idx, node.span, ctx);
        }
        // `transport(p, v)` call form — the `p # v` transport
        // eliminator under a plain identifier (apply-forms-unification
        // §9, decided: both spellings exist). Intercepted by callee
        // name before ordinary application — the `ua`/`Univalence`
        // precedent: `transport` stays an ordinary identifier
        // everywhere else (bare uses, other arities warn here rather
        // than silently capturing a binding — same as `ua`).
        if let Some(callee_node) = ast.get(AstRef::new(callee_idx))
            && callee_node.kind == "var"
            && node_name(callee_node).as_deref() == Some("transport")
        {
            if kids.len() != 3 {
                self.cub_diag(
                    ctx,
                    "warning",
                    "transport(p, v) needs exactly two operands — a path and a value".to_string(),
                    node.span,
                );
                return Ok(self.cub_unknown(ctx));
            }
            let path_idx = kids[1]
                .as_u64()
                .ok_or_else(|| LangError::engine("call: path index not a u64".to_string()))?
                as usize;
            let val_idx = kids[2]
                .as_u64()
                .ok_or_else(|| LangError::engine("call: value index not a u64".to_string()))?
                as usize;
            return self.transport_judgment_at(lang, ast, path_idx, val_idx, node.span, ctx);
        }
        let mut cur = self.synth_node(lang, ast, AstRef::new(callee_idx), ctx)?;
        // `f()` — zero-argument application. Legal only when the
        // callee's next domain is `Unit`/unknown (host ops declare
        // `Unit -> T` signatures for nullary ops — `math.e()`,
        // `proc.pid()`); `()` at eval passes zero args to the op.
        // Anything else is a missing argument.
        if kids.len() == 1 {
            let out = self.synth_call_no_args(lang, ast, &cur, ctx)?;
            self.retry_pending_instances(ctx, false)?;
            return Ok(zonk_index(&out, ctx));
        }
        // `<>` index application — `f<i, …>` (`dimension` post-migration,
        // `index_app` for pre-desugar/hand-built trees): each argument
        // fills the next implicit `type.pi` as an index term.
        let is_index_app = matches!(node.kind.as_str(), "index_app" | "dimension");
        for cv in kids.iter().skip(1) {
            let arg_idx = cv
                .as_u64()
                .ok_or_else(|| LangError::engine("call: arg index not a u64".to_string()))?
                as usize;
            let arg_ref = AstRef::new(arg_idx);
            cur = zonk_index(&cur, ctx);
            // `f<i, …>` index application (phase-2 WS-C): each argument
            // *fills* the next implicit `type.pi` as an index term —
            // `head<2, Bool>` supplies `n := 2`, `a := Bool` — where a
            // bare `f(…)` call would have inserted a fresh meta instead.
            // Once the implicit pis are exhausted the argument falls
            // through to the ordinary domain check (e.g. `Vec<2, Bool>`
            // on a `type.family` accumulates index terms the same way
            // `Vec(2, Bool)` does).
            if is_index_app
                && tag_of(&cur) == Some("type.pi")
                && cur.get("implicit") == Some(&Value::Bool(true))
            {
                let binder = cur
                    .get("binder")
                    .and_then(Value::as_str)
                    .unwrap_or("_i")
                    .to_string();
                let t = self.node_index_or_type(lang, ast, arg_idx, ctx)?;
                let mv = eval_index(&t, ctx);
                let body = cur.get("body").cloned().unwrap_or(Value::Null);
                cur = subst_index(&body, &var_term(&binder), &mv);
                continue;
            }
            // Implicit index binders (phase-2 WS-C `f<a, …}`): fill each
            // implicit `type.pi` before consuming an explicit argument.
            // Phase-3 WS-A: a domain `C(i⃗)` whose head has registered
            // instances is *evidence* — dictionary search runs first
            // (`implicit_pi_fill`), with the fresh-meta index fill as
            // the fallback for ordinary index binders.
            while tag_of(&cur) == Some("type.pi") && cur.get("implicit") == Some(&Value::Bool(true))
            {
                let binder = cur
                    .get("binder")
                    .and_then(Value::as_str)
                    .unwrap_or("_i")
                    .to_string();
                let param = cur
                    .get("param")
                    .or_else(|| cur.get("kind"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let fill = self.implicit_pi_fill(node, &param, ctx)?;
                let body = cur.get("body").cloned().unwrap_or(Value::Null);
                cur = subst_index(&body, &var_term(&binder), &fill);
                cur = zonk_index(&cur, ctx);
            }
            cur = match tag_of(&cur) {
                Some("type.ctor_sig") => self.instantiate_ctor_full(lang, ast, &cur, ctx)?,
                Some("type.function") => {
                    let param = cur.get("param").cloned().unwrap_or(Value::Null);
                    // Positional tuple domains: a multi-param arrow
                    // (`(x, y, …) ~> body`) mints `param` as
                    // `type.array{elements: [d1, …, dn]}` — each call
                    // argument checks against the *next* domain element
                    // and the tuple shrinks until exhausted, at which
                    // point the result is `body`. A real array-typed
                    // domain spells `type.array{element: t}` (uniform)
                    // and still checks the argument as a whole array.
                    let domains = if tag_of(&param) == Some("type.array") {
                        param.get("elements").and_then(Value::as_array).cloned()
                    } else {
                        None
                    };
                    match domains {
                        Some(domains) if !domains.is_empty() => {
                            self.check_node(lang, ast, arg_ref, &domains[0], ctx)?;
                            if domains.len() > 1 {
                                let mut rem = cur.as_object().cloned().unwrap_or_default();
                                let mut dom = Map::new();
                                dom.insert(
                                    "elements".to_string(),
                                    Value::Array(domains[1..].to_vec()),
                                );
                                rem.insert("param".to_string(), make_tag(ctx, "type.array", dom));
                                Value::Object(rem)
                            } else {
                                cur.get("body").cloned().unwrap_or(Value::Null)
                            }
                        }
                        _ => {
                            self.check_node(lang, ast, arg_ref, &param, ctx)?;
                            let body = cur.get("body").cloned().unwrap_or(Value::Null);
                            // Dependent application (the model's
                            // `subst_term_` rule): a `param_name`-
                            // carrying `type.function` — a ctor
                            // telescope layer (`VCons(k : Nat, x : a,
                            // xs : Vec(k, a))`) — substitutes the
                            // argument's index term for the bound name
                            // in the codomain, so `xs`'s domain
                            // `Vec(k, a)` sees the applied index and a
                            // declared result `Vec(k + 1, a)` computes
                            // rather than minting a fresh meta.
                            match cur.get("param_name").and_then(Value::as_str) {
                                Some(pname) if !pname.is_empty() => {
                                    let arg_node = ast.get(arg_ref).ok_or_else(|| {
                                        LangError::engine(
                                            "call: arg node out of bounds".to_string(),
                                        )
                                    })?;
                                    let term = self.node_to_index_term(lang, ast, arg_node, ctx);
                                    if term.is_null() {
                                        body
                                    } else {
                                        subst_index(
                                            &body,
                                            &var_term(pname),
                                            &eval_index(&term, ctx),
                                        )
                                    }
                                }
                                _ => body,
                            }
                        }
                    }
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
                    m.insert(
                        "name".to_string(),
                        cur.get("name").cloned().unwrap_or(Value::Null),
                    );
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
        // WS-A: retry evidence obligations deferred while index metas
        // were unsettled — the explicit arg checks may have solved
        // them. Determined-but-unresolvable goals requeue; the
        // end-of-module drain owns the error.
        self.retry_pending_instances(ctx, false)?;
        Ok(zonk_index(&cur, ctx))
    }

    /// `f()` — zero-argument application. Peels exactly one domain iff
    /// it is `Unit` (`type.unit`) or unresolved (`type.unknown`): host
    /// ops declare `Unit -> T` signatures for nullary ops (`math.e()`,
    /// `proc.pid()`), and `()` at eval passes zero args to the op, so
    /// the type-level peel mirrors that. A non-unit domain is a
    /// missing argument — the same failure the per-argument
    /// `check_node` reports for `f(x)` on a non-function.
    fn synth_call_no_args(
        &self,
        lang: &Language,
        ast: &Ast,
        callee: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let cur = zonk_index(callee, ctx);
        let missing = |domain: &Value| {
            LangError::engine(format!(
                "call: missing argument — callee domain is {}",
                display_type(domain, ctx)
            ))
        };
        match tag_of(&cur) {
            // Constructor telescope: instantiate, then re-dispatch —
            // an empty telescope lands on `type.family_app` (a bare
            // type, passthrough below), a `Unit`-headed one peels, and
            // anything else reports its first domain.
            Some("type.ctor_sig") => {
                let inst = self.instantiate_ctor_full(lang, ast, &cur, ctx)?;
                self.synth_call_no_args(lang, ast, &inst, ctx)
            }
            Some("type.function") => {
                let param = cur.get("param").cloned().unwrap_or(Value::Null);
                let param = zonk_index(&param, ctx);
                if nullary_domain(&param) {
                    Ok(cur.get("body").cloned().unwrap_or(Value::Null))
                } else {
                    Err(missing(&param))
                }
            }
            Some("type.pi") | Some("type.forall") => {
                let param = cur
                    .get("param")
                    .or_else(|| cur.get("kind"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let param = zonk_index(&param, ctx);
                if nullary_domain(&param) {
                    // Dependent codomain applied at the unit term —
                    // `()` carries no index information, so `Null` is
                    // the canonical "no argument" term.
                    self.dep_closure_apply_term(lang, ast, &cur, &Value::Null, ctx)
                } else {
                    Err(missing(&param))
                }
            }
            // `family`/`family_app`/`unknown` aren't functions — a
            // zero-arg "call" of one is just the type itself (the same
            // leniency the arg loop gives `type.unknown` callees).
            Some("type.family") | Some("type.family_app") | Some("type.unknown") => Ok(cur),
            _ => {
                if extract_meta_id(&cur).is_some() || is_unknown_ty(&cur) {
                    // Unresolved callee — stay permissive.
                    Ok(cur)
                } else {
                    Err(LangError::engine(format!(
                        "call: callee is not applicable (type {})",
                        display_type(&cur, ctx)
                    )))
                }
            }
        }
    }

    /// `synth_handle` — `handle <body> with <h>` typed-effect handler
    /// typing (phase-2 WS-F; ADR-0153 §4 "effects are objects in the
    /// codomain").
    ///
    /// The body's `IO(S)` former has its op set checked against `h`'s
    /// declared impls, and the result re-wraps as `IO(S')` mirroring the
    /// runtime's interpret-in-place semantics ([`exec-tree-walk`]
    /// `interpret_io_former`): each covered entry's declared type is
    /// replaced by the impl's declared *return* type (the op is
    /// interpreted, its result is data); each uncovered entry keeps its
    /// declared signature (the op stays `IO` data for an outer handler
    /// or user destructuring). The residual sig is therefore
    /// `S'` = `(S \ S_h) ∪ {op: ret_ty}` — the spec's
    /// `IO(S \ S_h) × results` shape.
    ///
    /// Coverage rules:
    /// - `h`'s op names are its record fields (`type.record{fields}`).
    ///   A covered impl `h_k : dom ~> ret` must accept the op's declared
    ///   sig entry — `S[k] <: dom` — and `ret` is the result type.
    /// - An impl `h` declares for an op `S` never names → diagnostic
    ///   (spurious — the impl can never fire).
    /// - `S` non-empty with `h` covering none of it → diagnostic (the
    ///   `handle` is a guaranteed no-op — almost certainly a bug).
    /// - `h` op-untyped (meta / `type.Obj` / `type.unknown`) or `S`
    ///   symbolic (a sig meta/var — row-polymorphic `IO(s)`) → the
    ///   check is undecidable → pass through unchanged (honest bound).
    /// - `body`'s type is `A ~> IO(S)` (a `type.function` whose codomain
    ///   is an `IO` former) — the deep-handler case (`handle f with h`
    ///   binds the handled arrow, runtime `attach_carried_handler`):
    ///   the codomain's sig is checked the same way, producing
    ///   `A ~> IO(S')`. A `type.pi` codomain (a closure) can't rewrite
    ///   statically → passthrough.
    pub(crate) fn synth_handle_judgment(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let target_idx = role_child_ref(lang, node, "target")
            .or_else(|| child_at(ast, node, 0))
            .ok_or_else(|| LangError::engine("handle: missing target child".to_string()))?;
        let impl_idx = role_child_ref(lang, node, "impl")
            .or_else(|| child_at(ast, node, 1))
            .ok_or_else(|| LangError::engine("handle: missing impl child".to_string()))?;
        let t_body = self.synth_node(lang, ast, target_idx, ctx)?;
        let t_impl = self.synth_node(lang, ast, impl_idx, ctx)?;
        // ADR-0158 `acc` clause: `handle f with h acc s0` — the optional
        // trailing child is the accumulator seed; synthesize it (for
        // validation) — the coverage rules are unchanged by `acc`.
        if let Some(acc_idx) = role_child_ref(lang, node, "acc") {
            let _t_acc = self.synth_node(lang, ast, acc_idx, ctx)?;
        }
        self.handle_result_type(&t_body, &t_impl, ctx)
    }

    /// The `IO(S')` result of `handle` given the body's and handler's
    /// types — see [`Self::synth_handle_judgment`] for the coverage
    /// rules.
    fn handle_result_type(
        &self,
        t_body: &Value,
        t_impl: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let tb = self.zonk(t_body, ctx);
        if io_headed(&tb) {
            return self.handle_sig_check(&tb, t_impl, ctx);
        }
        if tag_of(&tb) == Some("type.function") {
            let body = self.zonk(&tb.get("body").cloned().unwrap_or(Value::Null), ctx);
            if io_headed(&body) {
                let residual = self.handle_sig_check(&body, t_impl, ctx)?;
                let mut out = tb.as_object().cloned().unwrap_or_default();
                out.insert("body".to_string(), residual);
                return Ok(Value::Object(out));
            }
        }
        Ok(tb)
    }

    /// Compute `IO(S')` for an `IO(S)`-headed value and a handler type —
    /// the coverage check itself.
    fn handle_sig_check(
        &self,
        io_val: &Value,
        t_impl: &Value,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        // The sig record — `{"__former__": {…, "args": [sig]}}` (the
        // `dep_synth_type`/`quote` spelling) or the runtime-style
        // `{"indices": [sig]}` mint (`type.IO` in lib-type).
        let sig = io_val
            .get("__former__")
            .and_then(|f| f.get("args"))
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .or_else(|| {
                io_val
                    .get("indices")
                    .and_then(|a| a.as_array())
                    .and_then(|a| a.first())
            })
            .cloned()
            .unwrap_or(Value::Null);
        let sig = self.zonk(&sig, ctx);
        let Some(sig_obj) = sig.as_object() else {
            // `IO(⊤)` (arity-0, any sig) or a symbolic sig (a row
            // meta/var `IO(s)` — coverage is undecidable at this use;
            // solving happens at the arrow's call site).
            return Ok(io_val.clone());
        };
        // The handler's declared op set — `type.record{fields}`.
        let ti = self.zonk(t_impl, ctx);
        let fields = if tag_of(&ti) == Some("type.record") {
            // ADR-0158: `return` is the handler's unit clause and
            // `__*__` keys are reserved slots (`__acc__`, …) — neither
            // is an op impl, so both drop out of coverage/spurious
            // accounting.
            ti.get("fields")
                .and_then(Value::as_object)
                .cloned()
                .map(|f| {
                    f.into_iter()
                        .filter(|(k, _)| k != "return" && !k.starts_with("__"))
                        .collect::<Map<String, Value>>()
                })
        } else {
            None
        };
        let Some(fields) = fields else {
            // `type.Obj` / meta / unknown handler — its op set is
            // unknowable statically; the runtime still dispatches by
            // name. Passthrough (honest bound).
            return Ok(io_val.clone());
        };
        // Spurious impls — `h` declares ops the body can never perform.
        let spurious: Vec<&String> = fields
            .keys()
            .filter(|k| !sig_obj.contains_key(*k))
            .collect();
        if !spurious.is_empty() {
            return Err(LangError::engine(format!(
                "typecheck: handle: handler declares impls for ops the body's sig never names: {}",
                spurious
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let covered: Vec<&String> = sig_obj.keys().filter(|k| fields.contains_key(*k)).collect();
        if !sig_obj.is_empty() && covered.is_empty() {
            return Err(LangError::engine(format!(
                "typecheck: handle: handler covers none of the body's declared ops ({}) — the handle is a guaranteed no-op",
                sig_obj.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        let mut out_sig = Map::new();
        for (k, req) in sig_obj.iter() {
            match fields.get(k) {
                Some(it) => {
                    // Covered — interpret in place: the entry's declared
                    // type becomes the impl's declared return type.
                    let it = self.zonk(it, ctx);
                    let ret = if tag_of(&it) == Some("type.function") {
                        let dom = it.get("param").cloned().unwrap_or(Value::Null);
                        // `dispatch_op` feeds the sig entry's *value* to
                        // the impl — its domain must accept the declared
                        // sig entry.
                        if tag_of(req) == Some("type.unknown") {
                            // `IO<{ puts }>`-style unstated requests: skip
                            // the domain check, and an unsolved return
                            // meta defaults to the request (a shared
                            // `?m` inside the sig object can't solve in
                            // `conv` — Const compares structurally).
                            let body = it.get("body").cloned().unwrap_or(Value::Null);
                            if let Some(id) = extract_meta_id(&body) {
                                ctx.solve_meta(id, req.clone());
                            }
                            self.zonk(&body, ctx)
                        } else {
                            self.check_subtype(req, &dom, ctx).map_err(|_| {
                                LangError::engine(format!(
                                    "typecheck: handle: impl '{k}' domain does not accept the op's declared signature"
                                ))
                            })?;
                            it.get("body").cloned().unwrap_or(Value::Null)
                        }
                    } else {
                        // A non-arrow impl produces its declared type.
                        it
                    };
                    out_sig.insert(k.clone(), ret);
                }
                None => {
                    // Uncovered — the op request stays (still `IO` data
                    // for an outer handler).
                    out_sig.insert(k.clone(), req.clone());
                }
            }
        }
        // Re-wrap in the body's own spelling.
        let mut out = io_val.as_object().cloned().unwrap_or_default();
        if let Some(f) = out.get_mut("__former__").and_then(Value::as_object_mut) {
            if let Some(args) = f.get_mut("args").and_then(Value::as_array_mut)
                && let Some(slot) = args.first_mut()
            {
                *slot = Value::Object(out_sig);
            }
        } else if let Some(indices) = out.get_mut("indices").and_then(Value::as_array_mut)
            && let Some(slot) = indices.first_mut()
        {
            *slot = Value::Object(out_sig);
        }
        Ok(Value::Object(out))
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
    /// `name<b, …>` on an `assign`/`type-annot` subject — implicit index
    /// binders (phase-2 WS-C) — or `name<N, …>` — generic dimension
    /// parameters (ADR-0106 §12: `f<N> := body` means `f := N ~> body`;
    /// the `dimension` node's `args` scope over the bound value, same
    /// class of binding). Returns `(name, [(binder, sort_node?)])`
    /// where each arg is a `var` (unsorted) or
    /// `type-annot(var, sort)`; `None` for any other subject shape.
    fn implicit_binder_head(
        &self,
        lang: &Language,
        ast: &Ast,
        node: &AstNode,
    ) -> Option<(String, Vec<(String, Option<usize>)>)> {
        if node.kind != "index_app" && node.kind != "dimension" {
            return None;
        }
        let kids = children_of(node)?;
        let callee = ast.get(AstRef::new(kids.first()?.as_u64()? as usize))?;
        let name = node_name(callee)?;
        let mut binders = Vec::new();
        for cv in kids.iter().skip(1) {
            let b = ast.get(AstRef::new(cv.as_u64()? as usize))?;
            if b.kind == "type-annot" || role_child_ref(lang, b, "annot").is_some() {
                let bk = children_of(b)?;
                let vn = node_name(ast.get(AstRef::new(bk.first()?.as_u64()? as usize))?)?;
                binders.push((vn, bk.get(1).and_then(Value::as_u64).map(|i| i as usize)));
            } else {
                binders.push((node_name(b)?, None));
            }
        }
        Some((name, binders))
    }

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
        let mut alias_params: Option<Vec<String>> = None;
        let mut implicit_binders: Vec<(String, Option<usize>)> = Vec::new();
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
            let t_node = ast
                .get(AstRef::new(tidx))
                .ok_or_else(|| LangError::engine("synth_assign: lhs out of bounds".to_string()))?;
            let n =
                if t_node.kind == "type-annot" || role_child_ref(lang, t_node, "annot").is_some() {
                    let subj = lang
                        .child_role_index(&t_node.kind, "subject")
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                    children_of(t_node)
                        .and_then(|k| k.get(subj))
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(|s| {
                            self.implicit_binder_head(lang, ast, s)
                                .map(|(n, bs)| {
                                    implicit_binders = bs;
                                    n
                                })
                                .or_else(|| node_name(s))
                        })
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
            let lhs_node = ast
                .get(AstRef::new(lhs_idx))
                .ok_or_else(|| LangError::engine("synth_assign: lhs out of bounds".to_string()))?;
            if let Some((aname, aparams)) = self.type_alias_head(ast, lhs_node) {
                // `type.Name := T` / `type.Name(p…) := T` — a transparent
                // type alias (ADR-0154 §3): bind a `type.alias` expansion
                // record (params + the body's ast index); applications
                // substitute at use sites — no former mint.
                alias_params = Some(aparams);
                (aname, AstRef::new(vidx), None)
            } else if role_child_ref(lang, lhs_node, "annot").is_some()
                || lhs_node.kind == "type-annot"
            {
                let ta = children_of(lhs_node).ok_or_else(|| {
                    LangError::engine("synth_assign: type-annot children".to_string())
                })?;
                if ta.len() < 2 {
                    return Err(LangError::engine(
                        "synth_assign: type-annot needs 2 children".to_string(),
                    ));
                }
                let iv = ta[0]
                    .as_u64()
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(|s| {
                        self.implicit_binder_head(lang, ast, s)
                            .map(|(n, bs)| {
                                implicit_binders = bs;
                                n
                            })
                            .or_else(|| node_name(s))
                    })
                    .ok_or_else(|| LangError::engine("synth_assign: annot var".to_string()))?;
                let tidx = ta[1]
                    .as_u64()
                    .ok_or_else(|| LangError::engine("synth_assign: annot idx".to_string()))?
                    as usize;
                (iv.to_string(), AstRef::new(vidx), Some(AstRef::new(tidx)))
            } else if matches!(lhs_node.kind.as_str(), "dim_arrow" | "axis_flow" | "flow") {
                // `f : A ~[k]~> B` / `f : A >>>[k] B` — `:` binds
                // tighter than the arrow infix, so the annotation
                // parses as `arrow(type-annot(f, A), …)` with the
                // `type-annot` nested in the arrow's src operand. The
                // whole lhs is the annotation; the bound name lives in
                // the src `type-annot`'s subject.
                let src_node = child_at(ast, lhs_node, 0)
                    .and_then(|i| ast.get(i))
                    .ok_or_else(|| {
                        LangError::engine("synth_assign: arrow lhs has no src operand".to_string())
                    })?;
                let sinner = if src_node.kind == "paren" {
                    child_at(ast, src_node, 0)
                        .and_then(|i| ast.get(i))
                        .unwrap_or(src_node)
                } else {
                    src_node
                };
                let subj = lang
                    .child_role_index(&sinner.kind, "subject")
                    .ok()
                    .flatten()
                    .unwrap_or(0);
                let iv = children_of(sinner)
                    .and_then(|k| k.get(subj))
                    .and_then(Value::as_u64)
                    .and_then(|i| ast.get(AstRef::new(i as usize)))
                    .and_then(|s| {
                        self.implicit_binder_head(lang, ast, s)
                            .map(|(n, bs)| {
                                implicit_binders = bs;
                                n
                            })
                            .or_else(|| node_name(s))
                    })
                    .ok_or_else(|| LangError::engine("synth_assign: annot var".to_string()))?;
                (iv, AstRef::new(vidx), Some(AstRef::new(lhs_idx)))
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
                if pinner.kind == "type-annot" || role_child_ref(lang, pinner, "annot").is_some() {
                    let subj = lang
                        .child_role_index(&pinner.kind, "subject")
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                    let iv = children_of(pinner)
                        .and_then(|k| k.get(subj))
                        .and_then(Value::as_u64)
                        .and_then(|i| ast.get(AstRef::new(i as usize)))
                        .and_then(|s| {
                            self.implicit_binder_head(lang, ast, s)
                                .map(|(n, bs)| {
                                    implicit_binders = bs;
                                    n
                                })
                                .or_else(|| node_name(s))
                        })
                        .ok_or_else(|| LangError::engine("synth_assign: annot var".to_string()))?;
                    (iv, AstRef::new(vidx), Some(AstRef::new(lhs_idx)))
                } else {
                    return Err(LangError::engine(
                        "synth_assign: function lhs params is not a type-annot".to_string(),
                    ));
                }
            } else {
                // A bare `name<b, …>`/`name<N>` lhs (`dedupe<N> := v`)
                // declares generic binders — extract the head the same
                // way the annotated subject arms do, so the binders
                // scope over the initializer below.
                let n = self
                    .implicit_binder_head(lang, ast, lhs_node)
                    .map(|(n, bs)| {
                        implicit_binders = bs;
                        n
                    })
                    .or_else(|| node_name(lhs_node))
                    .ok_or_else(|| LangError::engine("synth_assign: var name".to_string()))?;
                (n, AstRef::new(vidx), None)
            }
        };
        let bound_type = if let Some(params) = alias_params {
            // The alias binds its expansion record, not a synthesized
            // type — `Name`/`Name(args)` in type position substitute
            // through it (the `type.alias` arms of `dep_synth_var_type`/
            // `dep_synth_call_type`).
            let mut m = Map::new();
            m.insert(
                "params".to_string(),
                Value::Array(params.into_iter().map(Value::from).collect()),
            );
            m.insert("body".to_string(), Value::from(value_ref.index() as u64));
            m.insert("name".to_string(), Value::from(name.clone()));
            make_tag(ctx, "type.alias", m)
        } else if let Some(ann_ref) = ann_ref {
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
            // Implicit index binders scope (phase-2 WS-C `name<b, …>`):
            // each binder is bound rigidly — `var{b}` in `results` and Γ —
            // so occurrences inside the annotation elaborate as the rigid
            // index variable they name (shadowing any outer `b`).
            let binder_mark = ctx.mark();
            let binder_results = ctx.results.clone();
            for (bn, _) in &implicit_binders {
                ctx.bind_result(bn.clone(), var_term(bn));
                ctx.extend(bn.clone(), var_term(bn));
            }
            let ann = match self
                .dep_synth_type(lang, ast, ann_ref.index(), ctx)
                .or_else(|_| crate::interpret_type_expr(lang, ast, type_expr_node, &ctx.registry))
            {
                // A term-valued annotation (e.g. `call` on a results-bound
                // name — `inc(4)` lowers to `App{var inc, [4]}`) reduces
                // against the bound-value env before classification.
                Ok(a) if is_ann_term(&a) => self.eval_ann_term(lang, ast, &a, ctx, ANN_FUEL)?,
                // `type.unknown` is the *unruled-kind fallback*, not a
                // real annotation — an ordered value annotation
                // (`x : E := v`, ADR-0103) evaluates `E` as a term
                // first (`5 >>> inc` → `6`); keep the placeholder only
                // when the term path fails too.
                Ok(a) if tag_of(&a) == Some("type.unknown") => {
                    match self.eval_ann_node(lang, ast, ann_ref.index(), ctx) {
                        Ok(v) => v,
                        Err(_) => a,
                    }
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
            // Wrap the elaborated annotation in one implicit `type.pi`
            // per `name<b, …>` binder — the binder name rides in
            // `binder`, `implicit` marks it non-lambda-consuming, and the
            // sort is the declared one or a fresh meta.
            let ann = if implicit_binders.is_empty() {
                ann
            } else {
                let mut w = ann;
                for (bn, sort) in implicit_binders.iter().rev() {
                    let param = match sort {
                        Some(si) => self.dep_synth_type(lang, ast, *si, ctx)?,
                        None => meta_value(ctx.fresh_meta()),
                    };
                    // Erasure-class marker (phase-3 WS-A, ADR-0106 §11):
                    // a former-app domain (`Eq a`) is *evidence* — the
                    // binder is discharged by dictionary search and the
                    // dictionary is a runtime argument (kept implicit),
                    // unlike index binders which erase at exec. The
                    // classification is declared data on the pi so the
                    // exec side can read it — no ad hoc kind check.
                    let evidence = tag_of(&param) == Some("type.family_app");
                    let mut m = Map::new();
                    m.insert("param".to_string(), param);
                    m.insert("body".to_string(), w);
                    m.insert("implicit".to_string(), Value::Bool(true));
                    m.insert("binder".to_string(), Value::String(bn.clone()));
                    if evidence {
                        m.insert("evidence".to_string(), Value::Bool(true));
                    }
                    w = make_tag(ctx, "type.pi", m);
                }
                w
            };
            ctx.restore(binder_mark);
            ctx.results = binder_results;
            // An explicit `x : type.unknown = v` annotation opts the
            // binding into the dynamic lane. It typechecks (top accepts
            // everything), but the silence is worth a warning — unlike
            // *inferred* `unknown` (host calls, `resolve` imports —
            // warning there would flag every dynamic value in the
            // program), the user wrote the annotation and should know
            // it erases the binding's static type.
            if is_unknown_ty(&ann) {
                ctx.diagnostics.push(TypecheckDiagnostic {
                    severity: "warning".to_string(),
                    message: format!(
                        "assign: '{name}' is annotated `type.unknown` — the dynamic top accepts any value and erases the binding's static type"
                    ),
                    span: node.span,
                });
            }
            // Self-recursion: the bound name is visible inside its own
            // initializer at the annotation's type — the bootstrap
            // scope pass's "a block binding's name is visible to
            // closures in its own initializer" (the runtime's
            // update-or-define ScopeStack resolves the wire at call
            // time). `vec_map{n, a, b} : T := … vec_map{…}(…) …`
            // resolves the recursive call against the implicit `Pi`s.
            ctx.extend(name.clone(), ann.clone());
            if matches!(
                ann.get("__tag__").and_then(Value::as_str),
                Some(
                    "type.pi"
                        | "type.family"
                        | "type.sigma"
                        | "Path"
                        | "type.forall"
                        | "type.family_app"
                        | "type.array"
                )
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
                    Ok(vv) => self.apply_ann_val(lang, ast, &ann, &[vv], ctx, ANN_FUEL)?,
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
            // A `name<b,…>`/`name<N>` head on a bare assign scopes its
            // generic binders over the initializer (ADR-0106 §12 —
            // `f<N> := body` means `f := N ~> body`; the bootstrap
            // scope pass's `dim_param_names`/`implicit_binder_names`
            // bind the same names over the value only). Bound rigidly —
            // `var{N}` in results and Γ — so the value's occurrences
            // (`… ~[N]~> …`) resolve to the bound dimension variable.
            let binder_mark = ctx.mark();
            let binder_results = ctx.results.clone();
            for (bn, _) in &implicit_binders {
                ctx.bind_result(bn.clone(), var_term(bn));
                ctx.extend(bn.clone(), var_term(bn));
            }
            let ty = self.synth_node(lang, ast, value_ref, ctx);
            ctx.restore(binder_mark);
            ctx.results = binder_results;
            ty?
        };
        ctx.extend(name.clone(), bound_type);
        // ADR-0103: record the bound *term* of the value — the bound-value
        // env ordered annotations evaluate against (the model's
        // `tc_bind_result`). `var{name}` inside a call annotation resolves
        // to this term (e.g. `inc ↦ Lam{params, body}`).
        if let Some(vn) = ast.get(value_ref)
            && let Some(t) = self.node_to_ann_term(lang, ast, vn)
        {
            ctx.bind_result(name.clone(), t);
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

// ---------------------------------------------------------------------------
// Instance-resolution helpers (phase-3 WS-A)
// ---------------------------------------------------------------------------

/// Split a constraint goal into `(head, index-args)` — a
/// `type.family_app` or a bare `type.family` (a 0-index constraint).
fn instance_goal_parts(v: &Value) -> Option<(String, Vec<Value>)> {
    match tag_of(v) {
        Some("type.family_app") => {
            let name = v.get("name").and_then(Value::as_str)?.to_string();
            let args = v
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            Some((name, args))
        }
        Some("type.family") => {
            let name = v.get("name").and_then(Value::as_str)?.to_string();
            if name.is_empty() {
                return None;
            }
            Some((name, Vec::new()))
        }
        _ => None,
    }
}

/// Whether `v` contains an *unsolved* tagged meta leaf anywhere.
fn has_unsolved_meta(v: &Value, ctx: &TypecheckContext) -> bool {
    if let Some(id) = extract_meta_id(v) {
        return ctx
            .metas
            .get(id as usize)
            .and_then(|o| o.as_ref())
            .is_none();
    }
    match v {
        Value::Object(m) => m.values().any(|x| has_unsolved_meta(x, ctx)),
        Value::Array(xs) => xs.iter().any(|x| has_unsolved_meta(x, ctx)),
        _ => false,
    }
}

/// Rewrite each unsolved tagged-meta leaf in `v` to a per-instance
/// rigid `var` (`$instvar{inst}_{pos}`) — a registered instance shape
/// must be *instantiated* per lookup, so a lookup can never solve the
/// declaration's shared meta cell. Solved metas keep their solution.
fn freeze_metas(v: &Value, inst: usize, pos: &mut usize) -> Value {
    if extract_meta_id(v).is_some() {
        *pos += 1;
        return var_term(&format!("$instvar{inst}_{}", *pos - 1));
    }
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), freeze_metas(x, inst, pos)))
                .collect(),
        ),
        Value::Array(xs) => Value::Array(xs.iter().map(|x| freeze_metas(x, inst, pos)).collect()),
        _ => v.clone(),
    }
}

/// Guard the *goal* side of an instance lookup: wrap each rigid `var`
/// leaf as `App{$rigid, [v]}` so `unify_index` treats it as an
/// unforgeable atom — a rigid goal index (`Eq a` inside a polymorphic
/// body) must not *solve* to a concrete candidate index (`Eq Nat`
/// would bind `a := Nat` and corrupt the enclosing binder's scope).
/// A generic candidate's own `var` holes still bind through the
/// `var`-arm, so `instance Eq (List a)` still matches `Eq (List n)`.
fn guard_rigid_vars(args: &[Value]) -> Vec<Value> {
    fn go(v: &Value) -> Value {
        if tag_of(v) == Some("var") {
            return app_term(var_term("$rigid"), vec![v.clone()]);
        }
        match v {
            Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), go(x))).collect()),
            Value::Array(xs) => Value::Array(xs.iter().map(go).collect()),
            _ => v.clone(),
        }
    }
    args.iter().map(go).collect()
}

/// Module-scope `Sig_m` population (phase-3 WS-A): an imported binding
/// whose type is a `type.family_app` registers as an instance
/// candidate under its head — the dict term is `var{name}` since the
/// import bound the dictionary under `name` in Γ. Unchecked
/// scope-level registration (no coherence — first-match wins); the
/// `instance` declaration's coherence rule governs the decl path.
fn register_imported_instance(ctx: &mut TypecheckContext, name: &str, ty: &Value) {
    let Some((head, args)) = instance_goal_parts(ty) else {
        return;
    };
    ctx.instances.entry(head).or_default().push(InstanceEntry {
        name: name.to_string(),
        args,
    });
}

/// Collect the `var` names in an AST subtree that bind to nothing —
/// the instance constraint's implicit index parameters
/// (`instance Eq (List a)` generalizes `a`). Γ/results/globals-bound
/// and builtin type names are excluded (family heads, sorts).
fn collect_unbound_index_vars(
    ast: &Ast,
    node_ref: AstRef,
    ctx: &TypecheckContext,
    out: &mut Vec<String>,
) {
    let Some(node) = ast.get(node_ref) else {
        return;
    };
    if node.kind == "var" {
        if let Some(name) = node_name(node)
            && !name.is_empty()
            && !ctx.results.contains_key(&name)
            && ctx.lookup(&name).is_none()
            && !is_builtin_type_name(&name)
            && !out.contains(&name)
        {
            out.push(name);
        }
        return;
    }
    if let Some(kids) = children_of(node) {
        for k in kids.clone() {
            if let Some(i) = k.as_u64() {
                collect_unbound_index_vars(ast, AstRef::new(i as usize), ctx, out);
            }
        }
    }
}

impl NbeTypecheckEngine {
    /// Lower an AST node to a closed annotation *term* (the model's
    /// `node_term_`). Returns `None` for kinds outside the evaluable
    /// fragment — the annotation then keeps the type-interpretation
    /// error path.
    fn node_to_ann_term(&self, lang: &Language, ast: &Ast, node: &AstNode) -> Option<Value> {
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
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
                // `dimension`/`index_app` share `call`'s `[callee, args…]`
                // layout — `inc<4>` in an ordered-value annotation evaluates
                // exactly like `inc(4)`.
                "call" | "index_app" | "dimension" => {
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
                    let params_idx =
                        role_child_ref(lang, node, "params").or_else(|| child_at(ast, node, 0))?;
                    let body_idx =
                        role_child_ref(lang, node, "body").or_else(|| child_at(ast, node, 1))?;
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
                    let cond = role_child_ref(lang, node, "condition")
                        .or_else(|| child_at(ast, node, 0))?;
                    let then =
                        role_child_ref(lang, node, "then").or_else(|| child_at(ast, node, 1))?;
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
                        m.insert("else".to_string(), self.node_to_ann_term(lang, ast, e)?);
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
        })
    }

    /// `node_to_ann_term` + `eval_ann_term` (the model's `eval_node_`).
    fn eval_ann_node(
        &self,
        lang: &Language,
        ast: &Ast,
        idx: usize,
        ctx: &mut TypecheckContext,
    ) -> Result<Value, LangError> {
        let node = ast
            .get(AstRef::new(idx))
            .ok_or_else(|| LangError::engine(format!("ann eval: node {idx} out of bounds")))?;
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
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
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
                        bin_term(term.get("op").and_then(Value::as_str).unwrap_or(""), l, r)
                    } else {
                        let o = self.eval_ann_term(
                            lang,
                            ast,
                            &term.get("operand").cloned().unwrap_or(Value::Null),
                            ctx,
                            fuel - 1,
                        )?;
                        un_term(term.get("op").and_then(Value::as_str).unwrap_or(""), o)
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
                            "binary.eq" if ln.is_some() && rn.is_some() => Some(Value::Bool(false)),
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
        })
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
        // Growable-stack seam (WS-H, see `crate::STACK_RED_ZONE`).
        stacker::maybe_grow(crate::STACK_RED_ZONE, crate::STACK_SEGMENT, || {
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
        })
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
                    let ty = sig.unwrap_or_else(|| self.type_unknown(ctx));
                    insert_module_sig_field(&mut fields, &name, ty, ctx);
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
    ///
    /// Phase-3 WS-A (instances ride `Sig_m`): an imported binding whose
    /// type is a `type.family_app` is a module-scope instance
    /// candidate — it registers in the instance table under its head so
    /// dictionary search sees dictionaries other modules exported.
    /// Registration is unchecked (scope-level candidates, not
    /// declarations): the `instance` decl's coherence rule governs the
    /// declaration path; first-match wins at search.
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
            ctx.extend(a.to_string(), bound.clone());
            register_imported_instance(ctx, a, &bound);
            return Ok(unit(ctx));
        }

        match names {
            // `from "<spec>" import *` — splice fields into Γ.
            Some(Value::String(s)) if s == "*" => {
                if let Some(fields) = sig.get("fields").and_then(Value::as_object) {
                    for (n, t) in fields {
                        ctx.extend(n.clone(), t.clone());
                        register_imported_instance(ctx, n, t);
                    }
                }
                // Bare module-name specs (`str`) also stay usable qualified.
                if is_bare_module_name(spec) {
                    ctx.extend(spec.to_string(), sig.clone());
                }
            }
            Some(Value::String(s)) => {
                let ty = self.import_field_type(&sig, spec, s, ctx);
                ctx.extend(s.clone(), ty.clone());
                register_imported_instance(ctx, s, &ty);
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
                    ctx.extend(bound.clone(), ty.clone());
                    register_imported_instance(ctx, &bound, &ty);
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
    let t_idx = lang.child_role_index(&node.kind, "target").ok().flatten()?;
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

// ---------------------------------------------------------------------------
// Ring-normalization + solver tests (phase-2 WS-D)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod index_solver_tests {
    use super::*;

    fn ctx() -> TypecheckContext {
        TypecheckContext::new("handl.canonical", HashMap::new())
    }

    fn var(n: &str) -> Value {
        var_term(n)
    }
    fn add(l: Value, r: Value) -> Value {
        bin_term("binary.add", l, r)
    }
    fn sub(l: Value, r: Value) -> Value {
        bin_term("binary.sub", l, r)
    }
    fn mul(l: Value, r: Value) -> Value {
        bin_term("binary.mul", l, r)
    }
    fn num(x: i64) -> Value {
        Value::from(x)
    }

    #[test]
    fn ring_unit_identities() {
        let c = ctx();
        // n+0 → n, 0+n → n, n*1 → n, 1*n → n, n*0 → 0
        assert_eq!(eval_index(&add(var("n"), num(0)), &c), var("n"));
        assert_eq!(eval_index(&add(num(0), var("n")), &c), var("n"));
        assert_eq!(eval_index(&mul(var("n"), num(1)), &c), var("n"));
        assert_eq!(eval_index(&mul(num(1), var("n")), &c), var("n"));
        assert_eq!(eval_index(&mul(var("n"), num(0)), &c), num(0));
        // nested: (n - 1) + 1 → n
        assert_eq!(
            eval_index(&add(sub(var("n"), num(1)), num(1)), &c),
            var("n")
        );
    }

    #[test]
    fn ring_sort_collect_and_distribute() {
        let c = ctx();
        // Commutativity: n+m ≡ m+n — same normal form.
        assert_eq!(
            eval_index(&add(var("n"), var("m")), &c),
            eval_index(&add(var("m"), var("n")), &c)
        );
        // Collect like terms: a+b+a → a*2+b.
        assert_eq!(
            eval_index(&add(add(var("a"), var("b")), var("a")), &c),
            add(mul(var("a"), Value::from(2.0)), var("b"))
        );
        // Distribution: n*(m+k) → n*m + n*k.
        assert_eq!(
            eval_index(&mul(var("n"), add(var("m"), var("k"))), &c),
            eval_index(&add(mul(var("n"), var("m")), mul(var("n"), var("k"))), &c)
        );
        // Constant pooling: (n + 2) + 3 → n + 5.
        assert_eq!(
            eval_index(&add(add(var("n"), num(2)), num(3)), &c),
            add(var("n"), Value::from(5.0))
        );
    }

    #[test]
    fn conv_vec_commutes_and_doubles() {
        let mut c = ctx();
        // Vec(n+m) ≡ Vec(m+n) — by normalization before meta-solving.
        assert_eq!(
            unify_index(&add(var("n"), var("m")), &add(var("m"), var("n")), &mut c),
            (true, false)
        );
        // Vec(2*n) ≡ Vec(n*2).
        assert_eq!(
            unify_index(&mul(num(2), var("n")), &mul(var("n"), num(2)), &mut c),
            (true, false)
        );
        // Vec(n+1+0) ≡ Vec(n+1).
        assert_eq!(
            unify_index(
                &add(add(var("n"), num(1)), num(0)),
                &add(var("n"), num(1)),
                &mut c
            ),
            (true, false)
        );
        // append-shaped: the `Vec(n,m) ~> Vec(n+m)` codomain commutes —
        // Vec(rows+cols) ≡ Vec(cols+rows) under two rigid binders.
        assert_eq!(
            unify_index(
                &add(var("rows"), var("cols")),
                &add(var("cols"), var("rows")),
                &mut c
            ),
            (true, false)
        );
    }

    #[test]
    fn ring_under_rigid_binders() {
        let mut c = ctx();
        // Rigid `var` indices bound under a Pi — `n` is scope-bound,
        // not a meta: `n + m + 0 ≡ m + n` holds definitionally.
        c.bind_result("n".to_string(), var("n"));
        c.bind_result("m".to_string(), var("m"));
        assert_eq!(
            unify_index(
                &add(add(var("n"), var("m")), num(0)),
                &add(var("m"), var("n")),
                &mut c
            ),
            (true, false)
        );
        // `var n` vs `n + 1`: the var arm reports ok without recording a
        // cyclic `n ↦ n+1` binding (the occurs guard fires) — same as
        // before normalization.
        let (ok, decided) = unify_index(&var("n"), &add(var("n"), num(1)), &mut c);
        assert!(ok);
        assert!(!decided);
    }

    #[test]
    fn solver_fragment_unchanged() {
        let mut c = ctx();
        // Linear: ?k + 3 ≡ 7 solves ?k := 4.
        c.metas.push(None);
        let k = meta_value(0);
        assert_eq!(unify_index(&add(k, num(3)), &num(7), &mut c), (true, false));
        assert_eq!(c.metas[0], Some(num(4)));
        // Exact division: ?k * 3 ≡ 9 → ?k := 3; non-divisible is decided.
        c.metas.push(None);
        let k2 = meta_value(1);
        assert_eq!(
            unify_index(&mul(k2, num(3)), &num(9), &mut c),
            (true, false)
        );
        c.metas.push(None);
        let k3 = meta_value(2);
        assert_eq!(
            unify_index(&mul(k3, num(3)), &num(10), &mut c),
            (false, true)
        );
        // Integer root: ?k * ?k ≡ 25 → ?k := 5.
        c.metas.push(None);
        let k4 = meta_value(3);
        assert_eq!(
            unify_index(&mul(k4.clone(), k4), &num(25), &mut c),
            (true, false)
        );
        // Distinct numerals: decided failure.
        assert_eq!(unify_index(&num(2), &num(5), &mut c), (false, true));
        // Two symbolic factors stay Unknown — never silently accepted.
        c.metas.push(None);
        c.metas.push(None);
        let (ok, decided) = unify_index(&mul(meta_value(4), meta_value(5)), &num(6), &mut c);
        assert!(!ok && !decided);
        // Solver verdicts directly.
        assert_eq!(
            INDEX_SOLVER.prove(&var("x"), &var("y"), &mut c),
            IndexVerdict::Unknown
        );
        assert_eq!(
            INDEX_SOLVER.prove(&add(var("n"), var("m")), &add(var("m"), var("n")), &mut c),
            IndexVerdict::Proven
        );
    }

    // ------------------------------------------------------------------
    // `data`-declared ctor index projections (phase-3 WS-D)
    // ------------------------------------------------------------------

    fn succ_proj(c: &mut TypecheckContext) {
        c.ctor_proj.insert(
            "Succ".to_string(),
            CtorProj {
                arity: 1,
                template: add(var_term(&format!("{CTOR_PROJ_HOLE}0")), num(1)),
            },
        );
        c.ctor_proj.insert(
            "Zero".to_string(),
            CtorProj {
                arity: 0,
                template: num(0),
            },
        );
    }

    fn succ(x: Value) -> Value {
        app_term(var("Succ"), vec![x])
    }

    #[test]
    fn ctor_projection_normalizes_succ_and_zero() {
        let mut c = ctx();
        succ_proj(&mut c);
        // eval: App{Succ, n} ↦ n + 1 (ring-canonical — constants write
        // back as f64), nested ↦ n + 2, var{Zero} ↦ 0,
        // App{Succ, 0+0} ↦ 1 (args evaluate first).
        assert_eq!(
            eval_index(&succ(var("n")), &c),
            add(var("n"), Value::from(1.0))
        );
        assert_eq!(
            eval_index(&succ(succ(var("n"))), &c),
            add(var("n"), Value::from(2.0))
        );
        assert_eq!(eval_index(&var("Zero"), &c), num(0));
        assert_eq!(eval_index(&succ(add(num(0), num(0))), &c), Value::from(1.0));
        // unify: `Vec(Succ n) ≡ Vec(n+1)` both directions, `≡ Vec(n+2)`
        // for the nested form.
        assert_eq!(
            unify_index(&succ(var("n")), &add(var("n"), num(1)), &mut c),
            (true, false)
        );
        assert_eq!(
            unify_index(&add(var("n"), num(1)), &succ(var("n")), &mut c),
            (true, false)
        );
        assert_eq!(
            unify_index(&succ(succ(var("n"))), &add(var("n"), num(2)), &mut c),
            (true, false)
        );
        // `Succ` inside a larger arithmetic term normalizes through.
        assert_eq!(
            unify_index(
                &add(succ(var("n")), var("m")),
                &add(add(var("n"), num(1)), var("m")),
                &mut c
            ),
            (true, false)
        );
    }

    #[test]
    fn ctor_projection_solver_and_decided_contract() {
        let mut c = ctx();
        succ_proj(&mut c);
        // Linear solving reaches through the projection:
        // `Succ ?k ≡ 2` solves `?k := 1`.
        c.metas.push(None);
        assert_eq!(
            unify_index(&succ(meta_value(0)), &num(2), &mut c),
            (true, false)
        );
        assert_eq!(c.metas[0], Some(num(1)));
        // Decided mismatch survives: `Succ Zero ≡ 0` is `1 ≡ 0` —
        // distinct numerals, a decided failure.
        let (ok, decided) = unify_index(&succ(var("Zero")), &num(0), &mut c);
        assert!(!ok && decided);
        // `Succ n ≡ n` — `n+1` vs `var n`: the var arm's occurs guard
        // keeps `n ↦ n+1` from binding (same as `var n` vs `n+1`
        // today), so this is never a decided failure.
        let (_ok, decided) = unify_index(&succ(var("n")), &var("n"), &mut c);
        assert!(!decided);
        // Unregistered heads stay honest atoms — `F n ≢ n + 1` is
        // Unknown, not accepted.
        let (ok, decided) = unify_index(
            &app_term(var("F"), vec![var("n")]),
            &add(var("n"), num(1)),
            &mut c,
        );
        assert!(!ok && !decided);
        // …but still ctor-injective against itself.
        assert_eq!(
            unify_index(
                &app_term(var("F"), vec![var("n")]),
                &app_term(var("F"), vec![var("n")]),
                &mut c
            ),
            (true, false)
        );
        // Arity mismatch on a registered head stays an atom too
        // (`Succ` applied to two args does not read as arithmetic).
        let over = app_term(var("Succ"), vec![var("n"), var("m")]);
        assert_eq!(eval_index(&over, &c), over);
    }
}
