//! Judgment ops over the `Val` domain — the [`OpHandler`] the checker
//! installs into `hnl_engine::nbe::eval` (phase-1 spec WS-B;
//! ADR-0154 §7: `types.yaml` demotes to a `node_kind → op name`
//! dispatch table and the judgment ops are the spec).
//!
//! Two families:
//!
//! - **Constructors** build domain values: `nominal`/`former`/`universe`/
//!   `pi`/`hom`/`meta`/`literal`/`lift`/`quote`.
//! - **Judges** decide or synthesize: `infer`/`check`/`unify`/`subtype`/
//!   `bind`/`lookup`.
//!
//! Pure index arithmetic (`add`/`sub`/`mul`/`div` over `Const`
//! naturals) lives here too: an index like `1 + 1` normalizes by
//! *evaluation*, so `Vec(1+1) ≡ Vec(2)` is ordinary conversion — the
//! bespoke index-rewriting solver of the legacy checker has no reason
//! to exist (ADR-0152 §3).
//!
//! The handler owns a `RefCell<Checker>` — `OpHandler::op` takes
//! `&self` but judging mutates (meta solving, Γ binding).

use std::cell::RefCell;
use std::rc::Rc;

use handl_language::{Ast, AstRef, Language};
use handl_value::nbe::{ArrowVal, Lvl, Spine, Val};
use handl_value::{Sym, TaggedValue};
use hnl_engine::Term;
use hnl_engine::nbe::{EvalError, OpHandler};
use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::checker::{Checker, lift, type_sym};

/// Evaluation fuel for [`eval_arrow_body`] — the same guaranteed-
/// termination bound every other checker recursion keeps.
const APPLY_FUEL: u32 = 256;

/// The judgment-op surface: evaluates type-level `Term`s into the
/// `Val` domain, solving metas and binding Γ in the owned `Checker`.
pub struct JudgmentOps {
    /// The checking context (mutated through `RefCell` — `OpHandler`
    /// is `&self`).
    pub checker: RefCell<Checker>,
    /// The ambient language, when the host knows it — needed by
    /// `apply_arrow` body evaluation for role lookups
    /// (`child_role_index`) and `Ast::instantiate`. `None` keeps the
    /// positional fallback and makes `function` bodies stuck.
    pub lang: Option<Rc<Language>>,
}

impl JudgmentOps {
    /// A fresh handler over an empty checker.
    pub fn new() -> Self {
        JudgmentOps {
            checker: RefCell::new(Checker::new()),
            lang: None,
        }
    }

    /// A fresh handler over a pre-seeded checker (formers/nominals
    /// already declared).
    pub fn over(checker: Checker) -> Self {
        JudgmentOps {
            checker: RefCell::new(checker),
            lang: None,
        }
    }

    /// Install the ambient language (arrow-body role resolution and
    /// `Ast::instantiate` for nested `function` values).
    pub fn with_lang(mut self, lang: Rc<Language>) -> Self {
        self.lang = Some(lang);
        self
    }
}

impl Default for JudgmentOps {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Arrow-body evaluation (`OpHandler::apply_arrow`)
// ---------------------------------------------------------------------------
//
// A `Val::Arrow` is self-contained data: `body_def` is a `{"__ast__":
// <arena>}` tagged value, `binds` the residual environment. Applying a
// *saturated* arrow evaluates the body root under `binds ∪ {param ↦
// arg}` over the small tagged fragment the checker needs — open
// subterms are legal inside checking and produce `Former{stuck.*}`
// neutrals rather than errors (the `infer` side then reports
// `CannotInfer`/`UnknownFormer` honestly). Partial application stays
// the residual closure (`None` from `apply_arrow`).

/// A `Former{stuck.*}` neutral — the open-term stand-in. `kind` names
/// the shape that could not reduce (`stuck.var`, `stuck.call`,
/// `stuck.binary.add`, `stuck.function`, …).
fn stuck_val(kind: &str, args: Vec<Val>) -> Val {
    Val::Former {
        head: type_sym(&format!("stuck.{kind}")),
        args,
    }
}

fn const_val(v: impl Into<serde_json::Value>) -> Val {
    Val::Const(TaggedValue::from(v.into()))
}

/// Decode a `body_def` `{"__ast__": <arena>}` value into its `Ast`.
fn decode_body(def: &TaggedValue) -> Result<Ast, EvalError> {
    handl_language::as_ast(&handl_value::tagged_to_value(def))
        .filter(|a| !a.is_empty())
        .ok_or_else(|| EvalError::Op("apply_arrow: body_def is not a __ast__ arena".into()))
}

/// The arena index of the child at `role` (ADR-0081) — `pos` is the
/// positional fallback when the node kind declares no roles or no
/// language is installed.
fn body_child(
    _ast: &Ast,
    node: &handl_language::AstNode,
    lang: Option<&Language>,
    role: &str,
    pos: usize,
) -> Option<AstRef> {
    let idx = lang
        .and_then(|l| l.child_role_index(&node.kind, role).ok().flatten())
        .unwrap_or(pos);
    node.payload
        .get("children")?
        .as_array()?
        .get(idx)?
        .as_u64()
        .map(|i| AstRef::new(i as usize))
}

/// Evaluate a saturated arrow's body: `body_def` under
/// `binds ∪ extra`. `extra` is the fresh binding for the argument
/// (`apply_arrow` passes `[param ↦ arg]`; the checker's `infer`
/// saturates every param with a rigid binder).
pub(crate) fn eval_arrow_body(
    arrow: &ArrowVal,
    extra: &[(Rc<str>, Val)],
    lang: Option<&Language>,
    lvl: Lvl,
) -> Result<Val, EvalError> {
    let ast = decode_body(&arrow.body_def)?;
    let mut env: FxHashMap<Rc<str>, Val> = (*arrow.binds).clone();
    for (k, v) in extra {
        env.insert(k.clone(), v.clone());
    }
    let mut fuel = APPLY_FUEL;
    eval_body_node(&ast, AstRef::new(ast.len() - 1), &env, lang, &mut fuel, lvl)
}

/// Evaluate an arena node to a `Val` under `env` (var/call/binary/
/// unary/if/array/object/paren/function + the cell forms).
fn eval_body_node(
    ast: &Ast,
    r: AstRef,
    env: &FxHashMap<Rc<str>, Val>,
    lang: Option<&Language>,
    fuel: &mut u32,
    lvl: Lvl,
) -> Result<Val, EvalError> {
    if *fuel == 0 {
        return Ok(stuck_val("fuel", vec![]));
    }
    *fuel -= 1;
    let node = ast
        .get(r)
        .ok_or_else(|| EvalError::Op("apply_arrow: node out of bounds".into()))?;
    let ev = |i: AstRef,
              env: &FxHashMap<Rc<str>, Val>,
              ast: &Ast,
              lang: Option<&Language>,
              fuel: &mut u32|
     -> Result<Val, EvalError> { eval_body_node(ast, i, env, lang, fuel, lvl) };
    match node.kind.as_str() {
        "value" | "lit" => {
            // A `lit`/`value` payload *is* a boundary literal — the
            // `instantiate` convention every exec engine follows
            // (`TaggedValue::from(payload)`). A `{"value": …}` envelope
            // unwraps first; any other object payload decodes as data
            // and lifts into the domain, so desugar-minted `lit{__sym__}`
            // (`push_sym_lit`) and `reify_ast`-produced
            // `lit{__former__}`/`lit{__type__}`/`lit{__neutral__}` nodes
            // evaluate to their nominal/former/universe/neutral payload
            // rather than `Null`.
            let payload = node
                .payload
                .get("value")
                .cloned()
                .unwrap_or_else(|| node.payload.clone());
            Ok(lift(lvl, &TaggedValue::from(payload)))
        }
        "var" => {
            let name = node
                .payload
                .as_str()
                .or_else(|| node.payload.get("name").and_then(Value::as_str))
                .unwrap_or("");
            Ok(env
                .get(name)
                .cloned()
                .unwrap_or_else(|| stuck_val("var", vec![const_val(name)])))
        }
        "paren" => match body_child(ast, node, lang, "body", 0) {
            Some(i) => ev(i, env, ast, lang, fuel),
            None => Ok(stuck_val("paren", vec![])),
        },
        "if" => {
            let cond = body_child(ast, node, lang, "condition", 0);
            let then = body_child(ast, node, lang, "then", 1);
            let els = body_child(ast, node, lang, "else", 2);
            let (Some(c), Some(t)) = (cond, then) else {
                return Ok(stuck_val("if", vec![]));
            };
            match ev(c, env, ast, lang, fuel)? {
                Val::Const(TaggedValue::Bool(true)) => ev(t, env, ast, lang, fuel),
                Val::Const(TaggedValue::Bool(false)) => match els {
                    Some(e) => ev(e, env, ast, lang, fuel),
                    None => Ok(Val::Const(TaggedValue::Null)),
                },
                c => Ok(stuck_val("if", vec![c])),
            }
        }
        "array" => {
            let kids = node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            let mut els = Vec::with_capacity(kids.len());
            let mut all_const = true;
            for k in kids {
                let Some(i) = k.as_u64() else { continue };
                let v = ev(AstRef::new(i as usize), env, ast, lang, fuel)?;
                all_const &= matches!(v, Val::Const(_));
                els.push(v);
            }
            if all_const {
                Ok(Val::Const(TaggedValue::Arr(
                    els.into_iter()
                        .map(|v| match v {
                            Val::Const(t) => t,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                )))
            } else {
                Ok(stuck_val("array", els))
            }
        }
        "object" => {
            // Object literal — each `entry` child contributes a field.
            let kids = node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            let mut fields = rustc_hash::FxHashMap::default();
            let mut all_const = true;
            let mut stuck = Vec::new();
            for k in kids {
                let Some(i) = k.as_u64() else { continue };
                let Some(entry) = ast.get(AstRef::new(i as usize)) else {
                    continue;
                };
                if entry.kind == "entry" {
                    let key = body_child(ast, entry, lang, "key", 0)
                        .and_then(|i| ast.get(i))
                        .and_then(|n| n.payload.as_str().map(str::to_string))
                        .unwrap_or_default();
                    let v = match body_child(ast, entry, lang, "value", 1) {
                        Some(vi) => ev(vi, env, ast, lang, fuel)?,
                        None => stuck_val("entry", vec![]),
                    };
                    all_const &= matches!(v, Val::Const(_));
                    fields.insert(Rc::from(key.as_str()), v);
                } else {
                    let v = ev(AstRef::new(i as usize), env, ast, lang, fuel)?;
                    all_const &= matches!(v, Val::Const(_));
                    stuck.push(v);
                }
            }
            if all_const {
                let obj = fields
                    .into_iter()
                    .map(|(k, v)| match v {
                        Val::Const(t) => (k, t),
                        _ => unreachable!(),
                    })
                    .collect();
                Ok(Val::Const(TaggedValue::Obj(Rc::new(obj))))
            } else {
                stuck.extend(fields.into_values());
                Ok(stuck_val("object", stuck))
            }
        }
        "function" => {
            // Nested arrows stay arrow values: params from the shape's
            // `params` role, the body subtree re-arenas through
            // `Ast::instantiate` with the env's serializable bindings.
            let (Some(lang), Some(params_idx), Some(body_idx)) = (
                lang,
                body_child(ast, node, lang, "params", 0),
                body_child(ast, node, lang, "body", 1),
            ) else {
                return Ok(stuck_val("function", vec![]));
            };
            let Some(pnode) = ast.get(params_idx) else {
                return Ok(stuck_val("function", vec![]));
            };
            let env_vals: std::collections::HashMap<Rc<str>, Value> = env
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        handl_value::tagged_to_value(&handl_value::nbe::quote(
                            &handl_value::nbe::MetaStore::default(),
                            0,
                            v,
                        )),
                    )
                })
                .collect();
            let sub = ast
                .instantiate(body_idx, lang, &env_vals)
                .map_err(|e| EvalError::Op(format!("apply_arrow: instantiate: {e}")))?;
            let body_def = TaggedValue::from(handl_language::ast_value(&sub));
            // Destructured/array-pattern params cannot be spelled as a
            // positional `params` list — mint the boundary arrow value
            // instead, carrying `param_defs`/`array_pattern` exactly
            // like `resolve_closure_params` does on the exec side
            // (ADR-0081). `params` holds the flattened bound names.
            // `apply_body_value` interprets them via `try_pattern_val`.
            if lang.param_contains_patterns(ast, pnode) || lang.is_array_pattern(pnode) {
                // `params` carries the flattened bound names (exec's
                // `ClosureParams::names`): `param_names` for the plain
                // array-pattern path, `param_bound_names` when patterns
                // are present (rest bindings live inside `param_defs`).
                let (names, rest) = if lang.param_contains_patterns(ast, pnode) {
                    (lang.param_bound_names(ast, pnode).unwrap_or_default(), None)
                } else {
                    lang.param_names(ast, pnode).unwrap_or_default()
                };
                let params_sub = ast
                    .subtree(params_idx)
                    .map_err(|e| EvalError::Op(format!("apply_arrow: params subtree: {e}")))?;
                let arrow = TaggedValue::Arrow {
                    params: names.into(),
                    rest: rest.map(Rc::from),
                    array_pattern: lang.param_single_sequence(ast, params_idx)
                        || lang.is_array_pattern(pnode),
                    param_defs: if lang.param_contains_patterns(ast, pnode) {
                        Some(Rc::new(TaggedValue::from(handl_language::ast_value(
                            &params_sub,
                        ))))
                    } else {
                        None
                    },
                    body_def: Rc::new(body_def),
                    binds: Rc::new(
                        env.iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    handl_value::nbe::quote(
                                        &handl_value::nbe::MetaStore::default(),
                                        0,
                                        v,
                                    ),
                                )
                            })
                            .collect(),
                    ),
                    handlers: Rc::from(Vec::new()),
                    sig: None,
                };
                return Ok(Val::Const(arrow));
            }
            let Ok((names, _rest)) = lang.param_names(ast, pnode) else {
                return Ok(stuck_val("function", vec![]));
            };
            Ok(Val::Arrow(Rc::new(ArrowVal {
                params: names.into(),
                body_def,
                binds: Rc::new(env.clone()),
            })))
        }
        "call" => {
            let callee_i = body_child(ast, node, lang, "callee", 0);
            let args_i = lang
                .and_then(|l| l.child_role_index(&node.kind, "args").ok().flatten())
                .unwrap_or(1);
            let kids = node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            let Some(callee_i) = callee_i else {
                return Ok(stuck_val("call", vec![]));
            };
            let callee = ev(callee_i, env, ast, lang, fuel)?;
            let mut args = Vec::new();
            for k in kids.iter().skip(args_i) {
                let Some(i) = k.as_u64() else { continue };
                args.push(ev(AstRef::new(i as usize), env, ast, lang, fuel)?);
            }
            apply_body_value(callee, args, lang, fuel, lvl)
        }
        // The cell forms evaluate to their canonical former data — the
        // same shapes `synth_dim_arrow`/`synth_axis_flow` emit.
        "dim_arrow" => {
            let (Some(s), Some(d), Some(t)) = (
                body_child(ast, node, lang, "src", 0),
                body_child(ast, node, lang, "dim", 1),
                body_child(ast, node, lang, "tgt", 2),
            ) else {
                return Ok(stuck_val("dim_arrow", vec![]));
            };
            let sv = ev(s, env, ast, lang, fuel)?;
            let dv = ev(d, env, ast, lang, fuel)?;
            let tv = ev(t, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("type.hom"),
                args: vec![dv, sv, tv],
            })
        }
        // `A ~[G]~> B` evaluates to the same canonical former data as
        // `dim_arrow` at dimension 1 — a graded arrow is a 1-cell; the
        // grade child still evaluates for well-formedness.
        "graded_arrow" => {
            let (Some(s), Some(g), Some(t)) = (
                body_child(ast, node, lang, "src", 0),
                body_child(ast, node, lang, "grade", 1),
                body_child(ast, node, lang, "tgt", 2),
            ) else {
                return Ok(stuck_val("graded_arrow", vec![]));
            };
            let sv = ev(s, env, ast, lang, fuel)?;
            let _gv = ev(g, env, ast, lang, fuel)?;
            let tv = ev(t, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("type.hom"),
                args: vec![const_val(1), sv, tv],
            })
        }
        "axis_flow" => {
            let (Some(a), Some(k), Some(b)) = (
                body_child(ast, node, lang, "left", 0),
                body_child(ast, node, lang, "axis", 1),
                body_child(ast, node, lang, "right", 2),
            ) else {
                return Ok(stuck_val("axis_flow", vec![]));
            };
            let av = ev(a, env, ast, lang, fuel)?;
            let kv = ev(k, env, ast, lang, fuel)?;
            let bv = ev(b, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("cell.comp"),
                args: vec![kv, av, bv],
            })
        }
        "flow" => {
            let (Some(a), Some(b)) = (
                body_child(ast, node, lang, "left", 0),
                body_child(ast, node, lang, "right", 1),
            ) else {
                return Ok(stuck_val("flow", vec![]));
            };
            let av = ev(a, env, ast, lang, fuel)?;
            let bv = ev(b, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("cell.comp"),
                args: vec![const_val(0), av, bv],
            })
        }
        "cell_id" => {
            let Some(x) = body_child(ast, node, lang, "operand", 0) else {
                return Ok(stuck_val("cell_id", vec![]));
            };
            let xv = ev(x, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("cell.id"),
                args: vec![const_val(2), xv],
            })
        }
        "cell_comp_v" | "cell_comp_h" => {
            let axis = if node.kind == "cell_comp_v" { 2 } else { 0 };
            let (Some(a), Some(b)) = (
                body_child(ast, node, lang, "left", 0),
                body_child(ast, node, lang, "right", 1),
            ) else {
                return Ok(stuck_val("cell_comp", vec![]));
            };
            let av = ev(a, env, ast, lang, fuel)?;
            let bv = ev(b, env, ast, lang, fuel)?;
            Ok(Val::Former {
                head: type_sym("cell.comp"),
                args: vec![const_val(axis), av, bv],
            })
        }
        kind if kind.starts_with("binary.") || kind.starts_with("unary.") => {
            let l = body_child(ast, node, lang, "left", 0)
                .or_else(|| body_child(ast, node, lang, "operand", 0));
            let r = body_child(ast, node, lang, "right", 1);
            let lv = l
                .map(|i| ev(i, env, ast, lang, fuel))
                .transpose()?
                .unwrap_or(Val::Const(TaggedValue::Null));
            let rv = r.map(|i| ev(i, env, ast, lang, fuel)).transpose()?;
            match fold_scalar(kind, &lv, rv.as_ref()) {
                Some(v) => Ok(v),
                None => {
                    let mut args = vec![const_val(kind), lv];
                    if let Some(r) = rv {
                        args.push(r);
                    }
                    Ok(stuck_val(kind, args))
                }
            }
        }
        kind => {
            // Out-of-fragment kinds: evaluate the children that reduce
            // and keep them inside a `stuck.<kind>` neutral — open
            // terms are legal inside checking.
            let kids = node
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            let mut args = Vec::new();
            for k in kids {
                let Some(i) = k.as_u64() else { continue };
                args.push(ev(AstRef::new(i as usize), env, ast, lang, fuel)?);
            }
            Ok(stuck_val(kind, args))
        }
    }
}

/// Fold a scalar binary/unary operation on `Const` operands —
/// arithmetic on ints/floats, comparisons, boolean `and`/`or`, `neg`/
/// `not`. `None` = the operation does not reduce (open operand).
fn fold_scalar(kind: &str, l: &Val, r: Option<&Val>) -> Option<Val> {
    let int = |v: &Val| match v {
        Val::Const(TaggedValue::Int(n)) => Some(*n),
        _ => None,
    };
    let num = |v: &Val| match v {
        Val::Const(TaggedValue::Int(n)) => Some(*n as f64),
        Val::Const(TaggedValue::Float(n)) => Some(*n),
        _ => None,
    };
    let bool = |v: &Val| match v {
        Val::Const(TaggedValue::Bool(b)) => Some(*b),
        _ => None,
    };
    let is_int = |v: &Val| matches!(v, Val::Const(TaggedValue::Int(_)));
    if let (Some(a), Some(Some(b))) = (int(l), r.map(int)) {
        let v = match kind {
            "binary.add" => Some(a + b),
            "binary.sub" => Some(a - b),
            "binary.mul" => Some(a * b),
            "binary.div" if b != 0 => Some(a / b),
            "binary.mod" if b != 0 => Some(a % b),
            "binary.pow" if b >= 0 => a.checked_pow(b as u32),
            _ => None,
        };
        if let Some(n) = v {
            return Some(const_val(n));
        }
    }
    if let (Some(a), Some(Some(b))) = (num(l), r.map(&num)) {
        let v = match kind {
            "binary.eq" => Some(const_val(a == b)),
            "binary.neq" => Some(const_val(a != b)),
            "binary.lt" => Some(const_val(a < b)),
            "binary.lte" => Some(const_val(a <= b)),
            "binary.gt" => Some(const_val(a > b)),
            "binary.gte" => Some(const_val(a >= b)),
            "binary.div" if b != 0.0 && !is_int(l) => Some(const_val(a / b)),
            _ => None,
        };
        if v.is_some() {
            return v;
        }
    }
    if let (Some(a), Some(Some(b))) = (bool(l), r.map(&bool)) {
        let v = match kind {
            "binary.and" => Some(a && b),
            "binary.or" => Some(a || b),
            _ => None,
        };
        if let Some(b) = v {
            return Some(const_val(b));
        }
    }
    match (kind, bool(l), num(l), r) {
        ("unary.not", Some(a), _, None) => Some(const_val(!a)),
        ("unary.neg", _, Some(a), None) => Some(const_val(-a)),
        _ => None,
    }
}

/// Apply an evaluated body value to evaluated arguments: arrows
/// saturate by evaluating their `body_def` under `binds ∪ params ↦
/// args` (residual on partial application); neutrals extend their
/// spine; formers extend their index args; anything else is a
/// `stuck.call` neutral.
fn apply_body_value(
    f: Val,
    args: Vec<Val>,
    lang: Option<&Language>,
    fuel: &mut u32,
    lvl: Lvl,
) -> Result<Val, EvalError> {
    let mut f = f;
    let mut args = args.into_iter();
    loop {
        let Some(arg) = args.next() else {
            return Ok(f);
        };
        let arrow: Rc<ArrowVal> = match f {
            Val::Arrow(a) => a,
            Val::Const(tv) => {
                // Boundary arrows carrying destructured params
                // (`param_defs` — `([x, y]) ~>`, `({a, b}) ~>`) or the
                // plain `array_pattern` flag (`[x, y] ~>`) cannot be
                // spelled as a positional `params` list — apply them
                // through the pattern matcher instead of degrading to
                // first-param-binds-all (the `ArrowVal` projection
                // would lose `param_defs`/`array_pattern`/`rest`).
                if matches!(
                    &tv,
                    TaggedValue::Arrow {
                        param_defs: Some(_),
                        ..
                    } | TaggedValue::Arrow {
                        array_pattern: true,
                        ..
                    }
                ) {
                    let mut all_args = Vec::with_capacity(args.size_hint().0 + 1);
                    all_args.push(arg);
                    all_args.extend(args);
                    return apply_destructured_arrow(tv, all_args, lang, fuel, lvl);
                }
                match tv {
                    TaggedValue::Arrow {
                        params,
                        body_def,
                        binds,
                        ..
                    } => Rc::new(ArrowVal {
                        params,
                        body_def: (*body_def).clone(),
                        binds: Rc::new(
                            binds
                                .iter()
                                .map(|(k, v)| (k.clone(), Val::Const(v.clone())))
                                .collect(),
                        ),
                    }),
                    other => {
                        let mut stuck = vec![Val::Const(other), arg];
                        stuck.extend(args);
                        return Ok(stuck_val("call", stuck));
                    }
                }
            }
            Val::Rigid { lvl, spine } => {
                let mut s = spine;
                s.push(arg);
                f = Val::Rigid { lvl, spine: s };
                continue;
            }
            Val::Flex { meta, spine } => {
                let mut s = spine;
                s.push(arg);
                f = Val::Flex { meta, spine: s };
                continue;
            }
            Val::Unfold { name, spine, value } => {
                let mut s = spine;
                s.push(arg);
                f = Val::Unfold {
                    name,
                    spine: s,
                    value,
                };
                continue;
            }
            Val::Former { head, args: fargs } => {
                let mut fargs = fargs;
                fargs.push(arg);
                f = Val::Former { head, args: fargs };
                continue;
            }
            other => {
                let mut stuck = vec![other, arg];
                stuck.extend(args);
                return Ok(stuck_val("call", stuck));
            }
        };
        // Bind `params ↦ args` greedily; leftover params stay residual,
        // a saturated arrow runs its body and the result may itself be
        // applied to further args.
        let mut env2: FxHashMap<Rc<str>, Val> = (*arrow.binds).clone();
        env2.insert(Rc::from(arrow.params[0].as_str()), arg);
        let mut consumed = 1;
        for p in arrow.params.iter().skip(1) {
            match args.next() {
                Some(v) => {
                    env2.insert(Rc::from(p.as_str()), v);
                    consumed += 1;
                }
                None => break,
            }
        }
        if consumed < arrow.params.len() {
            return Ok(Val::Arrow(Rc::new(ArrowVal {
                params: arrow.params[consumed..].to_vec().into(),
                body_def: arrow.body_def.clone(),
                binds: Rc::new(env2),
            })));
        }
        let ast = decode_body(&arrow.body_def)?;
        if *fuel == 0 {
            return Ok(stuck_val("fuel", vec![]));
        }
        *fuel -= 1;
        f = eval_body_node(&ast, AstRef::new(ast.len() - 1), &env2, lang, fuel, lvl)?;
    }
}

/// Apply a boundary arrow carrying destructured params — `param_defs`
/// (`([x, y]) ~>`, `({a, b}) ~>`, `[x, [y, z]] ~>`) or the plain
/// `array_pattern` flag (`[x, y] ~>`) — to `args`. Mirrors
/// `apply_arrow_patterns`/`resolve_closure_params` on the exec side
/// (ADR-0081): the `param_defs` arena's top-level patterns each match
/// their argument via [`try_pattern_val`], a trailing `rest` pattern
/// binds the surplus tail, and surplus args past the fixed patterns
/// curry through [`apply_body_value`]. Arity/pattern mismatches
/// produce `stuck.call`/`stuck.match` neutrals rather than errors —
/// the NBE domain's honest-open convention.
fn apply_destructured_arrow(
    arrow: TaggedValue,
    args: Vec<Val>,
    lang: Option<&Language>,
    fuel: &mut u32,
    lvl: Lvl,
) -> Result<Val, EvalError> {
    let TaggedValue::Arrow {
        params,
        rest,
        param_defs,
        body_def,
        binds,
        ..
    } = &arrow
    else {
        return Err(EvalError::Op("apply_arrow: not an arrow value".into()));
    };
    let (params, rest, param_defs, body_def, binds) = (
        params.clone(),
        rest.clone(),
        param_defs.clone(),
        body_def.clone(),
        binds.clone(),
    );
    let stuck_call = |args: Vec<Val>| {
        Ok(stuck_val(
            "call",
            std::iter::once(Val::Const(arrow.clone()))
                .chain(args)
                .collect(),
        ))
    };
    let mut env2: FxHashMap<Rc<str>, Val> = binds
        .iter()
        .map(|(k, v)| (k.clone(), Val::Const(v.clone())))
        .collect();
    let mut curry: Vec<Val> = Vec::new();
    if let Some(pd) = &param_defs {
        let Some(lang) = lang else {
            return stuck_call(args);
        };
        let pat_ast = decode_body(pd)?;
        if pat_ast.is_empty() {
            return stuck_call(args);
        }
        // The params-node subtree's root is the last arena node —
        // the same convention `body_def` uses.
        let params_ref = AstRef::new(pat_ast.len() - 1);
        let tops = lang
            .top_level_param_refs(&pat_ast, params_ref)
            .map_err(|e| EvalError::Op(format!("apply_arrow: param_defs: {e}")))?;
        let root = pat_ast
            .get(params_ref)
            .ok_or_else(|| EvalError::Op("apply_arrow: param_defs root".into()))?;
        let root_is_pattern = matches!(
            lang.pattern_role(&root.kind),
            Some("sequence") | Some("entries")
        );
        if root_is_pattern {
            // Single pattern (`[x, y] ~>`, `{a, b} ~>`): one scrutinee,
            // surplus args curry.
            let Some(scrut) = args.first() else {
                return stuck_call(args);
            };
            curry = args[1..].to_vec();
            if !try_pattern_val(lang, &pat_ast, params_ref, scrut, &mut env2)? {
                return Ok(stuck_val("match", vec![scrut.clone()]));
            }
        } else {
            // Params list: top-level children are positional patterns;
            // a `rest` pattern (at most one, last) binds the surplus
            // tail as an array.
            let mut fixed: Vec<AstRef> = Vec::with_capacity(tops.len());
            let mut rest_ref: Option<AstRef> = None;
            for t in tops {
                let n = pat_ast
                    .get(t)
                    .ok_or_else(|| EvalError::Op("apply_arrow: param_defs child".into()))?;
                if lang.pattern_role_is(&n.kind, "rest") == Some(true) {
                    if rest_ref.is_some() {
                        return Err(EvalError::Op("apply_arrow: multiple rest params".into()));
                    }
                    rest_ref = Some(t);
                } else {
                    fixed.push(t);
                }
            }
            if args.len() < fixed.len() {
                return stuck_call(args);
            }
            for (i, pat_ref) in fixed.iter().enumerate() {
                if !try_pattern_val(lang, &pat_ast, *pat_ref, &args[i], &mut env2)? {
                    return Ok(stuck_val("match", vec![args[i].clone()]));
                }
            }
            let split = fixed.len();
            if let Some(rr) = rest_ref {
                // The tail binds as an array *value* — `Val`s quote back
                // to tagged data (`__neutral__` for open terms).
                let tail_vals: Vec<TaggedValue> = args[split..]
                    .iter()
                    .map(|v| {
                        handl_value::nbe::quote(&handl_value::nbe::MetaStore::default(), lvl, v)
                    })
                    .collect();
                let tail = Val::Const(TaggedValue::Arr(Rc::from(tail_vals)));
                bind_rest_pattern_val(lang, &pat_ast, rr, tail, &mut env2)?;
            } else if args.len() > split {
                curry = args[split..].to_vec();
            }
        }
    } else {
        // `array_pattern` (`[x, y] ~>`, `[x, ...rest] ~>`): one array
        // argument destructured positionally — fixed names bind to
        // leading elements, the rest binding receives the tail.
        if args.len() != 1 {
            return stuck_call(args);
        }
        let Val::Const(TaggedValue::Arr(arr)) = &args[0] else {
            return stuck_call(args);
        };
        let fixed = params.len();
        if arr.len() < fixed || (rest.is_none() && arr.len() != fixed) {
            return stuck_call(args);
        }
        let (head, tail) = arr.split_at(fixed);
        for (i, p) in params.iter().enumerate() {
            env2.insert(Rc::from(p.as_str()), Val::Const(head[i].clone()));
        }
        if let Some(r) = &rest {
            env2.insert(
                r.clone(),
                Val::Const(TaggedValue::Arr(Rc::from(tail.to_vec()))),
            );
        }
    }
    let body_ast = decode_body(&body_def)?;
    if body_ast.is_empty() {
        return stuck_call(args);
    }
    if *fuel == 0 {
        return Ok(stuck_val("fuel", vec![]));
    }
    *fuel -= 1;
    let mut result = eval_body_node(
        &body_ast,
        AstRef::new(body_ast.len() - 1),
        &env2,
        lang,
        fuel,
        lvl,
    )?;
    if !curry.is_empty() {
        result = apply_body_value(result, curry, lang, fuel, lvl)?;
    }
    Ok(result)
}

/// Bind a `rest` pattern node's subject against the tail value — a
/// plain `var` subject binds the name directly (wildcards bind
/// nothing); anything else recurses through [`try_pattern_val`],
/// matching exec's `rest` handling in `apply_arrow_patterns`.
fn bind_rest_pattern_val(
    lang: &Language,
    ast: &Ast,
    rest_ref: AstRef,
    tail: Val,
    env: &mut FxHashMap<Rc<str>, Val>,
) -> Result<(), EvalError> {
    let rn = ast
        .get(rest_ref)
        .ok_or_else(|| EvalError::Op("apply_arrow: rest pattern".into()))?;
    let Some(sref) = crate::role_child_ref(lang, rn, "subject") else {
        return Ok(());
    };
    let snode = ast
        .get(sref)
        .ok_or_else(|| EvalError::Op("apply_arrow: rest subject".into()))?;
    let is_leaf = lang.pattern_role(&snode.kind) == Some("leaf")
        || (lang.pattern_role(&snode.kind).is_none() && snode.kind == "var");
    if is_leaf && let Some(name) = snode.payload.as_str() {
        if !crate::is_wildcard_binder(name) {
            env.insert(Rc::from(name), tail);
        }
        return Ok(());
    }
    try_pattern_val(lang, ast, sref, &tail, env).map(|_| ())
}

/// `pattern_match::try_pattern` over the `Val` domain: leaf binders
/// bind *any* scrutinee (irrefutable — including `Flex`/`Rigid`
/// neutrals); `sequence`/`entries` patterns need a `Const` array/object
/// scrutinee and recurse element/field-wise; `subject`-roled wrappers
/// (`type-annot`-like) unwrap; role-less `var` keeps the legacy binder
/// reading. Tag (`callee`-roled) and literal patterns report no match
/// rather than guess — the caller surfaces a `stuck.match` neutral.
fn try_pattern_val(
    lang: &Language,
    ast: &Ast,
    pat: AstRef,
    scrut: &Val,
    env: &mut FxHashMap<Rc<str>, Val>,
) -> Result<bool, EvalError> {
    let node = ast
        .get(pat)
        .ok_or_else(|| EvalError::Op("apply_arrow: pattern node out of bounds".into()))?;
    match lang.pattern_role(&node.kind) {
        Some("leaf") => {
            let Some(name) = node.payload.as_str() else {
                return Ok(false);
            };
            if crate::is_wildcard_binder(name) {
                return Ok(true);
            }
            env.insert(Rc::from(name), scrut.clone());
            Ok(true)
        }
        Some("rest") => {
            // A `rest` node reached directly matches its subject
            // against the (already tail-array) scrutinee.
            match crate::role_child_ref(lang, node, "subject") {
                Some(sref) => try_pattern_val(lang, ast, sref, scrut, env),
                None => Ok(true),
            }
        }
        Some("sequence") => {
            let Val::Const(TaggedValue::Arr(arr)) = scrut else {
                return Ok(false);
            };
            let mut positional: Vec<AstRef> = Vec::new();
            let mut rest_ref: Option<AstRef> = None;
            if let Some(children) = node.payload.get("children").and_then(|c| c.as_array()) {
                for entry in children {
                    let Some(idx) = entry.as_u64() else { continue };
                    let r = AstRef::new(idx as usize);
                    let Some(cn) = ast.get(r) else { continue };
                    if lang.pattern_role_is(&cn.kind, "rest") == Some(true) {
                        rest_ref = Some(r);
                    } else {
                        positional.push(r);
                    }
                }
            }
            let n = positional.len();
            if (rest_ref.is_none() && arr.len() != n) || arr.len() < n {
                return Ok(false);
            }
            for (i, pr) in positional.iter().enumerate() {
                if !try_pattern_val(lang, ast, *pr, &Val::Const(arr[i].clone()), env)? {
                    return Ok(false);
                }
            }
            if let Some(rr) = rest_ref {
                let tail = Val::Const(TaggedValue::Arr(Rc::from(arr[n..].to_vec())));
                bind_rest_pattern_val(lang, ast, rr, tail, env)?;
            }
            Ok(true)
        }
        Some("entries") => {
            let Val::Const(TaggedValue::Obj(obj)) = scrut else {
                return Ok(false);
            };
            let mut rest_ref: Option<AstRef> = None;
            let mut entries: Vec<(String, AstRef)> = Vec::new();
            if let Some(children) = node.payload.get("children").and_then(|c| c.as_array()) {
                for entry in children {
                    let Some(idx) = entry.as_u64() else { continue };
                    let r = AstRef::new(idx as usize);
                    let Some(cn) = ast.get(r) else { continue };
                    if lang.pattern_role_is(&cn.kind, "rest") == Some(true) {
                        rest_ref = Some(r);
                        continue;
                    }
                    // `entry` nodes: literal `key` child names the
                    // field, `value` child is the sub-pattern. A
                    // computed key cannot be resolved statically —
                    // report no match rather than guess.
                    let Some(kref) = crate::role_child_ref(lang, cn, "key") else {
                        continue;
                    };
                    let Some(knode) = ast.get(kref) else { continue };
                    let Some(key) = knode.payload.as_str().map(str::to_string) else {
                        return Ok(false);
                    };
                    let Some(vref) = crate::role_child_ref(lang, cn, "value") else {
                        continue;
                    };
                    entries.push((key, vref));
                }
            }
            let mut matched: Vec<String> = Vec::with_capacity(entries.len());
            for (key, vref) in &entries {
                let Some(field) = obj.get(key.as_str()) else {
                    return Ok(false);
                };
                matched.push(key.clone());
                if !try_pattern_val(lang, ast, *vref, &Val::Const(field.clone()), env)? {
                    return Ok(false);
                }
            }
            if let Some(rr) = rest_ref {
                let rest_obj: rustc_hash::FxHashMap<Rc<str>, TaggedValue> = obj
                    .iter()
                    .filter(|(k, _)| !matched.iter().any(|m| m.as_str() == k.as_ref()))
                    .map(|(k, v)| (Rc::clone(k), v.clone()))
                    .collect();
                bind_rest_pattern_val(
                    lang,
                    ast,
                    rr,
                    Val::Const(TaggedValue::Obj(Rc::new(rest_obj))),
                    env,
                )?;
            }
            Ok(true)
        }
        Some("entry") => {
            // Reached directly — the `value` child holds the pattern.
            match crate::role_child_ref(lang, node, "value") {
                Some(vref) => try_pattern_val(lang, ast, vref, scrut, env),
                None => Ok(false),
            }
        }
        _ => {
            // Role-less `var` keeps the legacy binder reading
            // (`is_binder_node`'s `kind == "var"` fallback).
            if lang.pattern_role(&node.kind).is_none() && node.kind == "var" {
                let Some(name) = node.payload.as_str() else {
                    return Ok(false);
                };
                if !crate::is_wildcard_binder(name) {
                    env.insert(Rc::from(name), scrut.clone());
                }
                return Ok(true);
            }
            // `subject`-roled wrappers (`type-annot`-like) unwrap.
            if let Some(sref) = crate::role_child_ref(lang, node, "subject") {
                return try_pattern_val(lang, ast, sref, scrut, env);
            }
            Ok(false)
        }
    }
}

/// Read a string argument: the op `lit`, the first `args` const, or
/// the threaded `input`.
fn str_arg<'a>(lit: Option<&'a Value>, args: &'a [Val], input: &'a Val) -> Option<String> {
    if let Some(s) = lit.and_then(Value::as_str) {
        return Some(s.to_string());
    }
    for v in args.iter().chain(std::iter::once(input)) {
        if let Val::Const(TaggedValue::Str(s)) = v {
            return Some(s.to_string());
        }
    }
    None
}

fn int_arg(lit: Option<&Value>, args: &[Val], input: &Val) -> Option<i64> {
    if let Some(n) = lit.and_then(Value::as_i64) {
        return Some(n);
    }
    for v in args.iter().chain(std::iter::once(input)) {
        if let Val::Const(TaggedValue::Int(n)) = v {
            return Some(*n);
        }
    }
    None
}

fn bool_val(b: bool) -> Val {
    Val::Const(TaggedValue::Bool(b))
}

impl OpHandler for JudgmentOps {
    fn op(
        &self,
        name: &str,
        args: &[Val],
        lit: Option<&Value>,
        dim: &Term,
        input: Val,
    ) -> Result<Val, EvalError> {
        let err = |m: String| EvalError::Op(format!("{name}: {m}"));
        match name {
            // -- constructors -------------------------------------------------
            "nominal" | "sym" => {
                let seed = str_arg(lit, args, &input)
                    .ok_or_else(|| err("expected a seed string".into()))?;
                Ok(Val::Nominal(type_sym(&seed)))
            }
            "former" => {
                let seed = str_arg(lit, args, &input)
                    .ok_or_else(|| err("expected a former seed".into()))?;
                Ok(Val::Former {
                    head: type_sym(&seed),
                    args: args.to_vec(),
                })
            }
            "universe" => {
                let i = int_arg(lit, args, &input).ok_or_else(|| err("expected a level".into()))?;
                if i < 0 {
                    return Err(err("negative universe level".into()));
                }
                Ok(Val::Universe(i as u32))
            }
            "pi" => {
                if args.len() != 2 {
                    return Err(err("expected [dom, cod]".into()));
                }
                Ok(Val::Former {
                    head: type_sym("type.pi"),
                    args: args.to_vec(),
                })
            }
            "hom" => {
                // `Hom(n, src, tgt)` — ADR-0153. `n` may also arrive as
                // the `dim` slot (`~[n]~>` reads the dimension index).
                let mut all: Vec<Val> = Vec::new();
                if let Some(n) = dim.nat_literal() {
                    all.push(Val::Const(TaggedValue::Int(n as i64)));
                }
                all.extend(args.iter().cloned());
                if all.len() != 3 {
                    return Err(err("expected [n, src, tgt]".into()));
                }
                Ok(Val::Former {
                    head: type_sym("type.hom"),
                    args: all,
                })
            }
            "decl_parent" => {
                // Declare a parent link on a seeded nominal:
                // `decl_parent` with lit `[child, parent]` or args
                // `[child, parent]` — the `Sym::seeded_with_parent`
                // declaration (ADR-0154 §2's `parent` link).
                let pair: Vec<String> = lit
                    .and_then(Value::as_array)
                    .map(|xs| {
                        xs.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        args.iter()
                            .filter_map(|v| match v {
                                Val::Const(TaggedValue::Str(s)) => Some(s.to_string()),
                                _ => None,
                            })
                            .collect()
                    });
                if pair.len() != 2 {
                    return Err(err("expected [child, parent] seed pair".into()));
                }
                let parent = Sym::seeded(&pair[1]);
                Ok(Val::Nominal(Sym::seeded_with_parent(&pair[0], &parent)))
            }
            "meta" => {
                let ty = args.first().cloned().unwrap_or(Val::Universe(0));
                let m = self.checker.borrow_mut().fresh_meta(ty);
                Ok(Val::Flex {
                    meta: m,
                    spine: Spine::default(),
                })
            }
            "literal" => {
                // The literal itself is `input` or the first arg's const.
                let tv = args
                    .iter()
                    .chain(std::iter::once(&input))
                    .find_map(|v| match v {
                        Val::Const(t @ (TaggedValue::Int(_) | TaggedValue::Float(_))) => {
                            Some(t.clone())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| err("expected a numeric literal".into()))?;
                Ok(self.checker.borrow_mut().infer_literal(&tv))
            }
            "lift" => {
                let lvl = self.checker.borrow().lvl();
                let tv = args
                    .iter()
                    .chain(std::iter::once(&input))
                    .find_map(|v| match v {
                        Val::Const(t) => Some(t.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| err("expected a tagged value".into()))?;
                Ok(lift(lvl, &tv))
            }
            "quote" => {
                let c = self.checker.borrow();
                let v = args.first().unwrap_or(&input);
                Ok(Val::Const(handl_value::nbe::quote(&c.metas, c.lvl(), v)))
            }

            // -- pure index arithmetic (normalization = evaluation) -----------
            "add" | "sub" | "mul" | "div" => {
                let mut acc: Option<i64> = None;
                for v in args {
                    let n = match v {
                        Val::Const(TaggedValue::Int(n)) => *n,
                        _ => return Err(err("non-integer index".into())),
                    };
                    acc = Some(match (acc, name) {
                        (None, _) => n,
                        (Some(a), "add") => a + n,
                        (Some(a), "sub") => a - n,
                        (Some(a), "mul") => a * n,
                        (Some(a), "div") => {
                            if n == 0 {
                                return Err(err("division by zero".into()));
                            }
                            a / n
                        }
                        _ => unreachable!(),
                    });
                }
                Ok(Val::Const(TaggedValue::Int(
                    acc.ok_or_else(|| err("no operands".into()))?,
                )))
            }

            // -- judges --------------------------------------------------------
            "infer" | "synth" => {
                let v = args.first().cloned().unwrap_or(input);
                self.checker
                    .borrow_mut()
                    .infer(&v)
                    .map_err(|e| err(e.to_string()))
            }
            "check" => {
                if args.len() != 2 {
                    return Err(err("expected [value, type]".into()));
                }
                self.checker
                    .borrow_mut()
                    .check(&args[0], &args[1])
                    .map_err(|e| err(e.to_string()))?;
                Ok(bool_val(true))
            }
            "unify" | "conv" => {
                if args.len() != 2 {
                    return Err(err("expected [a, b]".into()));
                }
                let b = self.checker.borrow_mut().conv(&args[0], &args[1]);
                Ok(bool_val(b))
            }
            "subtype" => {
                if args.len() != 2 {
                    return Err(err("expected [a, b]".into()));
                }
                let b = self.checker.borrow_mut().subtype(&args[0], &args[1]);
                Ok(bool_val(b))
            }
            "bind" => {
                // `bind("x", ty)` — extend Γ, return the rigid head.
                let name =
                    str_arg(lit, args, &input).ok_or_else(|| err("expected a name".into()))?;
                let ty = args
                    .iter()
                    .find(|v| !matches!(v, Val::Const(TaggedValue::Str(_))))
                    .cloned()
                    .ok_or_else(|| err("expected a type".into()))?;
                Ok(self.checker.borrow_mut().bind(name, ty))
            }
            "lookup" => {
                let name =
                    str_arg(lit, args, &input).ok_or_else(|| err("expected a name".into()))?;
                Ok(self
                    .checker
                    .borrow()
                    .lookup_name(&name)
                    .cloned()
                    .unwrap_or(Val::Const(TaggedValue::Null)))
            }

            // -- cell ops (ADR-0153 §2: `>>>[k]`, `id_k`) --------------------
            "cell_comp_axis" | "cell2_comp_axis" => {
                // Shapes: `Seq`-delegated (args `[a, b]`, the axis in
                // `lit{"axis"}`/`dim`), or the op spelling
                // `[k, a, b]` / `[k, dim, a, b]`.
                let (kv, a, b) = match args.len() {
                    2 => {
                        let k = lit
                            .and_then(|l| l.get("axis").and_then(Value::as_i64))
                            .or_else(|| dim.nat_literal().map(|n| n as i64))
                            .map(|n| Val::Const(TaggedValue::Int(n)))
                            .unwrap_or(Val::Const(TaggedValue::Null));
                        (k, args[0].clone(), args[1].clone())
                    }
                    3 => (args[0].clone(), args[1].clone(), args[2].clone()),
                    4 => (args[0].clone(), args[2].clone(), args[3].clone()),
                    _ => return Err(err("expected [a, b] (+lit axis) or [k, a, b]".into())),
                };
                // The boundary law is decided now (fail-fast); the
                // composite is canonical former data — `infer` of
                // `cell.comp` re-derives the same rule.
                self.checker
                    .borrow_mut()
                    .infer_comp_axis(&kv, &a, &b)
                    .map_err(|e| err(e.to_string()))?;
                Ok(Val::Former {
                    head: type_sym("cell.comp"),
                    args: vec![kv, a, b],
                })
            }
            "cell_id" | "refl" => {
                let (kv, x) = match args.len() {
                    1 => {
                        let k = lit
                            .and_then(Value::as_i64)
                            .or_else(|| dim.nat_literal().map(|n| n as i64))
                            .map(|n| Val::Const(TaggedValue::Int(n)));
                        (k, args[0].clone())
                    }
                    2 => (Some(args[0].clone()), args[1].clone()),
                    _ => return Err(err("expected [x] or [k, x]".into())),
                };
                self.checker
                    .borrow_mut()
                    .infer_id(kv.as_ref(), &x)
                    .map_err(|e| err(e.to_string()))?;
                Ok(Val::Former {
                    head: type_sym("cell.id"),
                    args: match kv {
                        Some(k) => vec![k, x],
                        None => vec![x],
                    },
                })
            }
            // -- trivial natives ----------------------------------------------
            "id" => {
                // `id` at dimension ≥ 2 is the identity cell `id_k`
                // (`eval.cell_id` emits `Op{id, dim:2}`); at dim ≤ 1 it
                // stays the trivial identity arrow.
                let k = dim.nat_literal().unwrap_or(1);
                if k >= 2 {
                    let v = args.first().cloned().unwrap_or(input);
                    let kv = Val::Const(TaggedValue::Int(k as i64));
                    self.checker
                        .borrow_mut()
                        .infer_id(Some(&kv), &v)
                        .map_err(|e| err(e.to_string()))?;
                    return Ok(Val::Former {
                        head: type_sym("cell.id"),
                        args: vec![kv, v],
                    });
                }
                Ok(input)
            }
            // `flow` — the `eval.flow` value dispatch (the engines'
            // `x >>> f`): two cell/morphism operands compose (∘₀ —
            // `Const` payloads take the kernel's `Arrow{first,
            // second}` mint, domain cells the `infer_comp_axis`
            // boundary law); anything else applies the right value to
            // the left. `nbe::eval`'s `Seq{0}` threads schemes; this
            // arm is the operand-reading counterpart.
            "flow" => {
                if args.len() != 2 {
                    return Err(err("expected [a, b]".into()));
                }
                let (a, b) = (args[0].clone(), args[1].clone());
                if hnl_engine::nbe::is_cell_val(&a) && hnl_engine::nbe::is_cell_val(&b) {
                    return hnl_engine::nbe::compose_cell_vals(0, 1, dim, &a, &b, self);
                }
                hnl_engine::nbe::apply(&b, a, self)
            }
            "const" => Ok(args
                .first()
                .cloned()
                .unwrap_or(Val::Const(TaggedValue::Null))),

            other => Err(EvalError::Op(format!("unknown judgment op `{other}`"))),
        }
    }

    fn apply_arrow(&self, arrow: &ArrowVal, arg: Val) -> Option<Result<Val, EvalError>> {
        // Fire only when this application saturates the arrow — a
        // partial application stays the residual closure (`None`).
        if arrow.params.len() > 1 {
            return None;
        }
        let extra: Vec<(Rc<str>, Val)> = arrow
            .params
            .first()
            .map(|p| vec![(Rc::from(p.as_str()), arg)])
            .unwrap_or_default();
        // Ambient level for `lift` inside body evaluation (boundary
        // `__neutral__` literal payloads restore rigid levels from it).
        let lvl = self.checker.borrow().lvl();
        Some(eval_arrow_body(arrow, &extra, self.lang.as_deref(), lvl))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hnl_engine::nbe::eval;
    use hnl_engine::term::ArrSource;

    fn jop(name: &str, args: Vec<Term>, lit: Option<Value>) -> Term {
        Term::Op {
            dim: Term::default_dim(),
            name: name.into(),
            args,
            source: ArrSource::Judgment,
            lit,
            attrs: None,
        }
    }

    fn ct(v: impl Into<Value>) -> Term {
        Term::Const { value: v.into() }
    }

    #[test]
    fn former_construct_and_conv() {
        let ops = JudgmentOps::new();
        // `former("vec.Vec", 2, nominal("type.int"))`
        let a = jop(
            "former",
            vec![ct(2), jop("nominal", vec![], Some("type.Int".into()))],
            Some("vec.Vec".into()),
        );
        let b = jop(
            "former",
            vec![ct(2), jop("nominal", vec![], Some("type.Int".into()))],
            Some("vec.Vec".into()),
        );
        let va = eval(&a, Val::Const(TaggedValue::Null), &ops).unwrap();
        let vb = eval(&b, Val::Const(TaggedValue::Null), &ops).unwrap();
        let eq = eval(
            &jop("unify", vec![a, b], None),
            Val::Const(TaggedValue::Null),
            &ops,
        )
        .unwrap();
        assert!(matches!(eq, Val::Const(TaggedValue::Bool(true))));
        assert!(matches!(va, Val::Former { .. }));
        assert!(matches!(vb, Val::Former { .. }));
    }

    #[test]
    fn index_normalizes_by_evaluation() {
        // `Vec(1 + 1, Int) ≡ Vec(2, Int)` — index equality is
        // evaluation + conv, not a rewriting solver (0152 §3).
        let ops = JudgmentOps::new();
        let int = || jop("nominal", vec![], Some("type.Int".into()));
        let a = jop(
            "former",
            vec![jop("add", vec![ct(1), ct(1)], None), int()],
            Some("vec.Vec".into()),
        );
        let b = jop("former", vec![ct(2), int()], Some("vec.Vec".into()));
        let eq = eval(
            &jop("unify", vec![a, b], None),
            Val::Const(TaggedValue::Null),
            &ops,
        )
        .unwrap();
        assert!(matches!(eq, Val::Const(TaggedValue::Bool(true))));
    }

    #[test]
    fn literal_synths_constrained_meta() {
        let ops = JudgmentOps::new();
        let v = eval(
            &jop("literal", vec![ct(7)], None),
            Val::Const(TaggedValue::Null),
            &ops,
        )
        .unwrap();
        assert!(matches!(v, Val::Flex { .. }));
    }

    #[test]
    fn bind_then_lookup() {
        let ops = JudgmentOps::new();
        let t = Term::seq(
            jop(
                "bind",
                vec![jop("universe", vec![], Some(0.into()))],
                Some("A".into()),
            ),
            jop("lookup", vec![], Some("A".into())),
        );
        let v = eval(&t, Val::Const(TaggedValue::Null), &ops).unwrap();
        // `lookup("A")` returns the bound *type* — Type(0).
        assert!(matches!(v, Val::Universe(0)));
        // And Γ has one entry named A.
        assert_eq!(ops.checker.borrow().gamma.len(), 1);
    }

    #[test]
    fn check_reports_mismatch_as_op_error() {
        let ops = JudgmentOps::new();
        let t = jop(
            "check",
            vec![
                jop("universe", vec![], Some(0.into())),
                jop("nominal", vec![], Some("type.Int".into())),
            ],
            None,
        );
        // `Type(0) : Int` — mismatch.
        assert!(matches!(
            eval(&t, Val::Const(TaggedValue::Null), &ops),
            Err(EvalError::Op(_))
        ));
    }

    /// Phase-3 WS-F: a `lit` whose payload is a *boundary-encoded* object
    /// (`__sym__`/`__former__`/`__type__` — what desugar's `push_sym_lit`
    /// mints into `data` blocks and `reify_ast` emits) evaluates through
    /// `eval_arrow_body` to the payload's lifted `Val`, not `Null`.
    #[test]
    fn lit_object_payload_lifts_into_domain() {
        let arrow_with_body = |payload: Value| ArrowVal {
            params: Rc::from(vec!["x".to_string()]),
            body_def: TaggedValue::from(serde_json::json!({
                "__ast__": {"nodes": [
                    {"kind": "lit", "payload": payload}
                ]}
            })),
            binds: Rc::new(FxHashMap::default()),
        };

        // `lit{__sym__}` — the desugar's ctor/former sym literal —
        // evaluates to the nominal, not `Const(Null)`.
        let sym_payload =
            handl_value::tagged_to_value(&TaggedValue::Sym(Sym::seeded("data.Option.Some")));
        let v = eval_arrow_body(&arrow_with_body(sym_payload), &[], None, 0).unwrap();
        match v {
            Val::Nominal(s) => assert_eq!(s.name(), "Some"),
            other => panic!("lit{{__sym__}} should lift to Nominal, got {other:?}"),
        }

        // `lit{__former__}` (quote's `{sym, args}` spelling) → `Val::Former`.
        let former_payload = serde_json::json!({
            "__former__": {
                "sym": handl_value::tagged_to_value(&TaggedValue::Sym(Sym::seeded("vec.Vec"))),
                "args": [2, handl_value::tagged_to_value(&TaggedValue::Sym(Sym::seeded("type.Int")))]
            }
        });
        let v = eval_arrow_body(&arrow_with_body(former_payload), &[], None, 0).unwrap();
        match v {
            Val::Former { head, args } => {
                assert_eq!(head.name(), "Vec");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Val::Const(TaggedValue::Int(2))));
                assert!(matches!(&args[1], Val::Nominal(s) if s.name() == "Int"));
            }
            other => panic!("lit{{__former__}} should lift to Former, got {other:?}"),
        }

        // `lit{__type__}` → `Val::Universe`.
        let v = eval_arrow_body(
            &arrow_with_body(serde_json::json!({"__type__": 1})),
            &[],
            None,
            0,
        )
        .unwrap();
        assert!(matches!(v, Val::Universe(1)));

        // A plain object payload is the literal itself — `Const(Obj)`,
        // not `Null`.
        let v =
            eval_arrow_body(&arrow_with_body(serde_json::json!({"a": 1})), &[], None, 0).unwrap();
        assert!(matches!(v, Val::Const(TaggedValue::Obj(_))));

        // The `{"value": …}` envelope still unwraps first.
        let v = eval_arrow_body(
            &arrow_with_body(serde_json::json!({"value": 7})),
            &[],
            None,
            0,
        )
        .unwrap();
        assert!(matches!(v, Val::Const(TaggedValue::Int(7))));

        // Scalar payloads are unchanged.
        let v = eval_arrow_body(&arrow_with_body(serde_json::json!(3)), &[], None, 0).unwrap();
        assert!(matches!(v, Val::Const(TaggedValue::Int(3))));
    }

    /// The same `lit{__sym__}` reached through the full `apply_arrow`
    /// op path (saturated single-param arrow application).
    #[test]
    fn apply_arrow_lit_sym_body_evaluates_payload() {
        let ops = JudgmentOps::new();
        let arrow = ArrowVal {
            params: Rc::from(vec!["x".to_string()]),
            body_def: TaggedValue::from(serde_json::json!({
                "__ast__": {"nodes": [
                    {"kind": "lit", "payload":
                        handl_value::tagged_to_value(
                            &TaggedValue::Sym(Sym::seeded("data.Nat.Succ")))}
                ]}
            })),
            binds: Rc::new(FxHashMap::default()),
        };
        let v = ops
            .apply_arrow(&arrow, Val::Const(TaggedValue::Int(0)))
            .expect("saturated arrow")
            .unwrap();
        match v {
            Val::Nominal(s) => assert_eq!(s.name(), "Succ"),
            other => panic!("apply_arrow lit{{__sym__}} body should be Nominal, got {other:?}"),
        }
    }
}
