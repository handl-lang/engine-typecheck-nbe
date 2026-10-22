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

use handl_value::nbe::{ArrowVal, Spine, Val};
use handl_value::{Sym, TaggedValue};
use hnl_engine::nbe::{EvalError, OpHandler};
use hnl_engine::Term;
use serde_json::Value;

use crate::checker::{Checker, lift, type_sym};

/// The judgment-op surface: evaluates type-level `Term`s into the
/// `Val` domain, solving metas and binding Γ in the owned `Checker`.
pub struct JudgmentOps {
    /// The checking context (mutated through `RefCell` — `OpHandler`
    /// is `&self`).
    pub checker: RefCell<Checker>,
}

impl JudgmentOps {
    /// A fresh handler over an empty checker.
    pub fn new() -> Self {
        JudgmentOps {
            checker: RefCell::new(Checker::new()),
        }
    }

    /// A fresh handler over a pre-seeded checker (formers/nominals
    /// already declared).
    pub fn over(checker: Checker) -> Self {
        JudgmentOps {
            checker: RefCell::new(checker),
        }
    }
}

impl Default for JudgmentOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Read a string argument: the op `lit`, the first `args` const, or
/// the threaded `input`.
fn str_arg<'a>(
    lit: Option<&'a Value>,
    args: &'a [Val],
    input: &'a Val,
) -> Option<String> {
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
                let i = int_arg(lit, args, &input)
                    .ok_or_else(|| err("expected a level".into()))?;
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
                Ok(Val::Const(handl_value::nbe::quote(
                    &c.metas,
                    c.lvl(),
                    v,
                )))
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
                    _ => {
                        return Err(err(
                            "expected [a, b] (+lit axis) or [k, a, b]".into(),
                        ))
                    }
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
            "const" => Ok(args.first().cloned().unwrap_or(Val::Const(TaggedValue::Null))),

            other => Err(EvalError::Op(format!("unknown judgment op `{other}`"))),
        }
    }

    fn apply_arrow(&self, _arrow: &ArrowVal, _arg: Val) -> Option<Result<Val, EvalError>> {
        // Body evaluation is the bound eval engine's job — wired when
        // `NbeTypecheckEngine` adopts this handler (the `EvalEngine`
        // boundary). `None` = residual closure for now.
        None
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
            vec![
                ct(2),
                jop("nominal", vec![], Some("type.Int".into())),
            ],
            Some("vec.Vec".into()),
        );
        let b = jop(
            "former",
            vec![
                ct(2),
                jop("nominal", vec![], Some("type.Int".into())),
            ],
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
}
