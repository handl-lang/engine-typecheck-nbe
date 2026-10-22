//! The `Val`-domain checking core (phase-1 spec WS-B; ADR-0154 §§2–5,
//! ADR-0152 §1, ADR-0153 §1).
//!
//! This module is the semantic kernel the `lang.types[kind]` judgment
//! ops dispatch into: types are *values* in `handl_value::nbe::Val` —
//! nominal `sym` formers with index args — and checking is
//!
//! - `infer` — synthesize the type of a domain value;
//! - `check` — `infer` then `subtype` (directed checking);
//! - `conv`  — nominal heads + structural bodies + meta solving
//!   (delegates to [`handl_value::nbe::conv`]);
//! - `subtype` — `conv` else the *parent* walk over nominal heads
//!   (ADR-0154 §2: subtyping is the `parent` link; membership `x : T`
//!   is the `type` link — two relations, not one chain).
//!
//! No host closures anywhere: a former's signature is an explicit
//! [`Telescope`] — `params` + `result` data where dependent positions
//! mention earlier params as `Rigid` level heads — and `Pi`'s universe
//! rule is `max(domain, codomain)` computed by [`Checker::infer_pi`]
//! after the caller has inferred the body under an extended Γ.
//!
//! Literals are constrained metas (ADR-0154 §5): `infer` on an
//! integer/float literal mints `Flex` constrained
//! `DescendantOf(type.number)`/`DescendantOf(type.float)`; unresolved
//! metas default to the constraint anchor (`default_unsolved`).

use std::collections::HashMap;
use std::rc::Rc;

use handl_value::nbe::{self, ArrowVal, Lvl, MetaConstraint, MetaId, MetaStore, Spine, Val};
use handl_value::{Sym, TaggedValue};

/// The canonical seed for a `type.*` nominal (ADR-0154 §1 — declared
/// nominals mint deterministically by qualified name).
pub fn type_sym(seed: &str) -> Rc<Sym> {
    Sym::seeded(seed)
}

/// A former signature — an explicit telescope (data, not a closure).
///
/// `params[i]` is `(name, type)`; a param's type may mention earlier
/// params as `Rigid{lvl: <position>}` heads (level = telescope
/// position). `result` likewise.
///
/// Scoping note: telescope-position rigids share the `Val::Rigid`
/// representation with Γ binders. A `Telescope` is therefore declared
/// *closed* — top-level formers (`data` declarations elaborate at
/// statement top level, where Γ is empty) — so every `Rigid` inside a
/// telescope is a param position. A former declared under open Γ
/// binders would need shifted levels; that is not yet needed.
/// Applying the former checks each
/// argument against the substituted param type and substitutes into
/// `result` — dependent *result* indices are ordinary unification of
/// the stored `Rigid`s (ADR-0152: index refinement is unification).
#[derive(Clone, Debug)]
pub struct Telescope {
    /// Index/parameter bindings, in order.
    pub params: Vec<(Rc<str>, Val)>,
    /// The family's result type (typically `Val::Universe(i)` for a
    /// type former; a `Former` value for a constructor former).
    pub result: Val,
}

/// Errors the `Val` checker reports.
#[derive(Debug, Clone, PartialEq)]
pub enum TcError {
    /// The value's type cannot be synthesized (needs an annotation).
    CannotInfer(&'static str),
    /// `got` is not a subtype of `want` — values shown via `quote`.
    Mismatch {
        /// The inferred type, rendered.
        got: String,
        /// The expected type, rendered.
        want: String,
    },
    /// A metavariable was still open where a decided value was needed.
    UnsolvedMeta(MetaId),
    /// A former was applied to the wrong number of indices.
    Arity {
        /// The former's display name.
        former: String,
        /// Indices declared.
        want: usize,
        /// Indices supplied.
        got: usize,
    },
    /// A former with no registered signature was applied.
    UnknownFormer(String),
    /// A value was expected to be a type (infer → `Type(i)`).
    NotAType(String),
}

impl std::fmt::Display for TcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CannotInfer(k) => write!(f, "cannot synthesize a type for `{k}`"),
            Self::Mismatch { got, want } => {
                write!(f, "type mismatch: got {got}, want {want}")
            }
            Self::UnsolvedMeta(m) => write!(f, "unsolved metavariable ?{m}"),
            Self::Arity { former, want, got } => {
                write!(f, "former `{former}` expects {want} indices, got {got}")
            }
            Self::UnknownFormer(n) => write!(f, "unknown former `{n}`"),
            Self::NotAType(v) => write!(f, "expected a type, got {v}"),
        }
    }
}

impl std::error::Error for TcError {}

/// The `Val`-domain checking context: Γ + meta store + the former and
/// nominal registries. Γ is indexed by de Bruijn *level* — `gamma[i]`
/// is the type of the binder at level `i` (levels count outward, the
/// smalltt convention the domain already follows).
#[derive(Debug, Default)]
pub struct Checker {
    /// The metavariable store.
    pub metas: MetaStore,
    /// Typing context: `(name, type)` per binder level.
    pub gamma: Vec<(Rc<str>, Val)>,
    /// Former telescopes, keyed by `Sym::handle`.
    pub formers: HashMap<u64, Telescope>,
    /// Declared types of nominals, keyed by `Sym::handle`
    /// (`Int : Type(0)` &c.); a nominal with no entry defaults to
    /// `Type(0)` — types are values and a plain declared type lives at
    /// level 0.
    pub nominal_ty: HashMap<u64, Val>,
    /// The type each meta ranges over (parallel to the store).
    meta_tys: Vec<Val>,
    /// The constraint each fresh meta was minted with — parallel to
    /// the store (the store's entries are private; defaults need the
    /// constraint back).
    meta_constraints: Vec<MetaConstraint>,
}

impl Checker {
    /// A fresh context with no bindings or declarations.
    pub fn new() -> Self {
        Self::default()
    }

    /// The ambient level — the number of binders in Γ.
    pub fn lvl(&self) -> Lvl {
        self.gamma.len() as Lvl
    }

    /// Extend Γ with a binder of the given type; returns the binder's
    /// `Rigid` head value (level = position).
    pub fn bind(&mut self, name: impl Into<Rc<str>>, ty: Val) -> Val {
        let lvl = self.lvl();
        self.gamma.push((name.into(), ty));
        Val::Rigid {
            lvl,
            spine: Spine::default(),
        }
    }

    /// Pop binders down to `mark` (a prior [`Checker::mark`]).
    pub fn mark(&self) -> usize {
        self.gamma.len()
    }

    /// Restore Γ to `mark`, dropping later binders.
    pub fn restore(&mut self, mark: usize) {
        self.gamma.truncate(mark);
    }

    /// Look up a binder's type by level.
    pub fn lookup(&self, lvl: Lvl) -> Option<&Val> {
        self.gamma.get(lvl as usize).map(|(_, t)| t)
    }

    /// Look up a binder's type by name (innermost last wins).
    pub fn lookup_name(&self, name: &str) -> Option<&Val> {
        self.gamma
            .iter()
            .rev()
            .find(|(n, _)| n.as_ref() == name)
            .map(|(_, t)| t)
    }

    /// Mint a fresh meta ranging over `ty` (unconstrained).
    pub fn fresh_meta(&mut self, ty: Val) -> MetaId {
        let m = self.metas.fresh();
        self.meta_tys.push(ty);
        self.meta_constraints.push(MetaConstraint::None);
        m
    }

    /// Mint a fresh meta ranging over `ty` with a solving constraint.
    pub fn fresh_meta_constrained(&mut self, ty: Val, c: MetaConstraint) -> MetaId {
        let m = self.metas.fresh_constrained(c.clone());
        self.meta_tys.push(ty);
        self.meta_constraints.push(c);
        m
    }

    /// The meta as a bare `Flex` head.
    pub fn meta_val(&self, m: MetaId) -> Val {
        Val::Flex {
            meta: m,
            spine: Spine::default(),
        }
    }

    /// The declared constraint of a meta.
    pub fn meta_constraint(&self, m: MetaId) -> Option<&MetaConstraint> {
        self.meta_constraints.get(m as usize)
    }

    /// The type a meta ranges over.
    pub fn meta_ty(&self, m: MetaId) -> Option<&Val> {
        self.meta_tys.get(m as usize)
    }

    /// A literal's type: a constrained meta (ADR-0154 §5). Integer
    /// literals range over `type.number` descendants, float literals
    /// over `type.float` descendants.
    pub fn infer_literal(&mut self, tv: &TaggedValue) -> Val {
        let c = match tv {
            TaggedValue::Float(_) => MetaConstraint::DescendantOf(type_sym("type.float")),
            _ => MetaConstraint::DescendantOf(type_sym("type.number")),
        };
        let m = self.fresh_meta_constrained(Val::Universe(0), c);
        self.meta_val(m)
    }

    /// `conv` at the ambient level — nominal heads, structural bodies,
    /// meta solving (kernel `conv`).
    pub fn conv(&mut self, a: &Val, b: &Val) -> bool {
        let lvl = self.lvl();
        nbe::conv(&mut self.metas, lvl, a, b)
    }

    /// Subtyping `a <: b` (ADR-0154 §2): conversion, else the `parent`
    /// walk on nominal heads. Universes are cumulative
    /// (`Type(i) <: Type(j)` for `i ≤ j`); indexed families are
    /// *invariant* in their indices (`Former` args convert, they do
    /// not subtype — `Vec(Nat)` is not a `Vec(Int)`).
    pub fn subtype(&mut self, a: &Val, b: &Val) -> bool {
        if self.conv(a, b) {
            return true;
        }
        let a = self.metas.force_head(a).clone();
        let b = self.metas.force_head(b).clone();
        match (&a, &b) {
            (Val::Universe(i), Val::Universe(j)) => i <= j,
            (Val::Nominal(x), Val::Nominal(y)) => x.has_ancestor(y),
            // A bare nominal can inhabit a family application only if it
            // *is* that family's head with no indices — `Nat <: Nat()` —
            // covered by conv elsewhere; nominal-vs-former is not a
            // subtype step.
            _ => false,
        }
    }

    /// Synthesize the type of `v`.
    pub fn infer(&mut self, v: &Val) -> Result<Val, TcError> {
        let v = self.metas.force_head(v).clone();
        match v {
            // `Type(i) : Type(i+1)` (0154 §4).
            Val::Universe(i) => Ok(Val::Universe(i + 1)),
            Val::Nominal(s) => Ok(self
                .nominal_ty
                .get(&s.handle())
                .cloned()
                .unwrap_or(Val::Universe(0))),
            Val::Former { head, args } => self.infer_former(&head, &args),
            Val::Rigid { lvl, spine } if spine.is_empty() => self
                .lookup(lvl)
                .cloned()
                .ok_or(TcError::CannotInfer("unbound rigid")),
            Val::Flex { meta, spine } if spine.is_empty() => self
                .meta_tys
                .get(meta as usize)
                .cloned()
                .ok_or(TcError::UnsolvedMeta(meta)),
            Val::Unfold { value, .. } => value
                .get()
                .ok_or(TcError::CannotInfer("unfold-forced"))
                .and_then(|v| self.infer(v)),
            Val::Const(tv) => self.infer_const(&tv),
            Val::Arrow(_) => Err(TcError::CannotInfer("arrow")),
            _ => Err(TcError::CannotInfer("spined neutral")),
        }
    }

    /// The type of a boundary constant (0154 §5 for numerics; the
    /// scalar tags for the rest).
    fn infer_const(&mut self, tv: &TaggedValue) -> Result<Val, TcError> {
        Ok(match tv {
            TaggedValue::Int(_) | TaggedValue::Float(_) => self.infer_literal(tv),
            TaggedValue::Str(_) => Val::Nominal(type_sym("type.string")),
            TaggedValue::Bool(_) => Val::Nominal(type_sym("type.bool")),
            TaggedValue::Null => Val::Nominal(type_sym("type.null")),
            TaggedValue::Arr(_) => Val::Nominal(type_sym("type.array")),
            TaggedValue::Obj(_) => Val::Nominal(type_sym("type.obj")),
            TaggedValue::Bytes(_) => Val::Nominal(type_sym("type.bytes")),
            TaggedValue::Sym(s) => {
                return self.infer(&Val::Nominal(s.clone()));
            }
            _ => return Err(TcError::CannotInfer("const")),
        })
    }

    /// Synthesize the result type of a former application —
    /// telescope checking with progressive index substitution.
    pub fn infer_former(&mut self, head: &Rc<Sym>, args: &[Val]) -> Result<Val, TcError> {
        let tel = self
            .formers
            .get(&head.handle())
            .cloned()
            .ok_or_else(|| TcError::UnknownFormer(head.name().to_string()))?;
        if tel.params.len() != args.len() {
            return Err(TcError::Arity {
                former: head.name().to_string(),
                want: tel.params.len(),
                got: args.len(),
            });
        }
        let mut subst_map: Vec<Val> = Vec::with_capacity(args.len());
        for ((name, ty), arg) in tel.params.iter().zip(args.iter()) {
            let ty = subst(&subst_map, ty);
            self.check(arg, &ty).map_err(|e| match e {
                TcError::Mismatch { got, want } => TcError::Mismatch {
                    got: format!("{name} : {got}"),
                    want,
                },
                e => e,
            })?;
            subst_map.push(arg.clone());
        }
        Ok(subst(&subst_map, &tel.result))
    }

    /// Directed checking: `v` against `ty` — infer then subtype.
    pub fn check(&mut self, v: &Val, ty: &Val) -> Result<(), TcError> {
        let got = self.infer(v)?;
        if self.subtype(&got, ty) {
            Ok(())
        } else {
            Err(TcError::Mismatch {
                got: self.show(&got),
                want: self.show(ty),
            })
        }
    }

    /// `Pi`'s universe rule (0154 §4): `Pi(A, B) : Type(max(l_A, l_B))`.
    /// `dom`/`cod` are type *values*; the caller obtains `cod` by
    /// inferring the body under Γ extended by the domain binder —
    /// body evaluation is the engine's job (defunctionalized arrows
    /// are interpreted, never host-called).
    pub fn infer_pi(&mut self, dom: &Val, cod: &Val) -> Result<Val, TcError> {
        let l = self.universe_level(dom)?;
        let r = self.universe_level(cod)?;
        Ok(Val::Universe(l.max(r)))
    }

    /// The universe level a type value lives at — `infer` must yield
    /// `Type(i)`.
    pub fn universe_level(&mut self, v: &Val) -> Result<u32, TcError> {
        match self.infer(v)? {
            Val::Universe(i) => Ok(i),
            other => Err(TcError::NotAType(self.show(&other))),
        }
    }

    /// Declare a nominal former: seeds the sym and registers its
    /// telescope. `Nat`, `Vec`, `Hom`, `Pi`… — one mechanism
    /// (0152/0153: every type-level entity is an index on a nominal
    /// former).
    pub fn declare_former(&mut self, seed: &str, tel: Telescope) -> Rc<Sym> {
        let s = Sym::seeded(seed);
        self.formers.insert(s.handle(), tel);
        s
    }

    /// Declare a nominal type with an optional parent (the `parent`
    /// link — subtyping, 0154 §2) and its type (default `Type(0)`).
    pub fn declare_nominal(
        &mut self,
        seed: &str,
        parent: Option<&Sym>,
        ty: Option<Val>,
    ) -> Rc<Sym> {
        let s = match parent {
            Some(p) => Sym::seeded_with_parent(seed, p),
            None => Sym::seeded(seed),
        };
        self.nominal_ty
            .insert(s.handle(), ty.unwrap_or(Val::Universe(0)));
        s
    }

    /// Default every still-unsolved meta to its constraint anchor
    /// (0154 §5: unresolved literal metas default — integer literals
    /// take the numeric anchor until a `Nat` former lands with WS-C).
    pub fn default_unsolved(&mut self) {
        for (i, c) in self.meta_constraints.iter().enumerate() {
            if self.metas.solution(i as MetaId).is_some() {
                continue;
            }
            if let MetaConstraint::DescendantOf(sym) = c {
                let sym = sym.clone();
                self.metas.solve(i as MetaId, Val::Nominal(sym));
            }
        }
    }

    /// Display a value via `quote` (diagnostics).
    pub fn show(&self, v: &Val) -> String {
        let tv = nbe::quote(&self.metas, self.lvl(), v);
        serde_json::to_string(&handl_value::tagged_to_value(&tv))
            .unwrap_or_else(|_| "<unprintable>".into())
    }
}

/// Substitute telescope parameters: `Rigid{lvl: i}` heads denote
/// param `i` of the telescope being instantiated; `spine` arguments
/// are re-applied after substitution.
pub fn subst(map: &[Val], v: &Val) -> Val {
    match v {
        Val::Rigid { lvl, spine } => {
            let head = map
                .get(*lvl as usize)
                .cloned()
                .unwrap_or_else(|| v.clone());
            spine.0.iter().fold(head, |f, a| spine_push(f, subst(map, a)))
        }
        Val::Flex { meta, spine } => Val::Flex {
            meta: *meta,
            spine: Spine(spine.0.iter().map(|a| subst(map, a)).collect()),
        },
        Val::Former { head, args } => Val::Former {
            head: head.clone(),
            args: args.iter().map(|a| subst(map, a)).collect(),
        },
        Val::Arrow(a) => Val::Arrow(Rc::new(ArrowVal {
            params: a.params.clone(),
            body_def: a.body_def.clone(),
            binds: Rc::new(
                a.binds
                    .iter()
                    .map(|(k, v)| (k.clone(), subst(map, v)))
                    .collect(),
            ),
        })),
        Val::Unfold { name, spine, value } => Val::Unfold {
            name: name.clone(),
            spine: Spine(spine.0.iter().map(|a| subst(map, a)).collect()),
            value: value.clone(),
        },
        Val::Nominal(_) | Val::Universe(_) | Val::Const(_) => v.clone(),
    }
}

/// Push one application argument onto a value head — spine extension
/// for neutrals, index extension for formers (the kernel-level half
/// of `hnl_engine::nbe::apply`, which also needs the op handler).
fn spine_push(f: Val, arg: Val) -> Val {
    match f {
        Val::Rigid { lvl, spine } => {
            let mut s = spine;
            s.push(arg);
            Val::Rigid { lvl, spine: s }
        }
        Val::Flex { meta, spine } => {
            let mut s = spine;
            s.push(arg);
            Val::Flex { meta, spine: s }
        }
        Val::Former { head, args } => {
            let mut args = args;
            args.push(arg);
            Val::Former { head, args }
        }
        Val::Unfold { name, spine, value } => {
            let mut s = spine;
            s.push(arg);
            Val::Unfold {
                name,
                spine: s,
                value,
            }
        }
        // A substituted head that is already a full value (e.g. an
        // arrow residual) cannot take further domain-level spine
        // arguments here — body application is the engine's job.
        other => other,
    }
}

/// Lift a boundary [`TaggedValue`] into the domain — the inverse of
/// [`nbe::quote`] (`__former__`/`__neutral__`/`__meta__`/`__type__`
/// encodings) plus the `__former__`/`__ctor__` surface spellings of
/// ADR-0152. `lvl` is the ambient level — quote reified rigids as de
/// Bruijn *indices*, lift restores levels (`l = lvl - 1 - i`).
pub fn lift(lvl: Lvl, tv: &TaggedValue) -> Val {
    match tv {
        TaggedValue::Sym(s) => Val::Nominal(s.clone()),
        TaggedValue::Arrow {
            params,
            body_def,
            binds,
            ..
        } => Val::Arrow(Rc::new(ArrowVal {
            params: params.clone(),
            body_def: (**body_def).clone(),
            binds: Rc::new(
                binds
                    .iter()
                    .map(|(k, v)| (k.clone(), lift(lvl, v)))
                    .collect(),
            ),
        })),
        TaggedValue::Obj(m) => {
            if let Some(f) = m.get("__former__") {
                // Two spellings: `{"__former__": {"sym": <sym>, "args":
                // [...]}}` (quote's encoding) and `{"__former__": <sym>,
                // "indices": [...]}` (0152's surface object).
                if let TaggedValue::Obj(fm) = f {
                    if let Some(TaggedValue::Sym(head)) = fm.get("sym") {
                        let args = fm
                            .get("args")
                            .or_else(|| fm.get("indices"))
                            .map(|a| lift_list(lvl, a))
                            .unwrap_or_default();
                        return Val::Former {
                            head: head.clone(),
                            args,
                        };
                    }
                }
                if let TaggedValue::Sym(head) = f {
                    let args = m
                        .get("indices")
                        .or_else(|| m.get("args"))
                        .map(|a| lift_list(lvl, a))
                        .unwrap_or_default();
                    return Val::Former {
                        head: head.clone(),
                        args,
                    };
                }
            }
            if let Some(t) = m.get("__type__") {
                if let TaggedValue::Int(i) = t {
                    return Val::Universe((*i).max(0) as u32);
                }
            }
            if let Some(TaggedValue::Int(id)) = m.get("__meta__") {
                return Val::Flex {
                    meta: *id as MetaId,
                    spine: Spine::default(),
                };
            }
            if let Some(TaggedValue::Obj(n)) = m.get("__neutral__") {
                let spine = Spine(
                    n.get("spine")
                        .map(|s| lift_list(lvl, s))
                        .unwrap_or_default(),
                );
                match n.get("head").and_then(|h| h.as_str()) {
                    Some("rigid") => {
                        let i = n
                            .get("index")
                            .and_then(|i| i.as_i64())
                            .unwrap_or(0)
                            .max(0) as u32;
                        return Val::Rigid {
                            lvl: lvl.saturating_sub(i + 1),
                            spine,
                        };
                    }
                    Some("meta") => {
                        let id = n
                            .get("id")
                            .and_then(|i| i.as_i64())
                            .unwrap_or(0)
                            .max(0) as MetaId;
                        return Val::Flex { meta: id, spine };
                    }
                    Some("unfold") => {
                        let name = n
                            .get("name")
                            .and_then(|s| s.as_str())
                            .unwrap_or("_")
                            .into();
                        return Val::Unfold {
                            name,
                            spine,
                            value: Rc::new(std::cell::OnceCell::new()),
                        };
                    }
                    _ => {}
                }
            }
            Val::Const(tv.clone())
        }
        _ => Val::Const(tv.clone()),
    }
}

fn lift_list(lvl: Lvl, tv: &TaggedValue) -> Vec<Val> {
    match tv {
        TaggedValue::Arr(xs) => xs.iter().map(|x| lift(lvl, x)).collect(),
        other => vec![lift(lvl, other)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number() -> Rc<Sym> {
        Sym::seeded("type.number")
    }

    fn int() -> Rc<Sym> {
        Sym::seeded_with_parent("type.int", &number())
    }

    fn nat() -> Rc<Sym> {
        Sym::seeded_with_parent("type.nat", &int())
    }

    #[test]
    fn universe_hierarchy() {
        let mut c = Checker::new();
        let t = c.infer(&Val::Universe(0)).unwrap();
        assert!(matches!(t, Val::Universe(1)));
        // Cumulativity: Type(0) <: Type(2).
        assert!(c.subtype(&Val::Universe(0), &Val::Universe(2)));
        assert!(!c.subtype(&Val::Universe(2), &Val::Universe(0)));
    }

    #[test]
    fn pi_level_is_max() {
        let mut c = Checker::new();
        int();
        // Pi(Int, Int) : Type(0); Pi(Type(0), Int) : Type(1).
        let p0 = c
            .infer_pi(&Val::Nominal(int()), &Val::Nominal(int()))
            .unwrap();
        assert!(matches!(p0, Val::Universe(0)));
        let p1 = c
            .infer_pi(&Val::Universe(0), &Val::Nominal(int()))
            .unwrap();
        assert!(matches!(p1, Val::Universe(1)));
    }

    #[test]
    fn literal_meta_solves_to_descendant() {
        let mut c = Checker::new();
        int();
        // `1 : Int` — the literal meta solves to a `type.number`
        // descendant (not via Nat <: Int subtyping).
        c.check(&Val::Const(TaggedValue::Int(1)), &Val::Nominal(int()))
            .unwrap();
        // `1 : Str` fails — `type.string` is not a number descendant.
        let strs = Sym::seeded("type.string");
        assert!(c
            .check(&Val::Const(TaggedValue::Int(1)), &Val::Nominal(strs))
            .is_err());
    }

    #[test]
    fn unsolved_literal_defaults_to_anchor() {
        let mut c = Checker::new();
        let m_ty = c.infer_literal(&TaggedValue::Int(3));
        let Val::Flex { meta, .. } = m_ty else {
            panic!("literal type is a flex meta")
        };
        c.default_unsolved();
        let sol = c.metas.solution(meta).unwrap();
        assert!(matches!(
            sol,
            Val::Nominal(s) if s.name() == "number"
        ));
    }

    #[test]
    fn former_application_telescope() {
        let mut c = Checker::new();
        int();
        let vec = c.declare_former(
            "vec.Vec",
            Telescope {
                params: vec![
                    ("n".into(), Val::Nominal(number())),
                    ("T".into(), Val::Universe(0)),
                ],
                result: Val::Universe(0),
            },
        );
        let app = Val::Former {
            head: vec.clone(),
            args: vec![Val::Const(TaggedValue::Int(2)), Val::Nominal(int())],
        };
        // Vec(2, Int) : Type(0).
        let t = c.infer(&app).unwrap();
        assert!(matches!(t, Val::Universe(0)));
        // Index equality is structural: Vec(2,Int) ≡ Vec(2,Int), ≢ Vec(3,Int).
        let app3 = Val::Former {
            head: vec.clone(),
            args: vec![Val::Const(TaggedValue::Int(3)), Val::Nominal(int())],
        };
        assert!(c.conv(&app, &app));
        assert!(!c.conv(&app, &app3));
        // Arity is checked.
        assert!(matches!(
            c.infer(&Val::Former {
                head: vec,
                args: vec![Val::Const(TaggedValue::Int(2))],
            }),
            Err(TcError::Arity { .. })
        ));
    }

    #[test]
    fn dependent_index_subst() {
        // Telescope where a later param's type mentions an earlier
        // index — `sig n x` with `x : Vec(n, Int)`.
        let mut c = Checker::new();
        int();
        let vec = c.declare_former(
            "vec.Vec",
            Telescope {
                params: vec![
                    ("n".into(), Val::Nominal(number())),
                    ("T".into(), Val::Universe(0)),
                ],
                result: Val::Universe(0),
            },
        );
        let sig = c.declare_former(
            "sig.Sig",
            Telescope {
                params: vec![
                    ("n".into(), Val::Nominal(number())),
                    (
                        "x".into(),
                        Val::Former {
                            head: vec.clone(),
                            args: vec![
                                Val::Rigid {
                                    lvl: 0,
                                    spine: Spine::default(),
                                },
                                Val::Nominal(int()),
                            ],
                        },
                    ),
                ],
                result: Val::Universe(0),
            },
        );
        // `x : Vec(n, Int)` takes a *value* of that type — supply
        // binder-bound rigids.
        let vec5 = Val::Former {
            head: vec.clone(),
            args: vec![Val::Const(TaggedValue::Int(5)), Val::Nominal(int())],
        };
        let x5 = c.bind("x5", vec5);
        let ok = Val::Former {
            head: sig.clone(),
            args: vec![Val::Const(TaggedValue::Int(5)), x5],
        };
        c.infer(&ok).unwrap();
        // `x : Vec(6, Int)` at `n := 5` must reject — index refinement
        // is ordinary unification of the substituted param type.
        let vec6 = Val::Former {
            head: vec.clone(),
            args: vec![Val::Const(TaggedValue::Int(6)), Val::Nominal(int())],
        };
        let x6 = c.bind("x6", vec6);
        let bad = Val::Former {
            head: sig.clone(),
            args: vec![Val::Const(TaggedValue::Int(5)), x6],
        };
        assert!(c.infer(&bad).is_err());
        // A meta index solves instead: `Sig(5, x)` with
        // `x : Vec(?m, Int)` solves ?m := 5.
        let m = c.fresh_meta(Val::Nominal(number()));
        let vecm = Val::Former {
            head: vec,
            args: vec![c.meta_val(m), Val::Nominal(int())],
        };
        let xm = c.bind("xm", vecm);
        let meta_app = Val::Former {
            head: sig,
            args: vec![Val::Const(TaggedValue::Int(5)), xm],
        };
        c.infer(&meta_app).unwrap();
        assert!(matches!(
            c.metas.solution(m),
            Some(Val::Const(TaggedValue::Int(5)))
        ));
    }

    #[test]
    fn nominal_subtype_via_parent() {
        let mut c = Checker::new();
        // Nat <: Int <: Number on the `parent` link.
        nat();
        assert!(c.subtype(&Val::Nominal(nat()), &Val::Nominal(int())));
        assert!(c.subtype(
            &Val::Nominal(nat()),
            &Val::Nominal(number())
        ));
        assert!(!c.subtype(
            &Val::Nominal(number()),
            &Val::Nominal(int())
        ));
    }

    #[test]
    fn gamma_binders_are_levels() {
        let mut c = Checker::new();
        int();
        let x = c.bind("x", Val::Nominal(int()));
        // `x : Int` — infer the rigid head.
        let t = c.infer(&x).unwrap();
        assert!(matches!(t, Val::Nominal(s) if s.name() == "int"));
        let _y = c.bind("y", Val::Nominal(Sym::seeded("type.string")));
        // The outer binder's level still resolves after extension.
        let t = c.infer(&x).unwrap();
        assert!(matches!(t, Val::Nominal(s) if s.name() == "int"));
    }

    #[test]
    fn import_binds_sig_record() {
        // `stmt.import` binds `m : Sig_m` — a module signature record
        // is a former over its field→type table (WS-B port of
        // `bind_imports_` → Γ).
        let mut c = Checker::new();
        int();
        let sig = c.declare_former(
            "type.sig",
            Telescope {
                params: vec![("fields".into(), Val::Nominal(type_sym("type.obj")))],
                result: Val::Universe(0),
            },
        );
        let fields = TaggedValue::from(serde_json::json!({"f": "Int"}));
        let sig_m = Val::Former {
            head: sig,
            args: vec![Val::Const(fields)],
        };
        let m = c.bind("m", sig_m);
        let t = c.infer(&m).unwrap();
        assert!(matches!(t, Val::Former { head, .. } if head.name() == "sig"));
        assert!(c.lookup_name("m").is_some());
    }

    #[test]
    fn lift_roundtrips_quote() {
        let vec = Sym::seeded("vec.Vec");
        let v = Val::Former {
            head: vec,
            args: vec![
                Val::Rigid {
                    lvl: 0,
                    spine: Spine::default(),
                },
                Val::Nominal(int()),
            ],
        };
        let store = MetaStore::default();
        let tv = nbe::quote(&store, 1, &v);
        let back = lift(1, &tv);
        let mut c = Checker::new();
        assert!(c.conv(&v, &back));
    }
}
