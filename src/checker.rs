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

/// The canonical `type.*` seed for a builtin type tag (the `__tag__`
/// spellings of legacy type values, and the `type.*` tags of the tagged
/// model). Unknown tags pass through as their own seed — they are former
/// heads, not builtin nominals.
pub fn canonical_tag_seed(tag: &str) -> &str {
    match tag {
        "int" | "type.int" => "type.Int",
        "nat" | "type.nat" => "type.Nat",
        "float" | "type.float" => "type.Float",
        "number" | "type.number" => "type.Number",
        "str" | "string" | "type.string" | "type.str" => "type.Str",
        "bool" | "type.bool" => "type.Bool",
        "null" | "type.null" => "type.Null",
        "obj" | "object" | "type.obj" | "type.object" => "type.Obj",
        "array" | "type.array" => "type.Array",
        "bytes" | "type.bytes" => "type.Bytes",
        "arrow" | "function" | "type.arrow" => "type.Arrow",
        "resource" | "type.resource" => "type.Resource",
        "unit" | "type.unit" => "type.Unit",
        "value" | "any" | "type.value" => "type.Value",
        "type" | "type.type" => "type.Type",
        "grade" | "type.grade" => "type.Grade",
        other => other,
    }
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

/// A former's signature — the *rule* that types its applications.
///
/// Most formers are plain telescopes; `Pi` and `Hom` carry judgment
/// rules instead (ADR-0154 §4's universe-max; ADR-0153's dimension
/// boundary law) — judgment ops are the spec, so these are honest
/// variants, not hardcoded kind checks.
#[derive(Clone, Debug)]
pub enum FormerSig {
    /// Ordinary telescope checking.
    Tel(Telescope),
    /// `Pi(dom, cod)` — `Type(max(level(dom), level(cod)))`. A
    /// dependent `cod` (a `Val::Arrow`) is rejected for now: its body
    /// type needs engine evaluation of `body_def`
    /// (`OpHandler::apply_arrow` — the engine-side hook).
    Pi,
    /// `Hom(n, src, tgt)` — the dimension-indexed hom former
    /// (ADR-0153). Checked for `n ≤ 3`: `n = 1` requires `src`/`tgt`
    /// to be types (`Type(max)` result); `n = 2` requires both to be
    /// 1-cells over convertible boundaries (`infer → Hom(1, a, b)`
    /// with `a ≡ a'`, `b ≡ b'`) or both objects (the degenerate-cell
    /// reading — `a ~[2]~> b` on types is the equation/`refl` type);
    /// `n = 3` requires both to be 2-cells over convertible arrows —
    /// the same boundary law one level up (`tgt_1 = src_1` recursion,
    /// [`Checker::cell_faces_at`]). `n ≥ 4` stores unchecked — the
    /// ADR-0153 honest bound ("parses, stores, does not check").
    Hom,
    /// `cell.comp(k, a, b)` — the `a >>>[k] b` composite *cell* value
    /// (a term-level former application; ADR-0153 §2). Its type is
    /// decided by the globular boundary law — see
    /// [`Checker::infer_comp_axis`].
    Comp,
    /// `cell.id(k, x)` — the identity k-cell on `x` (`id_k`). Its type
    /// is `Hom(k, x, x)` — see [`Checker::infer_id`].
    Id,
    /// `IO(sig)` — the typed-effect former (ADR-0153 §4): an `IO` type
    /// is an effect object whose index is the op-signature record.
    /// `IO` bare (arity 0) is `IO(⊤)` — the top signature, accepting
    /// any. Subtyping is record-width on the sig index — see
    /// [`Checker::subtype`].
    Io,
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
    /// A `>>>`/`cell.comp` operand is not a cell — its inferred type is
    /// not a `Hom` former application.
    NotACell(String),
    /// A composition axis is out of range for the operands' cell
    /// dimension (`k > m` — there is no axis-k boundary).
    Dim {
        /// The requested composition axis.
        axis: u64,
        /// The operands' cell dimension.
        dim: u64,
    },
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
            Self::NotACell(v) => write!(f, "expected a cell (Hom-typed), got {v}"),
            Self::Dim { axis, dim } => write!(
                f,
                "axis-{axis} composition requires cells of dimension > {axis} \
                 (operand dimension {dim})"
            ),
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
    /// Former signatures, keyed by `Sym::handle`.
    pub formers: HashMap<u64, FormerSig>,
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
    /// A fresh context with no bindings and the builtin former rules
    /// (`type.pi`, `type.hom`) registered — seeded syms, idempotent.
    /// The builtin `type.*` nominal graph (`Nat <: Int <: Number` and
    /// the `ty` links, ADR-0154 §2) is ensured so literal-meta solves
    /// and nominal subtyping see declared links regardless of mint
    /// order.
    pub fn new() -> Self {
        handl_value::ensure_builtin_type_nominals();
        let mut c = Self::default();
        c.declare_pi();
        c.declare_hom();
        c.declare_cell_formers();
        // `IO(sig)` — the typed-effect former (ADR-0153 §4).
        c.formers.insert(type_sym("type.io").handle(), FormerSig::Io);
        c
    }

    /// Register the cell formers `cell.comp`/`cell.id` (ADR-0153 §2) —
    /// term-level formers: an application is a cell *value*, typed by
    /// the `>>>[k]` boundary law / `id_k` rule.
    pub fn declare_cell_formers(&mut self) {
        self.formers
            .insert(type_sym("cell.comp").handle(), FormerSig::Comp);
        self.formers
            .insert(type_sym("cell.id").handle(), FormerSig::Id);
    }

    /// Register the `Pi` former rule (`type.pi`).
    pub fn declare_pi(&mut self) -> Rc<Sym> {
        let s = type_sym("type.pi");
        self.formers.insert(s.handle(), FormerSig::Pi);
        s
    }

    /// Register the `Hom` former rule (`type.hom`, ADR-0153).
    pub fn declare_hom(&mut self) -> Rc<Sym> {
        let s = type_sym("type.hom");
        self.formers.insert(s.handle(), FormerSig::Hom);
        s
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
            TaggedValue::Float(_) => MetaConstraint::DescendantOf(type_sym("type.Float")),
            _ => MetaConstraint::DescendantOf(type_sym("type.Number")),
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
        let af = self.metas.force_head_rc(a);
        let bf = self.metas.force_head_rc(b);
        match (af.as_val(), bf.as_val()) {
            (Val::Universe(i), Val::Universe(j)) => i <= j,
            (Val::Nominal(x), Val::Nominal(y)) => x.has_ancestor(y),
            // `IO(sa) <: IO(sb)` — signature subtyping: `sb` must name
            // at least the ops `sa` does (record-width on the sig
            // index, `sb <: sa`). An arity-0 expected (`IO` bare =
            // `IO(⊤)`) accepts any signature; an arity-0 actual is ⊤
            // itself and only satisfies ⊤.
            (Val::Former { head: ha, args: aa }, Val::Former { head: hb, args: ab })
                if ha.handle() == hb.handle()
                    && ha.handle() == type_sym("type.io").handle() =>
            {
                match (aa.first(), ab.first()) {
                    (_, None) => true,
                    (None, Some(_)) => false,
                    (Some(sa), Some(sb)) => self.sig_width_subtype(sb, sa),
                }
            }
            // A bare nominal can inhabit a family application only if it
            // *is* that family's head with no indices — `Nat <: Nat()` —
            // covered by conv elsewhere; nominal-vs-former is not a
            // subtype step.
            _ => false,
        }
    }

    /// Record-width subtyping on `IO` signature objects: `wider`
    /// supplies every op `narrower` names, with convertible field
    /// values (op signatures compare by `conv` — structural, per
    /// ADR-0154 §3). Non-record sigs fall back to conversion.
    fn sig_width_subtype(&mut self, wider: &Val, narrower: &Val) -> bool {
        let w = self.metas.force_head_rc(wider);
        let n = self.metas.force_head_rc(narrower);
        match (w.as_val(), n.as_val()) {
            (
                Val::Const(TaggedValue::Obj(wm)),
                Val::Const(TaggedValue::Obj(nm)),
            ) => nm.iter().all(|(k, v)| {
                wm.get(k).is_some_and(|wv| {
                    self.conv(&Val::Const(wv.clone()), &Val::Const(v.clone()))
                })
            }),
            _ => self.conv(w.as_val(), n.as_val()),
        }
    }

    /// Synthesize the type of `v`.
    pub fn infer(&mut self, v: &Val) -> Result<Val, TcError> {
        let vf = self.metas.force_head_rc(v);
        match vf.as_val() {
            // `Type(i) : Type(i+1)` (0154 §4).
            Val::Universe(i) => Ok(Val::Universe(*i + 1)),
            Val::Nominal(s) => Ok(self
                .nominal_ty
                .get(&s.handle())
                .cloned()
                .unwrap_or(Val::Universe(0))),
            Val::Former { head, args } => self.infer_former(head, args),
            Val::Rigid { lvl, spine } if spine.is_empty() => self
                .lookup(*lvl)
                .cloned()
                .ok_or(TcError::CannotInfer("unbound rigid")),
            Val::Flex { meta, spine } if spine.is_empty() => self
                .meta_tys
                .get(*meta as usize)
                .cloned()
                .ok_or(TcError::UnsolvedMeta(*meta)),
            Val::Unfold { value, .. } => value
                .get()
                .ok_or(TcError::CannotInfer("unfold-forced"))
                .and_then(|v| self.infer(v)),
            Val::Const(tv) => self.infer_const(tv),
            Val::Arrow(a) => {
                // Saturate the arrow on rigid binders (fresh domain
                // metas for the domains), evaluate the `{"__ast__"}`
                // body through the engine-owned arrow eval, and infer
                // the body's type — a `Pi` tower over the params.
                // Stuck bodies report `CannotInfer` honestly.
                let mark = self.mark();
                let mut doms: Vec<Val> = Vec::new();
                let mut extra: Vec<(Rc<str>, Val)> = Vec::new();
                for p in a.params.iter() {
                    let m = self.fresh_meta(Val::Universe(0));
                    let r = self.bind(p.as_str(), self.meta_val(m));
                    doms.push(self.meta_val(m));
                    extra.push((Rc::from(p.as_str()), r));
                }
                let lvl = self.lvl();
                let body_ty = crate::judgments::eval_arrow_body(a, &extra, None, lvl)
                    .ok()
                    .and_then(|body| self.infer(&body).ok());
                self.restore(mark);
                match body_ty {
                    Some(mut t) => {
                        for d in doms.into_iter().rev() {
                            t = Val::Former {
                                head: type_sym("type.pi"),
                                args: vec![d, t],
                            };
                        }
                        Ok(t)
                    }
                    None => Err(TcError::CannotInfer("arrow")),
                }
            }
            _ => Err(TcError::CannotInfer("spined neutral")),
        }
    }

    /// The type of a boundary constant (0154 §5 for numerics; the
    /// scalar tags for the rest).
    fn infer_const(&mut self, tv: &TaggedValue) -> Result<Val, TcError> {
        Ok(match tv {
            TaggedValue::Int(_) | TaggedValue::Float(_) => self.infer_literal(tv),
            TaggedValue::Str(_) => Val::Nominal(type_sym("type.Str")),
            TaggedValue::Bool(_) => Val::Nominal(type_sym("type.Bool")),
            TaggedValue::Null => Val::Nominal(type_sym("type.Null")),
            TaggedValue::Arr(_) => Val::Nominal(type_sym("type.Array")),
            TaggedValue::Obj(_) => Val::Nominal(type_sym("type.Obj")),
            TaggedValue::Bytes(_) => Val::Nominal(type_sym("type.Bytes")),
            TaggedValue::Sym(s) => {
                return self.infer(&Val::Nominal(s.clone()));
            }
            _ => return Err(TcError::CannotInfer("const")),
        })
    }

    /// Synthesize the result type of a former application —
    /// dispatch on the former's rule (telescope / Pi / Hom).
    pub fn infer_former(&mut self, head: &Rc<Sym>, args: &[Val]) -> Result<Val, TcError> {
        let sig = self
            .formers
            .get(&head.handle())
            .cloned()
            .ok_or_else(|| TcError::UnknownFormer(head.name().to_string()))?;
        match sig {
            FormerSig::Tel(tel) => self.infer_telescope(head, &tel, args),
            FormerSig::Pi => {
                let [dom, cod] = args else {
                    return Err(TcError::Arity {
                        former: head.name().to_string(),
                        want: 2,
                        got: args.len(),
                    });
                };
                // Dependent codomain: evaluate the arrow body on the
                // domain's rigid binder through the engine-owned arrow
                // eval (`{"__ast__"}`); stuck bodies report honestly.
                let cod = match cod {
                    Val::Arrow(a) => {
                        let mark = self.mark();
                        let x = self.bind("dom", dom.clone());
                        let lvl = self.lvl();
                        let applied = crate::judgments::eval_arrow_body(
                            a,
                            &[(Rc::from("dom"), x)],
                            None,
                            lvl,
                        );
                        self.restore(mark);
                        applied.map_err(|_| TcError::CannotInfer("dependent pi codomain"))?
                    }
                    other => other.clone(),
                };
                self.infer_pi(dom, &cod)
            }
            FormerSig::Hom => self.infer_hom(head, args),
            FormerSig::Comp => {
                let [k, a, b] = args else {
                    return Err(TcError::Arity {
                        former: head.name().to_string(),
                        want: 3,
                        got: args.len(),
                    });
                };
                self.infer_comp_axis(k, a, b)
            }
            FormerSig::Io => {
                // `IO(sig) : Type(0)` — the sig index is an
                // op-signature record; arity 0 is `IO(⊤)`.
                if args.len() > 1 {
                    return Err(TcError::Arity {
                        former: head.name().to_string(),
                        want: 1,
                        got: args.len(),
                    });
                }
                Ok(Val::Universe(0))
            }
            FormerSig::Id => match args {
                [x] => self.infer_id(None, x),
                [k, x] => self.infer_id(Some(k), x),
                _ => {
                    return Err(TcError::Arity {
                        former: head.name().to_string(),
                        want: 2,
                        got: args.len(),
                    })
                }
            },
        }
    }

    /// `Hom(n, src, tgt)` typing (ADR-0153), checked for `n ≤ 3`:
    /// - `n = 1`: `src`/`tgt` are types — `Hom(1, A, B) : Type(max)`.
    /// - `n = 2`: `src`/`tgt` are 1-cells over *convertible*
    ///   boundaries — `infer(src) ≡ infer(tgt) ≡ Hom(1, a, b)`;
    ///   `Hom(2, f, g) : Type(level of that hom)`.
    /// - `n = 3`: `src`/`tgt` are 2-cells over convertible arrows —
    ///   `infer ≡ Hom(2, f, g)` with `f ≡ f'`, `g ≡ g'`; the same law
    ///   one level up (coherence cells — `e >>>[2] eInv ~[3]~> id(f)`
    ///   *checks*).
    /// - `n ≥ 4` and non-literal `n`: stores unchecked (honest bound).
    pub fn infer_hom(&mut self, head: &Rc<Sym>, args: &[Val]) -> Result<Val, TcError> {
        let [n, src, tgt] = args else {
            return Err(TcError::Arity {
                former: head.name().to_string(),
                want: 3,
                got: args.len(),
            });
        };
        let k = match self.metas.force_head(n) {
            Val::Const(TaggedValue::Int(k)) => *k,
            _ => return Ok(Val::Universe(0)), // non-literal dim: store only
        };
        match k {
            1 => {
                let l = self.universe_level(src)?;
                let r = self.universe_level(tgt)?;
                Ok(Val::Universe(l.max(r)))
            }
            2 | 3 => {
                let st = self.infer(src)?;
                let tt = self.infer(tgt)?;
                // Both endpoints must be *parallel* (n-1)-cells: their
                // (n-2)-faces must convert. `Hom(2, f, g)` needs
                // `f, g : Hom(1, a, b)` over the same objects; `Hom(3,
                // e, e')` needs `e, e' : Hom(2, f, g)` over the same
                // arrows — the same law one level up. Objects and
                // lower cells promote to degenerate cells (iterated
                // identities — `a ~[n]~> b` requires `a ≡ b`, the refl
                // reading); an open endpoint (`Hom(?m, ·, ·)` or a
                // meta-typed value) is *stuck*, not rejected.
                let j = (k - 2) as u64;
                let (Some(sa), Some(ta)) = (
                    self.cell_faces_at(src, &st, j)?,
                    self.cell_faces_at(tgt, &tt, j)?,
                ) else {
                    return Ok(Val::Universe(0));
                };
                if !self.conv(&sa.0, &ta.0) || !self.conv(&sa.1, &ta.1) {
                    return Err(TcError::Mismatch {
                        got: self.show(&st),
                        want: self.show(&tt),
                    });
                }
                let l = self.universe_level(&st)?;
                Ok(Val::Universe(l))
            }
            // Honest bound (ADR-0153): n ≥ 4 parses and stores; the
            // boundary law past level 3 is not checked.
            _ => Ok(Val::Universe(0)),
        }
    }

    /// The `j`-face pair `(src_j, tgt_j)` of a cell value `v` whose
    /// inferred type is `vt` — the globular boundary at level `j`
    /// (ADR-0153, the `tgt_j = src_j` recursion). For a `Hom(m, s,
    /// t)`-typed cell: `j = m-1` reads the cell's own faces `(s, t)`;
    /// `j < m-1` recurses through `s` (source and target share all
    /// lower faces — globularity); `j ≥ m` is the degenerate
    /// self-boundary `(v, v)` — a lower cell promoted to an iterated
    /// identity. Objects (`vt` a `Universe`/`Nominal` type) are their
    /// own boundary at every level. `Ok(None)` is *stuck*: an open
    /// `Hom(?m, ·, ·)` dimension or an open endpoint type — store,
    /// don't decide (the open-cell convention). `Err` is a decided
    /// non-cell.
    fn cell_faces_at(
        &mut self,
        v: &Val,
        vt: &Val,
        j: u64,
    ) -> Result<Option<(Val, Val)>, TcError> {
        if let Val::Former { head, args } = self.metas.force_head(vt) {
            if head.handle() == type_sym("type.hom").handle() && args.len() == 3 {
                return match self.metas.force_head(&args[0]) {
                    Val::Const(TaggedValue::Int(m)) => {
                        let Ok(m) = u64::try_from(*m) else {
                            return Ok(None);
                        };
                        if j + 1 == m {
                            Ok(Some((args[1].clone(), args[2].clone())))
                        } else if j + 1 < m {
                            // Deeper faces live inside the (m-1)-cell
                            // boundaries — read through the source.
                            let s = args[1].clone();
                            match self.infer(&s) {
                                Ok(st) => self.cell_faces_at(&s, &st, j),
                                // An uninferrable boundary endpoint —
                                // stuck, not a decided non-cell.
                                Err(_) => Ok(None),
                            }
                        } else {
                            Ok(Some((v.clone(), v.clone())))
                        }
                    }
                    // Open dimension: at `j = 0` keep the endpoint's
                    // own faces (the n=2 rule read `Hom(?m, s, t)`
                    // boundaries as `(s, t)`); deeper levels can't be
                    // decided — stuck.
                    _ if j == 0 => Ok(Some((args[1].clone(), args[2].clone()))),
                    _ => Ok(None),
                };
            }
        }
        match self.metas.force_head(vt) {
            Val::Universe(_) | Val::Nominal(_) => Ok(Some((v.clone(), v.clone()))),
            Val::Flex { .. } | Val::Unfold { .. } => Ok(None),
            _ => Err(TcError::NotACell(self.show(v))),
        }
    }

    /// The `j`-face of a cell value `v` (type `vt`) on one side —
    /// `tgt_j(v)` when `tgt_side`, else `src_j(v)` — the deep-axis
    /// boundary law's projection. `Ok(None)` is stuck.
    fn cell_face_at(
        &mut self,
        v: &Val,
        vt: &Val,
        j: u64,
        tgt_side: bool,
    ) -> Result<Option<Val>, TcError> {
        Ok(self
            .cell_faces_at(v, vt, j)?
            .map(|(s, t)| if tgt_side { t } else { s }))
    }

    /// The globular 0-boundary of a cell value `v` whose inferred type
    /// is `vt`: for a `Hom(m, s, t)`-typed cell the boundary pair is
    /// `(s, t)`; an object (`vt` a `Universe`/nominal type) reads as a
    /// degenerate cell — `(v, v)`. `None` when `vt` is neither.
    fn cell_boundaries(&mut self, v: &Val, vt: &Val) -> Result<(Val, Val), TcError> {
        if let Val::Former { head, args } = self.metas.force_head(vt) {
            if head.handle() == type_sym("type.hom").handle() && args.len() == 3 {
                return Ok((args[1].clone(), args[2].clone()));
            }
        }
        // Objects are degenerate cells: `a` is its own boundary.
        if matches!(
            self.metas.force_head(vt),
            Val::Universe(_) | Val::Nominal(_)
        ) {
            return Ok((v.clone(), v.clone()));
        }
        Err(TcError::NotACell(self.show(v)))
    }

    /// The literal dimension of a `Hom(m, ·, ·)` former application —
    /// `None` for non-`Hom` types or non-literal `m`.
    fn hom_dim(&self, vt: &Val) -> Option<u64> {
        if let Val::Former { head, args } = self.metas.force_head(vt) {
            if head.handle() == type_sym("type.hom").handle() && args.len() == 3 {
                if let Val::Const(TaggedValue::Int(m)) = self.metas.force_head(&args[0]) {
                    return u64::try_from(*m).ok();
                }
            }
        }
        None
    }

    /// `a >>>[k] b` typing (ADR-0153 §2): compose cells along their
    /// shared k-boundary.
    ///
    /// - `k ≤ 1` (the surface `>>>`/`>>>[1]`): ordinary arrow
    ///   composition — `Hom(1, a, b) ∘ Hom(1, b, c) → Hom(1, a, c)`.
    ///   Objects promote to degenerate 1-cells, so `A >>> B` reads
    ///   `id_A ∘₀ id_B` and requires `A ≡ B`.
    /// - `k = 2`: vertical composition of 2-cells —
    ///   `Hom(2, f, g) ∘ Hom(2, g, h) → Hom(2, f, h)`; the boundary law
    ///   `tgt_1(fst) ≡ src_1(snd)` is enforced by `conv`.
    /// - `k = 3`: vertical composition of 3-cells —
    ///   `Hom(3, p, q) ∘ Hom(3, q, r) → Hom(3, p, r)`; the same law at
    ///   the 2-face level. Deep axes (`k < d-1`) compose the k-faces
    ///   (`tgt_k ≡ src_k` via [`Checker::cell_face_at`]) and paste the
    ///   (d-1)-faces pointwise.
    /// - `k ≥ 4` and non-literal `k` store unchecked (honest bound —
    ///   the same `m ≥ 4` bound `infer_hom` keeps).
    ///
    /// The composite's *value* is the former application
    /// `cell.comp(k, a, b)` the caller stores; this rule decides its
    /// type. Mixed-dimension operands compose at the higher dimension
    /// (the lower cell is read at its own level — the degenerate
    /// promotion is already inside its `Hom` boundary).
    pub fn infer_comp_axis(&mut self, k: &Val, a: &Val, b: &Val) -> Result<Val, TcError> {
        let ta = self.infer(a)?;
        let tb = self.infer(b)?;
        self.comp_axis_of(k, a, &ta, b, &tb)
    }

    /// The boundary law on already-inferred operand *types* — the
    /// tagged-checker bridge: operand *values* are unavailable there,
    /// so a non-`Hom` operand type stands in as its own degenerate
    /// object (the degenerate-cell reading, self-bounded). Decided
    /// errors ([`TcError::Dim`], [`TcError::Mismatch`]) are real
    /// rejections; anything else leaves the composite stored
    /// unchecked — the honest bound.
    pub fn infer_comp_axis_tys(&mut self, k: &Val, ta: &Val, tb: &Val) -> Result<Val, TcError> {
        let (a, b) = (ta.clone(), tb.clone());
        self.comp_axis_of(k, &a, ta, &b, tb)
    }

    /// The shared core of `infer_comp_axis`/`infer_comp_axis_tys`:
    /// `a`/`b` are the operand values used for degenerate promotion
    /// faces, `ta`/`tb` their inferred types.
    fn comp_axis_of(
        &mut self,
        k: &Val,
        a: &Val,
        ta: &Val,
        b: &Val,
        tb: &Val,
    ) -> Result<Val, TcError> {
        let axis = match self.metas.force_head(k) {
            Val::Const(TaggedValue::Int(n)) => u64::try_from(*n).ok(),
            _ => None,
        };
        let (ma, mb) = (self.hom_dim(&ta), self.hom_dim(&tb));
        // Effective dimension: a cell is at least its declared `Hom`
        // level; an object is a degenerate 1-cell.
        let d = ma.into_iter().chain(mb).max().unwrap_or(1).max(1);
        // The operand's `(d-1)`-faces: a `Hom(d, s, t)`-typed cell's own
        // faces; a lower-dimension cell (or object) promotes to the
        // degenerate `d`-cell `id` — faces are the operand itself.
        let face = |v: &Val, vt: &Val, lvl: Option<u64>| -> (Val, Val) {
            if lvl == Some(d) {
                if let Val::Former { args, .. } = self.metas.force_head(vt) {
                    return (args[1].clone(), args[2].clone());
                }
            }
            (v.clone(), v.clone())
        };
        let (sa, ta_b) = face(a, &ta, ma);
        let (sb, tb_b) = face(b, &tb, mb);
        // Non-cell operands: `infer` gave neither a `Hom` nor a type —
        // only the unchecked regime tolerates that.
        let cells_ok = matches!(ma, Some(_)) || self.cell_boundaries(a, &ta).is_ok();
        let cells_ok = cells_ok
            && (matches!(mb, Some(_)) || self.cell_boundaries(b, &tb).is_ok());
        let k = match axis {
            Some(k) => k,
            // Non-literal axis: store the composite unchecked at the
            // operand dimension.
            None => {
                return Ok(Val::Former {
                    head: type_sym("type.hom"),
                    args: vec![k.clone(), sa, tb_b],
                });
            }
        };
        if k >= 4 || d >= 4 {
            // Honest bound: cells past dimension 3 store unchecked.
            return Ok(Val::Former {
                head: type_sym("type.hom"),
                args: vec![Val::Const(TaggedValue::Int(d.max(k) as i64)), sa, tb_b],
            });
        }
        // `Hom(?d, ·, ·)` — a cell type whose dimension is a meta or
        // otherwise non-literal: the operand is a cell but its level
        // is *open* (`d` above is only the 1-default). `k > d` and the
        // boundary law can't be decided → store the composite at the
        // operand's open dimension — the honest bound.
        let open_dim = |vt: &Val, m: Option<u64>| -> Option<Val> {
            if m.is_some() {
                return None;
            }
            if let Val::Former { head, args } = self.metas.force_head(vt) {
                if head.handle() == type_sym("type.hom").handle() && args.len() == 3 {
                    return Some(args[0].clone());
                }
            }
            None
        };
        if let Some(dv) = open_dim(&ta, ma).or_else(|| open_dim(&tb, mb)) {
            return Ok(Val::Former {
                head: type_sym("type.hom"),
                args: vec![dv, sa, tb_b],
            });
        }
        if !cells_ok {
            let bad = if self.cell_boundaries(a, &ta).is_err() && ma.is_none() {
                self.show(&ta)
            } else {
                self.show(&tb)
            };
            return Err(TcError::NotACell(bad));
        }
        if k > d {
            return Err(TcError::Dim { axis: k, dim: d });
        }
        // `>>>[k]` at the *top* boundary of an m-cell means `k = m-1`
        // under the globular convention; the surface level-indexed
        // spelling `>>>[m]` denotes the same top-boundary composition
        // (ADR-0153: `e >>>[2] eInv` on 2-cells). Both readings — and
        // `>>>`/`>>>[1]` on 1-cells — compose along the top faces.
        let (src, tgt) = if k + 1 >= d {
            // top-boundary composition: `tgt(fst) ≡ src(snd)`.
            if !self.conv(&ta_b, &sb) {
                return Err(TcError::Mismatch {
                    got: self.show(&ta_b),
                    want: self.show(&sb),
                });
            }
            (sa, tb_b)
        } else {
            // Deep-axis composition `∘ₖ` (`k < d-1`): the globular law
            // is `tgt_k(a) ≡ src_k(b)` — the *k-faces*, which for
            // `k < d-1` live inside the operands' boundary cells (the
            // `cell_face_at` recursion), not the top faces. `k = 0` on
            // `Hom(2, f, g) ∘ Hom(2, f', g')` pastes along the shared
            // object `tgt_0 = src_0` and gives `Hom(2, f ∘₀ f', g ∘₀
            // g')`; the composite's own (d-1)-faces compose pointwise.
            let tk = self.cell_face_at(a, &ta, k, true)?;
            let sk = self.cell_face_at(b, &tb, k, false)?;
            let (Some(tk), Some(sk)) = (tk, sk) else {
                // Open k-faces — stuck; store the composite at the
                // operand dimension (the honest bound).
                return Ok(Val::Former {
                    head: type_sym("type.hom"),
                    args: vec![Val::Const(TaggedValue::Int(d as i64)), sa, tb_b],
                });
            };
            if !self.conv(&tk, &sk) {
                return Err(TcError::Mismatch {
                    got: self.show(&tk),
                    want: self.show(&sk),
                });
            }
            let sub = |x: Val, y: Val| Val::Former {
                head: type_sym("cell.comp"),
                args: vec![Val::Const(TaggedValue::Int(k as i64)), x, y],
            };
            (sub(sa, sb), sub(ta_b, tb_b))
        };
        Ok(Val::Former {
            head: type_sym("type.hom"),
            args: vec![Val::Const(TaggedValue::Int(d as i64)), src, tgt],
        })
    }

    /// `id_k(x)` — the identity cell at level `k` (ADR-0153 §2): the
    /// `Former{cell.id}` value carries `[k, x]`; its type is
    /// `Hom(k, x, x)`. `x` must be a `(k-1)`-cell (or a degenerate
    /// lower cell): `id_1` takes objects (`Hom(1, A, A)`), `id_2` takes
    /// 1-cells (`Hom(2, f, f)`); `id` on an object at level 2 is the
    /// `refl` equation `Hom(2, a, a)`; `id_3` takes 2-cells
    /// (`Hom(3, e, e)`) or lower cells by promotion (`Hom(3, f, f)`
    /// on a 1-cell, `Hom(3, a, a)` on an object). `k` absent defaults
    /// to `level(x) + 1` for cells and `2` for objects; `k ≥ 4` stores
    /// `Hom(k, x, x)` unchecked (honest bound).
    pub fn infer_id(&mut self, k: Option<&Val>, x: &Val) -> Result<Val, TcError> {
        let xt = self.infer(x)?;
        let m = self.hom_dim(&xt).unwrap_or(0); // 0 = object/0-cell
        let lvl = match k.and_then(|kv| match self.metas.force_head(kv) {
            Val::Const(TaggedValue::Int(n)) => u64::try_from(*n).ok(),
            _ => None,
        }) {
            Some(n) => n.max(1),
            None => {
                if m == 0 {
                    2
                } else {
                    m + 1
                }
            }
        };
        if lvl <= 3 {
            if m == 0 {
                // Object endpoint: `x` must be a type (`id_k` on an
                // object is the iterated refl — `Hom(k, a, a)`).
                self.universe_level(x)?;
            } else if m > lvl - 1 {
                return Err(TcError::Dim { axis: lvl, dim: m });
            }
        }
        Ok(Val::Former {
            head: type_sym("type.hom"),
            args: vec![
                Val::Const(TaggedValue::Int(lvl as i64)),
                x.clone(),
                x.clone(),
            ],
        })
    }

    /// Telescope checking with progressive index substitution.
    fn infer_telescope(
        &mut self,
        head: &Rc<Sym>,
        tel: &Telescope,
        args: &[Val],
    ) -> Result<Val, TcError> {
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
        self.formers.insert(s.handle(), FormerSig::Tel(tel));
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

    /// Snapshot the meta store (metas + the parallel `meta_tys`/
    /// `meta_constraints` tables) for backtracking a domain conversion
    /// attempt — the tagged bridge allocates fresh domain metas while
    /// lifting and a failed `conv` must not leak them.
    pub fn store_mark(&self) -> usize {
        self.metas.mark()
    }

    /// Restore the meta store to `mark` (see [`Self::store_mark`]).
    /// Solutions to pre-mark metas are monotone and kept (the NbE
    /// convention); only post-mark allocations drop.
    pub fn store_restore(&mut self, mark: usize) {
        self.metas.truncate(mark);
        self.meta_tys.truncate(mark);
        self.meta_constraints.truncate(mark);
    }

    /// Default every still-unsolved meta to its constraint anchor
    /// (0154 §5: unresolved literal metas default — integer literals →
    /// `Nat`, fraction literals → `Float`). The `type.Number` anchor
    /// defaults to `type.Nat` (its integral subtype); every other anchor
    /// is already its own default.
    pub fn default_unsolved(&mut self) {
        for (i, c) in self.meta_constraints.iter().enumerate() {
            if self.metas.solution(i as MetaId).is_some() {
                continue;
            }
            if let MetaConstraint::DescendantOf(sym) = c {
                // 0154 §5: the numeric anchor's default is Nat.
                let sym = if sym == &type_sym("type.Number") {
                    type_sym("type.Nat")
                } else {
                    sym.clone()
                };
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
            // The `__tag__` bridge (ADR-0154 §3 / spec WS-E): a legacy
            // tagged type value `{"__tag__": t}` lifts to `Nominal(t)`;
            // one with a payload lifts to `Former{t, <fields sorted by
            // key>}` — the tag names the former, the payload its
            // indices. `__tag__`/`__registry__`/`body_closure` are
            // structural keys, not indices.
            if let Some(TaggedValue::Str(tag)) = m.get("__tag__") {
                // `type.universe{level}` is `Type(i)` — a domain
                // universe, not a former application.
                if tag.as_ref() == "type.universe" {
                    let i = m
                        .get("level")
                        .and_then(|l| l.as_i64())
                        .unwrap_or(0)
                        .max(0) as u32;
                    return Val::Universe(i);
                }
                let mut keyed: Vec<(&Rc<str>, &TaggedValue)> = m
                    .iter()
                    .filter(|(k, _)| {
                        k.as_ref() != "__tag__"
                            && k.as_ref() != "__registry__"
                            && k.as_ref() != "body_closure"
                    })
                    .collect();
                if keyed.is_empty() {
                    // Builtin type tags lift to the canonical `type.*`
                    // seeded nominals (ADR-0154 §1 — the same seeds the
                    // runtime `type.<Name>` ops mint), so a legacy
                    // `{"__tag__": "int"}` value meets `type.Int`.
                    return Val::Nominal(type_sym(canonical_tag_seed(tag)));
                }
                keyed.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
                return Val::Former {
                    head: type_sym(tag),
                    args: keyed.into_iter().map(|(_, v)| lift(lvl, v)).collect(),
                };
            }
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
        Sym::seeded("type.Number")
    }

    fn int() -> Rc<Sym> {
        Sym::seeded_with_parent("type.Int", &number())
    }

    fn nat() -> Rc<Sym> {
        Sym::seeded_with_parent("type.Nat", &int())
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
        // `1 : Str` fails — `type.Str` is not a number descendant.
        let strs = Sym::seeded("type.Str");
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
        // 0154 §5: an unresolved integer literal defaults to `Nat`
        // (not its `type.Number` anchor).
        assert!(matches!(
            sol,
            Val::Nominal(s) if s.name() == "Nat"
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
        assert!(matches!(t, Val::Nominal(s) if s.name() == "Int"));
        let _y = c.bind("y", Val::Nominal(Sym::seeded("type.Str")));
        // The outer binder's level still resolves after extension.
        let t = c.infer(&x).unwrap();
        assert!(matches!(t, Val::Nominal(s) if s.name() == "Int"));
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
                params: vec![("fields".into(), Val::Nominal(type_sym("type.Obj")))],
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
    fn hom_dim_index_n_le_2() {
        // ADR-0153: `~[n]~>` = `Hom(n, ·, ·)`, checked for n ≤ 2.
        let mut c = Checker::new();
        let hom = type_sym("type.hom");
        let f = |a: Val, b: Val| Val::Former {
            head: hom.clone(),
            args: vec![Val::Const(TaggedValue::Int(1)), a, b],
        };
        // Hom(1, Int, Nat) : Type(0) — both endpoints are types.
        let h1 = f(Val::Nominal(int()), Val::Nominal(nat()));
        assert!(matches!(c.infer(&h1).unwrap(), Val::Universe(0)));
        // Hom(2, f, g) with f,g : Hom(1, Int, Nat) — same boundary.
        let arrow_ty = f(Val::Nominal(int()), Val::Nominal(nat()));
        let x = c.bind("f", arrow_ty.clone());
        let y = c.bind("g", arrow_ty);
        let h2 = Val::Former {
            head: hom.clone(),
            args: vec![Val::Const(TaggedValue::Int(2)), x, y],
        };
        assert!(matches!(c.infer(&h2).unwrap(), Val::Universe(0)));
        // Mismatched boundaries reject: g : Hom(1, Nat, Int).
        let z = c.bind(
            "h",
            f(Val::Nominal(nat()), Val::Nominal(int())),
        );
        let x0 = Val::Rigid {
            lvl: 1,
            spine: Spine::default(),
        };
        let bad = Val::Former {
            head: hom.clone(),
            args: vec![Val::Const(TaggedValue::Int(2)), x0, z],
        };
        assert!(c.infer(&bad).is_err());
        // n = 3 checks: `Hom(3, Int, Int)` is the object-equation
        // reading (degenerate promotion — endpoints `(Int, Int)`)
        // and `Hom(3, Int, Nat)` rejects on the boundary law.
        let h3 = Val::Former {
            head: hom.clone(),
            args: vec![
                Val::Const(TaggedValue::Int(3)),
                Val::Nominal(int()),
                Val::Nominal(int()),
            ],
        };
        assert!(c.infer(&h3).is_ok());
        nat();
        let h3_bad = Val::Former {
            head: hom.clone(),
            args: vec![
                Val::Const(TaggedValue::Int(3)),
                Val::Nominal(int()),
                Val::Nominal(nat()),
            ],
        };
        assert!(matches!(
            c.infer(&h3_bad),
            Err(TcError::Mismatch { .. })
        ));
        // n ≥ 4: stores unchecked (honest bound).
        let h4 = Val::Former {
            head: hom,
            args: vec![
                Val::Const(TaggedValue::Int(4)),
                Val::Nominal(int()),
                Val::Nominal(nat()),
            ],
        };
        assert!(matches!(c.infer(&h4).unwrap(), Val::Universe(0)));
    }

    #[test]
    fn pi_former_uses_max_level() {
        let mut c = Checker::new();
        // `Former{pi, [Type(0), Int]}` — infer via the Pi rule:
        // max(1, 0) = 1.
        let p = Val::Former {
            head: type_sym("type.pi"),
            args: vec![Val::Universe(0), Val::Nominal(int())],
        };
        assert!(matches!(c.infer(&p).unwrap(), Val::Universe(1)));
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

    // --- `>>>[k]` / `id_k` (ADR-0153 §2) ----------------------------------

    fn hom(n: i64, s: Val, t: Val) -> Val {
        Val::Former {
            head: type_sym("type.hom"),
            args: vec![Val::Const(TaggedValue::Int(n)), s, t],
        }
    }

    fn comp(k: i64, a: Val, b: Val) -> Val {
        Val::Former {
            head: type_sym("cell.comp"),
            args: vec![Val::Const(TaggedValue::Int(k)), a, b],
        }
    }

    fn idf(k: i64, x: Val) -> Val {
        Val::Former {
            head: type_sym("cell.id"),
            args: vec![Val::Const(TaggedValue::Int(k)), x],
        }
    }

    fn bool_s() -> Rc<Sym> {
        Sym::seeded("type.Bool")
    }

    #[test]
    fn comp_axis_0_and_1_compose_1cells() {
        let mut c = Checker::new();
        int();
        nat();
        bool_s();
        // f : Int ~> Nat, g : Nat ~> Bool — `f >>> g : Hom(1, Int, Bool)`
        // at both spellings (axis 0 = `>>>`, axis 1 = level-indexed).
        for k in [0, 1] {
            let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
            let g = c.bind("g", hom(1, Val::Nominal(nat()), Val::Nominal(bool_s())));
            let t = c.infer(&comp(k, f.clone(), g.clone())).unwrap();
            assert!(
                c.conv(&t, &hom(1, Val::Nominal(int()), Val::Nominal(bool_s()))),
                "axis {k}: got {}",
                c.show(&t)
            );
        }
        // Boundary violation: h : Bool ~> Nat — `f >>>[1] h` fails.
        let f = c.bind("f2", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let h = c.bind("h", hom(1, Val::Nominal(bool_s()), Val::Nominal(nat())));
        assert!(matches!(
            c.infer(&comp(1, f, h)),
            Err(TcError::Mismatch { .. })
        ));
    }

    #[test]
    fn comp_axis_2_is_vertical_2cell_composition() {
        let mut c = Checker::new();
        int();
        nat();
        bool_s();
        let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let g = c.bind("g", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let h = c.bind("h", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        // e : Hom(2, f, g), e2 : Hom(2, g, h) — `e >>>[2] e2 : Hom(2, f, h)`.
        let e = c.bind("e", hom(2, f.clone(), g.clone()));
        let e2 = c.bind("e2", hom(2, g.clone(), h.clone()));
        let t = c.infer(&comp(2, e.clone(), e2)).unwrap();
        assert!(c.conv(&t, &hom(2, f.clone(), h)), "got {}", c.show(&t));
        // Boundary violation: e3 : Hom(2, f, f) — `e >>>[2] e3` needs
        // src ≡ g, gets f.
        let e3 = c.bind("e3", hom(2, f.clone(), f));
        assert!(matches!(
            c.infer(&comp(2, e.clone(), e3)),
            Err(TcError::Mismatch { .. })
        ));
    }

    #[test]
    fn comp_promotes_lower_dim_operand() {
        let mut c = Checker::new();
        int();
        nat();
        // f (a 1-cell) `>>>[2]` with e : Hom(2, f, g): id₂(f) ∘ e — the
        // boundary law conv(f, src(e)) holds.
        let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let g = c.bind("g", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let e = c.bind("e", hom(2, f.clone(), g.clone()));
        let t = c.infer(&comp(2, f.clone(), e)).unwrap();
        assert!(c.conv(&t, &hom(2, f, g)), "got {}", c.show(&t));
    }

    #[test]
    fn comp_level_3_vertical_composition() {
        let mut c = Checker::new();
        int();
        nat();
        bool_s();
        // p : Hom(2, f, g), q : Hom(2, g, h), r : Hom(2, h, j) —
        // `x : Hom(3, p, q)`, `y : Hom(3, q, r)` compose at axis 3
        // (top boundary) into `Hom(3, p, r)`; axis 2 spells the same
        // top-boundary composition on 3-cells.
        let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let g = c.bind("g", hom(1, Val::Nominal(nat()), Val::Nominal(bool_s())));
        let h = c.bind("h", hom(1, Val::Nominal(bool_s()), Val::Nominal(int())));
        let j = c.bind("j", hom(1, Val::Nominal(int()), Val::Nominal(int())));
        let p = c.bind("p", hom(2, f.clone(), g.clone()));
        let q = c.bind("q", hom(2, g.clone(), h.clone()));
        let r = c.bind("r", hom(2, h.clone(), j.clone()));
        let x = c.bind("x", hom(3, p.clone(), q.clone()));
        let y = c.bind("y", hom(3, q.clone(), r.clone()));
        for k in [2, 3] {
            let t = c.infer(&comp(k, x.clone(), y.clone())).unwrap();
            assert!(
                c.conv(&t, &hom(3, p.clone(), r.clone())),
                "axis {k}: got {}",
                c.show(&t)
            );
        }
        // Boundary violation at level 3: `x2 : Hom(3, p, p)` —
        // `x >>>[3] x2` needs src ≡ q, gets p.
        let x2 = c.bind("x2", hom(3, p.clone(), p.clone()));
        assert!(matches!(
            c.infer(&comp(3, x.clone(), x2)),
            Err(TcError::Mismatch { .. })
        ));
        // `>>>[3]` on 2-cells is a dimension error — no axis-3
        // boundary on a 2-cell (same law as `>>>[2]` on 1-cells).
        let e = c.bind("e", hom(2, f.clone(), g.clone()));
        let e2 = c.bind("e2", hom(2, g.clone(), f.clone()));
        assert!(matches!(
            c.infer(&comp(3, e, e2)),
            Err(TcError::Dim { .. })
        ));
        // `>>>[2]` on 1-cells stays a dimension error.
        assert!(matches!(
            c.infer(&comp(2, f.clone(), g.clone())),
            Err(TcError::Dim { .. })
        ));
    }

    #[test]
    fn id_levels() {
        let mut c = Checker::new();
        int();
        // id₁ on an object: Hom(1, Int, Int).
        let t = c.infer(&idf(1, Val::Nominal(int()))).unwrap();
        assert!(c.conv(&t, &hom(1, Val::Nominal(int()), Val::Nominal(int()))));
        // id₂ on an arrow: Hom(2, f, f).
        let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(int())));
        let t = c.infer(&idf(2, f.clone())).unwrap();
        assert!(c.conv(&t, &hom(2, f.clone(), f.clone())));
        // id₂ on an object: refl — Hom(2, Int, Int).
        let t = c.infer(&idf(2, Val::Nominal(int()))).unwrap();
        assert!(c.conv(&t, &hom(2, Val::Nominal(int()), Val::Nominal(int()))));
        // And that Hom(2, Int, Int) itself now *checks* as a type —
        // the degenerate-endpoint reading; `Hom(2, Int, Nat)` rejects.
        assert!(c.infer(&hom(2, Val::Nominal(int()), Val::Nominal(int()))).is_ok());
        nat();
        assert!(c
            .infer(&hom(2, Val::Nominal(int()), Val::Nominal(nat())))
            .is_err());
        // id₁ on a 1-cell is a dimension error (id of a 1-cell is a 2-cell).
        assert!(matches!(
            c.infer(&idf(1, f)),
            Err(TcError::Dim { .. })
        ));
        // id₃ on an object: refl² — `Hom(3, Int, Int)`, itself a
        // checked type under the n = 3 boundary law.
        let t = c.infer(&idf(3, Val::Nominal(int()))).unwrap();
        assert!(c.conv(
            &t,
            &hom(3, Val::Nominal(int()), Val::Nominal(int()))
        ));
        // id₃ on a 1-cell promotes: `Hom(3, f, f)`; on a 2-cell:
        // `Hom(3, e, e)`; on a 3-cell it is a dimension error; id₄
        // stores unchecked.
        let f2 = c.bind("f2", hom(1, Val::Nominal(int()), Val::Nominal(int())));
        let t = c.infer(&idf(3, f2.clone())).unwrap();
        assert!(c.conv(&t, &hom(3, f2.clone(), f2.clone())));
        let e = c.bind("e", hom(2, f2.clone(), f2.clone()));
        let t = c.infer(&idf(3, e.clone())).unwrap();
        assert!(c.conv(&t, &hom(3, e.clone(), e.clone())));
        let x3 = c.bind("x3", hom(3, e.clone(), e.clone()));
        assert!(matches!(
            c.infer(&idf(3, x3)),
            Err(TcError::Dim { .. })
        ));
        let t = c.infer(&idf(4, e.clone())).unwrap();
        assert!(c.conv(&t, &hom(4, e.clone(), e.clone())));
    }

    #[test]
    fn hom3_boundary_law_and_coherence() {
        let mut c = Checker::new();
        int();
        nat();
        // e : Hom(2, f, g), e' : Hom(2, f, g) — `Hom(3, e, e')`
        // checks: endpoints' 1-faces convert.
        let f = c.bind("f", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let g = c.bind("g", hom(1, Val::Nominal(int()), Val::Nominal(nat())));
        let e = c.bind("e", hom(2, f.clone(), g.clone()));
        let e2 = c.bind("e2", hom(2, f.clone(), g.clone()));
        assert!(matches!(
            c.infer(&hom(3, e.clone(), e2.clone())).unwrap(),
            Val::Universe(0)
        ));
        // Mismatched boundaries reject: `e3 : Hom(2, g, f)` shares
        // no faces with `Hom(2, f, g)`.
        let e3 = c.bind("e3", hom(2, g.clone(), f.clone()));
        assert!(matches!(
            c.infer(&hom(3, e.clone(), e3)),
            Err(TcError::Mismatch { .. })
        ));
        // The coherence cell: `e >>>[2] eInv ~[3]~> id(f)` *checks*.
        let e_inv = c.bind("eInv", hom(2, g.clone(), f.clone()));
        let comp_v = comp(2, e.clone(), e_inv.clone());
        let id_f = idf(2, f);
        assert!(matches!(
            c.infer(&hom(3, comp_v, id_f)).unwrap(),
            Val::Universe(0)
        ));
        // And the other side: `eInv >>>[2] e ~[3]~> id(g)`.
        let comp_v2 = comp(2, e_inv, e2.clone());
        let id_g = idf(2, g.clone());
        assert!(matches!(
            c.infer(&hom(3, comp_v2, id_g)).unwrap(),
            Val::Universe(0)
        ));
    }

    #[test]
    fn hom3_interchange_boundary_law() {
        // `(al =>| al2) =>> (be =>| be2) ~[3]~>
        //   (al =>> be) =>| (al2 =>> be2)` — the interchange law as a
        // `Hom(3)` type: both sides *check* and have convertible
        // `Hom(2)` boundaries, so the coherence cell typechecks. The
        // two composites are NOT conv-equal as `cell.comp` data — the
        // law is asserted (proof-relevant 3-cell), not definitional.
        let mut c = Checker::new();
        int();
        nat();
        bool_s();
        // 1-cells: f1, g1, g3 : a→b; f2, g2, g4 : b→c.
        let ab = || hom(1, Val::Nominal(int()), Val::Nominal(nat()));
        let bc = || hom(1, Val::Nominal(nat()), Val::Nominal(bool_s()));
        let f1 = c.bind("f1", ab());
        let g1 = c.bind("g1", ab());
        let g3 = c.bind("g3", ab());
        let f2 = c.bind("f2", bc());
        let g2 = c.bind("g2", bc());
        let g4 = c.bind("g4", bc());
        let al = c.bind("al", hom(2, f1.clone(), g1.clone()));
        let al2 = c.bind("al2", hom(2, f2.clone(), g2.clone()));
        let be = c.bind("be", hom(2, g1.clone(), g3.clone()));
        let be2 = c.bind("be2", hom(2, g2.clone(), g4.clone()));
        let lhs = comp(2, comp(0, al.clone(), al2.clone()), comp(0, be.clone(), be2.clone()));
        let rhs = comp(0, comp(2, al, be), comp(2, al2, be2));
        // Both sides type to the same `Hom(2)` — parallel 2-cells —
        // so `Hom(3, lhs, rhs)` checks.
        let tl = c.infer(&lhs).unwrap();
        let tr = c.infer(&rhs).unwrap();
        assert!(c.conv(&tl, &tr), "{} vs {}", c.show(&tl), c.show(&tr));
        assert!(matches!(
            c.infer(&hom(3, lhs.clone(), rhs.clone())).unwrap(),
            Val::Universe(0)
        ));
        // But the composites differ as cell data — interchange is a
        // stated law (a 3-cell witness), not a conversion rule.
        assert!(!c.conv(&lhs, &rhs));
    }
}
