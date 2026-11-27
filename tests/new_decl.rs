//! `X := new T(args) { members }` — mode/algebra declaration
//! (graded-arrows.md §3 "Algebra declaration syntax", MTT migration).
//!
//! `synth_assign`'s `new` arm binds `X` as a `type.family` (so `X(p)`
//! mints `type.family_app{X, [p]}` — the call-head mode value) and
//! registers `{members}` as the `(X, [_]) ↦ dict` instance entry.
//! `Algebra` itself is M2's to define; these tests use a stub
//! record-family former (`Algebra<c>`) so `T(args)` elaborates.

use handl_language::{Ast, Language, Value};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex as lex_src;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest, TypecheckResult};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;

fn lang() -> Language {
    let kit = handl_kit_data_prelude::assemble().expect("canonical assembly");
    kit.lang().clone()
}

fn parse(lang: &Language, src: &str) -> Ast {
    PrattEngine::new()
        .parse(lang, &ParserRequest::new(lex_src(lang, src).expect("lex")))
        .expect("parse")
}

fn tc_full(src: &str) -> TypecheckResult {
    let lang = lang();
    let ast = parse(&lang, src);
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    nbe.typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck")
}

/// Stub `Algebra` former — a record-family in the `Eq<a>` spelling so
/// `Algebra(Nat)` elaborates to `type.family_app`. M2 replaces this
/// with the real algebra type; the `new` machinery only needs `T(args)`
/// to elaborate as a type.
const STUB: &str = "Algebra<c : Type> := {\n  mkAlgebra : (compose : c ~> c ~> c, unit : c, order : c ~> c ~> Bool) ~> Algebra\n}\n";

#[test]
fn new_decl_binds_mode_family_and_mode_value_types() {
    let src = format!(
        "{STUB}Cost := new Algebra(Nat) {{\n  compose: (a, b) ~> a + b,\n  unit: 0,\n  order: (a, b) ~> a <= b\n}}\nCost(3)\n"
    );
    let result = tc_full(&src);
    let typ = &result.typ;
    assert_eq!(
        typ.get("__tag__").and_then(Value::as_str),
        Some("type.family_app"),
        "Cost(3) synthesizes a family_app — got {typ}"
    );
    assert_eq!(
        typ.get("name").and_then(Value::as_str),
        Some("Cost"),
        "the mode value's head is the declared name — got {typ}"
    );
    let args = typ.get("args").and_then(Value::as_array).expect("args");
    assert_eq!(args.len(), 1, "Cost(3) carries one payload index");
}

#[test]
fn new_decl_registers_instance_for_head() {
    // Two `new` declarations over the same former must not collide —
    // the instance head is the declared name, not `T`. (Coherence is
    // per-(head, index-shape), so `Cost`/`Price` coexist.)
    let src = format!(
        "{STUB}Cost := new Algebra(Nat) {{ unit: 0 }}\nPrice := new Algebra(Nat) {{ unit: 0 }}\nCost(1)\n"
    );
    let result = tc_full(&src);
    let typ = &result.typ;
    assert_eq!(
        typ.get("name").and_then(Value::as_str),
        Some("Cost"),
        "second declaration did not disturb the first — got {typ}"
    );
}

#[test]
fn new_decl_bare_former_without_args() {
    // `new T { … }` — no argument list; `T` elaborates through the
    // family self-application and still binds a mode family.
    let src = format!(
        "{STUB}Session := new Algebra {{\n  compose: (a, b) ~> a,\n  unit: 0\n}}\nSession(0)\n"
    );
    let result = tc_full(&src);
    let typ = &result.typ;
    assert_eq!(
        typ.get("__tag__").and_then(Value::as_str),
        Some("type.family_app"),
        "Session(0) synthesizes a family_app — got {typ}"
    );
    assert_eq!(
        typ.get("name").and_then(Value::as_str),
        Some("Session"),
        "bare-former declaration still binds the declared name — got {typ}"
    );
}

#[test]
fn new_decl_duplicate_name_is_coherence_error() {
    // A second `Cost(_)` instance has a unifiable index shape — the
    // shared `register_instance` coherence scan rejects it.
    let src = format!(
        "{STUB}Cost := new Algebra(Nat) {{ unit: 0 }}\nCost := new Algebra(Nat) {{ unit: 1 }}\n"
    );
    let lang = lang();
    let ast = parse(&lang, &src);
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let err = nbe
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect_err("duplicate Cost declaration must fail coherence");
    let msg = format!("{err}");
    assert!(
        msg.contains("overlapping instance") || msg.contains("Cost"),
        "expected coherence error, got: {msg}"
    );
}

// --- `new Constraint(S…)` — the constraint schema
// (design/instance-unified-registry.md §2): the schema head chooses
// what registers; `Constraint` binds `X` as a `type.family` whose
// `base` elements are the index sorts, so `instance X(T) := dict` and
// implicit `<ev : X(a)>` evidence binders flow through the unchanged
// `synth_instance`/`resolve_instance` machinery.

/// `Nat`/`Bool` datatypes + the `Eq` constraint + its `Nat` instance +
/// a function demanding implicit `Eq` evidence.
const CONSTRAINT_SRC: &str = "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }\n\
     Bool := { True : Bool, False : Bool }\n\
     Eq := new Constraint(Type) { evidence : a ~> a ~> Bool }\n\
     instance Eq(Nat) := { eq: (x, y) ~> x == y }\n\
     f<a, ev : Eq(a)> : a ~> a := v ~> v\n";

#[test]
fn new_constraint_discharges_implicit_instance() {
    // `f(Zero)` — `ev : Eq(Nat)` resolves by dictionary search against
    // the declared `instance Eq(Nat)`.
    let src = format!("{CONSTRAINT_SRC}r := f(Zero)\n");
    tc_full(&src);
}

#[test]
fn new_constraint_missing_instance_is_honest_error() {
    // `f(True)` needs `Eq(Bool)` — only `Eq(Nat)` is declared, so
    // instance search fails honestly (no silent unsolved meta).
    let src = format!("{CONSTRAINT_SRC}r := f(True)\n");
    let lang = lang();
    let ast = parse(&lang, &src);
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let err = nbe
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect_err("missing Eq(Bool) instance must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("Eq") && (msg.contains("instance") || msg.contains("Bool")),
        "expected missing-instance error naming Eq/Bool, got: {msg}"
    );
}

/// The learn-doc example (docs/learn "Instances and evidence",
/// design/instance-unified-registry.md §2): `ev.evidence` projects
/// through the constraint's declared dictionary *shape* — the
/// `"<head>.shape"`/`"<head>.shape.vars"` results
/// `synth_new_constraint_decl` records — with the schematic vars
/// substituted by the `family_app`'s index args.
const LEARN_DOC_SRC: &str = "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }\n\
     Bool := { True : Bool, False : Bool }\n\
     Eq := new Constraint(Type) { evidence : a ~> a ~> Bool }\n\
     instance Eq(Nat) := { evidence: (x, y) ~> x == y }\n\
     eqv<a, ev : Eq(a)> : a ~> a ~> Bool :=\n  x ~> y ~> ev.evidence(x, y)\n\
     nat_eq : Bool := eqv(Zero)(Zero)\n";

#[test]
fn new_constraint_evidence_projection_typechecks() {
    // `ev.evidence` on the `Eq(a)` binder yields `a ~> a ~> Bool`;
    // `eqv(Zero)(Zero)` discharges `Eq(Nat)` by dictionary search and
    // the result checks against the `Bool` annotation.
    tc_full(LEARN_DOC_SRC);
}

#[test]
fn new_constraint_explicit_evidence_binder() {
    // The explicit `(ev : Eq(a))` form — no `<>` evidence binder —
    // projects through the same shape path.
    let src = format!(
        "{LEARN_DOC_SRC}eqvE<a> : (ev : Eq(a)) ~> a ~> a ~> Bool := ev ~> x ~> y ~> ev.evidence(x, y)\n"
    );
    tc_full(&src);
}

#[test]
fn new_constraint_shape_field_must_exist() {
    // `ev.other` — the shape declares `evidence` only; projection of
    // an undeclared field is an honest error.
    let src =
        format!("{LEARN_DOC_SRC}bad<a, ev : Eq(a)> : a ~> a ~> Bool := x ~> y ~> ev.other(x, y)\n");
    let lang = lang();
    let ast = parse(&lang, &src);
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let err = nbe
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect_err("ev.other must fail — no such shape field");
    let msg = format!("{err}");
    assert!(
        msg.contains("other") || msg.contains("shape"),
        "expected missing-shape-field error, got: {msg}"
    );
}
