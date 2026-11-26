//! PL-0003 I2 — inductives-as-data: `X := new Inductive {…}` lowers to
//! `data` at parse time, the `inductive.spec` value projects through
//! `F.spec`, and strict positivity runs as the `check_inductive_spec`
//! judgment over spec values (NBE hard-`Err` mirror of the model's
//! diagnostic channel).

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

fn tc_err(src: &str) -> String {
    let lang = lang();
    let ast = parse(&lang, src);
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let err = nbe
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect_err("typecheck must fail");
    format!("{err}")
}

#[test]
fn new_inductive_lowers_to_data_node() {
    // The parse layer rewrites `X := new Inductive {…}` (bare `Inductive`
    // head + record dict) to the `data` node `synth_data` consumes.
    let lang = lang();
    let ast = parse(
        &lang,
        "Nat := new Inductive { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
    );
    assert!(
        ast.nodes().iter().any(|n| n.kind == "data"),
        "new Inductive must lower to a data node — got kinds: {:?}",
        ast.nodes()
            .iter()
            .map(|n| n.kind.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn new_inductive_typechecks_like_data() {
    // Same declaration, both spellings — `data` record form and the
    // `new Inductive` sugar — must typecheck identically.
    tc_full("Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nnat_zero := Zero\n");
    tc_full("Nat := new Inductive { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nnat_zero := Zero\n");
}

#[test]
fn new_inductive_spec_projects_through_field_access() {
    // `Nat.spec` resolves via the `type.family` arm of
    // `record_field_type` (Γ scope binding `"Nat.spec"`); the spec
    // record itself is the projected *type* — the data-as-type
    // convention `results` values already use.
    let result =
        tc_full("Nat := new Inductive { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nNat.spec\n");
    let typ = &result.typ;
    assert_eq!(
        typ.get("__tag__").and_then(Value::as_str),
        Some("inductive.spec"),
        "Nat.spec must project the spec value — got {typ}"
    );
    assert_eq!(typ.get("name").and_then(Value::as_str), Some("Nat"));
    let ctors = typ.get("ctors").and_then(Value::as_array).expect("ctors");
    assert_eq!(ctors.len(), 2);
}

#[test]
fn new_inductive_spec_nested_projection() {
    // `Nat.spec.ctors` — a second `field-access` projection off the
    // `inductive.spec` container (`record_field_type`'s
    // `inductive.spec` arm).
    let result = tc_full(
        "Nat := new Inductive { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nNat.spec.ctors\n",
    );
    let typ = &result.typ;
    assert!(
        typ.as_array().is_some_and(|a| a.len() == 2),
        "Nat.spec.ctors must project the ctor list — got {typ}"
    );
}

#[test]
fn new_inductive_positivity_accepts_nested_positive() {
    // `W` occurs only in the *codomain* of `f`'s type — a positive
    // position; strict positivity accepts. (`Nat` is declared first so
    // `Nat ~> W` elaborates.)
    tc_full("Nat := { Zero : Nat }\nW := new Inductive { C : (f : Nat ~> W) ~> W }\n");
}

#[test]
fn new_inductive_positivity_rejects_negative_occurrence() {
    // `f : W ~> Nat` puts `W` in a domain (negative) position —
    // `check_inductive_spec` fails the declaration.
    let msg = tc_err("Nat := { Zero : Nat }\nW := new Inductive { C : (f : W ~> Nat) ~> W }\n");
    assert!(
        msg.contains("positivity") || msg.contains("check_inductive_spec"),
        "expected strict-positivity failure, got: {msg}"
    );
}

#[test]
fn new_inductive_with_schema_args_is_honest_error() {
    // `new Inductive(args)` does not lower (the inductive form takes no
    // schema arguments) — the surviving `new` node reaches
    // `synth_new_decl`, which reports honestly instead of minting a
    // mode family named `X`.
    let msg = tc_err("Bad := new Inductive(Nat) { C : Bad }\n");
    assert!(
        msg.contains("Inductive") || msg.contains("new"),
        "expected honest new-Inductive error, got: {msg}"
    );
}
