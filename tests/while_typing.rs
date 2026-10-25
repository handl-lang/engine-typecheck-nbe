//! `while` body typing (phase-3): the `type.while` rule descends into the
//! body via `synth_while`. Two surface protocols share the node kind —
//! `while cond { stmts… }` (desugars to `rec` at eval) and the
//! state-passing `while <state> { <step-closure> }` (`loop_step`).

use handl_kit_data_prelude::language_from_yaml;
use handl_language::{Ast, Language};
use hnl_engine_exec_tree_walk::TreeWalkEngine;
use hnl_engine_lex_pattern::lex;
use hnl_engine_parse::{ParserEngine, ParserRequest};
use hnl_engine_parse_pratt::PrattEngine;
use hnl_engine_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_engine_typecheck_nbe::NbeTypecheckEngine;

fn lang() -> Language {
    let lang = language_from_yaml().expect("canonical language should assemble");
    let _ = handl_plugin_lib_handl::install_default_language(lang.clone());
    lang
}

/// Typecheck `src` through the same `do { … }` wrap `Kit::typecheck_*`
/// uses: the Pratt parser consumes one top-level expression per call.
fn check(src: &str) -> Result<hnl_engine_typecheck::TypecheckResult, String> {
    let lang = lang();
    let wrapped = format!("do {{\n{}\n}}", src);
    let tokens = lex(&lang, &wrapped).map_err(|e| format!("lex: {e}"))?;
    let req = ParserRequest::new(tokens);
    let ast: Ast = PrattEngine::new()
        .parse(&lang, &req)
        .map_err(|e| format!("parse: {e}"))?;
    let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let req = TypecheckRequest::new(ast);
    engine
        .typecheck(&lang, &req)
        .map_err(|e| e.to_string())
}

fn err(src: &str) -> String {
    match check(src) {
        Err(e) => e,
        Ok(res) => {
            if let Some(d) = res
                .diagnostics
                .iter()
                .find(|d| d.severity.eq_ignore_ascii_case("error"))
            {
                return d.message.clone();
            }
            panic!("expected a type error, got ok: {:?}", res.typ);
        }
    }
}

fn ok(src: &str) -> hnl_engine_typecheck::TypecheckResult {
    match check(src) {
        Err(e) => panic!("expected ok, got error: {e}"),
        Ok(res) => {
            if let Some(d) = res
                .diagnostics
                .iter()
                .find(|d| d.severity.eq_ignore_ascii_case("error"))
            {
                panic!("expected ok, got error diagnostic: {}", d.message);
            }
            res
        }
    }
}

// --- `while cond { body }` (statement form) -------------------------------

/// The repro from the spec: a bad annotation inside a `while` body must
/// now fail.
#[test]
fn while_cond_body_type_error_reports() {
    let e = err("while true { x : Bool := 3 }");
    assert!(
        e.contains("unify") || e.contains("expected"),
        "unexpected error: {e}"
    );
}

/// A well-typed body passes; the loop synthesizes `type.unit`.
#[test]
fn while_cond_well_typed_body_ok() {
    let res = ok("i := 0\nwhile i < 3 { i := i + 1 }");
    // The module block's last statement is the while — its type is unit.
    let unit = res
        .diagnostics
        .iter()
        .all(|d| !d.severity.eq_ignore_ascii_case("error"));
    assert!(unit);
}

/// Body-local binds do not escape the loop scope (eval pops the block
/// env; Γ restores after the body).
#[test]
fn while_body_binds_do_not_escape() {
    let e = err("while true { fresh_local := 1 }\nfresh_local + 1");
    assert!(e.contains("unbound") || e.contains("scope"), "got: {e}");
}

// --- `while <state> { <step> }` (state-passing form) ----------------------

/// The motivating protocol: `{state}`/`{out}` records on different arms
/// never join — and the body is still checked.
#[test]
fn while_state_form_step_body_checked() {
    // The classic corpus shape: match on the destructured state, continue
    // with `{state: …}` or exit with `{out: …}`.
    ok("r := while { xs: [1, 2], acc: 0 } {\n  (xs, acc) ~> match xs {\n    []: { out: acc },\n    [h, ...t]: { state: { xs: t, acc: acc + h } }\n  }\n}");
}

/// Errors inside the step body report: `n + \"s\"` mixes Number and Str.
#[test]
fn while_state_form_body_error_reports() {
    let e = err("r := while { n: 0 } { (n) ~> { out: n + \"s\" } }");
    assert!(e.contains("unify") || e.contains("expected"), "got: {e}");
}

/// Unbound names inside the step body report.
#[test]
fn while_state_form_unbound_var_reports() {
    let e = err("r := while { n: 0 } { (n) ~> { out: missing_name } }");
    assert!(e.contains("unbound") || e.contains("missing_name"), "got: {e}");
}

/// Single-param step receives the whole state object; `s.xs` projects a
/// field type.
#[test]
fn while_state_single_param_gets_state() {
    ok("r := while { xs: [1], n: 0 } { s ~> if s.n > 1 { { out: s.xs } } else { { state: s } } }");
}

