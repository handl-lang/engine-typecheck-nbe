//! `type.unknown` is the dynamic TOP type (phase-3): for every `T`,
//! `unknown ≡ T` conv succeeds and `T <: unknown` subtype succeeds, on
//! either side of `unify`/`dep_unify`/`conv`/`subtype` — the
//! consistent-equality wildcard `types.yaml` documents ("unifies with
//! whatever the surrounding branches produce"). A value typed
//! `unknown` (host calls, `resolve` imports, `while` results) must not
//! poison the rest of the file: `x + 1`, `x == y`, `if x`, `x[i]`,
//! `x.f`, `f(x)` all accept it.
//!
//! An *explicit* `x : type.unknown = v` annotation still warns (the
//! user opted the binding out of static typing); inferred `unknown`
//! stays silent — host calls cannot warn on every use.

use handl_kit_data_prelude::language_from_yaml;
use handl_language::{Ast, Language};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;

fn lang() -> Language {
    let lang = language_from_yaml().expect("canonical language should assemble");
    let _ = handl_plugin_lib_handl::install_default_language(lang.clone());
    lang
}

/// Typecheck `src` through the same `do { … }` wrap `Kit::typecheck_*`
/// uses: the Pratt parser consumes one top-level expression per call.
fn check(src: &str) -> Result<hnl_tool_typecheck::TypecheckResult, String> {
    let lang = lang();
    let wrapped = format!("do {{\n{}\n}}", src);
    let tokens = lex(&lang, &wrapped).map_err(|e| format!("lex: {e}"))?;
    let req = ParserRequest::new(tokens);
    let ast: Ast = PrattEngine::new()
        .parse(&lang, &req)
        .map_err(|e| format!("parse: {e}"))?;
    let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let req = TypecheckRequest::new(ast);
    engine.typecheck(&lang, &req).map_err(|e| e.to_string())
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

fn ok(src: &str) -> hnl_tool_typecheck::TypecheckResult {
    match check(src) {
        Err(e) => panic!("expected ok, got error: {e}\n  src: {src}"),
        Ok(res) => {
            if let Some(d) = res
                .diagnostics
                .iter()
                .find(|d| d.severity.eq_ignore_ascii_case("error"))
            {
                panic!(
                    "expected ok, got error diagnostic: {}\n  src: {src}",
                    d.message
                );
            }
            res
        }
    }
}

// --- unknown-typed operands -----------------------------------------------

/// The dogfood-2 G2/G9 repro: an `unknown`-typed binding participates
/// in arithmetic, equality, and `if` conditions without poisoning the
/// file. `x : type.unknown = …` is the annotation spelling; the
/// `while` result is an inferred-`unknown` source.
#[test]
fn unknown_operand_arith_eq_if() {
    ok("x : type.unknown = 3\ny := x + 1\nz := x == 1\nw := if x { 1 } else { 2 }");
}

/// `x : type.unknown = v` binds `x` at `type.unknown`: the annotation
/// is the dynamic top — accepted, but warned (the user opted out of
/// static typing explicitly).
#[test]
fn unknown_annotation_binds_unknown_and_warns() {
    let res = ok("x : type.unknown = 3\nx + 1");
    assert!(
        res.diagnostics
            .iter()
            .any(|d| d.severity == "warning" && d.message.contains("type.unknown")),
        "expected an explicit-unknown-annotation warning: {:?}",
        res.diagnostics
    );
}

/// An `unknown` value checks against a concrete annotation — the
/// report's `a : Str = str.upper("hi")` case (`unify`/`dep_unify` see
/// `unknown` on the *synthesized* side).
#[test]
fn unknown_checks_against_concrete_annotation() {
    ok("x : type.unknown = 3\na : Str = x\nb : Number = x");
}

/// A `while` result (inferred `unknown` — no warning expected) flows
/// into operators the same way.
#[test]
fn inferred_unknown_from_while_flows() {
    // `(n, s)` is a multi-param destructure — each name binds the
    // same-named state record field (a single non-pattern param would
    // receive the whole state record instead, per `build_loop_step_args`).
    let res = ok(
        "u := while { n: 0, s: 0 } { (n, s) ~> if n < 2 { { state: { n: n + 1, s: s } } } else { { out: n } } }\nu + 1\nu == 1\nif u { 1 } else { 0 }",
    );
    // Inferred `unknown` must not warn — only explicit annotations do.
    assert!(
        !res.diagnostics
            .iter()
            .any(|d| d.severity == "warning" && d.message.contains("annotated `type.unknown`")),
        "inferred unknown should not warn: {:?}",
        res.diagnostics
    );
}

/// Projection and indexing off `unknown` stay `unknown` — the dynamic
/// lane is opaque, like `record_field_type`'s long-standing rule for
/// field access.
#[test]
fn unknown_projection_and_index_stay_unknown() {
    ok("x : type.unknown = 3\nf := x.field\ni := x[0]\ng := f + 1");
}

/// Applying an `unknown` callee yields `unknown` (pre-existing arm),
/// and passing `unknown` as an argument typechecks.
#[test]
fn unknown_callee_and_argument() {
    ok("x : type.unknown = 3\nf : Number ~> Number = n ~> n\nr := f(x)\ns := x(1)");
}

/// `match` arms of mixed `unknown`/concrete type join — the report's
/// `toStr` case (`expected Unknown, got Str`).
#[test]
fn unknown_match_arm_join() {
    ok("x : type.unknown = 3\nm := match x { _: \"s\", J: x }");
}

// --- controls: unknown must not mask real mismatches ----------------------

/// Two *concrete* mismatched types still fail — the top accept is not
/// a blanket wildcard.
#[test]
fn concrete_mismatch_still_fails() {
    let e = err("x : Bool = 3");
    assert!(
        e.contains("unify") || e.contains("expected"),
        "unexpected error: {e}"
    );
}

/// `if` on a concrete non-bool still fails.
#[test]
fn concrete_if_condition_still_fails() {
    let e = err("if 3 { 1 } else { 2 }");
    assert!(
        e.contains("unify") || e.contains("expected") || e.contains("bool"),
        "unexpected error: {e}"
    );
}
