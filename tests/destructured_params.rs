//! Destructured closure params (`param_defs`, ADR-0081) — the NBE
//! typecheck engine must bind pattern binders structurally instead of
//! degrading to `stuck_val`/empty names (myco I-008).
//!
//! Same harness as `parity.rs`: lex → Pratt parse (no desugar) →
//! `NbeTypecheckEngine` over `TreeWalkEngine`.

use handl_language::{LangError, Language};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex as lex_src;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;
use serde_json::Value;

fn lang() -> Language {
    let kit = handl_kit_data_prelude::assemble().expect("canonical assembly");
    kit.lang().clone()
}

fn tc(src: &str) -> Value {
    let lang = lang();
    let tokens = lex_src(&lang, src).expect("lex");
    let pratt = PrattEngine::new();
    let ast = pratt
        .parse(&lang, &ParserRequest::new(tokens))
        .expect("parse");
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    nbe.typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck")
        .typ
}

fn tc_err(src: &str) -> LangError {
    let lang = lang();
    let tokens = match lex_src(&lang, src) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let pratt = PrattEngine::new();
    let ast = match pratt.parse(&lang, &ParserRequest::new(tokens)) {
        Ok(a) => a,
        Err(e) => return e,
    };
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    nbe.typecheck(&lang, &TypecheckRequest::new(ast))
        .expect_err("expected a typecheck/parse/lex error")
}

/// `f := ([a, b]) ~> body` — a paren-wrapped `sequence` pattern is a
/// single array argument; the param side is `type.array{elements}`.
#[test]
fn paren_sequence_param_types_as_array() {
    let ty = tc("do { f := ([a, b]) ~> a; f }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let param = &ty["param"];
    assert_eq!(param["__tag__"], "type.array", "param: {param}");
    let elems = param["elements"].as_array().expect("elements array");
    assert_eq!(elems.len(), 2, "param: {param}");
}

/// `g := ({x, y}) ~> body` — an `entries` pattern is a record
/// argument; the param side is `type.record{fields}`.
#[test]
fn entries_param_types_as_record() {
    let ty = tc("do { g := ({x, y}) ~> x; g }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let param = &ty["param"];
    assert_eq!(param["__tag__"], "type.record", "param: {param}");
    let fields = param["fields"].as_object().expect("fields object");
    assert!(fields.contains_key("x"), "fields: {fields:?}");
    assert!(fields.contains_key("y"), "fields: {fields:?}");
}

/// `{x: a, y: b}` long-form entries bind `a`/`b` to the field types.
#[test]
fn entries_long_form_binds_value_patterns() {
    let ty = tc("do { g := ({x: a, y: b}) ~> a; g }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let param = &ty["param"];
    assert_eq!(param["__tag__"], "type.record", "param: {param}");
    let fields = param["fields"].as_object().expect("fields object");
    assert!(fields.contains_key("x"), "fields: {fields:?}");
    assert!(fields.contains_key("y"), "fields: {fields:?}");
}

/// `(n, [p, q]) ~> body` — a mixed params list keeps the positional
/// `type.array{elements}` encoding, with the pattern position typed
/// structurally.
#[test]
fn mixed_params_list_types_positionally() {
    let ty = tc("do { h := (n, [p, q]) ~> n; h }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let param = &ty["param"];
    assert_eq!(param["__tag__"], "type.array", "param: {param}");
    let elems = param["elements"].as_array().expect("elements array");
    assert_eq!(elems.len(), 2, "param: {param}");
    assert_eq!(elems[1]["__tag__"], "type.array", "elem: {}", elems[1]);
}

/// Nested patterns bind every leaf: `[x, [y, z]] ~>` types the inner
/// sequence recursively.
#[test]
fn nested_sequence_pattern_types_recursively() {
    let ty = tc("do { k := [x, [y, z]] ~> x; k }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let param = &ty["param"];
    assert_eq!(param["__tag__"], "type.array", "param: {param}");
    let elems = param["elements"].as_array().expect("elements array");
    assert_eq!(elems.len(), 2, "param: {param}");
    assert_eq!(elems[1]["__tag__"], "type.array", "inner: {}", elems[1]);
    assert_eq!(
        elems[1]["elements"].as_array().map(Vec::len),
        Some(2),
        "inner elements: {}",
        elems[1]
    );
}

/// A destructured-param body that references a bound name must not
/// report an unbound variable — the binder set matches exec's
/// `param_bound_names`.
#[test]
fn pattern_binders_resolve_in_body() {
    // Would fail with "unbound variable" if `a`/`b` were not bound.
    let ty = tc("do { f := ([a, b]) ~> a; f }");
    assert_eq!(ty["__tag__"], "type.function", "got {ty}");
    let _ = tc_err("do { f := ([a, b]) ~> c; f }"); // `c` unbound → error
}
