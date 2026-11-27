//! `@ [Algebra(payload), …]` ambient-mode ascription (PL-0004 E1;
//! design/effects-ambient-modes.md §1) — the judgement
//! `Γ ⊢ t : A @ μ` plumbed through NBE synth/check.
//!
//! Covers: `mode_ascribe` elaboration in type position (`dep_synth_type`
//! returns `A` unchanged and records the mode set), the ambient-mode
//! slot scoping over `type-annot` subjects and `assign` values
//! (default pure — un-annotated code unaffected), set semantics
//! (order irrelevant, one element per head, duplicate heads rejected),
//! head resolution (`mode.algebra.*` globals + `new Algebra`
//! declarations), the mode-metavariable honest bound, and the `IO<…>`
//! retired-former diagnostic (G2 — points at `T @ [Algebra(…)]`).
//!
//! Harness mirrors `surface_syntax.rs`: parse statements under a
//! synthetic `block` root and run the NBE engine.

use handl_kit_data_prelude::language_from_yaml;
use handl_language::{Ast, AstNode, AstRef, Language};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckDiagnostic, TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;
use serde_json::json;

fn canonical_lang() -> Language {
    let lang = language_from_yaml().expect("canonical language should assemble");
    let _ = handl_plugin_lib_handl::install_default_language(lang.clone());
    lang
}

fn parse_one(lang: &Language, src: &str) -> Ast {
    let tokens = lex(lang, src).expect("lex should succeed");
    let engine = PrattEngine::new();
    let req = ParserRequest::new(tokens);
    engine.parse(lang, &req).expect("parse should succeed")
}

/// Append all nodes of `other` to `base`, shifting child/body indices.
fn append_ast(base: &mut Ast, other: &Ast) {
    let offset = base.len();
    for node in other.nodes() {
        let mut payload = node.payload.clone();
        if let Some(children) = payload.get_mut("children").and_then(|c| c.as_array_mut()) {
            for child in children.iter_mut() {
                if let Some(idx) = child.as_u64() {
                    *child = json!(idx as usize + offset);
                }
            }
        }
        if let Some(body) = payload.get_mut("body").and_then(|b| b.as_u64()) {
            *payload.get_mut("body").unwrap() = json!(body as usize + offset);
        }
        base.push(AstNode {
            kind: node.kind.clone(),
            span: node.span,
            payload,
        });
    }
}

/// Parse several statements and combine them under a synthetic `block` root.
fn parse_block(lang: &Language, stmts: &[&str]) -> Ast {
    let mut combined = Ast::new();
    let mut roots = Vec::new();
    for src in stmts {
        let ast = parse_one(lang, src);
        let offset = combined.len();
        append_ast(&mut combined, &ast);
        let root_idx = ast.len() - 1 + offset;
        let root = combined.get(AstRef::new(root_idx)).expect("root");
        if root.kind == "block" {
            for c in root
                .payload
                .get("children")
                .and_then(|c| c.as_array())
                .expect("block children")
            {
                roots.push(c.as_u64().unwrap() as usize);
            }
        } else {
            roots.push(root_idx);
        }
    }
    combined.push(AstNode::new("block").with_payload(json!({ "children": roots })));
    combined
}

fn typecheck(lang: &Language, ast: Ast) -> hnl_tool_typecheck::TypecheckResult {
    let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let req = TypecheckRequest::new(ast);
    match engine.typecheck(lang, &req) {
        Ok(result) => result,
        Err(e) => hnl_tool_typecheck::TypecheckResult {
            typ: serde_json::Value::Null,
            diagnostics: vec![TypecheckDiagnostic {
                severity: "error".to_string(),
                message: e.to_string(),
                span: None,
            }],
            node_types: std::collections::HashMap::new(),
            signature: serde_json::Value::Null,
        },
    }
}

fn diags_of(result: &hnl_tool_typecheck::TypecheckResult) -> Vec<String> {
    result
        .diagnostics
        .iter()
        .map(|d| format!("{}: {}", d.severity, d.message))
        .collect()
}

fn has_error(result: &hnl_tool_typecheck::TypecheckResult) -> bool {
    result.diagnostics.iter().any(|d| d.severity == "error")
}

fn has_warning_containing(result: &hnl_tool_typecheck::TypecheckResult, needle: &str) -> bool {
    result
        .diagnostics
        .iter()
        .any(|d| d.severity == "warning" && d.message.contains(needle))
}

/// `x : T @ [Head] := v` checks cleanly: a `new Algebra`-declared head
/// resolves, the mode set elaborates, and the value checks against `T`
/// exactly as the un-annotated form (μ is judgement metadata, not part
/// of `A`). (`Effects` itself is no longer a registered global — the
/// `Effects(S)` row encoding retired with PL-0004 G2.)
#[test]
fn declared_head_ascription_checks() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "plain : Nat := 1",
            "eff : Nat @ [PutStr] := 1",
            "pure : Nat @ [] := 2",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "mode-ascribed bindings should check like plain ones, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "unknown algebra head"),
        "a `new Algebra` head resolves without warning, got {:?}",
        diags_of(&result)
    );
}

/// Stub `Algebra` former — a record-family so `Algebra(Nat)`
/// elaborates to `type.family_app` (same stub `new_decl.rs` uses; M2
/// replaces it with the real algebra type).
const ALGEBRA_STUB: &str = "Algebra<c : Type> := { mkAlgebra : (compose : c ~> c ~> c, unit : c, order : c ~> c ~> Bool) ~> Algebra }";

/// Set semantics: one element per algebra head — two `new
/// Algebra`-declared heads compose in one ascription, order irrelevant,
/// and both resolve without a warning.
#[test]
fn multi_algebra_set_and_declared_head() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "Cost := new Algebra(Nat) { compose: (a, b) ~> a + b, unit: 0, order: (a, b) ~> a <= b }",
            "FsRead := new Algebra { op: Unit ~> Str }",
            "f : Nat @ [FsRead, Cost(3)] := 1",
            "g : Nat @ [Cost(3), FsRead] := 2",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "multi-algebra mode sets should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "unknown algebra head"),
        "Cost declared via `new Algebra` must resolve, got {:?}",
        diags_of(&result)
    );
}

/// Duplicate algebra heads are *rejected* — a mode set holds one
/// element per algebra (order irrelevant, heads unique).
#[test]
fn duplicate_head_rejected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "FsRead := new Algebra { op: Unit ~> Str }",
            "bad : Nat @ [FsRead, FsRead] := 1",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "duplicate `FsRead` heads must be rejected, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("duplicate algebra head")),
        "expected a duplicate-head diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// An unresolvable head is the honest bound: the ascription passes
/// with a warning naming the gap, never a rejection.
#[test]
fn unknown_head_warns_but_passes() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["f : Nat @ [NoSuchAlgebra(1)] := 1"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "unknown algebra head must not reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "unknown algebra head `NoSuchAlgebra`"),
        "expected an unknown-head warning, got {:?}",
        diags_of(&result)
    );
}

/// Mode metavariables: `@ m` / `@ [m]` with `m` unbound passes with a
/// warning (the M3 honest bound) — a bound mode variable stays silent.
#[test]
fn mode_metavariable_honest_bound() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : Nat @ m := 1",
            "g : Nat @ [m] := 2",
            // A bound payload name resolves — `S` is results-bound, so
            // `Effects(S)` elaborates without a *metavariable* warning
            // (`Effects` itself is now an unknown head — the retired
            // `Effects(S)` row global — which warns separately; only
            // *unbound* bare vars warn as metavariables).
            "S := 0",
            "h : Nat @ [Effects(S)] := 3",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "mode metavariables must not reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "mode metavariable `m` does not resolve"),
        "expected a mode-metavariable warning, got {:?}",
        diags_of(&result)
    );
}

/// The ambient slot defaults pure: code without `@` checks
/// identically, and `@ []` is the explicit pure ascription.
#[test]
fn unannotated_code_unaffected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "a : Nat := 1",
            "b : (Nat ~> Nat) := n ~> n + 1",
            "c : Str := \"hi\"",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "un-annotated code must check identically, got {:?}",
        diags_of(&result)
    );
}

/// `IO(…)`/`IO<…>` surface sites carry the retired-former diagnostic
/// pointing at `T @ [Algebra(…)]` — warn + an open meta, don't reject
/// (the corpus migrates phase by phase; PL-0004 G2).
#[test]
fn io_former_retired_warning() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "narrow : IO<{ puts: Str ~> Unit }> ~> IO<{ puts: Str ~> Unit, gets: Unit ~> Str }> := x ~> x",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "IO<…> still checks (retirement is a warning + open meta), got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "`IO` is retired"),
        "expected the IO retired-former warning, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "T @ [Algebra(…)]"),
        "retirement must point at the ambient spelling, got {:?}",
        diags_of(&result)
    );
}

/// Term-position `e @ μ` — the `type.mode_ascribe` judgment
/// synthesizes the subject and records the mode set.
#[test]
fn term_position_ascription() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &["n : Nat := (1 @ [Effects({ puts: Str ~> Unit })])"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "term-position `e @ μ` should check, got {:?}",
        diags_of(&result)
    );
}
