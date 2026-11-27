//! Effects as ambient modes — E3 mode metavariables + the
//! `Val`-descriptor compose bridge (PL-0004;
//! design/effects-ambient-modes.md §3/§4.4): unbound `var`s inside
//! `@ μ` ascriptions mint elaboration-scoped tagged metas that
//! unify/substitute during checking (expected-side instantiation at
//! `check_mode_subsumption`), symbolic `Effects` payloads honest-bound
//! at every consumer, and non-`Effects` mode heads compose through
//! `Checker::mode_compose_vals` (`domain_lift`/`domain_quote`) instead
//! of the E2 fresh-meta bound.
//!
//! Harness mirrors `mode_effects.rs`: parse statements under a
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

/// Stub `Algebra` former — same stub `mode_ascribe.rs`/`new_decl.rs`
/// use so `new Algebra(Nat)` declarations elaborate.
const ALGEBRA_STUB: &str = "Algebra<c : Type> := { mkAlgebra : (compose : c ~> c ~> c, unit : c, order : c ~> c ~> Bool) ~> Algebra }";

/// An unbound `var` inside `Effects(…)` is a mode metavariable: it
/// warns (honest bound) but mints a unifiable meta, not a dead var
/// term — the payload stays a `__meta__` the subsumption check can
/// instantiate.
#[test]
fn effects_row_metavariable_mints_meta() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["f : (Str ~> Str) @ [Effects(s)] := t ~> t"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "mode metavariable must not reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "mode metavariable `s` does not resolve"),
        "expected the metavariable warning, got {:?}",
        diags_of(&result)
    );
}

/// Expected-side instantiation (E3 substitution): `x`'s ascription
/// `Effects(s)` holds an unsolved meta; checking `g` (row `{puts}`)
/// against it solves `?s := {puts}` — and the solved mode persists on
/// `x`'s binding, so a later concrete ascription sees the row.
#[test]
fn expected_side_metavariable_instantiates() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "g : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := t ~> t",
            "x : (Str ~> Str) @ [Effects(s)] := g",
            // If `?s` instantiated to `{puts}`, this concrete row
            // subsumes cleanly — no "effect not permitted" rejection.
            "y : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := x",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "expected-side mode metavariable should instantiate, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "mode metavariable `s` does not resolve"),
        "expected the metavariable warning, got {:?}",
        diags_of(&result)
    );
}

/// An unbound `var` *head* inside `@ […]` is a mode metavariable: it
/// warns (honest bound) but mints a unifiable meta — a head-less
/// element the head-set subsumption cannot enumerate, so widening
/// checks pass (never reject) while elaboration stays honest.
#[test]
fn bare_mode_metavariable_mints_meta() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [m] := t ~> t",
            "x : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "head-less mode metavariable must not reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "mode metavariable `m`"),
        "expected the metavariable warning, got {:?}",
        diags_of(&result)
    );
}

/// `perform` under a symbolic `Effects` payload: `puts` is not a
/// declared effect head (row keys are not heads under G1), so the
/// honest bound is warn + the legacy `IO` mint — the `IO<sig>`
/// spelling G2 retires.
#[test]
fn perform_under_symbolic_row_warns_and_mints() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &["x : IO<{ puts: Str }> @ [Effects(r)] := perform(puts, \"a\")"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "undeclared perform must fall back to the IO mint, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "not a declared effect algebra"),
        "expected the undeclared-op warning, got {:?}",
        diags_of(&result)
    );
}

/// The `Val`-descriptor bridge (E3): `Cost(1) ⋄ Cost(2)` composes
/// under the canonical default algebra (`mode.algebra.Cost` is not
/// registered — `new Algebra` decls are name-level — so payloads fall
/// back to `default`: numeric add) instead of E2's "no ambient
/// compose" honest bound.
#[test]
fn compose_bridges_non_effects_algebra() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "Cost := new Algebra(Nat) { compose: (a, b) ~> a + b, unit: 0, order: (a, b) ~> a <= b }",
            "f : (Str ~> Str) @ [Cost(1)] := s ~> s",
            "g : (Str ~> Str) @ [Cost(2)] := s ~> s",
            "h := g >>> f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "bridged mode compose should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "no ambient compose"),
        "the E3 bridge replaces the honest bound, got {:?}",
        diags_of(&result)
    );
}

/// `Effects` payload metas compose through the `Val`-descriptor
/// bridge: `?r ⋄ {puts}` cannot union without a solver — the retired
/// `row_union` opcode is an unknown compose op (honest bound +
/// warning) — and the composite still passes a wider concrete
/// ascription (the payload meta unifies at subsumption).
#[test]
fn compose_effects_meta_honest_bound() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects(r)] := t ~> t",
            "g : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := t ~> t",
            "h := g >>> f",
            "x : (Str ~> Str) @ [Effects({ puts: Str ~> Str, fs: Str ~> Str })] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "meta-row compose must honest-bound, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "unknown algebra compose op"),
        "expected the compose honest-bound warning, got {:?}",
        diags_of(&result)
    );
}

/// Bare modes compose through the same bridge: `@ [3] ⋄ @ [4]` runs
/// the canonical default algebra's numeric add — no "no ambient
/// compose" warning.
#[test]
fn compose_bare_modes_default_algebra() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [3] := s ~> s",
            "g : (Str ~> Str) @ [4] := s ~> s",
            "h := g >>> f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "bare-mode compose should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "no ambient compose"),
        "bare modes bridge to the default algebra, got {:?}",
        diags_of(&result)
    );
}

/// Un-annotated code is unaffected by E3: no `@`, no metas, no
/// warnings — identical checking.
#[test]
fn unannotated_code_unaffected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) := s ~> s",
            "g : (Str ~> Str) := s ~> s",
            "h := g >>> f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "un-annotated composition must check identically, got {:?}",
        diags_of(&result)
    );
}
