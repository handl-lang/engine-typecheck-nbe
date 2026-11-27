//! Effects as ambient modes — G0 granular effect algebras
//! (PL-0004; design/effects-ambient-modes.md §2/§4/§6): each effect
//! operation is its own `new Algebra` head carrying an `op : A ~> B`
//! signature; mode elements are *bare heads* (`@ [ReadFile]`) with the
//! payload slot reserved for parameters; `requires` closes the mode
//! set under declared algebra implication at elaboration; `perform`
//! resolves domain/codomain from the head's `op` field; composition
//! is set union and subsumption set inclusion for heads without
//! `compose`/`order`.
//!
//! Harness mirrors `mode_effects.rs`: parse statements under a
//! synthetic `block` root and run the NBE engine. `Algebra` itself is
//! stubbed exactly as `mode_ascribe.rs`/`new_decl.rs` do (M2 owns the
//! real former).

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

/// Stub `Algebra` former — a record-family so `Algebra(Nat)`/bare
/// `Algebra` declarations elaborate (same stub `mode_ascribe.rs` uses;
/// M2 owns the real former).
const ALGEBRA_STUB: &str = "Algebra<c : Type> := { mkAlgebra : (compose : c ~> c ~> c, unit : c, order : c ~> c ~> Bool) ~> Algebra }";

/// Two effect heads with `op` signatures: `perform` under a granular
/// `@ [Head, …]` mode resolves domain/codomain from the head's `op`
/// field — no `IO{op:T}` mint (a minted `IO` would fail the `Unit`
/// annotation's unification).
#[test]
fn granular_perform_via_op_field() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "f : Str @ [PutStr, GetStr] := perform(GetStr)",
            "g : Unit @ [PutStr] := perform(PutStr, \"x\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "granular perform should return the op codomain, got {:?}",
        diags_of(&result)
    );
}

/// The `op` field's declared domain is enforced:
/// `perform(PutStr, 1)` against `op: Str ~> Unit` rejects.
#[test]
fn granular_perform_checks_domain() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "bad : Unit @ [PutStr] := perform(PutStr, 1)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "perform with a wrong-typed argument must reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("does not match op `PutStr`'s domain")),
        "expected the op-domain diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// `requires` closes the mode set transitively at elaboration
/// (design §4.1): `ReadFile requires [FsRead]`, `FsRead requires
/// [Ioctl]` — `@ [ReadFile]` elaborates to `{ReadFile, FsRead,
/// Ioctl}`, so `perform` resolves every implied head.
#[test]
fn requires_closure_is_transitive() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "Ioctl := new Algebra { op: Str ~> Str }",
            "FsRead := new Algebra { op: Str ~> Str, requires: [Ioctl] }",
            "ReadFile := new Algebra { op: Str ~> Str, requires: [FsRead] }",
            "f : Str @ [ReadFile] := perform(FsRead, \"p\")",
            "g : Str @ [ReadFile] := perform(Ioctl, \"p\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "requires-closed heads must perform, got {:?}",
        diags_of(&result)
    );
}

/// An unresolvable `requires` head warns and passes — the honest
/// bound (the implied capability is simply absent).
#[test]
fn requires_unknown_head_warns_and_passes() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "ReadFile := new Algebra { op: Str ~> Str, requires: [NoSuchHead] }",
            "f : Str @ [ReadFile] := perform(ReadFile, \"p\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "unresolvable requires must not reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "requires unknown algebra head `NoSuchHead`"),
        "expected the requires warning, got {:?}",
        diags_of(&result)
    );
}

/// Bare-head mode elements: `@ [ReadFile]` names a declared algebra
/// head with no payload — it resolves (no "unknown algebra head"
/// warning) and the set holds exactly the granular heads.
#[test]
fn bare_head_elements_resolve() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "ReadFile := new Algebra { op: Str ~> Str }",
            "f : Str @ [ReadFile] := perform(ReadFile, \"p\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "bare-head ascription should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "unknown algebra head"),
        "declared bare head must resolve, got {:?}",
        diags_of(&result)
    );
}

/// Composition is set union for effect algebras (design §2):
/// `{PutStr} ⋄ {GetStr} = {PutStr, GetStr}` — the composed binding's
/// mode is the union and an ascription naming both heads accepts it
/// (set inclusion).
#[test]
fn granular_composition_unions_heads() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "g : (Str ~> Str) @ [GetStr] := s ~> s",
            "h := g >>> f",
            "both : (Str ~> Str) @ [PutStr, GetStr] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "composed granular modes should union and subsume, got {:?}",
        diags_of(&result)
    );
}

/// Subsumption is directional set inclusion: `{PutStr, GetStr} ≰
/// {PutStr}` — `GetStr` is not permitted by the narrower ascription.
#[test]
fn granular_subsumption_rejects_uncovered_head() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "g : (Str ~> Str) @ [GetStr] := s ~> s",
            "h := g >>> f",
            "narrow : (Str ~> Str) @ [PutStr] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "widening beyond the ascribed mode set must reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("effect head `GetStr` is not permitted")),
        "expected the granular subsumption diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// A narrower actual set fits a wider ascription — `{PutStr} ≤
/// {PutStr, GetStr}` is exactly set inclusion.
#[test]
fn granular_subsumption_accepts_narrower_set() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "wide : (Str ~> Str) @ [PutStr, GetStr] := f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "narrower mode set must fit wider ascription, got {:?}",
        diags_of(&result)
    );
}

/// The `Effects(S)` transition path still works alongside granular
/// heads: an `Effects` row ascription keeps its row element (G1
/// retired the row machinery — `Effects` is now an ordinary head
/// name), and a granular head in the same set composes by union.
/// `perform` names the granular head: row keys are not heads.
#[test]
fn effects_row_and_granular_heads_coexist() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "greet : Unit @ [Effects({ puts: Str ~> Unit }), PutStr] := perform(PutStr, \"hi\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Effects row + granular head should coexist, got {:?}",
        diags_of(&result)
    );
}

/// Compositional algebras keep payload composition: `Cost(3) ⋄
/// Cost(3)` still composes through the declared `compose` — the G0
/// set-union shortcut applies only to non-compositional heads.
#[test]
fn compositional_algebra_still_composes_payloads() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "Cost := new Algebra(Nat) { compose: (a, b) ~> a + b, unit: 0, order: (a, b) ~> a <= b }",
            "f : Nat @ [Cost(3)] := 1",
            "g : Nat @ [Cost(4)] := 2",
            "h := g >>> f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "compositional payloads should compose as before, got {:?}",
        diags_of(&result)
    );
}

/// Un-annotated code gets the same honest bound: `perform` with no
/// ambient mode warns and binds a fresh meta — G2 removed the silent
/// `IO{op:T}` mint (and the `IO<sig>` spelling) entirely.
#[test]
fn unannotated_perform_warns_and_binds_meta() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "x : Str := perform(puts, \"a\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "un-annotated perform must check via the honest bound, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "not a declared effect algebra"),
        "undeclared op warns even without an ambient mode, got {:?}",
        diags_of(&result)
    );
}
