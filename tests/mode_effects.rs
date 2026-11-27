//! Effects as ambient modes — G1 granular consumers (PL-0004;
//! design/effects-ambient-modes.md v2 §3/§4.2/§6): `perform(Head, arg)`
//! resolves domain/codomain from the head's declared `op : A ~> B` and
//! *requires* `Head` in the ambient mode set (absent → hard rejection;
//! undeclared op names get the honest bound — warn + a fresh type
//! meta, no `IO` mint since G2),
//! `handle` is a modality `μ → μ′` over mode *elements* — coverage,
//! `op ↦ ret` re-keying, `μ ∖ H`, and `μ_handler` union, requires-closed
//! — `>>>` composes mode sets by head union, and `@ μ` ascriptions
//! check widening by set inclusion over heads.
//!
//! The `Effects(S)` row encoding is retired: row *keys* are not mode
//! heads, so `perform(puts, …)` under `@ [Effects({ puts: … })]` gets
//! only the honest bound (warn + fresh meta — pinned by
//! `effects_row_keys_are_not_heads`).
//!
//! Harness mirrors `mode_granular.rs`: parse statements under a
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

/// `perform(Head, arg)` under `@ […, Head, …]` checks `arg` against
/// the head's declared `op : A ~> B` domain and returns the codomain
/// `B` — no `IO{op:T}` mint (a minted `IO` would fail the `Unit`
/// annotation's unification).
#[test]
fn ambient_perform_returns_codomain() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "greet : Unit @ [PutStr] := perform(PutStr, \"hi\")",
            "read : Str @ [GetStr] := perform(GetStr)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "ambient perform should return the op codomain, got {:?}",
        diags_of(&result)
    );
}

/// The declared domain is enforced: `perform(PutStr, 1)` against
/// `op: Str ~> Unit` rejects (the argument is not a `Str`).
#[test]
fn ambient_perform_checks_domain() {
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

/// G1 hard rejection: a *declared* op head absent from the ambient
/// mode set is not permitted — no `IO` mint fallback.
#[test]
fn ambient_perform_head_not_permitted_rejected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "GetStr := new Algebra { op: Unit ~> Str }",
            "bad : Str @ [PutStr] := perform(GetStr)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "perform of an unpermitted declared head must reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("effect head `GetStr` is not permitted")),
        "expected the not-permitted diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// An op *name* with no declared `op` field gets the honest bound —
/// warn + a fresh type meta (G2 retired the `IO{op:T}` mint with the
/// `type.io` former).
#[test]
fn ambient_perform_undeclared_op_warns_and_binds_meta() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Unit }",
            "x : Str @ [PutStr] := perform(fs, \"x\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "undeclared op falls back to warn + fresh meta (honest bound), got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "not a declared effect algebra"),
        "expected the undeclared-op warning, got {:?}",
        diags_of(&result)
    );
}

/// The retired `Effects(S)` row does not make its keys performable
/// heads: `puts` inside `Effects({ puts: … })` is a row entry, not a
/// declared algebra — `perform(puts, …)` gets only the honest bound
/// (warn + fresh meta), exactly like any undeclared op name.
#[test]
fn effects_row_keys_are_not_heads() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &["x : Str @ [Effects({ puts: Str ~> Unit })] := perform(puts, \"hi\")"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "row keys are not granular heads — warn + fresh meta is the honest bound, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "not a declared effect algebra"),
        "expected the undeclared-op warning, got {:?}",
        diags_of(&result)
    );
}

/// Un-annotated code gets the same honest bound: `perform` with no
/// ambient mode warns and binds a fresh meta — G2 removed the silent
/// `IO{op:T}` mint even without a mode ascription.
#[test]
fn unannotated_perform_warns_and_binds_meta() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["x : Str := perform(puts, \"a\")"]);
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

/// `handle m with h` as a modality `μ → μ′` over mode elements: the
/// handler's op fields are checked against the handled computation's
/// mode heads (the binding-carried mode), covered heads re-key to the
/// impl's return type, and the residual mode propagates to the
/// binding — a second `handle` still sees the covered head.
#[test]
fn ambient_handle_modality() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "puts_h : Str ~> Str := s ~> s",
            "handled := handle f with { PutStr: puts_h }",
            "handled_again := handle handled with { PutStr: puts_h }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "ambient handle should check coverage + residual, got {:?}",
        diags_of(&result)
    );
}

/// Coverage diagnostics are preserved on the granular path: a handler
/// declaring an op the handled mode never names is spurious.
#[test]
fn ambient_handle_spurious_impl_rejected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "FsWrite := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "fs_h : Str ~> Str := s ~> s",
            "bad := handle f with { FsWrite: fs_h }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "spurious handler impl must reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result).iter().any(|d| d.contains("never names")),
        "expected the spurious-impl diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// `μ_handler` unions into the residual (design §4.2): a handler impl
/// that itself performs `Ioctl` leaves `Ioctl` in the handled
/// computation's residual mode, so the ascription must permit it.
#[test]
fn handle_residual_unions_handler_effects() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "Ioctl := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "handled : (Str ~> Str) @ [PutStr, Ioctl] := handle f with { PutStr: s ~> perform(Ioctl, \"x\") }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "handler effects must union into the residual, got {:?}",
        diags_of(&result)
    );
}

/// Requires-closure of the residual (design §4.1/§6): `Ioctl
/// requires [FsRead]` — the residual mode set must carry `FsRead`
/// even though nothing performed or handled it, so an *outer*
/// `handle` sees `FsRead` as a coverable op rather than a spurious
/// impl.
#[test]
fn handle_residual_is_requires_closed() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "FsRead := new Algebra { op: Str ~> Str }",
            "Ioctl := new Algebra { op: Str ~> Str, requires: [FsRead] }",
            "PutStr := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "r_h : Str ~> Str := s ~> s",
            "handled : (Str ~> Str) @ [PutStr, Ioctl] := handle f with { PutStr: s ~> perform(Ioctl, \"x\") }",
            "outer := handle handled with { FsRead: r_h }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "requires-closed residual must expose implied heads to outer handles, got {:?}",
        diags_of(&result)
    );
}

/// `>>>` computes the mode set: `{PutStr} ⋄ {FsWrite} = {PutStr,
/// FsWrite}` — the composed binding's mode is the union, and an
/// ascription naming both heads accepts it (set inclusion).
#[test]
fn composition_unions_mode_heads() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "FsWrite := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "g : (Str ~> Str) @ [FsWrite] := s ~> s",
            "h := g >>> f",
            "both : (Str ~> Str) @ [PutStr, FsWrite] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "composed mode sets should union and subsume, got {:?}",
        diags_of(&result)
    );
}

/// Effect widening is directional: `{PutStr, FsWrite} ≰ {PutStr}` —
/// `FsWrite` is not permitted by the narrower ascription.
#[test]
fn subsumption_rejects_uncovered_head() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "FsWrite := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "g : (Str ~> Str) @ [FsWrite] := s ~> s",
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
            .any(|d| d.contains("effect head `FsWrite` is not permitted")),
        "expected the subsumption diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// A narrower actual set fits a wider ascription — `{PutStr} ≤
/// {PutStr, FsWrite}` is exactly set inclusion.
#[test]
fn subsumption_accepts_narrower_set() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            ALGEBRA_STUB,
            "PutStr := new Algebra { op: Str ~> Str }",
            "FsWrite := new Algebra { op: Str ~> Str }",
            "f : (Str ~> Str) @ [PutStr] := s ~> s",
            "wide : (Str ~> Str) @ [PutStr, FsWrite] := f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "narrower mode set must fit wider ascription, got {:?}",
        diags_of(&result)
    );
}
