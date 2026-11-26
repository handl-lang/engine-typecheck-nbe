//! Effects as ambient modes — E2 consumers (PL-0004;
//! design/effects-ambient-modes.md §3): `perform` extends the ambient
//! mode's `Effects` row and returns the op's codomain (no `IO{op:T}`
//! mint), `handle` is a modality `μ → μ′` with the coverage/residual
//! computation ported behind it, `>>>` composes the rows
//! (`{puts} ⋄ {fs} = {puts,fs}`), and `@ μ` ascriptions check effect
//! widening through the `Effects` algebra's `order`/`row_width`.
//!
//! Harness mirrors `mode_ascribe.rs`: parse statements under a
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

/// `perform(op, arg)` under `@ [Effects({op: A ~> B})]` checks `arg`
/// against the declared domain `A` and returns the codomain `B` —
/// no `IO{op:T}` mint (a minted `IO` would fail the `Unit`
/// annotation's unification).
#[test]
fn ambient_perform_returns_codomain() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "greet : Unit @ [Effects({ puts: Str ~> Unit })] := perform(puts, \"hi\")",
            "read : Str @ [Effects({ gets: Unit ~> Str })] := perform(gets)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "ambient perform should return the op codomain, got {:?}",
        diags_of(&result)
    );
}

/// The declared domain is enforced: `perform(puts, 1)` against
/// `puts : Str ~> Unit` rejects (the argument is not a `Str`).
#[test]
fn ambient_perform_checks_domain() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &["bad : Unit @ [Effects({ puts: Str ~> Unit })] := perform(puts, 1)"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "perform with a wrong-typed argument must reject, got {:?}",
        diags_of(&result)
    );
}

/// An op missing from the ambient `Effects` row keeps the legacy
/// `IO{op:T}` codomain path *with a warning* — the dual-path
/// transition (the smoke corpus still spells effects `IO<…>`).
#[test]
fn ambient_perform_missing_op_warns_and_mints_io() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &["x : IO<{ fs: Str }> @ [Effects({ puts: Str ~> Unit })] := perform(fs, \"x\")"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "missing op falls back to the IO mint (transition), got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning_containing(&result, "not in the ambient mode's `Effects` row"),
        "expected the missing-op transition warning, got {:?}",
        diags_of(&result)
    );
}

/// Un-annotated code is unaffected: `perform` with no ambient mode
/// mints `IO{op:T}` silently, exactly as before E2.
#[test]
fn unannotated_perform_unchanged() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["x : IO<{ puts: Str }> := perform(puts, \"a\")"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "un-annotated perform must check identically, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning_containing(&result, "ambient mode's `Effects` row"),
        "pure ambient must not warn, got {:?}",
        diags_of(&result)
    );
}

/// `handle m with h` as a modality `μ → μ′`: the handler's ops are
/// checked against the handled computation's `Effects` row (the
/// binding-carried mode), covered entries take the impl's return
/// type, and the residual mode propagates to the binding.
#[test]
fn ambient_handle_modality() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := s ~> s",
            "puts_h : Str ~> Str := s ~> s",
            "handled := handle f with { puts: puts_h }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "ambient handle should check coverage + residual, got {:?}",
        diags_of(&result)
    );
}

/// Coverage diagnostics are preserved on the ambient path: a handler
/// declaring an op the handled row never names is spurious.
#[test]
fn ambient_handle_spurious_impl_rejected() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := s ~> s",
            "fs_h : Str ~> Str := s ~> s",
            "bad := handle f with { fs: fs_h }",
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

/// `>>>` computes the effect rows: `{puts} ⋄ {fs} = {puts,fs}` — the
/// composed binding's mode is the union, and an ascription naming
/// both ops accepts it (row_width subsumption).
#[test]
fn composition_unions_effect_rows() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := s ~> s",
            "g : (Str ~> Str) @ [Effects({ fs: Str ~> Str })] := s ~> s",
            "h := g >>> f",
            "both : (Str ~> Str) @ [Effects({ puts: Str ~> Str, fs: Str ~> Str })] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "composed rows should union and subsume, got {:?}",
        diags_of(&result)
    );
}

/// Effect widening is directional: `h`'s `{puts, fs}` row does not
/// fit an ascription permitting only `{puts}` — `fs` is undecided
/// (`row_width`: every actual key must be in the expected row).
#[test]
fn subsumption_rejects_uncovered_effect() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := s ~> s",
            "g : (Str ~> Str) @ [Effects({ fs: Str ~> Str })] := s ~> s",
            "h := g >>> f",
            "narrow : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := h",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "widening beyond the ascribed row must reject, got {:?}",
        diags_of(&result)
    );
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("ambient mode subsumption")),
        "expected the subsumption diagnostic, got {:?}",
        diags_of(&result)
    );
}

/// A narrower actual row fits a wider ascription — `{puts} ≤
/// {puts, fs}` is exactly the `io_headed`/`row_width` direction.
#[test]
fn subsumption_accepts_narrower_row() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f : (Str ~> Str) @ [Effects({ puts: Str ~> Str })] := s ~> s",
            "wide : (Str ~> Str) @ [Effects({ puts: Str ~> Str, fs: Str ~> Str })] := f",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "narrower row must fit wider ascription, got {:?}",
        diags_of(&result)
    );
}
