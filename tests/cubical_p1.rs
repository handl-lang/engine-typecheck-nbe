//! ADR-0208 P1 — cubical interval & path types (NBE port).
//!
//! Mirrors `tool-typecheck-handl/tests/cubical_p1.rs`: the interval sort
//! `I`, the `Path`/`PathP` interval-function type formers, `refl`
//! introduction, lambda paths, `i0`/`i1` endpoints, and `#`/`j`
//! elimination over the `type.pi` + `path`-payload representation.

use handl_kit_data_prelude::language_from_yaml;
use handl_language::{Ast, AstNode, AstRef, Language};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;
use serde_json::json;

fn canonical_lang() -> Language {
    let lang = language_from_yaml().expect("canonical language should assemble");
    // Also install it process-wide: the NbeTypecheckEngine's HandlModule
    // bridge runtime resolves `default_language` lazily (ADR-0128 §0.8).
    let _ = handl_plugin_lib_handl::install_default_language(lang.clone());
    lang
}

fn parse_one(lang: &Language, src: &str) -> Ast {
    let tokens = lex(lang, src).expect("lex should succeed");
    let engine = PrattEngine::new();
    let req = ParserRequest::new(tokens);
    engine.parse(lang, &req).expect("parse should succeed")
}

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
    // The NBE engine reports fatal type errors as `Err(LangError)`; count
    // them as `error`-severity diagnostics (same convention as
    // `dependent_structure.rs`).
    match engine.typecheck(lang, &req) {
        Ok(result) => result,
        Err(e) => hnl_tool_typecheck::TypecheckResult {
            typ: serde_json::Value::Null,
            diagnostics: vec![hnl_tool_typecheck::TypecheckDiagnostic {
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

const NAT_VEC: &str = "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nVec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

/// `I` is the interval sort: `Pi(I, i ~> Bool)` is a dependent function
/// type over the interval, and a lambda checks against it (the binder
/// `i` is a dimension variable — an ordinary `var` index term).
#[test]
fn interval_sort_binds_dimension_variables() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "f : Pi(I, i ~> Bool) := i ~> true",
            "g : Pi(I, i ~> Nat) := i ~> Zero",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Pi over the interval sort should check with dimension binders, got {:?}",
        diags_of(&result)
    );
}

/// `Path(a, b)` is the cubical path type — a `type.pi` over `I` with a
/// `path` endpoint payload — not a `Path{frm,to}` record. `refl`
/// inhabits the reflexive case, `#` transports along it, and a
/// path-typed variable is itself a witness.
#[test]
fn path_type_is_interval_pi_with_endpoints() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "v : Vec(2, Bool) := VCons(true, VCons(false, VNil))",
            "p : Path(2, 2) := refl",
            "w : Vec(2, Bool) := p # v",
            "q : Path(2, 2) := p",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "refl/path-alias/transport over the interval Path should check, got {:?}",
        diags_of(&result)
    );
    // The recorded type of `p`'s annotation is a `type.pi` over
    // `type.interval` carrying `path` endpoints — never a `Path` tag.
    let saw_interval_path = result.node_types.values().any(|t| {
        t.get("__tag__").and_then(|g| g.as_str()) == Some("type.pi")
            && t.get("param")
                .and_then(|p| p.get("__tag__"))
                .and_then(|g| g.as_str())
                == Some("type.interval")
            && t.get("path").is_some()
    });
    assert!(
        saw_interval_path,
        "Path(2, 2) should synthesize a type.pi over type.interval with a path payload"
    );
}

/// `refl` proves only reflexive endpoints: `Path(2, 3)` is rejected,
/// and a heterogeneous `Path(Nat, Zero, Succ(Zero))` is rejected too.
#[test]
fn refl_rejects_non_refl_endpoints() {
    let lang = canonical_lang();
    let bad_index = parse_block(&lang, &[NAT_VEC, "bad : Path(2, 3) := refl"]);
    let result = typecheck(&lang, bad_index);
    assert!(
        has_error(&result),
        "refl should not inhabit Path(2, 3)"
    );

    let bad_term = parse_block(
        &lang,
        &[NAT_VEC, "bad : Path(Nat, Zero, Succ(Zero)) := refl"],
    );
    let result = typecheck(&lang, bad_term);
    assert!(
        has_error(&result),
        "refl should not inhabit Path(Nat, Zero, Succ(Zero))"
    );
}

/// `Path(A, x, y)` is the homogeneous interval function `(i:I) → A`:
/// `A` synthesizes as a type (constant family), `x`/`y` are endpoint
/// terms, and a lambda `i ~> body` is a genuine path introduction.
#[test]
fn homogeneous_path_accepts_lambda_and_refl() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "cp : Path(Bool, true, true) := i ~> true",
            "cr : Path(Bool, true, true) := refl",
            "cn : Path(Nat, Zero, Zero) := refl",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "lambda and refl should inhabit homogeneous Paths, got {:?}",
        diags_of(&result)
    );
}

/// `i0`/`i1` are the interval endpoints: closed terms of the interval
/// sort `I`, checked like any other typed constant.
#[test]
fn interval_endpoints_have_interval_type() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "e0 : I := i0",
            "e1 : Interval := i1",
            "f : Pi(I, i ~> I) := i ~> i0",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "i0/i1 should inhabit I (and Interval), got {:?}",
        diags_of(&result)
    );
}

/// `PathP(F, x, y)` is the heterogeneous path `(i:I) → F i` for
/// `F : I → Type` (written `Pi(I, i ~> …)`): `refl` inhabits the
/// reflexive endpoint pair and is rejected for distinct endpoints.
#[test]
fn pathp_heterogeneous_family() {
    let lang = canonical_lang();
    let ok = parse_block(
        &lang,
        &["p : PathP(Pi(I, i ~> Bool), true, true) := refl"],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "refl should inhabit PathP(Pi(I, i ~> Bool), true, true), got {:?}",
        diags_of(&result)
    );

    // `Zero`/`Succ(Zero)` are decided-distinct endpoint terms — `refl`
    // cannot prove them equal.
    let bad = parse_block(
        &lang,
        &[NAT_VEC, "p : PathP(Pi(I, i ~> Nat), Zero, Succ(Zero)) := refl"],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "refl should not inhabit PathP(…, Zero, Succ(Zero))"
    );
}
