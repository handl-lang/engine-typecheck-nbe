//! ADR-0208 P3 — cubical Glue & fibrancy: `glue`/`Glue`/`unglue`/
//! `isFibrant`/`fibrant` (NBE port).
//!
//! Mirrors `tool-typecheck-handl/tests/cubical_p3.rs`: CCHM-style glue
//! type formers over the P1 interval and P2 faces — `glue A, φ, T̂` /
//! `Glue(A, φ, T̂)` mint `type.glue{family, face, section}`, `unglue(G)`
//! projects back to the family, and `isFibrant(A)`/`fibrant A` form and
//! inhabit the fibrancy type via the structural `fibrant_verdict` —
//! against the NBE typecheck engine.

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
    // `cubical_p1.rs`/`cubical_p2.rs`).
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

fn has_warning(result: &hnl_tool_typecheck::TypecheckResult, needle: &str) -> bool {
    result
        .diagnostics
        .iter()
        .any(|d| d.severity == "warning" && d.message.contains(needle))
}

const NAT_VEC: &str = "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }\nVec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

/// `glue A, φ, T̂` — the keyword form of the glue type former. A
/// constant family `Bool`, the empty face `fF` (which normalizes to
/// `ff` — the degenerate "Glue ≡ family" case, warned not errored),
/// and an interval section `Pi(I, i ~> Bool)`. The synthesized node
/// type is the `type.glue` record it forms.
#[test]
fn glue_keyword_forms_type_glue() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "G := glue Bool, fF, Pi(I, i ~> Bool)"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "glue formation should check, got {:?}",
        diags_of(&result)
    );
    let saw_glue = result
        .node_types
        .values()
        .any(|t| t.get("__tag__").and_then(|g| g.as_str()) == Some("type.glue"));
    assert!(
        saw_glue,
        "glue should synthesize a type.glue record, got {:?}",
        diags_of(&result)
    );
}

/// `Glue(A, φ, T̂)` — the call form — plus `unglue(G)`: the type-level
/// projection back to the underlying family. `unglue(Glue(Bool, …))`
/// is `Bool`, so `true` inhabits it.
#[test]
fn glue_call_form_and_unglue_projection() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "y : unglue(Glue(Bool, fF, Pi(I, i ~> Bool))) := true",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Glue call form + unglue projection should check, got {:?}",
        diags_of(&result)
    );
}

/// `unglue` requires a `type.glue` operand — `unglue(Bool)` is an
/// error, not a silent `Bool`.
#[test]
fn unglue_rejects_non_glue() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "bad : unglue(Bool) := true"]);
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "unglue of a non-Glue type should be an error"
    );
}

/// The face operand must be a face formula — `42` is rejected (the
/// P2 `face_of_val_` discipline).
#[test]
fn glue_rejects_non_face_operand() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "bad := glue Bool, 42, Pi(I, i ~> Bool)"]);
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "glue with a non-face second operand should be an error"
    );
}

/// A section that is not an `I → Type` family warns (the
/// `Partial φ (Σ T:Type, T ≃ A i)` obligation is approximated) but
/// does not fail formation.
#[test]
fn glue_warns_on_non_family_section() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "G := glue Bool, fF, Bool"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "non-family section should warn, not error, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning(&result, "not an I → Type family"),
        "expected the non-family-section warning, got {:?}",
        diags_of(&result)
    );
}

/// `transp` over a `Glue` family: `fam_at` reads the off-face
/// family `A` (documented approximation — `φ(point)` is undecided
/// until P4). `Glue(Pi(I, i ~> Bool), …)` transports `Bool` along
/// `i0 → i1` giving the `Bool → Bool` arrow.
#[test]
fn transp_over_glue_family() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "t := transp Glue(Pi(I, i ~> Bool), fF, Pi(I, i ~> Bool)), i0, fF, true",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "transp over a Glue family should check, got {:?}",
        diags_of(&result)
    );
    let saw_arrow = result
        .node_types
        .values()
        .any(|t| t.get("__tag__").and_then(|g| g.as_str()) == Some("type.function"));
    assert!(
        saw_arrow,
        "transp over Glue should synthesize a type.function result, got {:?}",
        diags_of(&result)
    );
}

/// `isFibrant(A)` / `fibrant A` — the fibrancy type and its witness.
/// Base sorts (`Bool`) and dependent `Pi`/`Sigma` of fibrant parts
/// derive `yes`; `pf : isFibrant(A) := fibrant A` checks through the
/// `type.fibrant` unification arm.
#[test]
fn isfibrant_base_pi_and_sigma() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "pf : isFibrant(Bool) := fibrant Bool",
            "pf2 : isFibrant(Pi(I, i ~> Bool)) := fibrant Pi(I, i ~> Bool)",
            "pf3 : isFibrant(Pi(Bool, b ~> Bool)) := fibrant Pi(Bool, b ~> Bool)",
            "pf4 : isFibrant(Sigma(Bool, b ~> Bool)) := fibrant Sigma(Bool, b ~> Bool)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "isFibrant/fibrant over base sorts and Pi/Sigma of fibrant should check, got {:?}",
        diags_of(&result)
    );
}

/// `Glue` of fibrant parts is fibrant: `isFibrant(Glue(Bool, fF,
/// Pi(I, i ~> Bool)))` derives `yes` (family `Bool` fibrant, section
/// `I → Bool` fibrant) and the witness checks — exercising the
/// `type.glue` arm of `dep_unify`.
#[test]
fn isfibrant_glue_of_fibrant() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "pf : isFibrant(Glue(Bool, fF, Pi(I, i ~> Bool))) := fibrant Glue(Bool, fF, Pi(I, i ~> Bool))",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "isFibrant of a Glue of fibrant parts should check, got {:?}",
        diags_of(&result)
    );
}

/// A non-degenerate face: `f0(i)` inside an interval `Pi` body — the
/// dimension variable makes `φ` neither `tt` nor `ff`, so no
/// degenerate-face warning fires.
#[test]
fn glue_accepts_dimension_face() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "pf : isFibrant(Pi(I, i ~> Glue(Bool, f0(i), Pi(I, j ~> Bool)))) := fibrant Pi(I, i ~> Glue(Bool, f0(i), Pi(I, j ~> Bool)))",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "glue over a dimension face inside Pi should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        !has_warning(&result, "degenerate"),
        "a dimension face should not warn as degenerate, got {:?}",
        diags_of(&result)
    );
}

/// `fibrant Type` — the universe is not fibrant: the witness is a
/// hard error, and `isFibrant(Type)` leaves the fibrancy type open
/// (`type.unknown` + warning, never silent).
#[test]
fn fibrant_rejects_universe() {
    let lang = canonical_lang();
    let witness = parse_block(&lang, &[NAT_VEC, "bad := fibrant Type"]);
    let result = typecheck(&lang, witness);
    assert!(
        has_error(&result),
        "fibrant Type should be an error — Type is not fibrant"
    );

    let former = parse_block(&lang, &[NAT_VEC, "bad : isFibrant(Type) := fibrant Type"]);
    let result = typecheck(&lang, former);
    assert!(
        has_error(&result),
        "isFibrant(Type) with a fibrant Type witness should be an error"
    );
    assert!(
        has_warning(&result, "not fibrant"),
        "expected the isFibrant not-fibrant warning, got {:?}",
        diags_of(&result)
    );
}

/// Undecided fibrancy is `unknown` + warning, not silent acceptance:
/// `Vec(2, Bool)` is an indexed `type.family_app` — fibrancy depends
/// on the indices the checker does not inspect — so `isFibrant`
/// degrades to `type.unknown` + warning while `fibrant` still
/// synthesizes the witness optimistically.
#[test]
fn isfibrant_undecided_warns_not_silent() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "pf : isFibrant(Vec(2, Bool)) := fibrant Vec(2, Bool)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "undecided fibrancy should warn, not error, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning(&result, "undecided"),
        "expected the undecided-fibrancy warning, got {:?}",
        diags_of(&result)
    );
}

/// `Glue`/`isFibrant`/`unglue` are ordinary call-form type names —
/// no keywords needed (`Glue`, `isFibrant`, `unglue` stay identifiers
/// elsewhere). `Glue` with the wrong arity degrades to
/// `type.unknown` + warning.
#[test]
fn call_form_arity_is_checked() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "bad : Glue(Bool, fF) := true"]);
    let result = typecheck(&lang, ast);
    // `Glue(Bool, fF)` warns (needs 3 args) → type.unknown annotation
    // unifies with anything; the warning must be present.
    assert!(
        has_warning(&result, "Glue needs"),
        "expected the Glue arity warning, got {:?}",
        diags_of(&result)
    );
}
