//! ADR-0208 P2 — cubical judgments: `transp`/`hcomp`/`Comp`/`trans`/`symm`
//! (NBE port).
//!
//! Mirrors `tool-typecheck-handl/tests/cubical_p2.rs`: CCHM-style
//! transport in an `I → Type` family, homogeneous and heterogeneous
//! composition, face formulas (`f0(i)`/`f1(i)`/`fnot`/`f_or`/`f_and`/
//! `fT`/`fF`), the `i : I` and `i ∉ φ` side conditions, and the derived
//! path algebra (`trans`/`symm`) — against the NBE typecheck engine.

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
    // `cubical_p1.rs`).
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

/// `transp A, i, φ, u` — transport in the family `A : I → Type` along
/// `i`, partial on `φ`. CCHM direction: the result is the non-dependent
/// arrow `A(i1) → A(i0)`. `fF` is the empty face; `true` inhabits the
/// constant `Bool` fibre.
#[test]
fn transp_transports_partial_element() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "t := transp Pi(I, i ~> Bool), i0, fF, true"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "transp over a constant family should check, got {:?}",
        diags_of(&result)
    );
    // The transp node's recorded type is the arrow `A(i1) → A(i0)` —
    // here `Bool → Bool` (a `type.function`).
    let saw_arrow = result
        .node_types
        .values()
        .any(|t| t.get("__tag__").and_then(|g| g.as_str()) == Some("type.function"));
    assert!(
        saw_arrow,
        "transp should synthesize a type.function result, got {:?}",
        diags_of(&result)
    );
}

/// `transp` requires `i : I` — a `Bool` dimension operand is rejected.
#[test]
fn transp_rejects_non_interval_dimension() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "bad := transp Pi(I, i ~> Bool), true, fF, true"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "transp with a non-I dimension should be an error"
    );
}

/// The dimension must not occur in the face: `f0(i)` with `i` the
/// transported dimension variable is rejected; `f0(i1)` (a face over
/// the *other* endpoint) is admitted.
#[test]
fn transp_rejects_dimension_in_face() {
    let lang = canonical_lang();
    let bad = parse_block(
        &lang,
        &[
            NAT_VEC,
            "bad : Path(Bool, true, true) := i ~> transp Pi(I, j ~> Bool), i, f0(i), true",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "transp with i occurring in φ should be an error"
    );

    let ok = parse_block(
        &lang,
        &[NAT_VEC, "ok := transp Pi(I, i ~> Bool), i0, f0(i1), true"],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "transp with a face over a different dimension should check, got {:?}",
        diags_of(&result)
    );
}

/// Face formulas normalize: `f_and(fF, f0(i1))` is the empty face
/// (ff ∧ φ = ff), `f_or(fF, fT)` is full — both are valid faces not
/// containing the transported dimension.
#[test]
fn transp_accepts_normalized_faces() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "a := transp Pi(I, i ~> Bool), i0, f_and(fF, f0(i1)), true",
            "b := transp Pi(I, i ~> Bool), i0, f_or(fF, fT), true",
            "c := transp Pi(I, i ~> Bool), i0, fnot(fF), true",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "normalized face formulas should check, got {:?}",
        diags_of(&result)
    );
}

/// A non-face operand (`42`) is rejected — the face position requires
/// a face formula.
#[test]
fn transp_rejects_non_face_operand() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "bad := transp Pi(I, i ~> Bool), i0, 42, true"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "transp with a non-face third operand should be an error"
    );
}

/// `hcomp A, i, φ, u, cap` — homogeneous composition over a single
/// type: `cap : A` is the `i = 0` face (checked strictly), the result
/// is `: A`.
#[test]
fn hcomp_composes_over_single_type() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "h : Bool := hcomp Bool, i0, fF, true, false"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "hcomp over Bool should check, got {:?}",
        diags_of(&result)
    );
}

/// `hcomp`'s cap must inhabit `A` — a `Nat` cap over `Bool` is an
/// error.
#[test]
fn hcomp_rejects_wrong_cap() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "bad : Bool := hcomp Bool, i0, fF, true, Zero"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "hcomp with a cap outside A should be an error"
    );
}

/// `Comp F, i, φ, u, cap` — heterogeneous composition over
/// `F : I → Type`: `cap : F(i0)`, result `: F(i1)`. Over the constant
/// `Bool` family both endpoints are `Bool`.
#[test]
fn comp_composes_over_family() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "c : Bool := Comp Pi(I, i ~> Bool), i0, fF, true, false",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Comp over an I → Type family should check, got {:?}",
        diags_of(&result)
    );
}

/// `Comp`'s cap must inhabit `F(i0)` — `Zero : Nat` is not `Bool`.
#[test]
fn comp_rejects_wrong_cap() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "bad : Bool := Comp Pi(I, i ~> Bool), i0, fF, true, Zero",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "Comp with a cap outside F(i0) should be an error"
    );
}

/// `trans p, q` — derived path composition: `p : x ~ y`, `q : y ~ z`
/// yields `x ~ z`. Path-typed lambdas give decided-distinct endpoints.
#[test]
fn trans_composes_paths() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "p : Path(Nat, Zero, Succ(Zero)) := i ~> Zero",
            "q : Path(Nat, Succ(Zero), Succ(Succ(Zero))) := i ~> Succ(Zero)",
            "t : Path(Nat, Zero, Succ(Succ(Zero))) := trans p, q",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "trans of composable paths should check, got {:?}",
        diags_of(&result)
    );
}

/// `trans` requires the first path's target to unify with the second
/// path's source — `p : Zero ~ Succ(Zero)` does not compose with
/// `r : Zero ~ Succ(Zero)` (target `Succ(Zero)` ≠ source `Zero`).
#[test]
fn trans_rejects_incompatible_endpoints() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "p : Path(Nat, Zero, Succ(Zero)) := i ~> Zero",
            "r : Path(Nat, Zero, Succ(Zero)) := i ~> Zero",
            "bad : Path(Nat, Zero, Succ(Zero)) := trans p, r",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "trans of non-composable paths should be an error"
    );
}

/// `symm p` — derived path symmetry: `p : x ~ y` yields `y ~ x`.
#[test]
fn symm_swaps_path_endpoints() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "p : Path(Nat, Zero, Succ(Zero)) := i ~> Zero",
            "s : Path(Nat, Succ(Zero), Zero) := symm p",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "symm should swap path endpoints, got {:?}",
        diags_of(&result)
    );
}

/// `symm` of a non-path operand is an error.
#[test]
fn symm_rejects_non_path() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "bad := symm 42"]);
    let result = typecheck(&lang, ast);
    assert!(has_error(&result), "symm of a non-path should be an error");
}

/// `trans`/`symm` results are themselves path witnesses: a composed
/// path feeds `#` transport and `J`-style elimination positions.
#[test]
fn trans_result_is_a_path_witness() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "p : Path(Nat, Zero, Succ(Zero)) := i ~> Zero",
            "q : Path(Nat, Succ(Zero), Succ(Succ(Zero))) := i ~> Succ(Zero)",
            "t : Path(Nat, Zero, Succ(Succ(Zero))) := trans p, q",
            "s : Path(Nat, Succ(Succ(Zero)), Zero) := symm t",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "symm(trans p, q) should check as a composed witness, got {:?}",
        diags_of(&result)
    );
}

/// Lowercase `comp` remains an ordinary identifier — only the capital
/// `Comp` spelling is the heterogeneous composition keyword (the `J`
/// precedent).
#[test]
fn lowercase_comp_is_still_an_identifier() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "comp := 1", "use := comp"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "lowercase comp should still parse as an identifier, got {:?}",
        diags_of(&result)
    );
}

/// Production parity (ADR-0208 P1 gap): `hnl typecheck` seeds every
/// prelude wire name — including `i0`/`i1` — as a `type.unknown`
/// global placeholder. The placeholder must not shadow the endpoints'
/// `I` sort: `transp`/`hcomp`/`Comp` accept the literals directly,
/// exactly as the unseeded fixtures above.
#[test]
fn transp_accepts_seeded_interval_literals() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "t := transp Pi(I, i ~> Bool), i0, fF, true",
            "u := transp Pi(I, i ~> Bool), i1, fF, true",
        ],
    );
    let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    let mut engine: Box<dyn TypecheckEngine> = Box::new(engine);
    for name in ["i0", "i1"] {
        engine = engine.with_global(
            name,
            json!({"__tag__": "type.unknown", "__registry__": "handl.canonical"}),
        );
    }
    let req = TypecheckRequest::new(ast);
    let result = match engine.typecheck(&lang, &req) {
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
    };
    assert!(
        !has_error(&result),
        "seeded i0/i1 literals should still synthesize type.interval, got {:?}",
        diags_of(&result)
    );
}
