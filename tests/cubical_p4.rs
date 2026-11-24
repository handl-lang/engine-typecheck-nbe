//! ADR-0208 P4 — computing univalence: `univalence(e)` / `ua(e)` /
//! `Univalence(e)` / `Equiv(A, B)` (NBE port).
//!
//! Mirrors `tool-typecheck-handl/tests/cubical_p4.rs`: `ua(e)` for
//! `e : A ≃ B` synthesizes a path-carrying `type.pi` over `I` whose
//! `path` payload records the endpoints `x := A`, `y := B` and the
//! normalized `equiv{dom, cod, to, fro}` record. Transport `p # v`
//! along such a path applies the equivalence at the type level —
//! `v`'s type unified against the domain yields the codomain —
//! `symm` swaps the recorded direction, `trans` composes the recorded
//! endpoints, and `transp`/`fam_at` read the path-Pi as the CCHM
//! `λi. Glue B (i=0) T̂` family — against the NBE typecheck engine.

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
    // `cubical_p1.rs`–`cubical_p3.rs`).
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

/// An `Equiv(Bool, Nat)` inhabitant: the record `{to : Bool ~> Nat,
/// fro : Nat ~> Bool}` checked against the `type.equiv` former through
/// the `unify_equiv_record` arm.
const BOOL_NAT_EQUIV: &str = "e : Equiv(Bool, Nat) := { to: x ~> Zero, fro: y ~> true }";

/// Does any recorded node type carry a `path.equiv` payload — the
/// univalence path marker minted by `synth_univalence_core`?
fn saw_equiv_path(result: &hnl_tool_typecheck::TypecheckResult) -> bool {
    result.node_types.values().any(|t| {
        t.get("path")
            .and_then(|p| p.get("equiv"))
            .map(|e| !e.is_null())
            .unwrap_or(false)
    })
}

/// `univalence(e)` — the keyword form — mints the path-carrying
/// `type.pi` (`path{x: Bool, y: Nat, equiv: {…}}`).
#[test]
fn univalence_keyword_forms_path() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, BOOL_NAT_EQUIV, "p := univalence(e)"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "univalence(e) should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        saw_equiv_path(&result),
        "univalence should mint a path.equiv payload, got {:?}",
        diags_of(&result)
    );
}

/// `ua(e)` — the call form — is intercepted by callee name in the
/// `call` judgment and mints the same path.
#[test]
fn ua_call_form_mints_same_path() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, BOOL_NAT_EQUIV, "p := ua(e)"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "ua(e) should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        saw_equiv_path(&result),
        "ua(e) should mint a path.equiv payload, got {:?}",
        diags_of(&result)
    );
}

/// `Univalence(e)` — the capitalized call form — behaves identically.
#[test]
fn capitalized_univalence_call_form() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, BOOL_NAT_EQUIV, "p := Univalence(e)"]);
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Univalence(e) should check, got {:?}",
        diags_of(&result)
    );
    assert!(
        saw_equiv_path(&result),
        "Univalence(e) should mint a path.equiv payload, got {:?}",
        diags_of(&result)
    );
}

/// Transport along a `ua(e)` path applies the equivalence at the type
/// level: `p : Bool ~ Nat`, `v : Bool` transports to `Nat` —
/// `p # v : Nat` checks against `Nat` and is rejected against `Bool`.
#[test]
fn transport_along_univalence_applies_equivalence() {
    let lang = canonical_lang();
    let ok = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "v : Bool := true",
            "t : Nat := p # v",
        ],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "p # v should inhabit the codomain Nat, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "v : Bool := true",
            "t : Bool := p # v",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "p # v should not inhabit the domain Bool"
    );
}

/// `symm` swaps the recorded equivalence: `symm p : Nat ~ Bool`, so
/// transporting a `Nat` along it yields `Bool`.
#[test]
fn symm_swaps_univalence_direction() {
    let lang = canonical_lang();
    let ok = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "s := symm p",
            "w : Nat := Zero",
            "t : Bool := s # w",
        ],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "symm(ua e) # w should inhabit Bool, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "s := symm p",
            "w : Nat := Zero",
            "t : Nat := s # w",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(has_error(&result), "symm(ua e) # w should not inhabit Nat");
}

/// `trans` composes recorded equivalences: `ua(e₁) : Bool ~ Nat` and
/// `ua(e₂) : Nat ~ Bool` compose to `Bool ~ Bool` — transporting
/// `true : Bool` stays in `Bool`.
#[test]
fn trans_composes_univalence_paths() {
    let lang = canonical_lang();
    let nat_bool: &str = "e2 : Equiv(Nat, Bool) := { to: n ~> true, fro: b ~> Zero }";
    let ok = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            nat_bool,
            "p := ua(e)",
            "q := ua(e2)",
            "t := trans p, q",
            "r : Bool := t # true",
        ],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "trans(ua e, ua e2) # true should inhabit Bool, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            nat_bool,
            "p := ua(e)",
            "q := ua(e2)",
            "t := trans p, q",
            "r : Nat := t # true",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "trans(ua e, ua e2) # true should not inhabit Nat"
    );
}

/// `transp` over a `ua(e)` path family reads the CCHM Glue family:
/// `fam_at` maps `i0 ↦ A`, `i1 ↦ B` — transport of a `Bool` partial
/// along `i0 → i1` yields the `Nat → Bool`-shaped arrow without
/// error.
#[test]
fn transp_over_univalence_family() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, BOOL_NAT_EQUIV, "t := transp ua(e), i0, fF, true"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "transp over a ua(e) family should check, got {:?}",
        diags_of(&result)
    );
}

/// A bare record without `Equiv` annotation still works as an
/// equivalence operand when its `to`/`fro` fields are arrows in
/// opposite directions — `ua` reads the endpoints off the field
/// types.
#[test]
fn ua_accepts_unannotated_equivalence_record() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            "e := { to: x ~> Zero, fro: y ~> true }",
            "p := ua(e)",
            "v : Bool := true",
            "t : Nat := p # v",
        ],
    );
    let result = typecheck(&lang, ast);
    // The unannotated lambdas carry metavariable endpoints that the
    // `fro`/`to` cross-check instantiates to `Bool ~ Nat`; transport
    // then applies the equivalence as usual.
    assert!(
        !has_error(&result),
        "ua over a record literal should not error, got {:?}",
        diags_of(&result)
    );
}

/// `ua`/`univalence` reject non-equivalence operands: a literal and
/// a non-arrow record are errors, not silent `type.unknown` paths.
#[test]
fn univalence_rejects_non_equivalence() {
    let lang = canonical_lang();
    let lit = parse_block(&lang, &[NAT_VEC, "bad := ua(42)"]);
    let result = typecheck(&lang, lit);
    assert!(
        has_error(&result),
        "ua(42) should be an error, got {:?}",
        diags_of(&result)
    );

    let kw = parse_block(&lang, &[NAT_VEC, "bad := univalence(true)"]);
    let result = typecheck(&lang, kw);
    assert!(
        has_error(&result),
        "univalence(true) should be an error, got {:?}",
        diags_of(&result)
    );

    let rec = parse_block(&lang, &[NAT_VEC, "bad := ua({ to: 1 })"]);
    let result = typecheck(&lang, rec);
    assert!(
        has_error(&result),
        "ua of a non-arrow record should be an error, got {:?}",
        diags_of(&result)
    );
}

/// `Equiv(A, B)` inhabitants need both directions: a record missing
/// `fro` does not check against `type.equiv`.
#[test]
fn equiv_record_requires_both_directions() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[NAT_VEC, "bad : Equiv(Bool, Nat) := { to: x ~> Zero }"],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "Equiv record without `fro` should be an error, got {:?}",
        diags_of(&result)
    );
}

/// `Equiv` with the wrong arity warns rather than silently forming a
/// one-sided equivalence.
#[test]
fn equiv_former_arity_warning() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &[NAT_VEC, "bad : Equiv(Bool) := { to: x ~> x }"]);
    let result = typecheck(&lang, ast);
    assert!(
        has_warning(&result, "Equiv needs a domain and a codomain"),
        "Equiv(Bool) should warn about arity, got {:?}",
        diags_of(&result)
    );
}

/// Transport along a univalence path rewrites *dependent* occurrences
/// of the domain: `Vec(2, Bool)` mentions `Bool`, so `p # v` has type
/// `Vec(2, Nat)` — the `subst_index_` fallback inside the univalence
/// branch.
#[test]
fn transport_rewrites_dependent_domain_occurrences() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "v : Vec(2, Bool) := VCons(true, VCons(false, VNil))",
            "t : Vec(2, Nat) := p # v",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Vec(2, Bool) should transport to Vec(2, Nat), got {:?}",
        diags_of(&result)
    );
}

/// Transport along a univalence path whose value type does not
/// mention the domain warns and passes through — the same
/// approximation discipline as the P2 partial checks.
#[test]
fn transport_unrelated_type_warns() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            NAT_VEC,
            BOOL_NAT_EQUIV,
            "p := ua(e)",
            "v : Nat := Zero",
            "t := p # v",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "transport of an unrelated type should warn, not error, got {:?}",
        diags_of(&result)
    );
    assert!(
        has_warning(&result, "does not mention the equivalence domain"),
        "expected the pass-through warning, got {:?}",
        diags_of(&result)
    );
}
