//! ADR-0120 Phase 4 — dependent structure tests.
//!
//! Exercises the `Family(A)` / `Pi(A, f)` / `Sigma(A, f)` surface type
//! formers, dependent introduction rules, `data` family declarations,
//! reindexing-as-composition (`f >>> P`), and the `Σ_f ⊣ f* ⊣ Π_f`
//! adjunction law terms against the Handl-authored typecheck model.
//!
//! These tests pin *structural* support: index definitional equality on
//! computable terms (`Vec(1 + 1)` ≡ `Vec(2)`), constructor argument
//! checking through freshly instantiated `type.ctor_sig` signatures, and
//! constructor-pattern binding in `match`. Strict positivity is enforced
//! by the desugar native. *Dependent elimination* is covered by
//! `match_refines_scrutinee_index_per_branch` and friends below: each arm
//! unifies the constructor's instantiated result with the scrutinee's
//! family index — metavariable indices solve (linearly), rigid `var`
//! indices are rewritten through the branch-local `results` environment
//! with a `Path` witness bound as `eq_<name>`, and a decided literal
//! mismatch auto-discharges the arm (body unchecked but allowed).

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

/// Parse a single statement.
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

/// Parse several statements and combine them under a synthetic `block` root
/// so the model's `typecheck` sees them as one program.
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
    // The NBE engine reports fatal type errors as `Err(LangError)`; the
    // parity harness counts them as `error`-severity diagnostics so
    // `has_error` behaves identically to the Handl model.
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

/// The dependent surface formers and their introduction rules all check
/// cleanly: `Family` accepts an index predicate arrow, `Pi` checks a
/// function against the dependent function type, `Sigma` checks a pair
/// literal componentwise, and `data` binds the family and constructors.
#[test]
fn dependent_surface_forms_typecheck() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "x : Nat = 5",
            "fam : Family(Nat) = n ~> n > 0",
            "mk : Pi(Nat, n ~> Nat) = n ~> n",
            "mk2 : Pi(Nat, n ~> Nat) = n ~> n + 1",
            "p : Sigma(Nat, n ~> Nat) = [3, 4]",
            "Bool := { True : Bool, False : Bool }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "dependent surface forms should check cleanly, got {:?}",
        diags_of(&result)
    );
}

/// Reindexing needs no primitive: `f >>> P` is ordinary composition and
/// synthesizes a family over `f`'s source (the pullback `f* P`).
#[test]
fn reindexing_is_ordinary_composition() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "P : Family(Nat) = n ~> n > 0",
            "f = n ~> n + 1",
            "Q = f >>> P",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "f >>> P should typecheck as reindexing, got {:?}",
        diags_of(&result)
    );
}

/// The `Σ_f ⊣ f* ⊣ Π_f` adjunction laws are ordinary `~[2]~>` 2-cell
/// terms bound as arrows over their participants — the same mechanism as
/// every other law, no dedicated subsystem.
#[test]
fn adjunction_laws_are_2cell_terms() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "reindex = (f, P) ~> f >>> P",
            "sigma_unit = (Q, f, Sigma_f) ~> Q ~[2]~> (f >>> Sigma_f(Q))",
            "sigma_counit = (P, f, Sigma_f) ~> Sigma_f((f >>> P)) ~[2]~> P",
            "pi_unit = (P, f, Pi_f) ~> P ~[2]~> Pi_f((f >>> P))",
            "pi_counit = (Q, f, Pi_f) ~> (f >>> Pi_f(Q)) ~[2]~> Q",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "adjunction law terms should check cleanly, got {:?}",
        diags_of(&result)
    );
}

/// Index terms compute: `1 + 1`, `2 - 0`, `2 * 1` all normalize to `2`,
/// so a literal `Vec(2, Bool)` accepts a value constructed at those
/// indices. Constructor uses are freshly instantiated, so two uses of
/// `VCons` do not share solved metavariables.
#[test]
fn dependent_indices_compute_definitionally() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "v1 : Vec(1 + 1, Bool) = VCons(true, VCons(false, VNil))",
            "v2 : Vec(2 - 0, Bool) = VCons(true, VCons(false, VNil))",
            "v3 : Vec(2 * 1, Bool) = VCons(true, VCons(false, VNil))",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "computable index terms should be definitionally equal, got {:?}",
        diags_of(&result)
    );
}

/// A `data`-declared family shadows the builtin spelling, so
/// `Succ(pred : Nat)` takes the declared `Nat` family — `Succ(Zero)`
/// checks and `Succ(3)` does not.
#[test]
fn dependent_constructors_check_arguments() {
    let lang = canonical_lang();
    let good = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "x : Nat = Succ(Zero)",
            "y : Nat = Succ(Succ(Zero))",
        ],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "constructor applications at the right types should check, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "x : Nat = Succ(3)",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "Succ(3) should be rejected — pred : Nat, not Nat-number"
    );
}

/// Indexed constructors validate both the length index and the element
/// type: a 2-vector at `Vec(3, Bool)` and a string element at
/// `Vec(_, Bool)` are both rejected.
#[test]
fn dependent_constructors_reject_wrong_indices() {
    let lang = canonical_lang();
    let decl = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

    let wrong_len = parse_block(
        &lang,
        &[decl, "v : Vec(3, Bool) = VCons(true, VCons(false, VNil))"],
    );
    let result = typecheck(&lang, wrong_len);
    assert!(
        has_error(&result),
        "a 2-element vector should not check at Vec(3, Bool)"
    );

    let wrong_elem = parse_block(&lang, &[decl, "v : Vec(2, Bool) = VCons(\"s\", VNil)"]);
    let result = typecheck(&lang, wrong_elem);
    assert!(
        has_error(&result),
        "a string element should not check at Vec(1, Bool)"
    );
}

/// Dependent unification recurses through `type.function`: a family
/// application in the codomain still gets computational index equality.
#[test]
fn dependent_family_unifies_inside_function() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "f : Nat ~> Vec(1 + 1, Bool) = n ~> VCons(true, VCons(false, VNil))",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "family_app in a codomain should unify by normalized indices, got {:?}",
        diags_of(&result)
    );
}

/// Phase-3 WS-D — `data`-declared ctor index projections: `data Nat`'s
/// `Succ`/`Zero` register `App{Succ, x} ↦ x + 1` / `Zero ↦ 0` at
/// `synth_data` (the Nat-former shape — a unary same-family ctor is
/// successor arithmetic, the unique bare ctor is `0`), so `Vec(Succ n)`
/// and `Vec(n + 1)` are *definitionally* equal in both directions,
/// nested, and at literals.
#[test]
fn succ_index_bridges_to_arithmetic() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "fwd : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Vec(n + 1, Bool))) = n ~> v ~> v",
            "bwd : Pi(Nat, n ~> Pi(Vec(n + 1, Bool), v ~> Vec(Succ(n), Bool))) = n ~> v ~> v",
            "nested : Pi(Nat, n ~> Pi(Vec(Succ(Succ(n)), Bool), v ~> Vec(n + 2, Bool))) = n ~> v ~> v",
            "one : Vec(Succ(Zero), Bool) = VCons(true, VNil)",
            "nil : Vec(Zero, Bool) = VNil",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Vec(Succ n) ≡ Vec(n + 1) should hold definitionally, got {:?}",
        diags_of(&result)
    );
}

/// A `Vec(Succ n)` scrutinee refines through the bridge: the VCons
/// arm's instantiated `?k + 1` result index meets the bridged `n + 1`
/// — the arm is provably applicable and its fields check. (VNil stays
/// *possible* under the rigid `n` — coverage is WS-C's split tree; the
/// single-arm match is accepted either way.)
#[test]
fn match_on_succ_indexed_scrutinee_refines() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "head : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Bool)) = n ~> v ~> match v { VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "match on Vec(Succ n) should refine the VCons arm, got {:?}",
        diags_of(&result)
    );
}

/// The bridge does not over-accept: `Vec(Succ n)` vs `Vec(n + 2)` is a
/// real mismatch, `Vec(Succ Zero)` is decided `Vec(1)` (a 2-vector is
/// rejected), and an *undeclared* head `f` applied at index level
/// stays an honest `App` atom — `f(n) ≢ n + 1` is stuck, not silently
/// accepted.
#[test]
fn succ_projection_keeps_decided_contract() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let off_by_one = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "bad : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Vec(n + 2, Bool))) = n ~> v ~> v",
        ],
    );
    let result = typecheck(&lang, off_by_one);
    assert!(
        has_error(&result),
        "Vec(n + 1) vs Vec(n + 2) should not unify"
    );
    let decided = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "bad : Vec(Succ(Zero), Bool) = VCons(true, VCons(false, VNil))",
        ],
    );
    let result = typecheck(&lang, decided);
    assert!(
        has_error(&result),
        "a 2-vector at Vec(Succ Zero) ≡ Vec(1) is a decided mismatch"
    );
    // An opaque index function — results-bound, no declared projection —
    // keeps `App{f, n}` honest: never silently equal to `n + 1`.
    let opaque = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "g : Pi(Nat ~> Nat, f ~> Pi(Nat, n ~> Pi(Vec(f(n), Bool), v ~> Vec(n + 1, Bool)))) = f ~> n ~> v ~> v",
        ],
    );
    let result = typecheck(&lang, opaque);
    assert!(
        has_error(&result),
        "an undeclared index head should stay stuck, not silently accept"
    );
}

/// Constructor patterns in `match` bind their fields against the
/// instantiated constructor domains. On a non-indexed family there is
/// no index to refine; the dependent-elimination tests below cover the
/// indexed case.
#[test]
fn match_constructor_patterns_bind_arguments() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "f = v ~> match v { Zero: Zero, Succ(p): p }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "constructor patterns should check cleanly, got {:?}",
        diags_of(&result)
    );
}

/// Dependent elimination: `match` unifies each constructor's
/// instantiated result type with the scrutinee's family application
/// *inside that arm*. The scrutinee `v : Vec(n + 0, Bool)` carries a
/// rigid `var{n}` index (a compound index term evaluates `n` as a term
/// variable, bound symbolically by the outer `Pi`), so the `VNil` arm
/// carries the equation `n ≡ 0` — realized as a branch-local `results`
/// rewrite witnessed by a `Path` under `eq_n` — and the `VCons` arm
/// carries `n ≡ ?k + 1`. The `return (i, w) ~> Vec(i, Bool) with`
/// motive makes the refinement observable: `v`'s fixed type
/// `Vec(n + 0, Bool)` only checks at `Vec(0, Bool)`/`Vec(?k + 1, Bool)`
/// because `n` is rewritten under each arm. Without refinement both
/// bodies would fail (`var{n}` does not unify with `0` or `?k + 1`).
#[test]
fn match_refines_scrutinee_index_per_branch() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "id_vec : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Vec(n + 0, Bool))) = n ~> v ~> match v return (i, w) ~> Vec(i, Bool) with { VNil: v, VCons(x, xs): v }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "motive-indexed match should refine the rigid scrutinee index per branch, got {:?}",
        diags_of(&result)
    );
}

/// Index equations solve through the linear arithmetic fragment:
/// matching `v : Vec(2, Bool)` with `VCons(x, xs) : Vec(?k + 1, ?a)`
/// solves `?k := 1`, so `xs` is a `Vec(1, Bool)` inside the arm — the
/// nested match on `xs` then solves `?j + 1 ≡ 1` for its own `VCons`
/// arm, leaving `ys : Vec(0, Bool)` and `y : Bool`.
#[test]
fn match_refinement_solves_linear_index() {
    let lang = canonical_lang();
    let decl = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";
    let ast = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "g : Bool = match v { VCons(x, xs): match xs { VCons(y, ys): y } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "?k + 1 ≡ 2 should solve ?k := 1 so the nested Vec(1) match refines, got {:?}",
        diags_of(&result)
    );
}

/// A constructor whose result index *cannot* equal the scrutinee's is
/// *auto-discharged*: `VNil : Vec(0, ?a)` can never produce a
/// `Vec(2, Bool)` scrutinee, so the arm is skipped — coverage doesn't
/// require it, `absurd` isn't needed, and its body is unchecked but
/// allowed (here `VNil`, which would fail the arm-type join against
/// `Bool` if it were ever checked). Under `xs : Vec(1, Bool)` (refined
/// from `?k + 1 ≡ 2`) an inner `VNil` arm discharges the same way.
/// Contrast: the same ill-typed body *is* checked where the arm is
/// reachable.
#[test]
fn match_auto_discharges_decided_impossible_arm() {
    let lang = canonical_lang();
    let decl = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

    let direct = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "ok : Bool = match v { VNil: VNil, VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, direct);
    assert!(
        !has_error(&result),
        "a VNil arm on a Vec(2, Bool) scrutinee is decided-impossible — \
         auto-discharged with an unchecked body, got {:?}",
        diags_of(&result)
    );

    let nested = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "ok : Bool = match v { VCons(x, xs): match xs { VNil: VNil, VCons(y, ys): y } }",
        ],
    );
    let result = typecheck(&lang, nested);
    assert!(
        !has_error(&result),
        "under xs : Vec(1, Bool) the inner VNil arm is decided-impossible — \
         auto-discharged, got {:?}",
        diags_of(&result)
    );

    // Contrast: on a `Vec(1, Bool)` scrutinee the `VCons` arm is
    // reachable — the ill-typed body is checked and fails.
    let live = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(1, Bool) = VCons(true, VNil)",
            "bad : Bool = match v { VCons(x, xs): VNil, VNil: true }",
        ],
    );
    let result = typecheck(&lang, live);
    assert!(
        has_error(&result),
        "a reachable arm's body is checked — `VCons(x, xs): VNil` cannot join Bool"
    );
}

/// Transport consumes the `Path` witness bound by dependent
/// elimination: under the `VNil` arm `refine_scrutinee_` binds
/// `eq_n ↦ Path{var n, 0}` in `results`, and `eq_n # v` rewrites `v`'s
/// type `Vec(n + 0, Bool)` through that path to `Vec(0, Bool)` —
/// exactly what the motive requires. Under `VCons` the same term
/// transports along `Path{var n, ?k + 1}` to `Vec(?k + 1, Bool)`.
/// The `via` field on the transport node's recorded type keeps the
/// witness (proof-relevant — not erased).
#[test]
fn transport_consumes_branch_path_witness() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "id_vec : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Vec(n + 0, Bool))) = n ~> v ~> match v return (i, w) ~> Vec(i, Bool) with { VNil: eq_n # v, VCons(x, xs): eq_n # v }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "eq_n # v should transport v along the branch's Path witness, got {:?}",
        diags_of(&result)
    );
    let witnessed = result.node_types.values().any(|t| {
        t.get("via")
            .and_then(|p| p.get("__tag__"))
            .map(|tag| tag == "Path")
            .unwrap_or(false)
    });
    assert!(
        witnessed,
        "the transported type should record its Path witness under `via`"
    );
}

/// `refl` is the identity path: `p : Path(2, 2) = refl` checks (the
/// endpoints unify), `p # v` transports `v` along it, and a bare `refl`
/// in transport position is the identity coercion.
#[test]
fn refl_is_the_identity_path() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "p : Path(2, 2) = refl",
            "w : Vec(2, Bool) = p # v",
            "w2 : Vec(2, Bool) = refl # v",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "refl should inhabit Path(2, 2) and transport identically, got {:?}",
        diags_of(&result)
    );
}

/// The eliminators reject decided non-paths and absent paths:
/// `5 # v` and `v # v` are not transports over a `Path`, and `refl`
/// cannot prove a heterogeneous `Path(2, 3)`.
#[test]
fn transport_rejects_non_path_and_absent_paths() {
    let lang = canonical_lang();
    let decl = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

    let num_path = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "bad : Vec(2, Bool) = 5 # v",
        ],
    );
    let result = typecheck(&lang, num_path);
    assert!(
        has_error(&result),
        "a number is not a path — 5 # v should be rejected"
    );

    let vec_path = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "bad : Vec(2, Bool) = v # v",
        ],
    );
    let result = typecheck(&lang, vec_path);
    assert!(
        has_error(&result),
        "a Vec is not a path — v # v should be rejected"
    );

    let unbound = parse_block(
        &lang,
        &[
            decl,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "bad : Vec(2, Bool) = nosuchpath # v",
        ],
    );
    let result = typecheck(&lang, unbound);
    assert!(
        has_error(&result),
        "an absent path should be rejected — nosuchpath # v"
    );

    let bad_refl = parse_block(&lang, &[decl, "bad : Path(2, 3) = refl"]);
    let result = typecheck(&lang, bad_refl);
    assert!(
        has_error(&result),
        "refl proves only a reflexive Path — Path(2, 3) should be rejected"
    );
}

/// The `j` eliminator consumes a path with an explicit motive:
/// `J eq_n, (i, w) ~> Vec(i, Bool), v` checks `v` against the motive
/// applied at the path's source (`Vec(var n, Bool)`) and synthesizes
/// the motive at the target — `Vec(0, Bool)` under `VNil`,
/// `Vec(?k + 1, Bool)` under `VCons` — composing with `apply_motive`.
///
/// The surface form parses through `lex.keyword.j` (the CAPITAL `J`
/// spelling — `j` stays an identifier) into `j` nodes
/// (children `[path, motive, base]`); `synth_j` consumes them. A named
/// motive bound in `results` (`_motive := (i2, w2) ~> Vec(i2, Bool)`)
/// resolves to its `Lam` term through `apply_motive`'s var arm.
#[test]
fn j_eliminates_branch_path_through_motive() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "_motive := (i2, w2) ~> Vec(i2, Bool)",
            "id_vec : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Vec(n + 0, Bool))) = n ~> v ~> match v return (i, w) ~> Vec(i, Bool) with { VNil: J eq_n, _motive, v, VCons(x, xs): J eq_n, _motive, v }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "J eq_n, _motive, v should eliminate the branch's Path through the motive, got {:?}",
        diags_of(&result)
    );
}

/// `J` also takes a literal motive arrow: `J eq_n, (i, w) ~> Vec(i, Bool), v`
/// applies the function child directly.
#[test]
fn j_surface_literal_motive() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "id_vec : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Vec(n + 0, Bool))) = n ~> v ~> match v return (i, w) ~> Vec(i, Bool) with { VNil: J eq_n, (i2, w2) ~> Vec(i2, Bool), v, VCons(x, xs): J eq_n, (i2, w2) ~> Vec(i2, Bool), v }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "J eq_n, (i2, w2) ~> Vec(i2, Bool), v should eliminate the branch's Path, got {:?}",
        diags_of(&result)
    );
}

/// Bounded nonlinear index solving: `c * X ≡ t` solves `X` by exact
/// division. The `Dbl` family's constructor lands at `Dbl(2 * n)` —
/// commutative canonicalization already normalizes it to
/// `Bin{mul, ?n, 2}` — so checking `MkDbl` against `Dbl(6)` solves
/// `?n := 3` through the product, and matching a `Dbl(6)` scrutinee
/// refines the arm under `?k := 3`. A non-divisible target (`Dbl(5)`)
/// has no Nat solution: the equation is a *decided* failure, not stuck.
#[test]
fn nonlinear_product_solves_by_exact_division() {
    let lang = canonical_lang();
    let decl = "Dbl<n : Nat> := { MkDbl : (x : Nat) ~> Dbl(2 * n) }";

    let good = parse_block(
        &lang,
        &[
            decl,
            "d : Dbl(6) = MkDbl(7)",
            "r : Nat = match d { MkDbl(x): x }",
        ],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "2 * ?n ≡ 6 should solve ?n := 3 (in the annotation and per arm), got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(&lang, &[decl, "d : Dbl(5) = MkDbl(7)"]);
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "2 * ?n ≡ 5 has no Nat solution — a decided failure, not stuck"
    );
}

/// The square and small-power cases: `X * X ≡ t` (structurally equal
/// operands) solves by integer square root, and `X ** k ≡ t` for a
/// literal exponent solves by integer root. `Sq(9)` gives `?n := 3`
/// and `Cube(8)` gives `?n := 2`; `Sq(8)` and `Cube(9)` have no integer
/// root and are decided failures.
#[test]
fn nonlinear_square_and_pow_solve_by_integer_root() {
    let lang = canonical_lang();
    let sq = "Sq<n : Nat> := { MkSq : (x : Nat) ~> Sq(n * n) }";
    let cube = "Cube<n : Nat> := { MkCube : (x : Nat) ~> Cube(n ** 3) }";

    let good = parse_block(
        &lang,
        &[sq, cube, "s : Sq(9) = MkSq(7)", "c : Cube(8) = MkCube(1)"],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "?n * ?n ≡ 9 and ?n ** 3 ≡ 8 should solve by integer root, got {:?}",
        diags_of(&result)
    );

    let bad_sq = parse_block(&lang, &[sq, "s : Sq(8) = MkSq(7)"]);
    let result = typecheck(&lang, bad_sq);
    assert!(
        has_error(&result),
        "?n * ?n ≡ 8 has no integer root — decided failure"
    );

    let bad_cube = parse_block(&lang, &[cube, "c : Cube(9) = MkCube(1)"]);
    let result = typecheck(&lang, bad_cube);
    assert!(
        has_error(&result),
        "?n ** 3 ≡ 9 has no integer root — decided failure"
    );
}

/// `type.Name := T` / `type.Name(p…) := T` — the transparent type
/// alias (ADR-0154 §3): the `type.` namespace head on an `assign`
/// declares an alias that substitutes structurally at use sites. A
/// scalar alias expands in annotation position; a parameterized alias
/// substitutes its arguments into the body; a tuple body reads as the
/// `type.array{elements}` product convention.
#[test]
fn type_alias_expands_transparently() {
    let lang = canonical_lang();
    let good = parse_block(
        &lang,
        &[
            "type.Score := Nat",
            "s : Score = 3",
            "type.Id(A) := A",
            "x : Id(Nat) = 4",
            "y : type.Id(Bool) = true",
            "type.Pair(A, B) := (A, B)",
            "p : Pair(Nat, Bool) = (1, true)",
            "q : Pair(Nat, Bool) = p",
        ],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "alias declarations and uses should check, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(&lang, &["type.Id(A) := A", "x : Id(Bool) = 3"]);
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "Id(Bool) should expand to Bool and reject 3"
    );

    let bad_tuple = parse_block(
        &lang,
        &["type.Pair(A, B) := (A, B)", "p : Pair(Nat, Bool) = (1, 2)"],
    );
    let result = typecheck(&lang, bad_tuple);
    assert!(
        has_error(&result),
        "Pair(Nat, Bool) should reject (1, 2) — element 1 is not Bool"
    );
}

/// `IO(sig)` — the typed-effect former (ADR-0153 §4): an `IO` type is
/// an effect object indexed by the op-signature record. Checking
/// `IO(s)` against `IO(s')` is signature subtyping — `s'` names at
/// least the ops `s` does (record-width on the index). Bare `IO` is
/// `IO(⊤)` — it accepts any signature in expected position.
#[test]
fn io_signature_subtyping() {
    let lang = canonical_lang();
    let good = parse_block(
        &lang,
        &[
            "narrow : IO<{ puts: Str ~> Unit }> ~> IO<{ puts: Str ~> Unit, gets: Unit ~> Str }> = x ~> x",
            "top : IO<{ puts: Str ~> Unit }> ~> IO = x ~> x",
        ],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "IO<{{puts}}> should check against IO<{{puts,gets}}> and bare IO, got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(
        &lang,
        &[
            "widen : IO<{ puts: Str ~> Unit, gets: Unit ~> Str }> ~> IO<{ puts: Str ~> Unit }> = x ~> x",
        ],
    );
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "IO<{{puts,gets}}> must not check against IO<{{puts}}> — extra op not allowed"
    );
}

/// `Ctor: absurd` is the explicit dismissal of a decided-impossible arm:
/// `VNil : Vec(0, ?a)` can never produce a `Vec(2, Bool)` scrutinee, so
/// `VNil: absurd` is *discharged* — no unreachable-branch diagnostic —
/// while the reachable `VCons` arm checks normally and contributes the
/// join. `Dbl`'s constructor lands at `Dbl(2 * ?k)`, so on a `Dbl(5)`
/// scrutinee the arm is decided-impossible by the nonlinear solver and
/// `MkDbl(x): absurd` is accepted the same way.
#[test]
fn absurd_discharges_decided_impossible_arm() {
    let lang = canonical_lang();
    let vec = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";
    let dbl = "Dbl<n : Nat> := { MkDbl : (x : Nat) ~> Dbl(2 * n) }";

    let ast = parse_block(
        &lang,
        &[
            vec,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "x : Bool = match v { VNil: absurd, VCons(h, t): h }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "VNil: absurd on a Vec(2) scrutinee should be discharged, got {:?}",
        diags_of(&result)
    );

    let nonlinear = parse_block(
        &lang,
        &[
            dbl,
            "f : Pi(Dbl(5), d ~> Nat) = d ~> match d { MkDbl(x): absurd, y: 0 }",
        ],
    );
    let result = typecheck(&lang, nonlinear);
    assert!(
        !has_error(&result),
        "2 * ?k ≡ 5 is decided-unsatisfiable — the absurd arm should be discharged, got {:?}",
        diags_of(&result)
    );
}

/// `absurd` is only justified on a decided-impossible arm: marking a
/// *reachable* arm `absurd` is rejected. `VCons` on a `Vec(2)` scrutinee
/// solves `?k + 1 ≡ 2`, `VNil` on `Vec(0)` unifies outright, and a
/// rigid-index scrutinee (`v : Vec(n + 0, Bool)` — `n ≡ 0` binds under
/// the arm rather than failing) is reachable too, so all three `absurd`
/// arms error as reachable.
#[test]
fn absurd_rejects_reachable_arm() {
    let lang = canonical_lang();
    let vec = "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }";

    let reachable = parse_block(
        &lang,
        &[
            vec,
            "v : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "bad : Bool = match v { VCons(x, xs): absurd }",
        ],
    );
    let result = typecheck(&lang, reachable);
    assert!(
        has_error(&result),
        "VCons: absurd on a Vec(2) scrutinee is reachable — must be rejected"
    );

    let exact = parse_block(
        &lang,
        &[
            vec,
            "v0 : Vec(0, Bool) = VNil",
            "bad : Bool = match v0 { VNil: absurd }",
        ],
    );
    let result = typecheck(&lang, exact);
    assert!(
        has_error(&result),
        "VNil: absurd on a Vec(0) scrutinee is reachable — must be rejected"
    );

    let rigid = parse_block(
        &lang,
        &[
            vec,
            "f : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Bool)) = n ~> v ~> match v { VNil: absurd, VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, rigid);
    assert!(
        has_error(&result),
        "on Vec(n + 0) the VNil arm refines n ≡ 0 rather than failing — absurd must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Phase-3 WS-C — split-tree coverage for indexed `match`
// ---------------------------------------------------------------------------

/// `Vec(Succ n)` needs no `VNil` arm: the scrutinee's index evaluates to
/// `n + 1` with `n : Nat` known nonneg, so `0 ≡ n + 1` is
/// decided-unsatisfiable and `VNil` is out of the possible set — a
/// VCons-only match is complete *without* a catchall. Writing the arm
/// anyway makes it decided-unreachable: `VNil: absurd` discharges it,
/// a plain `VNil` body errors.
#[test]
fn match_succ_indexed_scrutinee_is_exhaustive_without_catchall() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let head = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "head : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Bool)) = n ~> v ~> match v { VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, head);
    assert!(
        !has_error(&result),
        "VCons-only match on Vec(Succ n) should be exhaustive, got {:?}",
        diags_of(&result)
    );

    let with_absurd = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "head : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Bool)) = n ~> v ~> match v { VNil: absurd, VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, with_absurd);
    assert!(
        !has_error(&result),
        "VNil: absurd should be discharged — the arm is decided-impossible, got {:?}",
        diags_of(&result)
    );

    // A written `VNil` arm is decided-impossible → *auto-discharged*:
    // allowed without `absurd`, its body unchecked (here `VNil`, which
    // would fail against the `Bool` motive if checked).
    let stray = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "head : Pi(Nat, n ~> Pi(Vec(Succ(n), Bool), v ~> Bool)) = n ~> v ~> match v { VNil: VNil, VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, stray);
    assert!(
        !has_error(&result),
        "a live VNil arm on Vec(Succ n) is decided-impossible — auto-discharged, got {:?}",
        diags_of(&result)
    );
}

/// The `zipWith` shape: `match v { VNil: _, VCons(x, xs): match ys {…} }`
/// over two `Vec<n>` arguments. The outer `VCons` arm refines
/// `n ↦ ?k + 1` (`?k` a `Nat`-sorted meta), so the inner match's `VNil`
/// arm is decided-impossible (`0 ≡ ?k + 1` by the nonnegativity gap) and
/// auto-discharges — coverage needs only the `VCons` arm, no `absurd`
/// anywhere. The dead `VNil` bodies stay ill-typed on purpose: they pin
/// the unchecked-but-allowed half of the contract.
#[test]
fn match_auto_discharge_zipwith_shape() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let zip = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            // `n + 0` spellings keep `n` a rigid index var (a bare `n`
            // resolves to the `Nat` sort — see collections.hnl's gap
            // log); the VCons arm's `n ↦ ?k + 1` rewrite then reaches
            // the inner match's scrutinee.
            "zw : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Pi(Vec(n + 0, Bool), w ~> Bool))) = n ~> v ~> w ~> match v { VNil: true, VCons(x, xs): match w { VNil: VNil, VCons(y, ys): y } }",
        ],
    );
    let result = typecheck(&lang, zip);
    assert!(
        !has_error(&result),
        "inner VNil arm under n ↦ ?k + 1 should auto-discharge, got {:?}",
        diags_of(&result)
    );
}

/// The same nested-match shape with an explicit `absurd` in the
/// decided-impossible inner arm (rather than relying on
/// auto-discharge): `absurd` checks under the `n ↦ ?k + 1` refinement
/// because the arm's scrutinee equation is decided-unsatisfiable.
#[test]
fn nested_absurd_inner_match_checks() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "zw : Pi(Nat, n ~> Pi(Vec(n + 0, Bool), v ~> Pi(Vec(n + 0, Bool), w ~> Bool))) = n ~> v ~> w ~> match v { VNil: true, VCons(x, xs): match w { VNil: absurd, VCons(y, ys): y } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "absurd in the decided-impossible inner VNil arm should check, got {:?}",
        diags_of(&result)
    );
}

/// `export f<n, …>` — implicit index binders on an exported assign:
/// `<n>` binds `var{n}` rigidly over the `Vec<n, …>` signature, and
/// the inner `absurd` arm discharges under `n ↦ ?k + 1` the same way
/// the `Pi`-annotated spelling does.
#[test]
fn export_implicit_binders_abstract_index() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export zw<n> : Vec<n, Bool> ~> Vec<n, Bool> ~> Bool = xs ~> ys ~> match xs { VNil: true, VCons(x, t): match ys { VNil: absurd, VCons(y, u): false } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "export zw<n> with absurd should check, got {:?}",
        diags_of(&result)
    );
}

/// The full `zipWith`: `export zw<n, a, b, c>` — three rigid index /
/// element binders, the inner decided-impossible arm, and a recursive
/// `zw(f)(t)(u)` call inside `VCons`. The tail of each recursive arm
/// reuses `t`/`u` at the refined index `?k`.
#[test]
fn export_zipwith_recursive_body_checks() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export zw<n, a, b, c> : (a ~> b ~> c) ~> Vec<n, a> ~> Vec<n, b> ~> Vec<n, c> = f ~> xs ~> ys ~> match xs { VNil: VNil, VCons(x, t): match ys { VNil: absurd, VCons(y, u): VCons(f(x)(y), zw(f)(t)(u)) } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "recursive zipWith body should check, got {:?}",
        diags_of(&result)
    );

    // Same shape without the function argument — `VCons(y, t)` rebuilds
    // at the arm's refined index.
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export zw<n, a, b, c> : Vec<n, a> ~> Vec<n, b> ~> Vec<n, c> = xs ~> ys ~> match xs { VNil: VNil, VCons(x, t): match ys { VNil: absurd, VCons(y, u): VCons(y, t) } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "no-fn zipWith shape should check, got {:?}",
        diags_of(&result)
    );
}

/// A `zipWith` arm returning the wrong type (`false` at `Vec<n, c>`)
/// is a decided mismatch, not a silent pass.
#[test]
fn export_zipwith_wrong_result_errors() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export zw<n, a, b, c> : (a ~> b ~> c) ~> Vec<n, a> ~> Vec<n, b> ~> Vec<n, c> = f ~> xs ~> ys ~> match xs { VNil: VNil, VCons(x, t): match ys { VNil: absurd, VCons(y, u): false } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "Bool in a Vec<n, c> arm should be a decided mismatch"
    );
}

/// `Sigma(Nat, m ~> Vec(m, a))` destructure: `match s { [m, ys]: … }`
/// binds `m` rigidly (`var{m}`) and `ys` at `Vec(var{m}, a)`. The
/// repack `[Succ(m), VCons(x, ys)]` checks `VCons(x, ys) :
/// Vec(var{m} + 1, a)` against the payload `Vec(m + 1, a)` — the
/// rigid-var-vs-meta index equation `?k + 1 ≡ m + 1` solves
/// `?k := var{m}` through `unify_index`'s same-op `Bin` congruence.
#[test]
fn sigma_destructure_repack_index() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export filter2<n, a> : (a ~> Bool) ~> Vec<n, a> ~> Sigma(Nat, m ~> Vec(m, a)) = f ~> v ~> match v { VNil: [Zero, VNil], VCons(x, xs): match filter2(f)(xs) { [m, ys]: [Succ(m), VCons(x, ys)] } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "sigma repack should solve ?k := var{{m}}, got {:?}",
        diags_of(&result)
    );

    // Same repack under an `if` on the predicate — both branches build
    // a Sigma pair, only the witness differs.
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export filter3<n, a> : (a ~> Bool) ~> Vec<n, a> ~> Sigma(Nat, m ~> Vec(m, a)) = f ~> v ~> match v { VNil: [Zero, VNil], VCons(x, xs): match filter3(f)(xs) { [m, ys]: if f(x) { [Succ(m), VCons(x, ys)] } else { [m, ys] } } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "sigma repack under if should check, got {:?}",
        diags_of(&result)
    );
}

/// `Vec(m + 1, ?a)` vs `Vec(?k + 1, ?a)`: a decided mismatch on a
/// *genuinely wrong* repack still reports — `VCons(Zero, VNil)` is
/// `Vec(1, Nat)`, never `Vec(m + 1, a)` for rigid `m`, `a`.
#[test]
fn sigma_destructure_wrong_payload_errors() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export bump<a> : Sigma(Nat, m ~> Vec(m, a)) ~> Sigma(Nat, k ~> Vec(k, a)) = s ~> match s { [m, ys]: [Succ(m), VCons(Zero, VNil)] }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "VCons(Zero, VNil) : Vec(1, Nat) against Vec(m + 1, a) should error"
    );
}

/// `exists m : S, T` — `Sigma` surface sugar, `forall`'s existential
/// twin. In annotation position the `exists` node mints the same
/// `type.sigma{param, body_closure}` `Sigma(S, m ~> T)` produces, so the
/// `[witness, value]` pair intro and the `match` pair destructure work
/// unchanged.
#[test]
fn exists_sugar_mints_sigma() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            // `exists` in a plain annotation — the pair intro checks
            // against the minted `type.sigma` exactly like `Sigma(...)`.
            "one_vec : exists m : Nat, Vec(m, Bool) = [Succ(Zero), VCons(true, VNil)]",
            // `exists` in an arrow codomain — the `filter` shape where
            // the output length can't be named.
            "export single_vec<a> : a ~> exists m : Nat, Vec(m, a) = x ~> [Succ(Zero), VCons(x, VNil)]",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "exists annotations should mint type.sigma and check cleanly, got {:?}",
        diags_of(&result)
    );
}

/// The lib-collections `filter` repro: `exists m : Nat, Vec(m, a)` as
/// the return type, the `[m, ys]` destructure binding `m` rigidly, and
/// the recursive repack solving `?k := var{m}` — the `Sigma(Nat, m ~>
/// Vec(m, a))` workaround written in the sugared spelling.
#[test]
fn exists_sugar_filter_repro() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export filter<n, a> : (a ~> Bool) ~> Vec<n, a> ~> exists m : Nat, Vec(m, a) = f ~> v ~> match v { VNil: [Zero, VNil], VCons(x, xs): match filter(f)(xs) { [m, ys]: if f(x) { [Succ(m), VCons(x, ys)] } else { [m, ys] } } }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "filter with an exists-typed result should check, got {:?}",
        diags_of(&result)
    );
}

/// The `exists` sugar still rejects a genuinely wrong repack — the same
/// decided-mismatch coverage `Sigma(...)` has (`VCons(Zero, VNil)` is
/// `Vec(1, Nat)`, never `Vec(m + 1, a)` for rigid `m`, `a`).
#[test]
fn exists_sugar_wrong_payload_errors() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let ast = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "export bump<a> : (exists m : Nat, Vec(m, a)) ~> exists k : Nat, Vec(k, a) = s ~> match s { [m, ys]: [Succ(m), VCons(Zero, VNil)] }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "VCons(Zero, VNil) : Vec(1, Nat) against Vec(m + 1, a) should error"
    );
}

/// `Vec(n + m)` splits: `VNil` is possible only at `n ↦ 0, m ↦ 0` and
/// `VCons` only when `n + m ≥ 1`. The `{VNil, VCons}` match is proved
/// exhaustive through the split tree; dropping either arm is a
/// non-exhaustive error naming the missing ctor (with the leaf
/// substitution that witnesses reachability).
#[test]
fn match_split_tree_decides_n_plus_m_coverage() {
    let lang = canonical_lang();
    let decls = [
        "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
        "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
    ];
    let good = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "f : Pi(Nat, n ~> Pi(Nat, m ~> Pi(Vec(n + m, Bool), v ~> Bool))) = n ~> m ~> v ~> match v { VNil: true, VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "{{VNil, VCons}} on Vec(n + m) should be exhaustive via the split tree, got {:?}",
        diags_of(&result)
    );

    let missing_nil = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "f : Pi(Nat, n ~> Pi(Nat, m ~> Pi(Vec(n + m, Bool), v ~> Bool))) = n ~> m ~> v ~> match v { VCons(x, xs): x }",
        ],
    );
    let result = typecheck(&lang, missing_nil);
    let diags = diags_of(&result);
    assert!(
        has_error(&result)
            && diags
                .iter()
                .any(|d| d.contains("non-exhaustive") && d.contains("VNil")),
        "VCons-only on Vec(n + m) must report VNil reachable under n ↦ 0, m ↦ 0, got {diags:?}"
    );

    let missing_cons = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "f : Pi(Nat, n ~> Pi(Nat, m ~> Pi(Vec(n + m, Bool), v ~> Bool))) = n ~> m ~> v ~> match v { VNil: true }",
        ],
    );
    let result = typecheck(&lang, missing_cons);
    let diags = diags_of(&result);
    assert!(
        has_error(&result)
            && diags
                .iter()
                .any(|d| d.contains("non-exhaustive") && d.contains("VCons")),
        "VNil-only on Vec(n + m) must report VCons reachable, got {diags:?}"
    );
}

/// Literal-index exhaustiveness: on `Vec(0)` the `VCons` index equation
/// `?k + 1 ≡ 0` is decided-unsatisfiable because `?k` is a `Nat`
/// metavariable (never negative) — so a VNil-only match is complete.
#[test]
fn match_literal_zero_index_needs_no_vcons_arm() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "v0 : Vec(0, Bool) = VNil",
            "x : Bool = match v0 { VNil: true }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "VCons cannot occur on Vec(0) — VNil-only match is exhaustive, got {:?}",
        diags_of(&result)
    );
}

/// Plain enum coverage comes free: a `data`-declared `Bool` scrutinee
/// makes `match { True: … }` non-exhaustive (`False` is decided
/// possible), while a catchall discharges it.
#[test]
fn match_enum_scrutinee_requires_all_ctors_or_catchall() {
    let lang = canonical_lang();
    let missing = parse_block(
        &lang,
        &[
            "Bool := { True : Bool, False : Bool }",
            "b : Bool = True",
            "x : Nat = match b { True: 1 }",
        ],
    );
    let result = typecheck(&lang, missing);
    let diags = diags_of(&result);
    assert!(
        has_error(&result)
            && diags
                .iter()
                .any(|d| d.contains("non-exhaustive") && d.contains("False")),
        "True-only match on Bool should report `False` not covered, got {diags:?}"
    );

    let caught = parse_block(
        &lang,
        &[
            "Bool := { True : Bool, False : Bool }",
            "b : Bool = True",
            "x : Nat = match b { True: 1, _: 0 }",
        ],
    );
    let result = typecheck(&lang, caught);
    assert!(
        !has_error(&result),
        "a catchall arm discharges False, got {:?}",
        diags_of(&result)
    );
}

/// Undecided coverage is a warning, never a silent accept nor an
/// error: a `Vec(f(n))` scrutinee with opaque `f` leaves every index
/// equation stuck, so the uncovered `VCons` reports `coverage unknown`
/// (the split bound is reached re-splitting `n`) while the check
/// itself still passes.
#[test]
fn match_undecided_coverage_warns_not_errors() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "g : Pi(Nat ~> Nat, f ~> Pi(Nat, n ~> Pi(Vec(f(n), Bool), v ~> Bool))) = f ~> n ~> v ~> match v { VNil: true }",
        ],
    );
    let result = typecheck(&lang, ast);
    let diags = diags_of(&result);
    assert!(
        !has_error(&result),
        "undecided coverage must not error, got {diags:?}"
    );
    assert!(
        result.diagnostics.iter().any(|d| d.severity == "warning"
            && d.message.contains("coverage unknown")
            && d.message.contains("VCons")),
        "expected a `coverage unknown` warning naming VCons, got {diags:?}"
    );
}

/// `handl-packages/type/dependent.hnl` — the packaged Phase-4 data — must
/// parse statement-by-statement and typecheck without errors.
#[test]
fn dependent_hnl_parses_and_typechecks() {
    let src = include_str!("../../lib-type/src/dependent.hnl");
    let stmts: Vec<&str> = src
        .lines()
        .map(|l| l.split_once("--").map(|(c, _)| c).unwrap_or(l))
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert!(
        stmts.len() >= 5,
        "dependent.hnl should carry the reindex arrow and four law terms"
    );
    let lang = canonical_lang();
    let ast = parse_block(&lang, &stmts);
    // The deeply-nested `Path` body needs ~3–4 MiB of check stack —
    // the judgment hops ride `stacker` growable segments (WS-H
    // `STACK_RED_ZONE`/`STACK_SEGMENT` seams), so the default test
    // thread's stack suffices (previously this spawned a 64 MiB
    // thread). `stack_probe` pins the bound explicitly.
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "dependent.hnl should typecheck cleanly, got {:?}",
        diags_of(&result)
    );
}

/// Ordered value annotations (ADR-0103): a `call`-shaped annotation
/// evaluates against the bound-value env (`inc := n ~> n + 1` records
/// `inc ↦ Lam{…}` in `results`), so `f : inc(4) := 5` checks
/// `5 == inc(4)` and `f : inc(4) := 6` is a mismatch — the WS-B
/// regression where the NBE checker silently accepted (or failed to
/// interpret) term-level `call` annotations.
#[test]
fn ordered_call_annotation_checks_equality() {
    let lang = canonical_lang();
    let good = parse_block(&lang, &["inc := n ~> n + 1", "f : inc(4) := 5"]);
    let result = typecheck(&lang, good);
    assert!(
        !has_error(&result),
        "f : inc(4) := 5 should check 5 == inc(4), got {:?}",
        diags_of(&result)
    );

    let bad = parse_block(&lang, &["inc := n ~> n + 1", "f : inc(4) := 6"]);
    let result = typecheck(&lang, bad);
    assert!(
        has_error(&result),
        "f : inc(4) := 6 should be a value-annotation mismatch"
    );

    let nested = parse_block(
        &lang,
        &[
            "inc := n ~> n + 1",
            "dbl := n ~> n * 2",
            "f : dbl(inc(4)) := 10",
        ],
    );
    let result = typecheck(&lang, nested);
    assert!(
        !has_error(&result),
        "dbl(inc(4)) = 2*(4+1) = 10 should check, got {:?}",
        diags_of(&result)
    );
}

// ---------------------------------------------------------------------------
// Phase-2 WS-F — typed-effect handler typing + effect-row polymorphism
// (ADR-0153 §4: effects are objects in the codomain; `IO(sig)` sig records
// are checked against `handle … with`'s declared impls)
// ---------------------------------------------------------------------------

/// A covering impl's declared return replaces the op's sig entry — the
/// interpret-in-place semantics of the runtime's `interpret_io_former`
/// (`handle print("x") with {puts: puts_impl}` produces
/// `IO<{ puts: Unit }>`).
#[test]
fn handle_covered_op_returns_impl_result() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "print : String ~> IO<{ puts: String ~> Unit }> = s ~> print(s)",
            "puts_impl : (String ~> Unit) ~> Unit = req ~> req(\"x\")",
            "x : IO<{ puts: Unit }> = handle print(\"x\") with {puts: puts_impl}",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "covered op should check, got {:?}",
        diags_of(&result)
    );
}

/// A handler that declares impls for ops the body's sig never names is a
/// diagnostic; a handler covering none of the body's ops is a guaranteed
/// no-op and also a diagnostic (phase-2 WS-F coverage rules).
#[test]
fn handle_missing_required_op_errors() {
    let lang = canonical_lang();
    let decls = [
        "print : String ~> IO<{ puts: String ~> Unit }> = s ~> print(s)",
        "puts_impl : (String ~> Unit) ~> Unit = req ~> req(\"x\")",
    ];
    let spurious = parse_block(
        &lang,
        &[
            decls[0],
            decls[1],
            "x := handle print(\"x\") with {gets: puts_impl}",
        ],
    );
    let result = typecheck(&lang, spurious);
    assert!(
        has_error(&result),
        "handler declaring a spurious op should error"
    );
    let empty = parse_block(&lang, &[decls[0], "x := handle print(\"x\") with {}"]);
    let result = typecheck(&lang, empty);
    assert!(
        has_error(&result),
        "handler covering none of the declared ops should error"
    );
}

/// An uncovered op passes through as `IO(rest)` — the residual stays an
/// op for an outer handler (the runtime leaves its request data intact).
#[test]
fn handle_uncovered_op_passes_through_as_io() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "send : String ~> IO<{ puts: String ~> Unit, gets: String ~> Unit }> = s ~> send(s)",
            "puts_impl : (String ~> Unit) ~> Unit = req ~> req(\"x\")",
            // `gets` uncovered → stays an op in the residual;
            // `puts` covered → its entry is replaced by the impl's
            // `Unit` return (annotating `puts: Unit` proves the
            // replacement — a stale `String ~> Unit` would fail conv).
            "x : IO<{ puts: Unit, gets: String ~> Unit }> = handle send(\"x\") with {puts: puts_impl}",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "residual IO should carry the uncovered op and the impl result, got {:?}",
        diags_of(&result)
    );
}

/// `handle f with h` on an arrow body — the deep-handler case (the
/// runtime's `attach_carried_handler`): the arrow's `IO` codomain is
/// checked the same way, producing `A ~> IO(S')`.
#[test]
fn handle_arrow_codomain_checked() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "print : String ~> IO<{ puts: String ~> Unit }> = s ~> print(s)",
            "puts_impl : (String ~> Unit) ~> Unit = req ~> req(\"x\")",
            "handled_print : String ~> IO<{ puts: Unit }> = handle print with {puts: puts_impl}",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "handled arrow should produce A ~> IO(S'), got {:?}",
        diags_of(&result)
    );
}

/// `perform(op, arg)` — the `IO<sig>` op-emission builtin (ADR-0153 §4):
/// it mints `IO{op: T}` from the op *name* and the arg's type — the
/// checked-side inverse of the runtime's `io.perform` mint.
/// `synth_perform` reads the `puts` operand as a name (a bare identifier
/// is deliberately never env-resolved or checked as a value).
#[test]
fn perform_mints_io_sig_from_op_name() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "x : IO<{ puts: Str }> = perform(puts, \"hi\")",
            "y : IO<{ puts: Str }> = io.perform(puts, \"hi\")",
            "z : IO<{ fs: Str }> = perform(\"fs\", \"path\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "perform should mint IO{{op: T}}, got {:?}",
        diags_of(&result)
    );
}

/// A wrong sig-entry type is a decided mismatch — `perform(puts, "hi")`
/// is `IO{puts: Str}`, never `IO{puts: Nat}`.
#[test]
fn perform_wrong_sig_entry_errors() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["x : IO<{ puts: Nat }> = perform(puts, \"hi\")"]);
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "IO{{puts: Str}} checked against IO{{puts: Nat}} should error"
    );
}

/// `handle` on a `perform`ed op runs the same `handle_sig_check`
/// coverage logic as a declared effect — the covered entry takes the
/// impl's declared return type.
#[test]
fn perform_through_handle_checks_sig_coverage() {
    let lang = canonical_lang();
    // Positive: the covered entry takes the impl's declared return type.
    let ok = parse_block(
        &lang,
        &[
            "puts_impl : Str ~> Str = s ~> s",
            "x : IO<{ puts: Str }> = handle perform(puts, \"hi\") with {puts: puts_impl}",
        ],
    );
    let result = typecheck(&lang, ok);
    assert!(
        !has_error(&result),
        "handled perform should check, got {:?}",
        diags_of(&result)
    );
    // A handler declaring ops the performed sig never names is the
    // WS-F spurious-impl diagnostic; an empty handler covering none of
    // the performed ops is the guaranteed-no-op diagnostic.
    let spurious = parse_block(
        &lang,
        &[
            "puts_impl : Str ~> Str = s ~> s",
            "z := handle perform(puts, \"hi\") with {fs: puts_impl}",
        ],
    );
    let result = typecheck(&lang, spurious);
    assert!(
        diags_of(&result).iter().any(|d| d.contains("never names")),
        "spurious impl should diagnose, got {:?}",
        diags_of(&result)
    );
    let none = parse_block(&lang, &["z := handle perform(puts, \"hi\") with {}"]);
    let result = typecheck(&lang, none);
    assert!(
        diags_of(&result)
            .iter()
            .any(|d| d.contains("none of the body's declared ops")),
        "empty handler should diagnose, got {:?}",
        diags_of(&result)
    );
}

/// `perform` is a *named* op emission — a non-name operand (a computed
/// expression) is a decided error.
#[test]
fn perform_computed_op_operand_errors() {
    let lang = canonical_lang();
    let ast = parse_block(&lang, &["x := perform(1 + 2, \"hi\")"]);
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "perform(1 + 2, …) should error — the op operand must be a name"
    );
}

/// A user `perform` binding shadows the builtin — `perform(puts, "hi")`
/// then checks as an ordinary call (which errors on the still-unbound
/// `puts` arg here, proving the builtin's name-read did not run).
#[test]
fn perform_is_shadowable() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "perform : Str ~> Str ~> Str = a ~> b ~> a",
            "x : Str = perform(\"a\", \"b\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "a source-level perform binding should shadow the builtin, got {:?}",
        diags_of(&result)
    );
}

/// Effect-row polymorphism: `IO(s)` with `s` a `{…}`-bound sig var — the
/// implicit `type.pi` binder inserts a fresh meta at application and the
/// record-width sig unification solves it to the concrete sig object.
#[test]
fn sig_row_polymorphism_solves_meta() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "print : String ~> IO<{ puts: String ~> Unit }> = s ~> print(s)",
            "id_io<s> : (String ~> IO(s)) ~> String ~> IO(s) = f ~> x ~> f(x)",
            "y : IO<{ puts: String ~> Unit }> = id_io(print)(\"x\")",
            "id_sig<s : Sig> : (String ~> IO(s)) ~> String ~> IO(s) = f ~> x ~> f(x)",
            "z : IO<{ puts: String ~> Unit }> = id_sig(print)(\"x\")",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "sig row-metas should solve to the concrete sig, got {:?}",
        diags_of(&result)
    );
}

// ---------------------------------------------------------------------------
// Phase-3 WS-A — instance resolution
// ---------------------------------------------------------------------------

/// `instance <C(i⃗)> := <dict>` registers a dictionary; an implicit
/// `<eq : Eq a>` binder's former-app domain discharges by dictionary
/// search — `f(Zero)` resolves `Eq Nat` from the declared instance
/// once the index meta `?a` solves to `Nat` through the explicit arg.
#[test]
fn instance_resolves_implicit_evidence_binder() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Eq<a : Type> := { mkEq : (f : a ~> a ~> Bool) ~> Eq(a) }",
            "instance Eq(Nat) := {eq: (x, y) ~> x == y}",
            "f<a, eq : Eq(a)> : a ~> a := v ~> v",
            "r := f(Zero)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Eq Nat should resolve the implicit evidence binder, got {:?}",
        diags_of(&result)
    );
}

/// Missing instance — `f(true)` needs `Eq Bool` but only `Eq Nat` is
/// declared: the deferred obligation drains at end-of-module as an
/// honest `no instance for Eq(Bool)` diagnostic, never a silent meta.
#[test]
fn instance_missing_produces_named_error() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Bool := { True : Bool, False : Bool }",
            "Eq<a : Type> := { mkEq : (f : a ~> a ~> Bool) ~> Eq(a) }",
            "instance Eq(Nat) := {eq: (x, y) ~> x == y}",
            "f<a, eq : Eq(a)> : a ~> a := v ~> v",
            "r := f(True)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(has_error(&result), "Eq Bool should be a missing instance");
    let diags = diags_of(&result);
    assert!(
        diags
            .iter()
            .any(|d| d.contains("no instance") && d.contains("Eq")),
        "diagnostics should name the constraint, got {:?}",
        diags
    );
}

/// Coherence — a second `instance` declaration for the same head whose
/// index shape unifies with a registered one is a declaration-time
/// error (decided policy: no overlap/priority).
#[test]
fn instance_overlap_is_a_coherence_error() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Eq<a : Type> := { mkEq : (f : a ~> a ~> Bool) ~> Eq(a) }",
            "instance Eq(Nat) := {eq: (x, y) ~> x == y}",
            "instance Eq(Nat) := {eq: (x, y) ~> true}",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "duplicate Eq Nat instance should be a coherence error"
    );
    let diags = diags_of(&result);
    assert!(
        diags.iter().any(|d| d.contains("overlapping instance")),
        "diagnostics should name the overlap, got {:?}",
        diags
    );
}

/// Non-overlapping instances coexist: `Eq Nat` and `Eq Bool` for one
/// head are distinct index shapes — both register and each resolves
/// its own goal.
#[test]
fn distinct_instance_index_shapes_coexist() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Bool := { True : Bool, False : Bool }",
            "Eq<a : Type> := { mkEq : (f : a ~> a ~> Bool) ~> Eq(a) }",
            "instance Eq(Nat) := {eq: (x, y) ~> x == y}",
            "instance Eq(Bool) := {eq: (x, y) ~> x == y}",
            "f<a, eq : Eq(a)> : a ~> a := v ~> v",
            "r1 := f(Zero)",
            "r2 := f(True)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Eq Nat and Eq Bool should both resolve, got {:?}",
        diags_of(&result)
    );
}

/// Generic instance (stretch): `instance Eq (List a)` — the unbound
/// `a` implicitly generalizes as a rigid index var, so the registered
/// shape `Eq (List a)` matches `Eq (List Nat)` at the call site via
/// pointwise `family_app` index unification (`a := Nat`).
#[test]
fn generic_indexed_instance_resolves() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "List<a : Type> := { LNil : List(a), LCons : (h : a, t : List(a)) ~> List(a) }",
            "Eq<a : Type> := { mkEq : (f : a ~> a ~> Bool) ~> Eq(a) }",
            "instance Eq(List(a)) := {eq: (x, y) ~> true}",
            "f<a, eq : Eq(List(a))> : a ~> a := v ~> v",
            "r := f(LCons(Zero, LNil))",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Eq (List Nat) should resolve through the generic Eq (List a) instance, got {:?}",
        diags_of(&result)
    );
}

/// The bare-name sig shorthand: `IO<{ puts }>` declares op `puts` with an
/// unstated request — handler typing still checks coverage.
#[test]
fn io_bare_name_sig_checks_coverage() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "print : String ~> IO<{ puts }> = s ~> print(s)",
            "x : IO<{ puts }> = handle print(\"x\") with {puts: (s) ~> s}",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "bare-name sig should check coverage, got {:?}",
        diags_of(&result)
    );
}

/// The dogfood `append` equation (F5): inside the `VCons` arm the
/// expected index `?k + 1 + m` must unify with the produced
/// `m + ?k + 1` — commuted sums mixing rigid `var`s and unsolved
/// metas canonicalize to one atom order (metas are honest ring atoms).
#[test]
fn commuted_sum_index_unifies_through_metas() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "append<n, m, a> : Vec<n, a> ~> Vec<m, a> ~> Vec<n + m, a> = xs ~> ys ~> match xs { VNil: ys, VCons(x, t): VCons(x, append(t)(ys)) }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "append should typecheck — ?k+1+m ≡ m+?k+1 canonicalizes, got {:?}",
        diags_of(&result)
    );
}

/// Implicit index binders scope into body annotations (F3): the
/// `<n, a>` names are rigid index vars inside the body, so
/// `(v : Vec(n, a))` — the annotation the runtime's `match` desugar
/// needs — unifies with `v`'s stored `Vec<n, a>` domain instead of
/// reading `n`/`a` as unsolved sort metas.
#[test]
fn implicit_binder_names_scope_into_body_annotation() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "f<n, a> : Vec<n, a> ~> Vec<n, a> = v ~> (v : Vec(n, a))",
            "g<n, a> : Vec<n + 1, a> ~> Vec<n + 1, a> = v ~> match (v : Vec(n + 1, a)) { VCons(x, xs): xs }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "implicit binder names should resolve as rigid index vars in body annotations, got {:?}",
        diags_of(&result)
    );
}

/// `F.rec` is visible to the checker (F4-adjacent): `field_access` on
/// a `data` family resolves the family's eliminator binding — the
/// `__splice__` block's `F.rec` — as a nested `type.function`, so
/// `Vec.rec(motive, c0..)(indices.., v)` checks its case arguments
/// against the ctors' instantiated domains.
#[test]
fn family_rec_resolves_eliminator_type() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "v1 : Vec(2, Bool) = VCons(true, VCons(false, VNil))",
            "step : Bool ~> Vec(1, Bool) ~> Number ~> Number = x ~> xs ~> ih ~> ih + 1",
            "len : Number = Vec.rec(null, 0, step)(2, Bool, v1)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "Vec.rec should resolve its eliminator type, got {:?}",
        diags_of(&result)
    );
}

/// Annotated-arm `VCons` reconstruction (dogfood gap): inside
/// `push<n, a> : a ~> Vec<n, a> ~> Vec<n + 1, a>` the arm's own `n`/`a`
/// rigid vars collide with the ctor telescope's family params of the
/// same name. The params bind internally as `__fam_<name>` so
/// `v : Vec<n, a>` refines `n ≡ k + 1` (not `?k + 1`) and
/// `VCons(k + 1, x, v)` reconstructs at `Vec<n + 1, a>` — previously
/// the instantiation bound `n`/`a` in `results` directly and the
/// scrutinee's rigids rewrote to the fresh metas.
#[test]
fn annotated_arm_family_params_dont_shadow_rigids() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (k : Nat, x : a, xs : Vec(k, a)) ~> Vec(k + 1, a) }",
            "push<n, a> : a ~> Vec<n, a> ~> Vec<n + 1, a> = x ~> v ~> match v { VNil: VCons(0, x, VNil), VCons(k, x2, xs): VCons(k + 1, x2, v) }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "annotated-arm VCons reconstruction should typecheck — __fam_ params must not shadow the user's rigids, got {:?}",
        diags_of(&result)
    );
}

/// Same collision through the *element* parameter only: `dup<n, a>`
/// matches on `Vec<n, a>` and rebuilds each ctor — `var{n}`/`var{a}`
/// refine per arm while the telescope's `n`/`a` stay internal.
#[test]
fn annotated_arm_elem_param_scopes_cleanly() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (k : Nat, x : a, xs : Vec(k, a)) ~> Vec(k + 1, a) }",
            "dup<n, a> : Vec<n, a> ~> Vec<n, a> = v ~> match v { VNil: VNil, VCons(k, x, xs): VCons(k, x, xs) }",
            "bump<n> : Vec<n, Bool> ~> Vec<n + 1, Bool> = v ~> match v { VNil: VCons(0, true, VNil), VCons(k, x, xs): VCons(k + 1, x, v) }",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "same-named rigid params inside annotated match arms should check, got {:?}",
        diags_of(&result)
    );
}

/// The F1 soundness hole stays closed: an un-annotated ctor result
/// index mints the unrecoverable `lit null`, so `VCons(Zero, true,
/// VNil) : Vec(5, Bool)` is a decided mismatch, not a free meta.
#[test]
fn unannotated_ctor_result_index_is_decided_mismatch() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec, VCons : (k : Nat, x : a, xs : Vec(k, a)) ~> Vec }",
            "v : Vec(5, Bool) = VCons(Zero, true, VNil)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        has_error(&result),
        "VCons(0, true, VNil) : Vec(5, Bool) must not check"
    );
}

/// WS-H stats probe — sorts last (`zzz_`), so with `--test-threads=1`
/// it prints the accumulated `HNL_TC_STATS`/`HNL_CONV_STATS` counters
/// for the whole file. With the env vars unset it prints zeros and is
/// a no-op.
#[test]
fn zzz_tc_stats_dump() {
    if hnl_tool_typecheck_nbe::stats_enabled() {
        eprintln!("{}", hnl_tool_typecheck_nbe::stats_report());
    }
}
