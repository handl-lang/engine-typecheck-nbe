//! Qualified `Family.ctor` names in *index position*.
//!
//! `synth_data` binds each constructor signature under both the bare
//! name and the `"F.c"` qualified spelling, and `eval_index` projects
//! `F.c` ctor apps through the same `CtorProj` templates. These tests
//! pin the index-term elaboration (`node_to_index_term` /
//! `node_index_or_type`) for the qualified spellings: a Sigma witness
//! `[Nat.succ(Nat.zero), …]`, a nullary `[Nat.zero, …]` witness, and a
//! `forall` implicit index argument `polyv(Nat.succ(Nat.zero))` must
//! elaborate to the same index terms the bare `Succ(Zero)` spelling
//! produced — not `var{}`/`Null`.

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

const NAT_VEC: &[&str] = &[
    "Nat := new Inductive { zero : Nat, succ : (pred : Nat) ~> Nat }",
    "Vec<n : Nat, a : Type> := new Inductive { vnil : Vec(0, a), vcons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
];

/// `[Nat.succ(Nat.zero), Vec.vcons(…)]` / `[Nat.zero, Vec.vnil]` checked
/// against `Sigma(Nat, m ~> Vec(m, Bool))` — the qualified ctor apps
/// must elaborate to `0 + 1` / `0` index terms.
#[test]
fn qualified_ctor_sigma_witness() {
    let lang = canonical_lang();
    let mut stmts: Vec<&str> = NAT_VEC.to_vec();
    stmts.extend([
        "one_vec : Sigma(Nat, m ~> Vec(m, Bool)) := [Nat.succ(Nat.zero), Vec.vcons(true, Vec.vnil)]",
        "empty_vec : Sigma(Nat, m ~> Vec(m, Bool)) := [Nat.zero, Vec.vnil]",
    ]);
    let result = typecheck(&lang, parse_block(&lang, &stmts));
    assert!(
        !has_error(&result),
        "qualified ctor sigma witnesses should check: {:?}",
        diags_of(&result)
    );
}

/// The dependent-pair `(m : Nat, v : Vec(m, Bool))` surface form — the
/// same witnesses through the `type.sigma` chain intro.
#[test]
fn qualified_ctor_dependent_pair() {
    let lang = canonical_lang();
    let mut stmts: Vec<&str> = NAT_VEC.to_vec();
    stmts.extend([
        "one_vec : (m : Nat, v : Vec(m, Bool)) := [Nat.succ(Nat.zero), Vec.vcons(true, Vec.vnil)]",
        "empty_vec : (m : Nat, v : Vec(m, Bool)) := [Nat.zero, Vec.vnil]",
    ]);
    let result = typecheck(&lang, parse_block(&lang, &stmts));
    assert!(
        !has_error(&result),
        "qualified ctor dependent pairs should check: {:?}",
        diags_of(&result)
    );
}

/// `polyv(Nat.succ(Nat.zero))` — a qualified ctor app in a `forall`
/// implicit index-argument position.
#[test]
fn qualified_ctor_forall_index_arg() {
    let lang = canonical_lang();
    let mut stmts: Vec<&str> = NAT_VEC.to_vec();
    stmts.extend([
        "polyv : forall n : Nat, Vec(n, Bool) ~> Vec(n, Bool) := n ~> v ~> v",
        "v2_b : Vec<1, Bool> := polyv(Nat.succ(Nat.zero))(Vec.vcons(true, Vec.vnil))",
        "v2_c : Vec<2, Bool> := polyv(Nat.succ(Nat.succ(Nat.zero)))(Vec.vcons(true, Vec.vcons(false, Vec.vnil)))",
    ]);
    let result = typecheck(&lang, parse_block(&lang, &stmts));
    assert!(
        !has_error(&result),
        "qualified ctor forall index args should check: {:?}",
        diags_of(&result)
    );
}

/// `Vec(Nat.succ n, a)` / `Vec(Nat.zero, a)` in type-index position —
/// the qualified spelling must elaborate to the same index term the
/// bare `Succ n` / `Zero` produced.
#[test]
fn qualified_ctor_type_index() {
    let lang = canonical_lang();
    let mut stmts: Vec<&str> = NAT_VEC.to_vec();
    stmts.extend([
        "hd<n, a> : Vec<Nat.succ n, a> ~> a := v ~> match v { Vec.vcons(x, xs): x, Vec.vnil: x }",
        "tl<n, a> : Vec<Nat.succ n, a> ~> Vec<n, a> := v ~> match v { Vec.vcons(x, xs): xs, Vec.vnil: xs }",
        "zvec<a> : Vec<Nat.zero, a> := Vec.vnil",
    ]);
    let result = typecheck(&lang, parse_block(&lang, &stmts));
    assert!(
        !has_error(&result),
        "qualified ctor type indices should check: {:?}",
        diags_of(&result)
    );
}
