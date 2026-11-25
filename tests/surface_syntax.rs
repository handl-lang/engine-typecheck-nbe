//! Surface-syntax upgrade coverage (typed-binder Π arrows, dependent
//! pairs, `~>[n]` dimension arrows, brace-free `IO<ops>` sigs).
//!
//! Harness mirrors `dependent_structure.rs`: parse statements under a
//! synthetic `block` root and run the NBE engine.

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

/// `(x : A) ~> B` in type position mints the same `type.pi` `Pi(A, x ~>
/// B)` produces — single and multiple typed binders (right-nested Π),
/// with the binder visible in the body.
#[test]
fn typed_binder_arrow_is_pi() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "idp : (x : Nat) ~> Nat := x ~> x",
            "constp : (x : Nat) ~> Bool := x ~> true",
            "same_as_pi : Pi(Nat, x ~> Nat) := x ~> x",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "ty_at : (n : Nat) ~> Type := n ~> Vec(n, Bool)",
            "curried : (x : Nat, y : Bool) ~> Nat := x ~> y ~> x",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "typed-binder arrows should check as Pi, got {:?}",
        diags_of(&result)
    );
}

/// `(x : A, y : B(x), …)` in type position is a dependent pair — the
/// same `type.sigma` `Sigma(A, x ~> Sigma(B, y ~> …))` produces.
/// `[w, v]`/`[w, [v, u]]` literals check componentwise; a non-dependent
/// binder list stays the `type.array` product.
#[test]
fn dependent_pair_is_sigma() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "Nat := { Zero : Nat, Succ : (pred : Nat) ~> Nat }",
            "Vec<n : Nat, a : Type> := { VNil : Vec(0, a), VCons : (x : a, xs : Vec(n, a)) ~> Vec(n + 1, a) }",
            "one_vec : (m : Nat, v : Vec(m, Bool)) := [Succ(Zero), VCons(true, VNil)]",
            "empty_vec : (m : Nat, v : Vec(m, Bool)) := [Zero, VNil]",
            "same_as_sigma : Sigma(Nat, m ~> Vec(m, Bool)) := [Succ(Zero), VCons(true, VNil)]",
            "prod : (x : Bool, y : String) := [true, \"hi\"]",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "dependent pairs should check as Sigma, got {:?}",
        diags_of(&result)
    );
}

/// `A ~>[n] B` parses to the same `dim_arrow` node `A ~[n]~> B` mints —
/// every consumer (term judgment, type-position `Hom`) reads it
/// unchanged.
#[test]
fn dim_arrow_bare_spelling() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "f := n ~> n",
            "g := n ~> n + 1",
            "cell_bare := f ~>[2] g",
            "cell_legacy := f ~[2]~> g",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "~>[n] should check like ~[n]~, got {:?}",
        diags_of(&result)
    );
}

/// `IO<puts, fs>` — the brace-free sig record: bare names declare ops
/// (`type.unknown` payload), `name : T` declares the payload type.
/// Widening/coverage rules are identical to `IO<{ … }>`.
#[test]
fn io_bare_sig_record() {
    let lang = canonical_lang();
    let ast = parse_block(
        &lang,
        &[
            "export printLn : Str ~> IO<puts> := s ~> printLn(s)",
            "export readFile : Str ~> IO<fs> := p ~> readFile(p)",
            "export both : Str ~> IO<puts, fs> := s ~> both(s)",
            "puts_handler : (Str ~> Unit) ~> Unit := req ~> req(\"handled\")",
            "fs_handler : (Str ~> Str) ~> Str := req ~> req(\"contents\")",
            "handled_partial : Str ~> IO<puts: Unit, fs> := handle both with {puts: puts_handler}",
            "handled_full : Str ~> IO<puts: Unit, fs: Str> := handle both with {puts: puts_handler, fs: fs_handler}",
            "widen_ok : Str ~> IO<puts, fs> := s ~> printLn(s)",
            "braced : Str ~> IO<{ puts, fs }> := s ~> both(s)",
        ],
    );
    let result = typecheck(&lang, ast);
    assert!(
        !has_error(&result),
        "IO<puts, fs> should check like IO<{{ puts, fs }}>, got {:?}",
        diags_of(&result)
    );
}
