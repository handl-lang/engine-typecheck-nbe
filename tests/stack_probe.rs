//! WS-H stack-depth probe — `dependent.hnl` typechecks on a bounded
//! stack.
//!
//! Historically `dependent_hnl_parses_and_typechecks` ran on a spawned
//! 64 MiB thread because `eval_judgment`'s judgment hops are several
//! deep frames each. WS-H's growable-stack seams
//! (`stacker::maybe_grow` in `lib.rs`/`dependent.rs`) make the check
//! depth memory-bounded rather than stack-size-bound, so the suite
//! runs on the default thread stack.
//!
//! `STACK_PROBE_MIB` overrides the probe's stack budget (dev-time
//! measurement of the true minimum); the default is 8 MiB — the
//! libtest thread size this regression is pinned to.

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

/// `dependent.hnl` must typecheck on a small stack — WS-H's exit
/// criterion is the default 8 MiB thread (previously the test spawned
/// a 64 MiB thread).
#[test]
fn dependent_hnl_checks_on_small_stack() {
    let mib: usize = std::env::var("STACK_PROBE_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let src = include_str!("../../lib-type/src/dependent.hnl");
    let stmts: Vec<&str> = src
        .lines()
        .map(|l| l.split_once("--").map(|(c, _)| c).unwrap_or(l))
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let lang = canonical_lang();
    let ast = parse_block(&lang, &stmts);
    let result = std::thread::Builder::new()
        .stack_size(mib * 1024 * 1024)
        .spawn(move || typecheck(&lang, ast))
        .expect("spawn typecheck thread")
        .join()
        .expect("typecheck thread panicked");
    let errors: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.severity == "error")
        .map(|d| d.message.clone())
        .collect();
    assert!(
        errors.is_empty(),
        "dependent.hnl should typecheck cleanly on {mib} MiB stack, got {errors:?}"
    );
}
