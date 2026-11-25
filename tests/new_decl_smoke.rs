use handl_language::{Ast, Language};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex as lex_src;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;

#[test]
fn smoke_new_decl_typechecks() {
    let src = include_str!("../../host/tests/smoke/new_decl.hnl");
    let kit = handl_kit_data_prelude::assemble().expect("canonical assembly");
    let lang: &Language = kit.lang();
    let ast: Ast = PrattEngine::new()
        .parse(lang, &ParserRequest::new(lex_src(lang, src).expect("lex")))
        .expect("parse");
    let nbe = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
    nbe.typecheck(lang, &TypecheckRequest::new(ast))
        .expect("typecheck");
}
