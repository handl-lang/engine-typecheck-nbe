//! `stmt.import` binds a signature record `Sig_m` (phase-1 spec WS-B):
//! `import m` binds `m : Sig_m` (`type.module_sig{module, fields}`), named
//! imports are field projections, `* as m` binds the record, `*` splices
//! fields, and `resolve`/dynamic specs degrade to `type.unknown`.
//! `stmt.export` accumulates the module's exported names + types into
//! `TypecheckResult.signature`.

use handl_kit_data_prelude::language_from_yaml;
use handl_language::{Ast, Language, Value};
use hnl_exec_tree_walk::TreeWalkEngine;
use hnl_tool_lex_pattern::lex;
use hnl_tool_parse::{ParserEngine, ParserRequest};
use hnl_tool_parse_pratt::PrattEngine;
use hnl_tool_typecheck::{TypecheckEngine, TypecheckRequest};
use hnl_tool_typecheck_nbe::NbeTypecheckEngine;
use std::collections::HashMap;
use std::rc::Rc;

fn canonical_lang() -> Language {
    language_from_yaml().expect("canonical language should assemble")
}

fn parse(lang: &Language, src: &str) -> Ast {
    let tokens = lex(lang, src).expect("lex should succeed");
    PrattEngine::new()
        .parse(lang, &ParserRequest::new(tokens))
        .expect("parse should succeed")
}

/// A tiny module registry: one module `mymod` exporting `double` (a
/// function with a declared type signature) and `flag` (an unsigned
/// value export).
struct TestModule;
impl handl_platform::Module for TestModule {
    fn name(&self) -> &str {
        "mymod"
    }
    fn function(&self, name: &str) -> Option<&dyn handl_platform::Function> {
        if name == "double" {
            Some(&DOUBLE)
        } else {
            None
        }
    }
    fn functions(&self) -> Vec<(&str, &dyn handl_platform::Function)> {
        vec![("double", &DOUBLE)]
    }
    fn value_names(&self) -> Vec<String> {
        vec!["flag".to_string()]
    }
    fn value(&self, name: &str) -> Option<Value> {
        (name == "flag").then_some(Value::Bool(true))
    }
}

struct Double;
impl handl_platform::Function for Double {
    fn call(
        &self,
        _args: &[Value],
        _ctx: &mut handl_platform::EvalContext,
    ) -> Result<Value, handl_platform::EvalError> {
        Ok(Value::Null)
    }
    fn as_type_signature(&self) -> Option<&dyn handl_platform::TypeSignature> {
        Some(self)
    }
}
impl handl_platform::TypeSignature for Double {
    fn type_signature(&self) -> Value {
        serde_json::json!({"__tag__":"type.function","__registry__":"handl.canonical","param":{"__tag__":"type.number","__registry__":"handl.canonical"},"body":{"__tag__":"type.number","__registry__":"handl.canonical"}})
    }
}
static DOUBLE: Double = Double;

fn engine() -> Box<dyn TypecheckEngine> {
    let registry = handl_plugin_module::ModuleRegistry::new().with(TestModule);
    NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new())).with_modules(Rc::new(registry))
}

fn tag(v: &Value) -> Option<&str> {
    v.get("__tag__").and_then(Value::as_str)
}

/// `import m` binds `m : Sig_m` — a `type.module_sig` record whose fields
/// are the module's exports (`double` with its declared signature,
/// `flag` at `type.unknown`).
#[test]
fn import_binds_sig_record() {
    let lang = canonical_lang();
    // Wrap statements in a block so the module sees them in order.
    let ast = parse(&lang, "do {\nimport mymod\nx := mymod.double\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let t = result
        .node_types
        .values()
        .find(|v| tag(v) == Some("type.function"))
        .cloned();
    assert!(
        t.is_some(),
        "mymod.double should project double's signature: {result:?}"
    );
}

/// `from "m" import f as g` binds `g : Sig_m.f` — here the projection's
/// declared signature.
#[test]
fn named_import_binds_field_projection() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nfrom \"mymod\" import double as d\nx := d\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let t = result
        .node_types
        .values()
        .find(|v| tag(v) == Some("type.function"))
        .cloned();
    assert!(t.is_some(), "d should carry double's signature: {result:?}");
}

/// `from "m" import *` splices `Sig_m`'s fields into Γ and also binds the
/// bare module name (qualified access parity with the model).
#[test]
fn star_import_splices_fields() {
    let lang = canonical_lang();
    let ast = parse(
        &lang,
        "do {\nfrom \"mymod\" import *\nx := flag\ny := mymod.double\n}",
    );
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let has_fn = result
        .node_types
        .values()
        .any(|v| tag(v) == Some("type.function"));
    assert!(has_fn, "mymod.double projection should work: {result:?}");
}

/// A dynamic/unknown spec binds `type.unknown` — `resolve`-style imports
/// are statically opaque (phase-1 spec WS-B).
#[test]
fn dynamic_import_is_unknown() {
    let lang = canonical_lang();
    let ast = parse(
        &lang,
        "do {\nimport no_such_module_xyz\nx := no_such_module_xyz\n}",
    );
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    assert!(
        result
            .node_types
            .values()
            .any(|v| tag(v) == Some("type.unknown")),
        "no_such_module_xyz should be bound to type.unknown: {result:?}"
    );
}

/// `export x` records `(x, type-of-x)` into the module signature surfaced
/// on `TypecheckResult.signature`.
#[test]
fn export_builds_module_signature() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nx := 1\nexport x\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let fields = result
        .signature
        .get("fields")
        .and_then(Value::as_object)
        .expect("signature should carry a fields record");
    assert_eq!(
        tag(fields.get("x").expect("x exported")),
        Some("type.number"),
        "x : type.number in Sig_m: {fields:?}"
    );
}

/// `export x :: expr` binds `x` and records the value's type.
#[test]
fn export_binding_records_type() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nexport x := 2\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let fields = result
        .signature
        .get("fields")
        .and_then(Value::as_object)
        .expect("signature should carry a fields record");
    assert_eq!(
        tag(fields.get("x").expect("x exported")),
        Some("type.number")
    );
}

/// A `type.module_sig` bound in Γ projects fields through the ordinary
/// `field-access`/`record_field_type` machinery; a missing field probes
/// `<module>.<name>` through the module chain (opaque function type when
/// the op has no declared signature), erroring when nothing resolves.
#[test]
fn module_sig_field_projection() {
    let lang = canonical_lang();
    // `flag` IS in the sig (value export at type.unknown); `double` has
    // a declared signature — both project.
    let ast = parse(&lang, "do {\nimport mymod\nx := mymod.double(4)\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    // The assign itself records type.unit; the call node's recorded type
    // is the projected signature's body.
    assert!(
        result
            .node_types
            .values()
            .any(|v| tag(v) == Some("type.number")),
        "mymod.double(4) should synth type.number: {result:?}"
    );
}

/// An absent field on `type.module_sig` that resolves through the module
/// chain without a signature mints an opaque meta-function type (the
/// `lookup` host-fn fallback's shape).
#[test]
fn missing_sig_field_probes_module_chain() {
    // `mymod.flag` is a *value* export — `lookup("mymod.flag")` is None
    // (functions only), so `mymod.unknown_op` is a real miss → error.
    let lang = canonical_lang();
    let bad = parse(&lang, "do {\nimport mymod\nx := mymod.unknown_op\n}");
    let res = engine().typecheck(&lang, &TypecheckRequest::new(bad));
    assert!(
        res.is_err(),
        "mymod.unknown_op should fail field resolution: {res:?}"
    );
}

/// A `resolve`-style dynamic import of a non-existent module binds
/// `type.unknown` and its projections stay unknown — the spec's dynamic
/// fallback.
#[test]
fn resolve_dynamic_binds_unknown() {
    let lang = canonical_lang();
    let ast = parse(
        &lang,
        "do {\nfrom \"./local/file.hnl\" import thing\nx := thing\n}",
    );
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    assert!(
        result
            .node_types
            .values()
            .any(|v| tag(v) == Some("type.unknown"))
    );
}

/// Globals map smoke: the engine's own `with_global` path is untouched.
#[test]
fn empty_module_has_empty_signature() {
    let lang = canonical_lang();
    let ast = parse(&lang, "1");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    let fields = result
        .signature
        .get("fields")
        .and_then(Value::as_object)
        .expect("signature should be a fields record");
    assert!(fields.is_empty(), "no exports → empty Sig_m: {fields:?}");
    let _: HashMap<String, Value> = HashMap::new();
}
