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
        match name {
            "double" => Some(&DOUBLE),
            "unit_op" => Some(&UNIT_OP),
            "rust.resolve" => Some(&RESOLVE),
            _ => None,
        }
    }
    fn functions(&self) -> Vec<(&str, &dyn handl_platform::Function)> {
        vec![
            ("double", &DOUBLE),
            ("unit_op", &UNIT_OP),
            ("rust.resolve", &RESOLVE),
        ]
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

/// A nullary host op — signature domain `Unit`, like `math.e()` /
/// `proc.pid()`: `unit_op() : Number`.
struct UnitOp;
impl handl_platform::Function for UnitOp {
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
impl handl_platform::TypeSignature for UnitOp {
    fn type_signature(&self) -> Value {
        serde_json::json!({"__tag__":"type.function","__registry__":"handl.canonical","param":{"__tag__":"type.unit","__registry__":"handl.canonical"},"body":{"__tag__":"type.number","__registry__":"handl.canonical"}})
    }
}
static UNIT_OP: UnitOp = UnitOp;

/// A namespaced op local name — `"rust.resolve"` inside module `mymod`,
/// the `ffi.rust.resolve` shape (P-013 T-152): `String ~> Number`.
struct Resolve;
impl handl_platform::Function for Resolve {
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
impl handl_platform::TypeSignature for Resolve {
    fn type_signature(&self) -> Value {
        serde_json::json!({"__tag__":"type.function","__registry__":"handl.canonical","param":{"__tag__":"type.string","__registry__":"handl.canonical"},"body":{"__tag__":"type.number","__registry__":"handl.canonical"}})
    }
}
static RESOLVE: Resolve = Resolve;

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

/// `f()` on a `Unit -> T` signature peels the unit domain — host ops
/// like `math.e()`/`proc.pid()` declare nullary signatures this way,
/// and `()` at eval passes zero args to the op.
#[test]
fn zero_arg_call_on_unit_domain() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nimport mymod\nx := mymod.unit_op()\n}");
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    assert!(
        result
            .node_types
            .values()
            .any(|v| tag(v) == Some("type.number")),
        "mymod.unit_op() should synth type.number: {result:?}"
    );
}

/// `f()` on a non-unit domain is a missing argument — `double :
/// Number -> Number` cannot be applied with zero args.
#[test]
fn zero_arg_call_on_non_unit_domain_fails() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nimport mymod\nx := mymod.double()\n}");
    let res = engine().typecheck(&lang, &TypecheckRequest::new(ast));
    assert!(
        res.is_err(),
        "mymod.double() should fail with a missing argument: {res:?}"
    );
}

/// `f()` where the domain is an unsolved meta is still a missing
/// argument — a meta is not `Unit`.
#[test]
fn zero_arg_call_on_meta_domain_fails() {
    let lang = canonical_lang();
    let ast = parse(&lang, "do {\nf := (x ~> x)\ny := f()\n}");
    let res = engine().typecheck(&lang, &TypecheckRequest::new(ast));
    assert!(
        res.is_err(),
        "f() on a meta domain should fail with a missing argument: {res:?}"
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

/// A dotted export name (`"rust.resolve"` — the `ffi.rust.resolve`
/// local-name shape, P-013 T-152) nests as `type.record` prefixes in
/// `Sig_m`: `mymod.rust` projects the record, `.resolve` its field, and
/// the call synthesizes the declared signature's body.
#[test]
fn dotted_export_name_nests_module_sig() {
    let lang = canonical_lang();
    let ast = parse(
        &lang,
        "do {\nimport mymod\nx := mymod.rust.resolve(\"a\")\n}",
    );
    let result = engine()
        .typecheck(&lang, &TypecheckRequest::new(ast))
        .expect("typecheck should succeed");
    assert!(
        result
            .node_types
            .values()
            .any(|v| tag(v) == Some("type.number")),
        "mymod.rust.resolve(\"a\") should synth type.number: {result:?}"
    );
}

/// The nested record is exact: a sibling under the same prefix that was
/// never exported is still a `record_field_type` miss.
#[test]
fn dotted_export_missing_sibling_fails() {
    let lang = canonical_lang();
    let bad = parse(&lang, "do {\nimport mymod\nx := mymod.rust.missing\n}");
    let res = engine().typecheck(&lang, &TypecheckRequest::new(bad));
    assert!(
        res.is_err(),
        "mymod.rust.missing should fail field resolution: {res:?}"
    );
}
