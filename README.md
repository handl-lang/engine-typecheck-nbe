# engine-typecheck-nbe

Normalization-by-evaluation typecheck engine: reads `lang.types[kind]` rules,
interprets them as `Term` judgments over a `Val` domain, and mints tagged type
values.

```rust
use hnl_engine_exec_tree_walk::TreeWalkEngine;
use hnl_engine_typecheck_nbe::NbeTypecheckEngine;

let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
let report = engine.typecheck(&lang, &request)?;
```

The bound eval engine supplies value-level computation (“types are values”);
the host owns the purity boundary (effectful `Foreign` ops are rejected there).

## Docs

- [Reference](https://handl-lang.github.io/docs/engine-typecheck-nbe/reference/) — public API
- [How-to](https://handl-lang.github.io/docs/engine-typecheck-nbe/how-to/) — task recipes
- [Explanation](https://handl-lang.github.io/docs/engine-typecheck-nbe/explanation/) — design

## Development

```bash
cargo test     # deps resolve from git; no sibling checkouts needed
```

`ci.sh` is the gate: `fmt --check` → `clippy -D warnings` → `test` → `doc -D warnings`.

## License

MIT.
