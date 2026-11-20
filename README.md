# engine-typecheck-nbe

Normalization-by-evaluation typecheck engine: reads `lang.types[kind]` rules,
interprets them as `Term` judgments over a `Val` domain, and mints tagged type
values.

An `engine-*` crate: a concrete `TypecheckEngine` — "types are values". The
bound eval engine supplies value-level computation; the host owns the purity
boundary (effectful `Foreign` ops are rejected there).

```rust
use hnl_engine_exec_tree_walk::TreeWalkEngine;
use hnl_engine_typecheck_nbe::NbeTypecheckEngine;

let engine = NbeTypecheckEngine::new(Box::new(TreeWalkEngine::new()));
let report = engine.typecheck(&lang, &request)?;
```

## Development

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

The repo's `xtask/` crate provides the gate tasks (`fmt`, `clippy`, `test`, `doc`,
`check`, `gate`); `ci.sh` is the gate entry. Deps resolve from git, so no
sibling checkouts are needed; a plain clone test-runs as-is.

## License

MIT OR Apache-2.0.

## Documentation

<https://handl-lang.github.io/docs/engine-typecheck-nbe/>
