# Agent guide — tool-typecheck-nbe

`hnl-tool-typecheck-nbe` — normalization-by-evaluation typecheck
engine: reads `lang.types[kind]` rules, interprets them as `Term`
judgments over a `Val` domain, and mints tagged type values ("types
are values"). The bound eval engine supplies value-level computation;
the host owns the purity boundary. `publish = false` (git-install
only). Rust 2024.

## Layout

| path | role |
| --- | --- |
| `src/lib.rs` | crate docs + engine entry |
| `src/checker.rs`, `src/judgments.rs` | judgment interpretation + checking |
| `src/dependent.rs`, `src/predicate_eval.rs` | dependent types + predicate evaluation |
| `src/totality_check.rs`, `src/stats.rs`, `src/tests.rs` | totality, stats, unit tests |
| `tests/` | `parity`, `dependent_structure`, `grade_type_dispatch`, `import_sig`, `cubical_p1`–`p4` |
| `project/`, `docs/` | project data + engine doc |
| `xtask/`, `workflows/gate.yaml`, `bupha.toml`, `.cargo/` | vendored repo gate + CI wiring |

## Contract

- Deps resolve from git — `handl-language`/`handl-value` →
  `handl-lang/kernel`, `hnl-exec` → `handl-lang/exec`,
  `hnl-tool-typecheck` → sibling repo, `handl-platform` →
  `handl-lang/host`, all `branch = "main"`.
- Implements `hnl-tool-typecheck`'s `TypecheckEngine` — rule
  changes belong in `lang.types` data, not Rust matches.

## Verification

```bash
cargo xtask gate   # fmt --check → clippy -D warnings → test → doc
```


<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
