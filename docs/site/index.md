# engine-typecheck-nbe

A normalization-by-evaluation typecheck engine: reads `lang.types[kind]` rules and interprets them as `Term` judgments over a `Val` domain.

## What it is

A **typecheck engine** in the `engine-*` family. Types are values (ADR-0032): checking reduces to evaluating the language-supplied judgment schemes, so a new typing rule is data, not a new Rust arm.

## Start here

- **Reference** — [public API](/engine-typecheck-nbe/reference/)
- **How-to** — [task recipes](/engine-typecheck-nbe/how-to/)
- **Explanation** — [design and rationale](/engine-typecheck-nbe/explanation/)

No tutorials: this repo is a building block used from Rust. End-to-end
walkthroughs live with the product that composes it — see the
[Handl platform](/handl/).
