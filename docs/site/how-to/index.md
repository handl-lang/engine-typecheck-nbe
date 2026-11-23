# How-to guides

Task recipes for `tool-typecheck-nbe`.

## Run the gate

```bash
./ci.sh          # fmt -> clippy -D warnings -> test -> doc -D warnings
./ci.sh test     # one stage
```

## Use it from Rust

```bash
cargo add --git https://github.com/handl-lang/tool-typecheck-nbe --branch main
```

See the [reference](/tool-typecheck-nbe/reference/) for the types involved and the
[explanation](/tool-typecheck-nbe/explanation/) for why the crate is shaped this way.
