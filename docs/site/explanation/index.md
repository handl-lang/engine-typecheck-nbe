# Explanation

## Why this crate exists

Types are values (ADR-0032): checking reduces to evaluating the language-supplied judgment schemes, so a new typing rule is data, not a new Rust arm.

## Where it sits

This crate belongs to the `engine-*` family. Plugins, kits and hosts reach it
through the kit's slot, so it can be swapped without touching a language
definition.

See the [reference](/engine-typecheck-nbe/reference/) for the API and the
[how-to guides](/engine-typecheck-nbe/how-to/) for tasks.
