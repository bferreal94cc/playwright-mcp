# init

My AI agent harness

## Behavioral rules

- Use the harness's MCP tools (`mcp__init__*`) for orchestration
- Memory ranking and routing are kernel primitives (`@metaharness/kernel`) the harness uses internally — there are no separate CLI subcommands for them
- Defer destructive operations to the user

## Commands

After `init init`, the following are available:

| Command | What it does |
|---|---|
| `init init` | Scaffold the harness into the current project |
| `init doctor` | Health check the install |
| `init --version` | Print the kernel version |

Run `init --help` for the full list. Memory ranking and routing are kernel
primitives used programmatically (see [@metaharness/kernel](https://www.npmjs.com/package/@metaharness/kernel)),
not CLI subcommands.

## Architecture

This harness uses [@metaharness/kernel](https://www.npmjs.com/package/@metaharness/kernel) for its primitives. The kernel is a Rust-compiled WASM module with a NAPI-RS native fallback — same code runs identically on every platform.
