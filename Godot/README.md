# Schola for Godot

This directory contains the Godot port of AMD Schola. The design keeps reusable reinforcement-learning logic independent from Godot, gRPC, and ONNX so that each part can be developed and tested separately.

## Directory structure

```text
Godot/
├── addons/
│   ├── schola/                  Runtime Godot plugin
│   └── schola_training/         Training-only Godot plugin
├── src/
│   ├── Schola.Core/             Engine-independent interfaces and behavior
│   │   ├── Environments/        Environment and agent contracts
│   │   ├── Spaces/              Observation/action spaces and point values
│   │   └── Connector/           Step, reset, and auto-reset coordination
│   ├── Schola.Godot/            Godot nodes and Inspector bindings
│   ├── Schola.Grpc/             gRPC transport implementation
│   └── Schola.Onnx/             ONNX policy implementation
├── examples/
│   └── basic_environment/       End-to-end demonstration environment
└── tests/
    ├── unit/                    Fast tests for individual modules
    └── integration/             Cross-module and Python compatibility tests
```

## Module boundaries

### `Schola.Core`

Contains the engine-independent environment interfaces, space and point types, agent state, connector loop, and transport/policy abstractions. It must not reference Godot, gRPC, or ONNX libraries.

### `Schola.Godot`

Adapts the core abstractions to Godot nodes, resources, the Inspector, and the engine lifecycle. This is the only project that may reference Godot APIs.

### `Schola.Grpc`

Implements the core transport abstraction using Schola's existing Protocol Buffer and gRPC contracts. Keeping it separate prevents networking dependencies from leaking into the core or inference-only builds.

### `Schola.Onnx`

Implements the core policy abstraction using ONNX Runtime. It supports local inference without a running Python process or training connection.

### Godot add-ons

- `addons/schola/` contains runtime functionality required by both development projects and exported games.
- `addons/schola_training/` contains training and communication integration. Exported games must be able to exclude this add-on while retaining runtime inference.

## Dependency direction

```text
Schola.Godot ──→ Schola.Core ←── Schola.Grpc
                       ↑
                  Schola.Onnx
```

Dependencies must point toward `Schola.Core`. The core must never depend on an adapter project.

## User-story mapping

| User story | Primary location |
| --- | --- |
| US1: Define an environment | `src/Schola.Core/Environments/` |
| US2: Declare spaces | `src/Schola.Core/Spaces/` |
| US3: Connect Python training tools | `src/Schola.Grpc/` and `addons/schola_training/` |
| US4: Execute the episode lifecycle | `src/Schola.Core/Connector/` |
| US5: Configure through Godot | `src/Schola.Godot/` and `examples/basic_environment/` |
| US6: Export a policy to ONNX | Existing Python Schola package under `Resources/python/` |
| US7: Run and ship an ONNX policy | `src/Schola.Onnx/` and `addons/schola/` |

The detailed acceptance criteria and current planning scope are in [`deliverables/D1/user-stories.md`](../deliverables/D1/user-stories.md). Engineering tasks, dependencies, assignees, and progress are tracked on the team's [Trello board](https://trello.com/b/Ry0Qkx2R).

## Working on a story

1. Branch from an up-to-date `main` using the branch name recorded on the Trello card.
2. Keep changes inside the module responsible for the story. Introduce cross-module dependencies only when they follow the dependency direction above.
3. Add unit tests for engine-independent behavior and integration tests for interactions between Godot, transport, Python, and inference.
4. Open a pull request and link its Trello card. Do not merge until the relevant tests pass and another team member has reviewed it.

Generated Godot state, build output, local models, and editor-specific files must not be committed.
