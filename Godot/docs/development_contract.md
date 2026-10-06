# Godot Development Contract

This document defines shared implementation rules for the Schola Godot modules. It intentionally does not define environment, agent, space, lifecycle, transport, binding, or policy APIs; the owners of those user stories define those interfaces in their feature work.

Changes to this contract require a focused pull request and review from at least one contributor working in an affected module.

## C++ conventions

- Engine-independent project code uses the `schola` namespace. Godot-facing classes follow the form required by `godot-cpp` and Godot class registration.
- Project C++ filenames use `snake_case.h` and `snake_case.cpp`. Types use `PascalCase`; functions and local variables use `snake_case`.
- New headers use `#pragma once`, matching the existing Schola source tree.
- A source file includes its corresponding header first, followed by project headers and then third-party or standard-library headers.
- Code under `src/core` must not include Godot, gRPC, Protocol Buffer, or ONNX Runtime headers. Technology-specific types are converted at module boundaries rather than leaking into core headers.
- Project code does not throw or depend on C++ exceptions. It also does not rely on C++ RTTI across GDExtension boundaries; Godot-facing type checks use Godot's supported mechanisms.

## Ownership and lifetime

- Godot owns the lifetime of `Object` and `Node` instances. Engine-independent code does not delete Godot-owned objects.
- Godot `RefCounted` values and `godot::Ref<T>` remain in Godot-facing modules.
- Prefer values and references for call-scoped data. Use `std::unique_ptr` for exclusive ownership and `std::shared_ptr<const T>` only when immutable data must share a lifetime, such as a cross-thread handoff.
- Raw pointers are non-owning unless ownership is explicitly documented beside the declaration.
- Technology adapters convert their native values into engine-independent values before passing them toward `src/core`.

## Threading boundary

- Only the Godot main thread may access the scene tree, read or mutate Godot objects, emit signals, or invoke user callbacks.
- Transport and inference implementations may perform blocking or expensive technology-specific work on worker threads, but those workers must not retain or access Godot objects.
- Values crossing a thread boundary must be engine-independent or technology-owned immutable data whose lifetime extends through the handoff.
- Worker results are handed back to the main thread before they affect an environment, agent, node, resource, or other user-facing Godot state.
- Shutdown must stop new work and join owned worker threads before destroying the state they can access.

These rules define where work may execute, not the queue type, lifecycle sequence, transport interface, or policy API. Those choices remain with their user-story owners. Godot-specific constraints take precedence over parallelism patterns used by the Unreal implementation.

## Error and logging boundary

- Recoverable failures are represented explicitly; project code does not use exceptions for normal configuration, validation, transport, model-loading, or shutdown failures.
- Assertions are reserved for programmer errors and broken internal invariants. User input, missing resources, incompatible data, unavailable services, and invalid project configuration must produce recoverable failures.
- Engine-independent code does not call Godot logging or error macros and does not write directly to a technology-specific logger.
- Godot-facing bindings translate failures into appropriate Godot diagnostics at the user-facing boundary. Transport and inference adapters add relevant technology context before returning a failure toward that boundary.
- Diagnostics must say what failed and provide enough context for a developer to correct it. They must not expose secrets or emit the same failure repeatedly on every frame.
- Cleanup and shutdown paths must remain safe after partial initialization and must not discard failures that require developer action.

This contract does not select a shared `Status`, `Result<T>`, error enum, exception wrapper, or logging class. A concrete type becomes shared only through a separately reviewed change involving every user story that would consume it.

## Protocol Buffer generation

- The existing definitions under the repository-level `Proto/` directory are the shared Schola wire contract and are authoritative for every engine and the Python package. They are not Unreal-only. Godot integration code must not create `Godot/Proto/` or maintain another divergent copy of those definitions.
- Engine-specific generated bindings remain separate: Unreal's generated C++ is under `Source/ScholaProtobuf/`, Python's generated bindings are under `Resources/python/schola/generated/`, and Godot's generated C++ uses the path below.
- `Godot/generated/proto/` is reserved for generated Godot C++ Protocol Buffer and gRPC sources. Generated output is build state, is ignored by Git, and must not be edited manually.
- Handwritten conversion and transport code belongs under `src/transport/grpc/`, outside the generated directory.
- The US3 compatibility spike will select compatible gRPC, Protocol Buffers, and generator versions together. Until that decision is reviewed, feature branches must not add their own generator or dependency versions.
- Once selected, generation must be deterministic and exposed through the shared build rather than requiring developers to run undocumented commands.

These rules assign ownership and paths without choosing the transport API or resolving the versions that remain under investigation with AMD.

## Shared-file policy

The extension entry points, `SConstruct`, `.gdextension` descriptors, dependency pins, CI workflow, and this contract are shared infrastructure. Feature branches should use the established module hooks and source discovery. A change to shared infrastructure should be isolated in a prerequisite commit or pull request rather than bundled into a user-story API change.

## Tests

- Engine-independent C++ unit tests use the pinned Catch2 release under `tests/unit/` and build through the shared `unit_tests` SCons target.
- Godot scene, binding, and extension integration tests use the pinned GdUnit4 release under `tests/integration/`.
- Python compatibility tests remain in the existing pytest suite.
- Tests assert externally observable behavior. A user-story branch must not introduce another test framework without a focused change to this contract and the shared CI workflow.
