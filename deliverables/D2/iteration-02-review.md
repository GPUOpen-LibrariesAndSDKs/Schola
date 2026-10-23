# Team 20: D2 Review and Retrospective

## What worked well

### Stable contracts enabled parallel work

Each user-story owner first contributed the interfaces needed by adjacent components. Shared abstractions such as `Agent`, `Environment`, `Policy`, `Space`, and `Status` let the team implement spaces, lifecycle coordination, connector behavior, and inference in parallel. When AMD's upstream policy-state design changed, the team updated the US7 boundary without redesigning unrelated modules.

### The shared foundation kept implementations consistent

The initial Godot foundation established the SCons targets, dependency pins, runtime and training add-ons, Catch2 and GdUnit4 tests, and GitHub Actions checks. The development contract also fixed rules such as keeping the core independent from Godot and returning failures as `Status` values. These constraints made independently developed user stories fit into the same architecture.

### A focused demo made integration problems visible

The team built 1 small grid-world instead of several disconnected feature samples. Connecting environment definitions, training messages, episode resets, reward reporting, policy export, and local inference exposed integration issues that unit tests alone would not show. The demo also gave the team a concrete boundary between completed production code and temporary stand-ins.

## What did not work well

### Cross-story dependencies were resolved too late

Several implementations depended on types owned by other stories. The team merged contracts first, but concrete adapters and conversions arrived close to the demo deadline. This left insufficient time to connect every production component, so the D2 demo uses GDScript stand-ins for the Godot environment nodes and JSON over local TCP in place of the final gRPC adapter.

### The team optimized individual branches before the integrated product

Work progressed inside user-story branches, while the end-to-end path was assembled late. Components could pass their own tests without proving that a user could move from environment definition to training and inference. The release branch and focused demo corrected this for D2, but the team should have created that integration target earlier.

### The architecture change consumed iteration time

D1 used C# to prototype the Godot interface. After confirming the production requirements, the team moved to C++ GDExtensions for runtime performance, native ONNX integration, and packaging. The decision improved the production architecture but required the team to rebuild the foundation and replace parts of the original prototype during D2.

## Process changes for the next iteration

- Maintain 1 integration branch and runnable example throughout the iteration, rather than assembling it near the deadline.
- Merge shared dependencies before the features that consume them, with internal dates that leave time for integration testing.
- Split implementations into reviewable increments and merge them once their tests pass.
- Add a packaged-demo workflow early so every milestone can be tested from a clean machine without a developer checkout.
- Record the production status of every demo component so temporary stand-ins are not mistaken for completed user stories.

## Demo review

The team prepared the D2 presentation by rehearsing 1 complete grid-world workflow and keeping the controls, connection state, reward graph, and inference status visible in the scene. During the course Demo Day on October 22, 2026, the team showed the environment configuration, training exchange, episode progression, reward improvement, policy export, and ONNX inference running in Godot after the trainer closed.

The presentation used the integrated D2 prototype, not a claim that every production adapter was complete. US2 space validation and US7 ONNX inference were represented by production C++ components. The Godot-facing environment configuration, episode lifecycle, and training transport were identified as stand-ins for unfinished US1, US3, US4, and US5 integration. The production US4 lifecycle was implemented and tested separately on the release branch.

The Demo Day presentation was accepted without a required redo. The team did not record D2-specific AMD change requests during that course presentation. Earlier AMD feedback warned against over-engineering reward configuration and asked for a small end-to-end trained example; the grid-world and its compact reward model followed that direction. The main lesson was that a polished interface helps explain the workflow, but the team must label integration boundaries clearly and prioritize replacing stand-ins in the next iteration.
