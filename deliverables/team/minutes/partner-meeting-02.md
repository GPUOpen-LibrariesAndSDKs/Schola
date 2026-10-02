# AMD Schola Partner Kickoff Meeting

## Meeting details

- **Date:** Oct 2, 2026
- **Time:** 11:00 a.m.–12:00 p.m.
- **Duration:** Approximately 50 minutes
- **Location:** Microsoft Teams
- **Meeting type:** Partner Meeting 2
- **Minutes prepared by:** Sanjay Ram Co-authored by Granola

## Attendees

### AMD

- Alexander Cann -- MTS @ AMD

### CSC301 team

- Guneev Pannu
- Sanjay Ram
- Bohdan Zmeul
- Vansh Sehrawat
- Shahyar Anfaz
- Isaac Tilahun
- Jimmy Zhu

## Meeting objectives

-   Plan for the term: what our team aims to achieve by the end of the term
-   MVP check: Reviewing our current user stories and discussing potential changes
-   Demo: a quick look at our Godot frontend demo.
-   Plugin language: C#, C++, or Rust? Each have their own trade-offs
-   Multi-agent support

## Discussion

### Plan for the Term: User Stories Overview

**User Story 1 (Isaac and Sanjay): define a simple RL environment**

- Initialize, reset, step, and observe
- Direction confirmed: multi-agent support (more desirable for game settings; enables self-play)
- "Engine-independent" wording flagged as unclear; should reflect independence from networking/backend, not the engine itself

**User Story 2: Godot proto handling**

- Build and parse protos for Box, Discrete, MultiDiscrete, MultiBinary spaces
- Send observations, receive actions, describe spaces at startup

**User Story 3: gRPC connection layer**

- Extend Python environment to Godot without changing Python side
- Three functions: establish connection, ensure clean architecture, isolate transport
- "Engine-independent" wording flagged here too; should be "opinionated toward Godot constraints," not mirroring Unreal decisions

**User Story 4: environment lifecycle (split between two contributors)**

- One owner: single-environment lifecycle (init, reset, step, truncation, episode leakage prevention)
- Other owner: multi-environment support and auto-reset modes
- Shared: interfaces and contracts
- Unreal's Parallel For should not be directly ported; Godot threading rules differ, parallelism is a stretch goal
- Interface structure in Unreal is heavily workaround-driven (Blueprint constraints); Godot has different constraints, so don't mirror those decisions blindly

**User Story 5 (Vansh): Godot frontend**

- Nodes and inspector values for configuring training environments
- Custom actuators are a stretch goal; scope kept small initially

**User Story 6: model export validation**

- Train a basic CartPole model, export via Skola's existing export functions
- Visualize ONNX output to confirm input/output shapes
- Prep step for User Story 7; export code already exists on the Python side

**User Story 7: inference in Godot**

- Port inference architecture from Unreal plugin into Godot C++ module
- Reference: NNE (Neural Network Engine, internally called "Nini" by Epic) policy as exact reference
- No mature ONNX inference library confirmed for Godot; solo-dev repo last updated 2 years ago flagged as risky
- Alternatives if ONNX is a bottleneck: LibTorch (C++ PyTorch); export format could be adapted on the Python side
- ONNX preferred for universality; will continue searching for a suitable library

### Godot Demo Walkthrough

- Demo scene shown: player, platform, goal node
- Skola environment node configurable in inspector: progress reward, step reward, goal reward, penalty, episode limit
- Agent node links to environment; no actuators yet
- Hardcoded logic for course deliverable, no ML running
- Feedback on reward fields (goal reward, failure penalty): risk of over-engineering
- Defining "goal" requires a language for conditions, which can spiral into reinventing a DSL
- Step reward is more reasonable to keep as a built-in
- Preferred pattern: modular helper components (e.g. frame stacking) that users compose, rather than baking everything into the environment node

### Plugin Language: C#, C++, or Rust

- Three options considered: C# (.NET Godot only), C++ (official GDExtension), Rust (community-maintained)
- C# limitation: restricts plugin to the .NET build of Godot, excluding the standard build
- C++ concern raised: existing Skola architecture is already C++, making it feel like a less novel port
- Counterpoint: adaptation needs are still significant due to Godot-specific constraints (threading, interfaces, etc.)
- Rust: unknown gRPC compatibility and ecosystem maturity; needs research
- Key factor: inference library availability, especially for User Story 7
- If no C# inference library exists, C# becomes a blocker
- Decision deferred; team leaning toward C++ but will research Rust
- Will update and confirm at next meeting (aim for earlier in the week)

### Scope and Velocity Check

- Overall user story scope looks good on first review
- Ways to scale down if needed: focus on training or inference only, cut certain space types
- Ways to scale up: add more actuator/observer types, more advanced example scenarios, end-to-end trained model demo
- Rough velocity target: ~3 user stories/month to finish on time (10 total)

### Coding Standards and AI Use

**Godot style guide**

- Follow Godot's own style guide and tooling, not Skola's Unreal-based Clang file
- Unreal conventions (e.g. bBool, FStruct, AActor prefixes) should not carry over to Godot
- Python side uses Black and PEP-adjacent style; each engine branch should look idiomatic to its own ecosystem
- Official Godot naming conventions exist; team to research and document

**Repository structure**

- Top-level pod directory with plugin source and tests is fine
- Follow Godot's recommended plugin folder structure for internal organization
- Skola may eventually consolidate into a monorepo or multi-repo (one per engine); team's structure will be adapted at that point

**AI use policy**

- No restrictions on using AI for coding
- Strong recommendation: read, understand, and take ownership of all generated code
- Code quality matters as a reusable package; "slop" affects usability and integration

## Next Steps

- Research and decide on plugin language (C++, C#, or Rust)
  - Check gRPC compatibility, inference library availability, and Godot ecosystem support for each; confirm decision before next meeting.
- Research Godot style guide and document conventions for the team
  - Find official or widely adopted Godot style; do not carry over Unreal naming conventions.
- Schedule next meeting for earlier in the week
  - Avoid a Friday slot so the plugin language decision can be confirmed without a multi-day wait

