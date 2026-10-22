# Schola for Godot (Team 20, The Hard Workers) - Individual Contributions (D2)

> For each team member: name, role, and specific contributions. Pull request numbers refer to this repository.

**Shahyar Anfaz**, Episode Lifecycle Developer and build foundation owner
> - Built the Godot GDExtension foundation (#1): SCons build, runtime and training add-ons, pinned `godot-cpp` and Catch2 submodules, dependency lock, GitHub Actions CI, PR template, and contributing guide.
> - Fixed CI so docs-only PRs report required checks (#5).
> - Defined the shared `Status` and identifier types used by every module (#8).
> - Defined the US4 episode lifecycle API (#6) and is implementing it with tests on `Shahyar/us4-lifecycle`.
> - Wrote the development contract and updated the project READMEs, plus the partner meeting 3 and TA meeting 2 minutes (#13).

**Jimmy Zhu**, Developer and QA
> - Defined the US2 space and point API (#7): Box, Discrete, MultiDiscrete, MultiBinary, and Dict spaces and points, with a header compile test.

**Isaac Tilahun**, Backend Developer
> - Defined the US1 core environment and agent API (#11): `Agent`, `Environment`, and `InteractionDefinition`, plus the `ScholaAgent` and `ScholaEnvironment` interfaces.

**Sanjay Ram**, Developer and Meeting Manager
> - Defined the US3 connector and gRPC transport API (#12): the connector loop, the gRPC connector, training messages, and the training-side Godot bindings.
> - Recorded the TA meeting 1 minutes (#3).

**Vansh Sehrawat**, Developer and Partner Liaison
> - Defined the US5 Godot environment binding API (#9): `ScholaAgent` and `ScholaEnvironment` nodes, the Inspector space resources, and the Variant conversion interface.

**Bohdan Zmeul**, AI Integration Developer
> - Defined the US7 inference API (#2) and aligned it with AMD's upstream policy state redesign (#15).
> - Merged AMD's latest upstream Schola changes into the fork (#14).
> - Implemented US7 on `Bohdan/us7-onnx-inference-packaging` (PR open, waiting on US1, US2, US5, and interop implementations):
>   - pinned ONNX Runtime 1.30.0 with a checksum-verified setup script, build and CI integration, and export packaging
>   - implemented the policy, per-agent policy state, and stepper
>   - implemented ONNX model loading, validation, and inference with recurrent state
>   - implemented the Godot `ScholaOnnxPolicy`, `ScholaPolicyState`, and `ScholaInferenceStepper` classes
>   - added test models, including models exported by Schola's Python exporter and checked against Python results
>   - wrote an inference guide

**Guneev Pannu**, Developer and Project Manager
> - Managed project coordination for D2: team scheduling, task tracking, and keeping deliverables on schedule.
