# AMD Schola Partner Meeting 3

## Meeting details

- **Date:** Oct 9, 2026
- **Time:** 11:00 a.m.–12:00 p.m.
- **Meeting type:** Partner Meeting 3
- **Minutes prepared by:** Sanjay Ram / Shahyar Anfaz
- **Minutes source:** Shared Granola notes

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

## Announcements

- AMD will share a frozen commit early the following week as the cutoff baseline for the Godot plugin implementation.

## Progress review

- The team switched from C# to a C++ GDExtension. Rust was ruled out because of licensing concerns around the evaluated option.
- The C++ repository structure is in place, and each team member opened a pull request defining the initial API for their user story.
- The APIs use interfaces and abstract types so dependent user stories can begin implementation before all concrete implementations are complete.

## Decisions and technical discussion

### Language and extension model

- The team confirmed C++ as the implementation language instead of Rust or GDScript.

### Lifecycle API

- Each training step requires an asynchronous, two-phase operation because Godot applies physics updates at frame boundaries.
- A consistent one-frame delay is acceptable for the MVP because the model can adapt to a stable delay.
- Strict observation validation should be opt-in or limited to debug builds. Validating every value can be prohibitively expensive for large observations.
- The lifecycle API owner will revise the validation proposal to follow Godot conventions.

### ONNX Runtime

- The team will target the latest ONNX Runtime release even though it no longer supports Intel-based macOS, because AMD is the project partner.
- ONNX Runtime binaries should not be committed through Git LFS because of free-tier quota limits. The preferred approach is a setup script that downloads Microsoft's prebuilt package.
- GPU inference is a stretch goal rather than an MVP requirement. For small in-game models, CPU inference may be competitive once GPU transfer overhead is considered.

### Recurrent and multi-agent support

- The upstream recurrent and multi-agent API was not yet merged.
- The Godot API should leave room for those features without blocking current implementation work.

## Action items

| Owner | Action | Status at meeting end |
| --- | --- | --- |
| AMD partner | Merge the recurrent and multi-agent API and share the frozen commit. | Open |
| All team members | Continue defining and implementing APIs from the current repository baseline. | In progress |
| ONNX user-story owner | Finalize the setup-script approach and preserve extension points for recurrent and multi-agent models. | Open |
| Isaac | Revise strict observation validation to follow Godot conventions and the performance rationale discussed. | Open |
| Team | Use the frozen upstream commit before finalizing multi-agent portions of the APIs. | Open |
