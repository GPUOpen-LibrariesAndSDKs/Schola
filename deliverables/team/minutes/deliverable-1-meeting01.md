# AMD Schola General Meeting

## Meeting details

- **Date:** September 28, 2026
- **Time:** 9:00 p.m.–11:00 p.m.
- **Duration:** Approximately 2 hours
- **Location:** Discord
- **Meeting type:** Completing Deliverable 1
- **Minutes prepared by:** Sanjay Ram

## Attendees

### CSC301 team

- Guneev Pannu
- Sanjay Ram
- Bohdan Zmeul
- Vansh Sehrawat
- Shahyar Anfaz
- Isaac Tilahun
- Jimmy Zhu

## Meeting objectives

- Review the Deliverable 1 requirements and rubric.
- Divide the remaining written and prototype work.
- Convert the partner's MVP into implementable user-story lanes.
- Agree on the initial Godot-side module boundaries.

## Discussion

### 1. Deliverable outcomes

The team reviewed the Quercus handout and confirmed that the submission requires a completed planning document, an interactive prototype or equivalent developer workflow, an architecture diagram, team records, meeting minutes, and an updated repository README. The team agreed to track the implementation work and its status on Trello.

### 2. MVP and user stories

The team converted the workflow described by AMD at the kickoff meeting into seven user-story lanes: environment definition, spaces and points, transport, the connector lifecycle, Godot bindings and a demonstration environment, ONNX export, and ONNX inference and packaging. Acceptance criteria would be recorded in the D1 user-story artifact, while engineering subtasks, dependencies, assignees, and progress would remain on Trello.

### 3. Initial code organization

The team agreed to keep environment-independent C# logic separate from Godot bindings and infrastructure. The proposed projects are `Schola.Core`, `Schola.Godot`, `Schola.Grpc`, and `Schola.Onnx`. Runtime functionality will live in `addons/schola`, while training-only functionality will live in `addons/schola_training` so it can be excluded from exported games.

### 4. Work allocation

- Sanjay Ram and Jimmy Zhu will work on the D1 prototype.
- Isaac Tilahun, Vansh Sehrawat, Shahyar Anfaz, Guneev Pannu, and Bohdan Zmeul will develop and review the user stories.
- Vansh Sehrawat will serve as the partner liaison and coordinate formal communication with AMD.

## Decisions

- Trello will be the source of truth for task status, ownership, and dependencies.
- The D1 planning document will link to a separate user-story artifact containing acceptance criteria.
- The MVP will prioritize a clean end-to-end workflow over full Unreal feature parity.
- Engine-independent core code must not reference Godot, gRPC, or ONNX libraries.
- Training-only dependencies must remain separable from runtime inference.

## Action items

| Owner | Action | Target | Status at meeting end |
| --- | --- | --- | --- |
| Sanjay and Jimmy | Produce the interactive D1 prototype and make it accessible to the TA and AMD. | Before D1 submission | In progress |
| Isaac, Vansh, Shahyar, Guneev, and Bohdan | Refine the seven MVP user stories and add testable acceptance criteria. | Before D1 submission | In progress |
| Vansh | Send the user stories and architecture to AMD for review and retain evidence of the communication. | Before D1 submission | Open |
| Team | Complete assigned planning-document sections and remove the template instructions. | Before D1 submission | In progress |
| Team | Confirm the recurring weekly AMD meeting time. | Before the next partner meeting | Open |

## Team-building activity

No separate team-building activity was recorded during this working meeting. The team must complete one and add its evidence and three member fun facts to the D1 planning document before submission.

## Open questions for the next meeting

- What exact weekly meeting time works for both AMD and the student team?
- Does AMD approve the proposed MVP stories and module boundaries?
- Which Godot version and .NET runtime should the team support first?
- Is gRPC viable inside the selected Godot runtime, or should the team formally evaluate a fallback transport?


