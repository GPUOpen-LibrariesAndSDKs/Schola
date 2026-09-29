# AMD Schola Partner Kickoff Meeting

## Meeting details

- **Date:** September 25, 2026
- **Time:** 11:00 a.m.–12:00 p.m.
- **Duration:** Approximately 41 minutes
- **Location:** Microsoft Teams
- **Meeting type:** Initial partner meeting and project-scoping discussion
- **Minutes prepared by:** Shahyar Anfaz
  
## Attendees

### AMD

- Alexander Cann -- MTS @ AMD. Primary representative for the meeting.
- TianYue Liu (Michael) -- Senior SWE @ AMD. Per Alex, the main contact as time goes on.

### CSC301 team

- Guneev Pannu
- Sanjay Ram
- Bohdan Zmeul
- Vansh Sehrawat
- Shahyar Anfaz
- Isaac Tilahun
- Jimmy Zhu

## Meeting objectives

- Introduce the AMD representatives and student team.
- Confirm the scope of the Godot port in relation to the published Unity project description.
- Establish the intended scope and minimum viable product (MVP).
- Discuss technical constraints and a practical way to divide the work.
- Establish communication expectations and immediate next steps.

## Discussion

### 1. Project assignment and Godot scope

The published CSC301 project description names Unity, but the teaching team recommended that this team work on a Godot port. Alex confirmed that AMD had originally considered separate Unity and Godot projects and that the published specification can be interpreted as referring to Godot for this team.

The Godot port should not be a line-for-line translation of the Unreal implementation. It should preserve Schola's recognizable high-level concepts and feature set while using Godot's native conventions and extension mechanisms. The team will need to balance consistency across engines with an idiomatic Godot developer experience.

### 2. Existing Schola architecture and reuse

The team described Schola as having an engine-side integration and a Python reinforcement-learning stack connected through gRPC and Protocol Buffers. AMD confirmed that the main expected contribution is the Godot engine-side integration.

The existing Python stack should be reused wherever practical. Small Python or protocol changes may be necessary when Schola moves from supporting one engine to supporting multiple engines. AMD can help address constraints that appear in the maintained Python and Unreal components.

gRPC is the existing communication mechanism, but it is not an absolute requirement if it proves unsuitable for Godot. Schola's Python side is designed to permit other communication backends. Choosing another protocol would, however, require corresponding Python-side implementation and would increase project scope.

### 3. Product and API design expectations

For this developer-facing product, the API and abstractions are the primary user interface. AMD prefers a smaller, well-designed, extensible core over broad but shallow feature coverage. The port should hide reinforcement-learning plumbing behind clear APIs and integrate naturally with Godot.

The design should retain enough conceptual similarity to Schola for Unreal that users can transfer their knowledge between engines. At the same time, Unreal-specific decisions—particularly those made for Blueprints or Unreal's object and interface restrictions—should not be copied when Godot offers a more appropriate mechanism.

The team should prototype important Godot abstractions early. This will help reveal engine-specific constraints before the full framework depends on an unsuitable interface or object model.

### 4. MVP workflow

AMD described the MVP as a high-quality, general-purpose core that can demonstrate a complete workflow with one deliberately simple environment:

1. A developer defines a simple Godot reinforcement-learning environment and agent through reusable APIs.
2. Observations, actions, rewards, terminal states, and resets are transmitted correctly to the existing Schola Python stack.
3. A policy is trained for a small number of steps and demonstrably learns the simple task.
4. The trained policy is exported as an ONNX model.
5. The model is loaded in Godot and used successfully for inference.

The suggested validation environment gives the agent a small reward for moving backward, a larger reward for moving forward, and a penalty for remaining still. The environment is intentionally simple: its purpose is to verify that the engine integration, training loop, model export, and inference path all work end to end.

The core must remain general-purpose rather than being hard-coded specifically for this example. Numerous specialized sensors or edge-case features are lower priority than clean foundational APIs.

### 5. Training and inference separation

AMD recommended treating training and inference as separable parts of the architecture. Training-only functionality, including communication dependencies such as gRPC, should be modular and easy to exclude from a shipped game. A deployed game should retain only the core types, necessary utilities, and inference functionality.

This separation also offers a possible initial division of work:

- One stream can investigate engine-to-Python communication and the training/stepping loop.
- Another can use an already-trained ONNX model to prototype loading and inference in Godot.

A model could initially be trained using Schola's existing Unreal example, exported, and then used to develop and validate the Godot inference path independently.

### 6. Long-term product direction

AMD intends Schola to evolve from an Unreal-only project into a maintained, open-source, multi-engine platform. Unity and Godot should not permanently be secondary ports; the longer-term aim is a shared user experience and transferable developer knowledge across supported game engines.

The desired experience is that users can use the same Python-facing entry point and choose the game engine hosting their environment. AMD expects successful student work to contribute toward the official open-source project and continue to be maintained.

### 7. Communication

Microsoft Teams will be the primary communication channel because AMD is more likely to see and respond to Teams messages promptly than email. Alex offered to create a group chat containing the AMD representatives and the student team.

The team should appoint one member as the partner liaison and formal point of contact. That person will coordinate meetings, deadlines, presentations, and other formal requests. Team members should post their own technical questions, bugs, and blockers directly in the shared Teams chat rather than routing them through the liaison.

The attendees agreed in principle to a recurring weekly partner meeting of approximately one hour. Once selected, the partner liaison will coordinate a mutually suitable time with Alex.

### 8. Repository access and confidentiality

The team asked whether the AMD representatives should be added to the CSC301 repository. Alex said they could be added and that access would be appreciated if the repository were private. He also stated that Schola development is conducted openly and that AMD did not identify anything in this project that needs to remain private. The team's CSC301 repository is public, consistent with AMD's preference for open development, so no additional repository access is required.

## Decisions and agreements

- The team will work on the **Godot port of AMD Schola**.
- The Unity wording in the published specification applies at a high level, with Unity replaced by Godot and implementation details adapted to Godot's native patterns.
- The main deliverable is the Godot engine-side integration; the existing Python stack should be reused with only necessary compatibility changes.
- The port should preserve Schola's high-level concepts while providing an idiomatic Godot API.
- A clean, extensible core and complete end-to-end workflow take priority over implementing many specialized features.
- Training dependencies and inference functionality must be modular and separable.
- Microsoft Teams will be the primary partner communication channel.
- The team will designate one partner liaison.
- The team and AMD will hold a recurring weekly meeting of approximately one hour, subject to scheduling.
- Technical questions should be asked directly in the shared Teams chat; the liaison is primarily responsible for formal coordination and scheduling.
- AMD has no confidentiality requirement for the project and is comfortable with open development. The team's CSC301 repository is public.

## Action items

| Owner | Action | Target date | Status |
| --- | --- | --- | --- |
| Alex | Create a Microsoft Teams group chat for the AMD representatives and student team. | After the meeting | Planned |
| Student team | Select and document the partner liaison. | As soon as possible | Open |
| Partner liaison | Contact Alex and arrange the recurring weekly one-hour partner meeting. | Before the next partner meeting | Open |
| Student team | Propose the initial Godot architecture, including the boundary between core, training, communication, and inference modules. | Before the next partner meeting | Open |
| Student team | Investigate Godot-native extension and API patterns and prototype any high-risk abstractions. | During initial planning | Open |
| Student team | Define MVP user stories and acceptance criteria around environment definition, training, ONNX export, and Godot inference. | Before the next partner meeting | Open |
| Student team | Create a plan for independently prototyping training communication and ONNX inference. | During initial planning | Open |
| Student team and AMD | Review the proposed architecture and MVP. | Next partner meeting | Open |

## Open questions for the next meeting

- Which Godot version and language/runtime combination should the project officially support?
- Which Godot extension mechanism best meets AMD's maintenance and distribution expectations?
- Which observations and action-space types are mandatory for the first milestone?
- Is gRPC expected for the first prototype, or should the team formally evaluate alternatives?
- What existing Schola interfaces and terminology must remain consistent across engines?
- What performance, headless-training, multi-environment, and multi-agent capabilities are required versus optional?
- What repository structure and upstream contribution process does AMD prefer for the Godot integration?
- What automated tests and documentation are required for AMD to accept and maintain the contribution?
- Which license and contribution arrangement should be recorded formally for D1, given AMD's preference for open development?
- What exact recurring meeting time works for both AMD and the student team?
