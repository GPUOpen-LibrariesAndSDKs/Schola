# AMD Schola Godot MVP User Stories

These stories describe the minimum end-to-end product agreed upon during the initial AMD partner meeting: a Godot developer can define a simple reinforcement-learning environment, train an agent through Schola's existing Python ecosystem, export the learned policy, and run that policy inside Godot without Python.

Implementation tasks, ownership, dependencies, and progress are tracked on the team's [Trello board](https://trello.com/b/Ry0Qkx2R). This document states the user value and acceptance boundary; Trello breaks each story into engineering tasks.

## US1: Define a reinforcement-learning environment

**Story:** As a Godot developer, I want to define a reinforcement-learning environment through a small engine-independent interface so that I can make an environment trainable without writing networking code.

**Acceptance criteria:**

- A developer can implement or configure the environment's initialization, reset, observation, reward, and terminal-state behavior.
- The environment can contain at least one agent.
- Environment code does not directly manage sockets, RPC calls, or serialized protocol messages.
- The environment accepts a reproducible random seed and optional reset configuration.
- Invalid or incomplete environment configuration produces a clear error.

## US2: Declare observation and action spaces

**Story:** As a Godot developer, I want to define observation and action spaces using reusable types so that Schola can validate and communicate the agent's available inputs and outputs.

**Acceptance criteria:**

- The core supports Box, Discrete, MultiDiscrete, and MultiBinary spaces and their corresponding point values.
- Spaces expose their shape, bounds, and data type where applicable.
- An omitted Box bound represents an unbounded dimension rather than zero.
- An observation or action that does not match its declared space is rejected with a useful error.
- Space definitions can be translated to the representation expected by the existing Schola Python package.
- Round-trip tests serialize and deserialize spaces, points, interaction definitions, and agent states without changing their values.

## US3: Connect to existing Python training tools

**Story:** As an ML practitioner, I want a Godot environment to connect to Schola's existing Python training tools so that I can train policies without maintaining a separate Godot-specific Python workflow.

**Acceptance criteria:**

- The Godot integration completes the connection and environment-definition exchange with the existing Python client.
- The integration uses Schola's existing protocol and gRPC services unless an alternative is approved by AMD.
- The transport implements `StartGymConnector`, `RequestTrainingDefinition`, and `UpdateState` from the existing Gym connector protocol.
- Python can discover the available environment, agents, observation spaces, and action spaces.
- The transport is isolated behind a core interface so the engine-independent code does not depend directly on gRPC.
- Connection failures and incompatible protocol data produce actionable errors instead of hanging the game or training process.

## US4: Execute the episode lifecycle

**Story:** As an ML practitioner, I want Schola to coordinate observations, actions, rewards, terminal states, and resets so that training proceeds correctly across complete episodes.

**Acceptance criteria:**

- For each step, Godot supplies an observation and accepts a compatible action from Python.
- Each step returns the resulting observation, reward, and termination or truncation state.
- Reset restores the demonstration environment to a valid initial state and returns an initial observation.
- The connector supports the existing disabled, same-step, and next-step auto-reset modes with the same externally observable behavior as Schola's Python API.
- The connector can coordinate more than one environment in a running scene.
- An integration test completes multiple episodes without lifecycle deadlock or state leakage between episodes.

## US5: Configure Schola through Godot-native tools

**Story:** As a Godot developer, I want to configure agents and environments through nodes and the Inspector so that I can use familiar Godot workflows instead of editing protocol or networking code.

**Acceptance criteria:**

- A developer can add the required Schola components to a scene using Godot's normal node workflow.
- Essential settings are visible and editable in the Inspector with understandable names and defaults.
- The demonstration project can be configured without modifying Schola's internal source code.
- The demonstration agent receives a small reward for moving backward, a larger reward for moving forward, and a penalty for remaining still.
- A configurable maximum step count truncates an episode that does not otherwise terminate.
- Running the scene reports missing or conflicting configuration clearly.

## US6: Export a trained policy to ONNX

**Story:** As a Godot developer, I want to export a trained policy to ONNX so that I can transfer the learned policy from the Python training process into Godot.

**Acceptance criteria:**

- A policy trained with Stable-Baselines3 can be exported through Schola's existing Python export workflow.
- The resulting file is a valid ONNX model that can be opened by an independent model-inspection tool.
- The model's input and output names, shapes, and data types are documented for the inference implementation.
- A policy trained for the Godot demonstration environment is exported with inputs and outputs matching that environment's declared spaces.

## US7: Run and ship an ONNX policy

**Story:** As a Godot developer, I want a trained policy to drive my agent with Python closed and to exclude training-only dependencies from exported games so that I can ship an autonomous agent without unnecessary training infrastructure.

**Acceptance criteria:**

- Godot loads the exported ONNX model and validates that its inputs and outputs match the agent's declared spaces.
- On each physics step, the agent follows an observe-infer-act loop that applies model output as its action.
- The demonstration agent performs the intended learned behavior while Python is not running.
- Missing, invalid, or incompatible model files produce a clear error.
- Training and transport code is packaged separately from the core environment API and inference code.
- A Godot export containing the demonstration environment runs its trained policy without Python or a live gRPC connection.
- The exported build excludes the training add-on without preventing the project from loading or using inference.
- The documentation identifies which modules are required for training and which are required in a shipped game.

## Partner review

The team will send this artifact and the accompanying architecture to AMD through the shared Microsoft Teams channel. Evidence of that communication and any requested revisions will be linked here after the review.
