# US1–US5 Unity architecture (UML)

US1 (observations), US2 (actions), US3 (episodes), US4 (train through Schola), and US5 (run a trained policy) are implemented on `main`.

## Overview

The Unity package has three cooperating layers:

1. **Runtime agent and environment (US1–US3)** — Sensors collect observations, actuators apply actions, `ScholaAgent` aggregates them, and `ScholaEnvironment` runs episode step/reset with optional reward and status.
2. **Training connection (US4)** — Python hosts Schola’s existing `GymService` gRPC API. A TCP/JSON bridge talks to Unity’s `TrainingServer` → `TrainingSession` → `TrainingWire` → `IMultiAgentEnvironment` (`ScholaMultiAgentEnvironment`, or `ScholaEnvironment` via `SingleAgentEnvironmentAdapter`). Unity does **not** host `GymService` on the shipped path.
3. **Local inference (US5)** — `PolicyRunner` / `SentisPolicy` drive the same `IAgent.Observe` / `IAgent.Act` loop without rewards, episode flags, or Python.

Both training and inference share the agent’s observation and action path. Only training uses the environment’s reward, terminated/truncated, reset, and wire protocol.

```mermaid
flowchart LR
  subgraph shared ["Shared Runtime US1 to US3"]
    Agent["ScholaAgent"]
    Env["ScholaEnvironment"]
    Env --> Agent
  end

  subgraph train ["Training US4"]
    Py["Python GymService bridge"]
    Server["TrainingServer"]
    Py --> Server --> Env
  end

  subgraph infer ["Inference US5"]
    Policy["SentisPolicy"]
    Stepper["SimpleStepper"]
    Stepper --> Agent
    Stepper --> Policy
  end
```

## Contents

1. [Legend](#legend)
2. [High-level architecture](#1-high-level-architecture)
3. [Observations and actions (US1–US2)](#2-observations-and-actions-us1us2)
4. [Episodes and environment (US3)](#3-episodes-and-environment-us3)
5. [Training through Schola (US4)](#4-training-through-schola-us4)
6. [Local inference (US5)](#5-local-inference-us5)
7. [Training vs inference](#6-training-vs-inference)
8. [Unity vs Unreal](#7-unity-vs-unreal)
9. [Type inventory](#8-type-inventory)

---

## Legend

| Marker | Mermaid cue | Meaning |
| --- | --- | --- |
| implemented | Solid box, no extra stereotype | Type on `main` in Runtime, Training, Inference, or Samples |
| placeholder | `<<placeholder>>` | Contract on `main` with no Runtime framework class (`IReward`, `IEpisodeStatus`, `ITrainingStateSink`) |
| Unreal | `<<Unreal>>` | Comparison type only; not a Unity type |
| optional / future | `<<optional>>` | Direct Unity `GymService`, `Training/Generated/`, or `-ScholaPort=` smoke contract — tooling only, not the shipped path |

Dashed associations mean optional, missing, or unused by the shipped training connector.

---

## 1. High-level architecture

**What to take away:** Unity Runtime owns agents and episodes. Training reaches that Runtime through Python’s gRPC adapter and Unity’s TCP server. Inference stays entirely inside Unity.

```mermaid
flowchart TB
  subgraph pythonSide ["Python — implemented"]
    trainers["SB3 / RLlib"]
    gymEnv["GymEnv / VecEnv / RayEnv"]
    grpcClient["GrpcProtocol client"]
    gymService["UnityGymService gRPC host"]
    bridge["schola.unity.bridge TCP JSON"]
    trainers --> gymEnv --> grpcClient --> gymService --> bridge
  end

  subgraph unityTraining ["Unity Training US4 — implemented"]
    server["TrainingServer"]
    session["TrainingSession"]
    wire["TrainingWire"]
    server --> session --> wire
  end

  subgraph unityRuntime ["Unity Runtime US1 to US3 — implemented"]
    env["ScholaEnvironment"]
    agent["ScholaAgent"]
    sensors["ISensor / PositionSensor"]
    actuators["IActuator / TranslateActuator"]
    reward["IReward placeholder"]
    status["IEpisodeStatus placeholder"]
    sink["ITrainingStateSink placeholder"]
    env --> agent
    sensors --> agent
    agent --> actuators
    env -.-> reward
    env -.-> status
    env -.-> sink
  end

  subgraph unityInference ["Unity Inference US5 — implemented"]
    policyRunner["PolicyRunner"]
    runner["StepperRunner"]
    stepper["SimpleStepper"]
    policy["IPolicy / SentisPolicy"]
    policyRunner --> runner --> stepper
    policyRunner --> policy
    stepper --> policy
  end

  proto["Proto GymService contract"]
  bridge -->|"loopback TCP"| server
  wire --> env
  stepper --> agent
  gymService --- proto
  grpcClient --- proto
```

Shipped training entry: `schola unity train` starts `UnityGymService` (default gRPC port 8000) and a TCP client to `TrainingServer` (default port 8001). States return in wire responses. `ITrainingStateSink` is unused by the training connector. Native Unity gRPC under `Training/Generated/` remains optional tooling.

Sources: `Unity/com.amd.schola/Runtime/`, `Inference/`, `Training/`, `Training/README.md`, `Proto/GymConnector.proto`, `Resources/python/schola/unity/bridge.py`.

---

## 2. Observations and actions (US1–US2)

**What to take away:** Developers attach sensors and actuators to a `ScholaAgent`. The agent exposes Dict-shaped observation and action spaces/points. Default sensors/actuators are Box-only; Discrete and MultiBinary exist as data types for the wire and nesting under Dict.

### 2.1 Agent composition

`ScholaAgent` implements `IAgent` and stores **concrete** `List<PositionSensor>` / `List<TranslateActuator>` (exposed as `IReadOnlyList`), not `ISensor` / `IActuator`. Adding a new sensor or actuator type requires a code change on the agent.

```mermaid
classDiagram
  direction TB

  class ISensor {
    <<interface>>
    +InitSensor()
    +GetObservationSpace() Space
    +CollectObservations() Point
  }
  class PositionSensor {
    +ObservationKey string
  }
  class IActuator {
    <<interface>>
    +InitActuator()
    +GetActionSpace() Space
    +TakeAction(Point)
  }
  class TranslateActuator {
    +ActionKey string
  }
  class IAgent {
    <<interface>>
    +Observe() Point
    +Act(Point)
  }
  class ScholaAgent {
    +Sensors IReadOnlyList~PositionSensor~
    +Actuators IReadOnlyList~TranslateActuator~
    +Define() Space
    +DefineActionSpace() Space
    +Observe() Point
    +Act(Point)
  }

  PositionSensor ..|> ISensor
  TranslateActuator ..|> IActuator
  ScholaAgent ..|> IAgent
  ScholaAgent "1" *-- "*" PositionSensor : Sensors
  ScholaAgent "1" *-- "*" TranslateActuator : Actuators
  PositionSensor ..> BoxSpace : observation space
  PositionSensor ..> BoxPoint : observations
  TranslateActuator ..> BoxSpace : action space
  TranslateActuator ..> BoxPoint : actions
  ScholaAgent ..> DictSpace : Define spaces
  ScholaAgent ..> DictPoint : Observe and Act
```

### 2.2 Spaces and points

Spaces describe legal shapes; points carry values. Dict types nest named child spaces/points. Training wire kinds: `"dict"` | `"box"` | `"discrete"` | `"multi_binary"`.

```mermaid
classDiagram
  direction TB

  class Space {
    <<abstract>>
  }
  class BoxSpace {
    +Dimensions List~BoxSpaceDimension~
    +Shape List~int~
  }
  class BoxSpaceDimension {
    +Low floatNullable
    +High floatNullable
  }
  class DictSpace {
    +Spaces Dict~string Space~
  }
  class DiscreteSpace {
    +High int
  }
  class MultiBinarySpace {
    +Shape int
  }

  class Point {
    <<abstract>>
  }
  class BoxPoint {
    +Values List~float~
    +Shape List~int~
  }
  class DictPoint {
    +Points Dict~string Point~
  }
  class DiscretePoint {
    +Value int
  }
  class MultiBinaryPoint {
    +Values List~bool~
  }

  Space <|-- BoxSpace
  Space <|-- DictSpace
  Space <|-- DiscreteSpace
  Space <|-- MultiBinarySpace
  BoxSpace *-- BoxSpaceDimension
  DictSpace o-- Space

  Point <|-- BoxPoint
  Point <|-- DictPoint
  Point <|-- DiscretePoint
  Point <|-- MultiBinaryPoint
  DictPoint o-- Point
```

### 2.3 Behavior and limits

- `Define()` builds a `DictSpace` keyed by `PositionSensor.ObservationKey` (default `"position"`).
- `Observe()` builds a matching `DictPoint` of 3-float world `BoxPoint`s.
- `DefineActionSpace()` builds a `DictSpace` keyed by `TranslateActuator.ActionKey` (default `"translate"`). Duplicate keys warn and last-write-wins.
- `Act(Point)` requires a `DictPoint`, checks **every** actuator key before any `TakeAction`, then applies 3-float `BoxPoint` deltas to `transform.position`. Wrong type or arity throws.
- `MultiBinarySpace` / `MultiBinaryPoint` mirror Gymnasium `MultiBinary(n)` (`Shape`, `List<bool> Values`). They nest under Dict and round-trip on the training wire (`shape: [n]`, `boolValues`). There is no dedicated MultiBinary sensor or actuator; default agents stay Box-only. `SentisPolicy` does not support MultiBinary.
- Proto `MultiDiscrete` and `Text` remain unimplemented in Unity.
- No `Space.Contains` / bound enforcement on Runtime types; training validation lives in `TrainingWire.Validate`.
- Unity `IAgent` is Observe/Act only. It has no `Define` / `GetStatus` / `SetStatus` (those exist on Unreal `IAgent`; see [Unity vs Unreal](#7-unity-vs-unreal)). `ScholaAgent.Define` is a concrete method, not part of `IAgent`.

Sources: `Runtime/Sensors/`, `Runtime/Actuators/`, `Runtime/Agent/`, `Runtime/Points/`, `Runtime/Spaces/`, `Training/TrainingWire.cs`.

---

## 3. Episodes and environment (US3)

**What to take away:** One `ScholaEnvironment` owns one `ScholaAgent`. Callers drive `Initialize`, `Step`, and `Reset`. Reward, episode status, and the training-state sink are optional contracts; the shipped training path does not use the sink.

### 3.1 Environment types

`IEnvironment.Step` takes one `Point` and returns one `AgentState` with no agent-id field. `IReward`, `IEpisodeStatus`, and `ITrainingStateSink` have no Runtime framework class; tests supply fakes. Sample `TrainingDemoTask` (`Samples~/`) implements `IReward` and `IEpisodeStatus`.

```mermaid
classDiagram
  direction TB

  class IEnvironment {
    <<interface>>
    +Initialize() InteractionDefinition
    +Seed(int)
    +SetOptions(IReadOnlyDictionary)
    +Reset() InitialAgentState
    +Step(Point) AgentState
  }
  class ScholaEnvironment {
    +DefaultAgentId SingleAgent
    +SetAgent(ScholaAgent)
    +SetReward(IReward)
    +SetEpisodeStatus(IEpisodeStatus)
    +SetTrainingStateSink(ITrainingStateSink)
  }
  class InteractionDefinition {
    +ObservationSpace Space
    +ActionSpace Space
  }
  class AgentState {
    +Observations Point
    +Reward float
    +Terminated bool
    +Truncated bool
    +Info Dict~string string~
  }
  class InitialAgentState {
    +Observations Point
    +Info Dict~string string~
  }
  class IReward {
    <<interface>>
    <<placeholder>>
    +OnEpisodeReset()
    +ComputeReward() float
  }
  class IEpisodeStatus {
    <<interface>>
    <<placeholder>>
    +OnEpisodeReset()
    +GetStatus(out terminated out truncated)
  }
  class ITrainingStateSink {
    <<interface>>
    <<placeholder>>
    +SubmitState(AgentState)
    +SubmitInitialState(InitialAgentState)
    +SubmitStates(map)
    +SubmitInitialStates(map)
  }
  class ScholaAgent

  ScholaEnvironment ..|> IEnvironment
  ScholaEnvironment "1" *-- "1" ScholaAgent
  ScholaEnvironment ..> InteractionDefinition : Initialize
  ScholaEnvironment ..> AgentState : Step
  ScholaEnvironment ..> InitialAgentState : Reset
  ScholaEnvironment ..> IReward : optional
  ScholaEnvironment ..> IEpisodeStatus : optional
  ScholaEnvironment ..> ITrainingStateSink : optional unused by TrainingSession
```

`IEnvironment` mirrors Unreal `ISingleAgentScholaEnvironment` with C# return values instead of out-params.

### 3.2 Step flow

Order inside `Step`: Act → reward → status → Observe → optional sink. The environment does **not** auto-reset; `TrainingSession` may reset around this call.

```mermaid
sequenceDiagram
  actor Caller
  participant Env as ScholaEnvironment
  participant Agent as ScholaAgent
  participant Actuator as TranslateActuator
  participant Reward as IReward
  participant Status as IEpisodeStatus
  participant Sensor as PositionSensor
  participant Sink as ITrainingStateSink

  Caller ->> Env: Step with DictPoint action
  Env ->> Agent: Act
  Agent ->> Agent: require every ActionKey
  Agent ->> Actuator: TakeAction BoxPoint
  Actuator ->> Actuator: apply position delta

  alt IReward bound
    Env ->> Reward: ComputeReward
  else unbound
    Env ->> Env: reward equals 0
  end

  alt IEpisodeStatus bound
    Env ->> Status: GetStatus
  else unbound
    Env ->> Env: both flags false
  end

  Env ->> Agent: Observe
  Agent ->> Sensor: CollectObservations
  Sensor -->> Agent: BoxPoint xyz
  Agent -->> Env: DictPoint

  opt ITrainingStateSink bound
    Env ->> Sink: SubmitState
  end

  Env -->> Caller: AgentState
```

### 3.3 Reset flow

```mermaid
sequenceDiagram
  actor Caller
  participant Env as ScholaEnvironment
  participant Reward as IReward
  participant Status as IEpisodeStatus
  participant Agent as ScholaAgent
  participant Sink as ITrainingStateSink

  Caller ->> Env: Reset
  Env ->> Env: RestorePoses
  opt IReward bound
    Env ->> Reward: OnEpisodeReset
  end
  opt IEpisodeStatus bound
    Env ->> Status: OnEpisodeReset
  end
  Env ->> Agent: Observe
  opt ITrainingStateSink bound
    Env ->> Sink: SubmitInitialState
  end
  Env -->> Caller: InitialAgentState
```

### 3.4 Behavior details

- `Initialize()` inits sensors/actuators, snapshots local poses of the agent hierarchy plus configured actuator transforms, and returns `InteractionDefinition(agent.Define(), agent.DefineActionSpace())`.
- `Reset()` calls `Initialize()` if not yet initialized, restores the snapshot, calls `OnEpisodeReset` on reward/status, observes, and returns `InitialAgentState` (obs + info only). No reward or done flags on reset.
- `Seed` and `SetOptions` are no-ops. `AgentState.Info` is allocated empty and never populated.
- `DefaultAgentId = "SingleAgent"` is the key `SingleAgentEnvironmentAdapter` injects when a single-agent environment is viewed as a multi-agent map. `ScholaEnvironment` itself still steps one agent and does not implement `IMultiAgentEnvironment`. `TrainingSession` speaks only `IMultiAgentEnvironment`; the `IEnvironment` constructor wraps through that adapter. The Python bridge forwards the agent ids Unity declares (env index `0`) and no longer injects `"SingleAgent"` itself.
- Reward and status may be assigned in the inspector (`GetComponentInChildren`) or via setters. The sink is **not** auto-resolved; it must be `SetTrainingStateSink`. The shipped training connector does not set or use the sink.

Sources: `Runtime/Environment/`, `Runtime/TrainingDataTypes/`, `Runtime/Common/InteractionDefinition.cs`, `Samples~/TrainingDemo/Runtime/TrainingDemoTask.cs`.

---

## 4. Training through Schola (US4)

**What to take away:** Trainers speak the usual Schola `GymService` RPCs. Python hosts that gRPC service and translates to Unity TCP/JSON. Unity runs `TrainingServer` on the main thread against `IMultiAgentEnvironment`.

### 4.1 Shipped connection

```mermaid
flowchart LR
  subgraph pythonSide ["Python"]
    cli["schola unity train"]
    gymEnv["GymEnv / VecEnv / RayEnv"]
    grpcClient["GrpcProtocol"]
    gymService["UnityGymService"]
    bridge["TCP JSON client"]
    cli --> gymEnv --> grpcClient --> gymService --> bridge
  end

  subgraph unitySide ["Unity"]
    server["TrainingServer"]
    session["TrainingSession"]
    wire["TrainingWire"]
    multi["IMultiAgentEnvironment"]
    server --> session
    session --> wire
    session --> multi
  end

  bridge -->|"commands"| server
```

Roles:

1. **`TrainingServer`** — MonoBehaviour; loopback TCP (default port 8001); one request per `Update` on the main thread. Serves either `ScholaEnvironment` or `ScholaMultiAgentEnvironment`.
2. **`TrainingSession`** — Commands `start`, `definition`, `reset`, `step`, `close` against `IMultiAgentEnvironment`. Autoreset: `0=SAME_STEP`, `1=NEXT_STEP`, `2=DISABLED`. The environment is done only when every agent is done.
3. **`TrainingWire`** — Version 2 JSON encode/decode/validate, with per-agent definition, action and state lists. Kinds: `"dict"` | `"box"` | `"discrete"` | `"multi_binary"` (`shape: [n]`, point `boolValues`).
4. **`schola.unity.bridge`** — Hosts `GymService` gRPC (default port 8000); maps proto ↔ TCP JSON using the agent ids Unity declared.

Python `UpdateState` still carries `Reset` or `Step`. The bridge maps those to Unity `reset` / `step`. Auto-reset is a `TrainingSession` concern; environments do not auto-reset. States return in the JSON response; `ITrainingStateSink` is unused.

Proto RPCs still used by trainers: `StartGymConnector`, `RequestTrainingDefinition`, `UpdateState`.

### 4.2 Protocol cardinality

**What to take away:** Proto and Unreal nest many envs and many agents. Unity’s shipped training stack is one env (index `0`) with one or many agents. `TrainingSession` drives `IMultiAgentEnvironment`. `ScholaMultiAgentEnvironment` returns `map<string, AgentState>` (and the matching definition and initial-state maps). `SingleAgentEnvironmentAdapter` injects the key `SingleAgent`, the same key Unreal's single-agent adapter uses, so a `ScholaEnvironment` stays one entry on that path. Python `GymEnv` still assumes 1 env, 1 agent. `GymVectorEnv` / SB3 flatten `(env, agent)` slots (shared spaces, finish together); `RayEnv` uses true multi-agent dicts.

Proto message nesting (identity lives in maps and repeated env slots):

```mermaid
classDiagram
  direction TB

  class ProtoTrainingState {
    <<protocol>>
    +environment_states EnvironmentState[]
  }
  class ProtoEnvironmentState {
    <<protocol>>
    +agent_states map~string AgentState~
  }
  class ProtoAgentState {
    <<protocol>>
    +observations Point
    +reward float
    +terminated bool
    +truncated bool
    +info map~string string~
  }
  class ProtoTrainingDefinition {
    <<protocol>>
    +environment_definitions EnvironmentDefinition[]
  }
  class ProtoEnvironmentDefinition {
    <<protocol>>
    +agent_definitions map~string AgentDefinition~
  }
  class ProtoStateUpdate {
    <<protocol>>
    +reset Reset
    +step Step
  }
  class ProtoStep {
    <<protocol>>
    +environments EnvironmentStep[]
  }
  class ProtoEnvironmentStep {
    <<protocol>>
    +updates map~string Point~
  }

  ProtoTrainingState *-- ProtoEnvironmentState : env index
  ProtoEnvironmentState *-- ProtoAgentState : agent_id
  ProtoTrainingDefinition *-- ProtoEnvironmentDefinition : env index
  ProtoStateUpdate *-- ProtoStep
  ProtoStep *-- ProtoEnvironmentStep : env index
  ProtoEnvironmentStep --> ProtoAgentState : agent_id keys actions
```

How Unity fits that nesting:

```mermaid
flowchart LR
  bridge["Python bridge forwards agent ids and env 0"]
  session["TrainingSession one env"]
  multiIface["IMultiAgentEnvironment"]
  state["AgentState has no agent-id field"]
  adapter["SingleAgentEnvironmentAdapter injects SingleAgent"]
  single["ScholaEnvironment one agent"]
  multi["ScholaMultiAgentEnvironment many agents by id"]
  bridge --> session --> multiIface --> state
  adapter --> multiIface
  adapter --> single
  multi --> multiIface
```

`IEnvironment` stays one agent. `SingleAgentEnvironmentAdapter` is what injects the key `SingleAgent` onto the `TrainingSession` path. `ScholaMultiAgentEnvironment` is the many-agent environment on that same path.

| Layer | Environments | Agents per env | How identity is keyed |
| --- | --- | --- | --- |
| Proto `TrainingState` / `TrainingDefinition` | Many (`repeated`) | Many (`map<string, …>`) | env index, then `agent_id` |
| Unreal `IScholaEnvironment` | Connector holds `TArray` of envs | Many (`TMap<FString, …>`) | `agent_id` string |
| Unreal `ISingleAgentScholaEnvironment` | One | One, adapter injects `"SingleAgent"` | injected key |
| Unity `ScholaEnvironment` | One | One | none on the C# state; adapter injects `"SingleAgent"` |
| Unity `TrainingSession` (US4) | One | One or many | JSON agent lists; bridge maps to proto keys |
| Unity `ScholaMultiAgentEnvironment` | One | Many | `agent_id` string |
| Unity `SingleAgentEnvironmentAdapter` | One | One | injected key `SingleAgent` |
| Python `GymEnv` | One | One | 1:1 wrap |
| Python `GymVectorEnv` / `sb3.VecEnv` | Many slots | Flattened `(env, agent)`; agents in one env finish together | slot index |
| Python `RayEnv` | Many | True multi-agent dicts | `agent_id` |
| Unity `SimpleStepper` (US5) | N/A | Many `IAgent`s, one `IPolicy` | list index, no episode |

No competitive/cooperative configs exist in Unity.

### 4.3 Optional direct gRPC (`<<optional>>`)

`Training/README.md` documents generating C# from `Proto/` into `Training/Generated/` and a future `-ScholaPort=` player smoke contract that would host `GymService` inside Unity (matching Unreal). That path is **not required** by the shipped TCP-plus-adapter implementation and has no committed generated baseline.

### 4.4 Historical notes

Current tree is the authority. Context only:

1. **`origin/US4`** — native gRPC spike: generated proto C# plus `GymStartupServer` implementing `StartGymConnector` only. Step/reset unimplemented.
2. **PR #17 / revert PR #18** — early TCP + JSON bridge was reverted, then restored and completed as the shipped `TrainingServer` / `TrainingSession` / `TrainingWire` + `schola.unity.bridge` path. It calls `IEnvironment` directly and does not implement `ITrainingStateSink`.

Sources: `Proto/GymConnector.proto`, `Proto/State.proto`, `Proto/Definitions.proto`, `Proto/StateUpdates.proto`, `Training/README.md`, `Training/TrainingServer.cs`, `TrainingSession.cs`, `TrainingWire.cs`, `Resources/python/schola/unity/bridge.py`, `Resources/python/schola/gym/env.py`.

---

## 5. Local inference (US5)

**What to take away:** Load an ONNX model in Unity and drive agents without Python. Same Observe/Act path as training; no reward, terminated/truncated, reset, or training sink.

### 5.1 Components

```mermaid
classDiagram
  direction TB

  class PolicyRunner {
    +Agents IReadOnlyList~ScholaAgent~
    +Model ModelAsset
    +StartRunning()
  }
  class StepperRunner {
    +IsRunning bool
  }
  class IStepper {
    <<interface>>
    +Step()
  }
  class SimpleStepper {
  }
  class IPolicy {
    <<interface>>
    +Init(InteractionDefinition)
    +Think(obs out action)
  }
  class SentisPolicy {
  }
  class IAgent {
    <<interface>>
    +Observe() Point
    +Act(Point)
  }

  PolicyRunner --> StepperRunner : requires
  PolicyRunner ..> SentisPolicy : builds from ModelAsset
  PolicyRunner ..> SimpleStepper : wires
  SimpleStepper ..|> IStepper
  StepperRunner ..> IStepper : drives each frame interval
  SimpleStepper ..> IPolicy : Think
  SimpleStepper ..> IAgent : Observe and Act
  SentisPolicy ..|> IPolicy
```

`PolicyRunner` is the inspector entry: assign agents and a `ModelAsset`, enter Play Mode. It builds `SentisPolicy`, wraps agents in `SimpleStepper`, and starts `StepperRunner`.

`SentisPolicy` supports Dict spaces with **Box** observations and **Box or Discrete** actions. It does **not** support MultiBinary. `SimpleStepper` can drive many `IAgent`s with one shared `IPolicy` (inference-only multi-agent by list index, not a training multi-agent environment).

### 5.2 Inference step flow

```mermaid
sequenceDiagram
  participant PolicyRunner as PolicyRunner
  participant Runner as StepperRunner
  participant Stepper as SimpleStepper
  participant Agent as IAgent
  participant Policy as SentisPolicy

  Note over PolicyRunner,Policy: Start wires SentisPolicy and SimpleStepper onto StepperRunner
  Runner ->> Stepper: Step
  loop each living IAgent by list index
    Stepper ->> Agent: Observe
  end
  loop each observation
    Stepper ->> Policy: Think obs to action
  end
  loop each agent
    Stepper ->> Agent: Act
  end
  Stepper -->> Runner: success or fail
  Note over Stepper,Policy: No reward, episode flags, Reset, or sink. Agents by list index.
```

Sources: `Inference/PolicyRunner.cs`, `Inference/SentisPolicy.cs`, `Runtime/Steppers/SimpleStepper.cs`, `Runtime/Steppers/StepperRunner.cs`, `Runtime/Policies/IPolicy.cs`.

---

## 6. Training vs inference

Both loops share `IAgent.Observe` / `IAgent.Act`. Only the environment/training path has reward, episode flags, reset, and the wire protocol.

| Concern | Training (US3 + US4) | Inference (US5) |
| --- | --- | --- |
| Entry | `TrainingServer` / `TrainingSession` or direct `ScholaEnvironment` | `PolicyRunner` / `StepperRunner` |
| Action source | Python trainer via bridge | `SentisPolicy.Think` |
| Reward / terminated / truncated | Yes, via optional `IReward` / `IEpisodeStatus` | No |
| Reset | `ScholaEnvironment.Reset`; autoreset in `TrainingSession` | No |
| Agent identity | Unity-declared agent ids (adapter injects `"SingleAgent"`) | List index |
| Multi-agent | `ScholaMultiAgentEnvironment` on the training path | Inference-only many agents, one policy |
| MultiBinary on wire | Supported in `TrainingWire` + bridge | Not supported in `SentisPolicy` |

---

## 7. Unity vs Unreal

**What to take away:** Unity mirrors the single-agent environment idea but splits reward/status into interfaces, keeps `IAgent` minimal, and ships TCP/JSON instead of in-process `GymService`.

```mermaid
classDiagram
  direction LR

  class UnityIAgent {
    +Observe() Point
    +Act(Point)
  }
  class UnrealIAgent {
    <<Unreal>>
    +Define()
    +GetStatus() EAgentStatus
    +SetStatus(EAgentStatus)
    +Observe()
    +Act()
  }

  class UnityIEnvironment {
    +Step(Point) AgentState
  }
  class UnrealISingleAgent {
    <<Unreal>>
    +InitializeEnvironment()
    +Step()
    +Reset()
  }
  class UnrealIScholaEnvironment {
    <<Unreal>>
    +Step() AgentStateMap
  }
  class UnrealIMultiAgent {
    <<Unreal>>
    +Step() AgentStateMap
  }

  class UnityIReward {
    <<placeholder>>
  }
  class UnityIEpisodeStatus {
    <<placeholder>>
  }
  class UnityITrainingStateSink {
    <<placeholder>>
  }
  class URPCGymConnector {
    <<Unreal>>
  }
  class TrainingServer
  class schola_unity_bridge {
    <<Python module>>
    +hosts GymService
    +TCP JSON to Unity
  }

  UnrealISingleAgent ..|> UnrealIScholaEnvironment : adapter key SingleAgent
  UnrealIMultiAgent ..|> UnrealIScholaEnvironment
  URPCGymConnector ..> UnrealIScholaEnvironment : TArray of envs
  TrainingServer ..> UnityIEnvironment : via SingleAgent adapter
  schola_unity_bridge ..> TrainingServer : bridge
  UnityIEnvironment ..> UnityIReward
  UnityIEnvironment ..> UnityIEpisodeStatus
```

Differences that matter for Unity training:

- Unreal `IAgent` also has `Define`, `GetStatus`, `SetStatus`. Unity `IAgent` is Observe/Act only; `ScholaAgent.Define` is concrete.
- Unreal environments fill `FAgentState.Reward` / `bTerminated` / `bTruncated` inside `Step`. They have no `IReward` / `IEpisodeStatus`.
- Unity `IMultiAgentEnvironment` matches Unreal `IMultiAgentScholaEnvironment`: maps keyed by agent id, with reward and episode flags filled per agent inside `Step`. `TrainingSession` drives that multi-agent interface; a single-agent environment is wrapped by `SingleAgentEnvironmentAdapter`.
- Unreal hosts `GymService` via `URPCGymConnector` / `UCommunicationManager`. Unity hosts TCP JSON; Python `schola.unity.bridge` hosts `GymService`.

Sources: `Source/Schola/Public/Agent/AgentInterface.h`, `Source/ScholaTraining/Public/Environment/`, `Source/ScholaProtobuf/Public/GymConnectors/gRPC/gRPCGymConnector.h`.

---

## 8. Type inventory

| Type | Story | Status | Notes |
| --- | --- | --- | --- |
| `ISensor` | US1 | implemented | `InitSensor`, `GetObservationSpace`, `CollectObservations` |
| `PositionSensor` | US1 | implemented | 3-float world `BoxPoint`, key `"position"` |
| `IActuator` | US2 | implemented | `InitActuator`, `GetActionSpace`, `TakeAction` |
| `TranslateActuator` | US2 | implemented | 3-float `BoxPoint` delta to `transform.position` |
| `IAgent` | US1–US2 | implemented | Observe / Act only |
| `ScholaAgent` | US1–US2 | implemented | Concrete sensor/actuator lists (public `IReadOnlyList`) |
| `Point` / `BoxPoint` / `DictPoint` / `DiscretePoint` | US1–US2 | implemented | No MultiDiscrete or Text |
| `MultiBinaryPoint` | US1–US4 | implemented | `List<bool> Values`; training wire + bridge; no Sentis |
| `Space` / `BoxSpace` / `DictSpace` / `DiscreteSpace` | US1–US2 | implemented | No `Contains` / bound enforcement on Runtime types |
| `MultiBinarySpace` | US1–US4 | implemented | `Shape` = Gymnasium `MultiBinary(n)`; wire kind `multi_binary` |
| `IEnvironment` | US3 | implemented | Mirrors Unreal `ISingleAgentScholaEnvironment` |
| `ScholaEnvironment` | US3 | implemented | One `ScholaAgent`; `Seed` / `SetOptions` no-ops |
| `IReward` | US3 | placeholder | No Runtime class; test fakes; sample `TrainingDemoTask` |
| `IEpisodeStatus` | US3 | placeholder | Independent terminated / truncated; sample `TrainingDemoTask` |
| `AgentState` / `InitialAgentState` | US3 | implemented | Hand-written proto mirrors; no agent-id field |
| `InteractionDefinition` | US3 | implemented | ObservationSpace + ActionSpace |
| `ITrainingStateSink` | US4 | placeholder | Singular submit plus `SubmitStates` / `SubmitInitialStates` for agent-id maps; unused by the training connector |
| `TrainingServer` | US4 | implemented | Loopback TCP; one request per `Update` |
| `TrainingSession` | US4 | implemented | Speaks `IMultiAgentEnvironment`; autoreset when every agent is done |
| `TrainingWire` | US4 | implemented | Version 2 per-agent lists; JSON encode/decode/validate including `multi_binary` |
| Python `schola.unity.bridge` | US4 | implemented | Hosts `GymService`; forwards Unity agent ids; TCP JSON ↔ proto |
| `IPolicy` / `SentisPolicy` | US5 | implemented | Local ONNX; Box obs; Box/Discrete actions; no MultiBinary |
| `PolicyRunner` | US5 | implemented | Inspector entry: agents + `ModelAsset` → Sentis + stepper |
| `IStepper` / `SimpleStepper` / `StepperRunner` | US5 | implemented | Observe-think-act; no reward or reset |
| Unity-hosted `GymService` | US4 | optional | Direct gRPC / `Training/Generated/` / `-ScholaPort=`; not shipped |
| `IMultiAgentEnvironment` | multi-agent | implemented | Mirrors Unreal `IMultiAgentScholaEnvironment`; finished agents keep terminal state until `Reset` |
| `ScholaMultiAgentEnvironment` | multi-agent | implemented | One environment, many `ScholaAgent`s keyed by id. On the `TrainingSession` path |
| `SingleAgentEnvironmentAdapter` | multi-agent | implemented | Wraps `IEnvironment` under the key `SingleAgent` for `TrainingSession` |
| `IMultiAgentScholaEnvironment` | Unreal | Unreal | Comparison type. Unity equivalent is `IMultiAgentEnvironment` |
