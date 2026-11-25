# Advanced NPCs for AMD Schola

[Schola](README-Schola.md) is AMD's Unreal Engine plugin that connects game environments and agents to
reinforcement learning (RL). Out of the box, it trains policies in Python (Stable Baselines3, RLlib) over
gRPC and runs trained ONNX policies inside Unreal through the Neural Network Engine (NNE), one agent
decision at a time.

Game developers asked for NPCs that this model does not cover well. Some NPCs should keep learning from
the player after the game ships. Some crowds have thousands of agents that cannot each run a model on the
game thread. Some NPC teams should share one network across specialized roles. Some NPCs should decide
only when a game event happens. This fork adds those NPC types to Schola as reusable C++/Blueprint
modules, with examples, tests and measurements.

| Already in Schola | Built by this project |
| --- | --- |
| Python training over gRPC (SB3, RLlib), ONNX export | **Adaptive NPCs:** in-game SAC/TD3/PPO learning in C++ (RLtools), no Python |
| `IAgent`, `IPolicy`, `IStepper`; synchronous `Think` per agent | **Background NPCs:** batched, non-blocking inference for Actors and Mass entities |
| NNE inference of one ONNX policy (`UNNEPolicy`) | **Shared Policy NPCs:** one network, role-specific output heads, NNE benchmarks |
| Spaces, points, sensors, StateTree helpers | **Event NPCs:** inference triggered by gameplay events (in progress) |

## Contents

- [Features and status](#features-and-status)
- [Core workflows](#core-workflows)
- [Architecture](#architecture)
- [Install and access](#install-and-access)
- [Testing and CI](#testing-and-ci)
- [Non-functional requirements](#non-functional-requirements)
- [Known limitations](#known-limitations)
- [Maintaining this project](#maintaining-this-project)
- [Deliverables and contributions](#deliverables-and-contributions)

## Features and status

| Feature | What a developer gets | Where | Verified on |
| --- | --- | --- | --- |
| Adaptive NPCs | `URLToolsNPCPolicy` learns during play with SAC, TD3 or PPO; `UAdaptiveStepper` runs the observe–act–learn loop; configuration assets for custom shapes; checkpoints; snapshots | `Source/ScholaRLTools`, `Source/ScholaInferenceUtils` | UE 5.8.4 / Windows (policy and world service); macOS and Linux (native learner) |
| Background NPCs | A bounded batching scheduler that runs ONNX policies on worker threads for many Actors (`UBackgroundStepper`) or Mass entities (`ScholaMass`) | `Source/ScholaInferenceUtils/.../Inference`, `Source/ScholaMass`, `Source/ScholaNNE` | UE 5.8.3 / Linux |
| Shared Policy NPCs | One batched call for several NPCs, routed to each NPC's role head; five ONNX architectures (A–E) benchmarked for correctness and latency | `Source/ScholaNNE/.../NNESharedPolicy*`, `SharedPolicyStepper`, `Prototypes/SharedPolicy` | UE 5.8.4 / macOS; Python 3.12 |
| Event NPCs | `UEventStepper`: inference only when gameplay triggers it | Branch `feature/event` (not merged) | Blueprint prototype |

## Core workflows

### Adaptive NPC: learn from the player during play

**User story:** as a game developer, I want NPC policies to learn from player interactions during
gameplay, so that NPCs adapt their strategies to the player's behavior.

1. **Describe the task.** The NPC implements `IAgent` (observe, act) and `IAdaptiveAgent` (reward since
   the last decision, episode ended, reset). Both can be implemented in Blueprint.
2. **Choose a learner.** Create a `URLToolsNPCPolicy` and pick SAC, TD3 or PPO plus a compiled
   configuration, or a configuration asset for your own observation/action sizes (generate it, then
   rebuild). Set `SharedGroup` if several NPCs should train one learner.
3. **Run the loop.** `UAdaptiveStepper::Init(Agents, Policy)` binds one policy copy per NPC.
   Call `Step()` every decision tick. Training runs on worker threads and never blocks the frame.
4. **Observe and control.** Read learning statistics, freeze learning, switch to deterministic actions,
   capture a frozen snapshot to compare against, and save or load checkpoints.

Guides: [adaptive stepper](Docs/Sphinx/guides/adaptive_stepper.rst),
[native adaptive policy](Docs/Sphinx/guides/native_adaptive_policy.rst),
[configuration generation](Docs/Sphinx/guides/rltools_configuration_generation.rst),
[checkpoints](Docs/Sphinx/guides/native_rltools_checkpoints.rst).

**Try it without Unreal** (macOS or Linux, C++20 compiler). This is the adaptation test on the plugin's
learner: it trains on one runner, switches to another, and checks that the learner recovers while a
frozen copy does not.

```sh
SEEDS="1 2 3" Tools/ScholaRLToolsNative/run-chase.sh     # ~6 min per seed; ends with PASS criterion5
Tools/ScholaRLToolsNative/build.sh                       # learner and checkpoint checks (460)
Tools/ScholaRLToolsNative/run-scaling.sh                 # cost at 1-200 NPCs
```

Results: with a 10k replay buffer the learner meets all five adaptation criteria on six seeds
([CHASE-ADAPTATION.md](Tools/ScholaRLToolsNative/CHASE-ADAPTATION.md)). The scaling numbers are in
[SCALING.md](Tools/ScholaRLToolsNative/SCALING.md).

**Try it in Unreal** (Windows, UE 5.8.4, Visual Studio 2022 with *Game development with C++*):

```powershell
./Examples/NativeLearning/Test-NativeLearning.ps1 -EnginePath 'D:\Apps\Epic Games\UE_5.8'
./Examples/NativeLearning/Test-NativeLearning.ps1 -EnginePath 'D:\Apps\Epic Games\UE_5.8' -Filter 'NativeLearning.ChaseAdaptation' -ChaseSeed 1
```

The first command builds a host with the whole plugin and runs the policy, stepper and world-service
tests. The second runs the same adaptation test through `UAdaptiveStepper` and `URLToolsNPCPolicy`.

**Playable demos.** `chase_sim` is an interactive terminal simulator of the chase task
(`Examples/AdaptiveTag`, [demo video](https://drive.google.com/file/d/1J_AV8wG49tRTUB4X1vTzty05bYBZ5JJi/view?usp=sharing)):

```sh
cd Examples/AdaptiveTag
sh Tools/ChaseSim/Build-ChaseSim.sh
Saved/ChaseSim/chase_sim --seed 1 --script Evidence/SAC/d3/demo.script
```

The original tag arena (UE 5.7, `Examples/AdaptiveTag/Play-TagArena.ps1 -Build -Paused`) shows a tagger
learning a runner's route preference and adapting when it flips. It uses an example-owned learner, not the
plugin API; see its [README](Examples/AdaptiveTag/README.md) for controls.

### Background NPCs: batch inference across many agents

**User story:** as a researcher, I want to batch inference requests across many background agents,
including Mass entities, so that I can study large-scale interactions within the simulation's
performance budget.

1. Set a `UNPCInferencePolicy` to `Paradigm = Background` with an ONNX model and a backend
   (ORT CPU or BasicCpu).
2. For Actors, use `UBackgroundStepper` (or write rows directly with `INPCBackgroundRowAgent`).
   For Mass, derive gather/apply processors from `UScholaMassGatherProcessor` /
   `UScholaMassApplyProcessor` and give entities the schedule, state and bucket fragments.
3. The game thread only gathers observation rows and applies action rows; workers run batched model calls.
   Results older than `MaxActionAgeMs` are discarded. Over capacity, requests are rejected as Busy rather
   than queued without limit.

Guide: [NPC inference paradigms](Docs/Sphinx/guides/mock_inference_paradigms.rst). Complete Mass example:
[`CrowdProcessors.cpp`](Examples/BackgroundNPCs/Source/BackgroundNPCs/CrowdProcessors.cpp).

**Try it** (Linux, UE 5.8 with NNERuntimeORT; Windows steps are in each example's README):

```bash
export UE_ENGINE_ROOT=/path/to/UnrealEngine-5.8
cd Examples/BackgroundNPCs && Scripts/build.sh && Scripts/run.sh     # Tab: API on/off
cd ../CampusCrowd && git lfs pull && Scripts/build.sh && Scripts/run.sh
```

[BackgroundNPCs](Examples/BackgroundNPCs/README.md) runs the same 20,000-agent crowd with and without the
API: 10.4 FPS without it, 60 FPS (capped) with it, at 200,000 decisions/s
([recording](deliverables/D2/evidence/background-npc-side-by-side.mp4)).
[CampusCrowd](Examples/CampusCrowd/README.md) places up to 100,000 students on a North Campus model built
from City of Toronto massing data ([demo video](https://utoronto-my.sharepoint.com/:v:/g/personal/maxh_xu_mail_utoronto_ca/IQDEpcI8CITpT6vs0vv20yCwAU4lH9gFIKJzbEZhq47R4p8?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=R3tV6g)).

### Shared Policy NPCs: one network, role-specific heads

**User story:** as a machine learning engineer, I want NPC policies with a shared backbone and specialized
output heads, so that I produce multiple specialized decisions with less computation than running each
policy independently.

1. Export a model with a shared layer and named role heads (the reference model has Combat and Support).
2. Bind each NPC to a head with the shared-policy configuration types, and use `USharedPolicyStepper`,
   which collects observations, makes one `ISharedPolicy` batch call, and routes each action back to its NPC.
3. Choose an architecture. D3 compared five designs (A–E). C (complete model per expert) had lower request
   latency in 45/48 trials on one Mac, by 0.6–2.9 µs. Memory and gameplay integration are still open.

**Try it** (Python 3.12 and [`uv`](https://docs.astral.sh/uv/)):

```bash
cd Prototypes/SharedPolicy
uv sync --python 3.12
uv run pytest -q
uv run python -m benchmarks.export_models
uv run python -m benchmarks.benchmark --config benchmarks/config.json --smoke --output benchmarks/runs/smoke.json
```

The Unreal NNE matrix and full design are in the [benchmark guide](Prototypes/SharedPolicy/benchmarks/README.md).

### Event NPCs: decide when the game says so

**User story:** as a game developer, I want NPCs to subscribe to gameplay events and run inference when
those events occur, so that their decisions follow turn changes and other game events.

`UEventStepper` implements `IStepper` and runs Observe → BatchedThink → Act only when `TriggerInference()`
is called. A Blueprint demonstration triggers it from a simulated gameplay event. The stepper is on branch
`feature/event` and is not merged into `main`. Automatic event subscription is the next step.

## Architecture

All four NPC types plug into the same Schola contract: an **agent** observes and acts, a **policy** maps
observations to actions, and a **stepper** decides when that happens. Each feature adds a new policy or
stepper. Game code never depends on RLtools, ONNX Runtime or worker threads directly.

| Layer | Adaptive | Background | Shared Policy | Event |
| --- | --- | --- | --- | --- |
| Stepper (when) | `UAdaptiveStepper` | `UBackgroundStepper`, Mass processors | `USharedPolicyStepper` | `UEventStepper` |
| Policy (how) | `URLToolsNPCPolicy` → `UAdaptiveNPCPolicy` | `UNPCInferencePolicy` (Background) | `UNNESharedPolicy`, `UNNEPerHeadSharedPolicy` | any `IPolicy` |
| Engine (where) | `URLToolsWorldSubsystem` worker pool, RLtools in C++ | `BackgroundInferenceScheduler`, NNE worker contexts | NNE | caller's thread |

Diagrams: [adaptive online-learning loop](Docs/Diagrams/adaptive-online-learning-loop.svg),
[background NPC architecture](Docs/Diagrams/background-npc-architecture.svg),
[inference API before/after](Docs/Diagrams/inference-api-before-after.svg). Design decisions and their
alternatives are in the [D3 product evolution report](deliverables/D3/product-evolution.md).

## Install and access

Schola is a source-distributed Unreal plugin, not a hosted service. To use these features in your own
project:

1. Copy or link this repository into your project as `Plugins/Schola`.
2. Enable the plugin, regenerate project files and build the editor target. This needs Unreal Engine 5.8
   and a C++ toolchain: Visual Studio 2022 on Windows, Xcode on macOS
   ([macOS setup](Docs/macOS-Setup.md)) or clang on Linux.
3. Use the workflows above. Upstream installation, Python tooling and engine support are in
   [README-Schola.md](README-Schola.md).

RLtools is vendored at a pinned revision under `Source/ThirdParty/RLTools`; nothing is downloaded at build
time. Adaptive learning needs no Python service or pre-trained model. Background and Shared Policy
inference need an ONNX model, and the examples include theirs.

The example projects under `Examples/` are ready-made hosts. Each README has its build, run, test and
packaging commands. Tagged releases (`D2`, `D3`, `D4`) mark each submitted snapshot.

## Testing and CI

| Component | Tests | Where they run |
| --- | --- | --- |
| Native RLtools learner and checkpoints | `Tools/ScholaRLToolsNative/build.sh` (460 checks) | GitHub Actions (`native-learner-ci.yml`) and locally |
| Chase simulator | `Examples/AdaptiveTag/Tools/ChaseSim/ci-smoke.sh` (7 checks) | GitHub Actions (`chase-sim-ci.yml`) and locally |
| Adaptive policy, stepper, RLtools world service | `Schola.Inference.Adaptive.*`, `Schola.RLTools.*` | Locally: `Examples/NativeLearning/Test-NativeLearning.ps1` |
| Configuration generator | `Test/rltools_generation` (43 checks) | Locally: `Test/rltools_generation/Run.ps1` |
| Background scheduler, NNE backend, Mass bridge | `Schola.Inference.Background.*`, `Schola.Mass.*` | Locally from `Examples/BackgroundNPCs` |
| Shared Policy | `Schola.Inference.SharedPolicy.*`, `Prototypes/SharedPolicy` pytest | Locally |
| Upstream Schola | `Schola.*` automation tests, `Test/` pytest | Locally |

Unreal automation tests do not run in GitHub Actions. Hosted runners have no Unreal installation, and
provisioning one does not fit the course's job time limits and minute budget. Everything that builds
without Unreal runs in CI. The [maintainer guide](DEVELOPMENT.md#testing) lists every command and its
expected result.

## Non-functional requirements

| Concern | How it is addressed | Evidence |
| --- | --- | --- |
| Frame-time budget | Model calls and training run on worker threads. The game thread only gathers and applies rows, about 1.6 ms per frame for 20,000 agents. An adaptive decision costs 5–11 µs. | [Background comparison](deliverables/D2/evidence/background-npc-comparison.md), [SCALING.md](Tools/ScholaRLToolsNative/SCALING.md) |
| Bounded resources | Fixed pending-row, byte, queue and replay limits. Over capacity, requests are rejected or dropped and counted, never queued without limit. An independent adaptive learner costs about 8.6 MB. | [SCALING.md](Tools/ScholaRLToolsNative/SCALING.md), [CampusCrowd NFRs](Examples/CampusCrowd/README.md#non-functional-requirements) |
| Correct attribution | Every adaptive outcome must match its decision's session, agent, episode and policy version, or it is rejected. Stale actions expire. | [Adaptive NFR evidence](deliverables/D3/adaptive-npc-evidence.md#non-functional-requirements), `Schola.Inference.Adaptive.*` |
| Reliability | Checkpoints are written atomically and validated (checksum, profile, algorithm) before replacing a learner; a failed load changes nothing. Teardown, cancellation and shutdown races are tested. | [Checkpoint guide](Docs/Sphinx/guides/native_rltools_checkpoints.rst), `CheckpointTests.cpp` |
| Correctness of results | Shared Policy benchmarks stop if any architecture's routed actions differ from the oracle. Adaptation is judged against a frozen copy of the learner on fixed evaluation starts, using criteria fixed in advance. | [Benchmark guide](Prototypes/SharedPolicy/benchmarks/README.md), [CHASE-ADAPTATION.md](Tools/ScholaRLToolsNative/CHASE-ADAPTATION.md) |
| Usability | Blueprint-callable controls, actionable configuration errors, and one stepper call per tick for each NPC type. | Sphinx guides under `Docs/Sphinx/guides` |
| Portability | The native learner builds with clang and g++ on macOS and Linux. Unreal features target UE 5.8. | `native-learner-ci.yml`, per-feature "Verified on" above |

Scheduling is cooperative: no hard real-time guarantee is claimed.

## Known limitations

- **Adaptive:** CPU feedforward networks with Box observations and bounded Box actions only. A shared
  learner trains on one worker, about 650–900 updates/s, so it keeps up with roughly 60–90 NPCs at 10 Hz.
  After adapting, the chase learner's catch rate still dips at times. The stepper and the Unreal chase test
  have not yet been run on Windows. The tag arena uses UE 5.7 and its own learner.
- **Background:** CPU execution only; measured on one Linux machine with a synthetic policy; actions arrive
  at least one frame after their observation. The Mass gather processor's `MaxRowsPerFrame` must not exceed
  the scheduler's `MaxPendingRows`.
- **Shared Policy:** synthetic models; memory, AMD hardware and gameplay integration are not yet evaluated.
- **Event:** not merged; events are connected manually, and the demo uses a test policy.
- **Engine versions:** examples target UE 5.8 except `Examples/AdaptiveTag` (UE 5.7). UE 5.7 support for the
  new modules is not tested.

## Maintaining this project

The [maintainer guide (DEVELOPMENT.md)](DEVELOPMENT.md) covers the repository layout, toolchain setup on
each platform, building and testing every component, CI rules, how to extend each NPC type, conventions,
releases, and open work in priority order. Upstream Schola's contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Deliverables and contributions

| Deliverable | Contents |
| --- | --- |
| [D1](deliverables/D1/planning.md) | Planning, user stories, roles |
| [D2](deliverables/D2/README.md) | First prototypes, [iteration review](deliverables/D2/iteration-02-review.md), [contributions](deliverables/D2/Individual_Contributions_Deliverable2.md) |
| [D3](deliverables/D3/README.md) | Plugin integration, benchmarks, [slides](deliverables/D3/index.html), [product evolution](deliverables/D3/product-evolution.md), [contributions](deliverables/D3/Individual_Contributions_Deliverable3.md) |

Team records (stakeholders, meeting minutes) are in [deliverables/team](deliverables/team).

## License

Schola is MIT-licensed ([LICENSE.txt](LICENSE.txt)). The vendored RLtools is also MIT-licensed; its
notice is kept under `Source/ThirdParty/RLTools`.
