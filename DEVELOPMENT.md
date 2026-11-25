# Maintainer guide

How to set up, build, test, extend and release this fork of AMD Schola. Read the [README](README.md) first
for what each NPC type does. Upstream Schola's contribution rules ([CONTRIBUTING.md](CONTRIBUTING.md)) and
installation notes ([README-Schola.md](README-Schola.md)) still apply. This guide covers what this project
added.

## Contents

- [Repository layout](#repository-layout)
- [Toolchains](#toolchains)
- [First-time setup](#first-time-setup)
- [Testing](#testing)
- [Continuous integration](#continuous-integration)
- [Extending each NPC type](#extending-each-npc-type)
- [Conventions](#conventions)
- [Releases](#releases)
- [Open work](#open-work)

## Repository layout

The repository root is the Schola plugin (`Schola.uplugin`). Unreal compiles every module under `Source/`.

| Path | Contents | Feature |
| --- | --- | --- |
| `Source/ScholaInferenceUtils` | Policies, steppers and the background scheduler. Adaptive base policy (`Policies/Adaptive*`), `UAdaptiveStepper`, `UBackgroundStepper`, `USharedPolicyStepper`, `Inference/*` | All |
| `Source/ScholaRLTools` | `URLToolsNPCPolicy`, world learner service, configuration assets, checkpoints. Engine-independent learner in `Private/Learning` | Adaptive |
| `Source/ScholaNNE` | NNE policies, the background batch backend, `UNNESharedPolicy` / `UNNEPerHeadSharedPolicy` | Background, Shared Policy |
| `Source/ScholaMass` | Mass gather/apply processors, fragments and the inference subsystem | Background |
| `Source/ThirdParty/RLTools` | RLtools headers pinned to `b32d9985c65a5e098a6bbf190fd994962d288b99`, with its license | Adaptive |
| `Source/Schola*` (other) | Upstream Schola: core, training, imitation, protobuf, StateTree, editor | Upstream |
| `Tools/ScholaRLToolsNative` | Engine-free builds of the RLtools learner: checkpoint tests, chase adaptation, scaling benchmark | Adaptive |
| `Examples/NativeLearning` | Minimal UE 5.8 host that links the whole plugin and runs the adaptive and RLtools tests | Adaptive |
| `Examples/AdaptiveTag` | Tag arena game (UE 5.7), `chase_sim` simulator, and the example-owned SAC runtime shared with AdaptiveSAC | Adaptive (demos) |
| `Examples/AdaptiveSAC` | UE 5.8 host for the example SAC learner behind `UAdaptiveNPCPolicy` | Adaptive (legacy) |
| `Examples/BackgroundNPCs` | 20,000-agent Mass crowd with and without the background API; Shared Policy NNE matrix scripts | Background, Shared Policy |
| `Examples/CampusCrowd` | North Campus crowd (Git LFS assets) | Background |
| `Prototypes/SharedPolicy` | Python reference model, ONNX export, A–E benchmarks | Shared Policy |
| `Prototypes/NativeLearning` | CMake build of the plugin learner for Windows | Adaptive |
| `Test/` | Upstream Python tests; `Test/rltools_generation` checks the configuration generator in C# | Upstream, Adaptive |
| `Docs/Sphinx/guides` | Developer guides; `Docs/Diagrams` holds architecture diagrams | All |
| `Resources/python` | Upstream Schola Python package (training, gym adapters) | Upstream |
| `deliverables/` | Course submissions and evidence (D1–D4), team records | Course |

Native test programs live under `Tools/`, never under `Source/`: Unreal compiles every `.cpp` under
`Source/`, so a `main()` there breaks the plugin build.

## Toolchains

| Work | Platform | Tools |
| --- | --- | --- |
| Unreal modules, NativeLearning, AdaptiveSAC | Windows | UE **5.8.4**, Visual Studio 2022 with *Game development with C++*, a Windows SDK |
| BackgroundNPCs, CampusCrowd | Linux (validated), Windows (untested) | UE **5.8.3+** with NNERuntimeORT, clang. Recording also needs X11, ffmpeg and xdotool |
| Unreal on Apple Silicon | macOS | UE 5.8, Xcode; see [Docs/macOS-Setup.md](Docs/macOS-Setup.md) |
| Tag arena | Windows | UE **5.7**, Visual Studio 2022, CMake 3.20+ for its native tests |
| Native learner, chase, scaling | macOS, Linux | clang++ or g++ with C++20 |
| Shared Policy prototype | Any | Python 3.12, [`uv`](https://docs.astral.sh/uv/) |
| Configuration generator tests | Windows | The .NET runtime bundled with UE 5.8 (`Run.ps1` finds it from `-EnginePath`, or pass `-DotNetPath`) |
| Docs | Any | Sphinx (`Docs/Sphinx`) |

Scripts find the engine through the Epic launcher registry, `-EnginePath`, or `UE_ENGINE_ROOT`.

## First-time setup

```sh
git clone https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs.git
cd 9-AMD-Advanced-NPCs
git lfs pull          # only needed for Examples/CampusCrowd
```

Then pick the part you are working on. Every example host links the plugin itself, so no copy step is
needed:

- **Adaptive, Unreal:** `./Examples/NativeLearning/Test-NativeLearning.ps1 -EnginePath <UE_5.8>`. This
  creates `Examples/NativeLearning/Plugins/Schola` as junctions to this repository, builds
  `NativeLearningEditor`, and runs the tests. Open `NativeLearning.uproject` in the editor afterwards to
  work interactively.
- **Adaptive, no Unreal:** `Tools/ScholaRLToolsNative/build.sh`.
- **Background:** `cd Examples/BackgroundNPCs && export UE_ENGINE_ROOT=<UE_5.8> && Scripts/build.sh`
  (it links `Plugins/Schola`). On Windows, create the junction once:
  `New-Item -ItemType Junction -Path Plugins/Schola -Target ../../..`.
- **Shared Policy:** `cd Prototypes/SharedPolicy && uv sync --python 3.12`.
- **Tag arena:** `cd Examples/AdaptiveTag && ./Prepare-LearningDemo.ps1 && ./Build.ps1`.

The first Unreal C++ build takes several minutes. Generated `Binaries/`, `Intermediate/`, `Saved/` and
`Plugins/` links are git-ignored. Never commit them.

## Testing

Run the checks for every area you touch before opening a PR. Expected results are the current baseline;
a lower number means a test went missing.

| Area | Command (from the repository root unless noted) | Expected |
| --- | --- | --- |
| Native learner + checkpoints | `Tools/ScholaRLToolsNative/build.sh` | `TOTAL: 460 passed, 0 failed` |
| Chase adaptation (plugin learner) | `SEEDS="1 2 3" Tools/ScholaRLToolsNative/run-chase.sh` | `PASS criterion5`; CSVs match `Evidence/chase-adaptation/replay-10k` |
| Scaling | `Tools/ScholaRLToolsNative/run-scaling.sh` (idle machine) | Same shape as `SCALING.md` |
| Adaptive + RLtools in Unreal | `./Examples/NativeLearning/Test-NativeLearning.ps1 -EnginePath <UE_5.8>` | All selected tests pass; the script fails if any registered test is missing |
| Unreal chase adaptation | `... -Filter 'NativeLearning.ChaseAdaptation' -ChaseSeed <1..3>` | Passes on seeds 1–3 (several minutes each) |
| Configuration generator | `./Test/rltools_generation/Run.ps1 -EnginePath <UE_5.8>` | 43 checks pass |
| Chase simulator | `cd Examples/AdaptiveTag && sh Tools/ChaseSim/Build-ChaseSim.sh && sh Tools/ChaseSim/ci-smoke.sh Saved/ChaseSim/chase_sim` | 7 checks pass |
| Tag arena | `cd Examples/AdaptiveTag && ./Test-Learning.ps1 -Mode Native && ./Test-TagArena.ps1` | All checks pass |
| Background + Mass | From `Examples/BackgroundNPCs`: `"$UE_ENGINE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd" "$PWD/BackgroundNPCs.uproject" -nullrhi -unattended -nosound "-ExecCmds=Automation RunTests Schola.Inference.Background+Schola.Mass;Quit"` | 43 tests pass |
| CampusCrowd | `cd Examples/CampusCrowd`, then the commands in its [README](Examples/CampusCrowd/README.md#local-validation-and-deployment-d3) | As documented there |
| Shared Policy (Python) | `cd Prototypes/SharedPolicy && uv run pytest -q` | All pass |
| Shared Policy (NNE matrix) | `Examples/BackgroundNPCs/Scripts/run_cd_isolated_matrix.sh` | 48/48 reports pass correctness |
| Upstream Schola | `pytest` (see `pytest.ini`; it builds Unreal, so set `engine_path`) | Upstream baseline |

Write Unreal tests as `IMPLEMENT_SIMPLE_AUTOMATION_TEST` in the module's `Private/Test` folder, named
`Schola.<Area>.<Feature>.<Case>`. `Test-NativeLearning.ps1` discovers expected tests by scanning those
registrations, so a new test is checked automatically. Prefer controlled fakes for policies and clocks
(for example `UAdaptiveDecisionTestPolicy`) over sleeps and real timing.

## Continuous integration

| Workflow | Runs on | Does |
| --- | --- | --- |
| `.github/workflows/native-learner-ci.yml` | PRs and pushes to `main` touching the learner, RLtools headers or `Tools/ScholaRLToolsNative` | Builds and runs the native learner and checkpoint checks with g++ |
| `.github/workflows/chase-sim-ci.yml` | PRs and pushes to `main` touching `Examples/AdaptiveTag` | Builds `chase_sim` and runs 7 smoke checks |

Course rules that every workflow must follow:

- Every job sets `timeout-minutes` (the current jobs use 5).
- Stay within the GitHub Actions minute budget. Trigger on narrow `paths`, and never build Unreal in
  Actions: hosted runners have no engine, and installing one takes longer than any allowed job.
- CI must stay green on `main`. A failing workflow is treated as a broken build, not ignored.

Unreal tests run locally with the scripts above. AMD runs its own Unreal CI/CD internally; connecting it to
this repository is pending with the partner.

## Extending each NPC type

**Adaptive: a new learner backend.** Subclass `UAdaptiveNPCPolicy`. After the backend is ready, call
`StartSession`. Implement `DispatchDecision` (complete with `CompleteDecision`, synchronously or later on
the game thread), `DispatchExperience`, `ReleaseLearnerBinding` and `OnEpisodeReset`. The base class owns
tickets, expiry, attribution checks and operation history; [adaptive_decision_lifecycle.rst](Docs/Sphinx/guides/adaptive_decision_lifecycle.rst)
specifies the contract. `UAdaptiveStepper` then drives the backend with no changes.

**Adaptive: a new observation/action shape or network.** In a game project, use an RLtools Configuration
Asset and **Generate Project Configuration**, or edit `Config/ScholaRLTools.json`, then rebuild.
`ScholaRLTools.Build.cs` validates the manifest and generates the C++ profile; the generator is covered by
`Test/rltools_generation`. Plugin-wide default profiles are in
`Source/ScholaRLTools/Private/Learning/Generated/RLToolsLearnerProfiles.inl`. For engine-free experiments,
inject profiles with `-DSCHOLA_RLTOOLS_PROJECT_PROFILES` (see `ChaseProfiles.inl`).

**Adaptive: changing the learner.** `RLToolsLearner.cpp` is engine-independent. Run `build.sh` and the
chase test on every change. A checkpoint format change must bump the format version and keep the refusal
tests passing. Upgrading RLtools means replacing `Source/ThirdParty/RLTools` and updating the pinned
revision in `FProfile::Revision`, which checkpoints record.

**Background: a new inference backend.** Subclass `UNPCBackgroundBackendProvider` so each worker gets its
own model context. Extend `NPCRowCodec` for new space types. Keep `MaxPendingRows` and byte limits bounded.

**Shared Policy: a new architecture.** Implement `ISharedPolicy` (see `UNNESharedPolicy`,
`UNNEPerHeadSharedPolicy`), and add it to the benchmark's correctness oracle before comparing performance.

**New NPC type or scheduling style.** Implement `IStepper`. Keep game-thread work bounded, and put
expensive work behind a policy that runs on workers.

## Conventions

- **Branches:** `main` is the integration branch. Work on `feature/…`, `fix/…`, `docs/…`, `ci/…` or
  `test/…` branches and merge through pull requests.
- **Reviews:** every PR needs an approving review from another team member before merge, and its checks
  must pass. Untested Unreal code says so in the PR, with reviewer test steps.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) with a scope, e.g.
  `feat(adaptive): …`, `fix(background): …`, `docs: …`.
- **Code:** follow the surrounding Unreal style (`F`/`U`/`I` prefixes, PascalCase). Game-thread APIs
  `check(IsInGameThread())`. Workers never touch UObjects.
- **Evidence:** a claimed result links to a reproducible command and its raw output, committed under the
  feature's `Evidence/` folder, with no machine-specific paths. Pass criteria are fixed before a run and
  are never loosened afterwards.
- **Docs:** user-facing changes update the README workflow and the matching Sphinx guide in the same PR.

## Releases

Each course deliverable is a GitHub release tagged `D2`, `D3`, `D4` on the merged `main` commit. To make
one:

1. Merge all PRs for the deliverable and confirm CI is green on `main`.
2. Run the local Unreal test scripts for the features in the release, and record the results in the PR or
   the deliverable folder.
3. `git tag D4 <commit> && git push origin D4`, then create the GitHub release from the tag with a short
   summary and links to `deliverables/D4`.

Packaged games, where they exist, are built with each example's `Package-*.ps1` or `Scripts/package.sh`
and attached to the release instead of being committed.

## Open work

In priority order for each feature.

**Adaptive**
1. Compile and run `UAdaptiveStepper` and `NativeLearning.ChaseAdaptation` on Windows UE 5.8.4
   (seeds 1–3), and fix whatever fails.
2. Define a post-recovery stability criterion for the chase test, then tune the replay size and learning
   rate against it with `run-chase.sh` (`--replay`, `--learning-rate`, `SEEDS=…`).
3. Collect observations, actions and rewards automatically through the Schola coordinator, so games do
   not hand-write `IAdaptiveAgent`.
4. Package the adaptive host as a separate project, and run `Schola.RLTools.Checkpoint.*` on Windows.
5. Move the tag arena from its example-owned learner to `URLToolsNPCPolicy` and UE 5.8.

**Background**
1. Validate BackgroundNPCs and CampusCrowd on Windows.
2. Run with a trained policy and measure scaling with more workers and larger crowds.
3. Resolve the Actor-adapter pool growth noted in D2.

**Shared Policy**
1. Compare C and D's model storage and baseline-subtracted steady-state memory.
2. Integrate the chosen design with trained policies and role-assigned NPCs in gameplay (the Lyra
   integration on `feat/lyra-reproducible-setup` is in progress).

**Event**
1. Review and merge `feature/event`.
2. Add configurable gameplay event subscription and unsubscription.
