# Adaptive Tag: native learning prototype

A small Unreal Engine tag game that updates its NPC policy during play using RLtools. This is an isolated experiment: Schola is disabled, and no Python service, external model, plugin junction or separate host checkout is required.

## Clone and play (Windows)

Install **Unreal Engine 5.7** (tested with 5.7.4) and **Visual Studio 2022** with the Game development with C++ workload and a Windows SDK. The Unreal installer and C++ compiler are external prerequisites, not files distributed in this repository. CMake 3.20+ is needed only for the standalone native tests.

```powershell
git clone https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs.git
cd 9-AMD-Advanced-NPCs/Examples/AdaptiveTag
./Play-TagArena.ps1 -Build -Paused
```

The script builds the editor target, opens the game, and starts paused. Click the window, press **Space**, and use **WASD** to control the cyan runner. Reach the green escape area while avoiding the red tagger.

The scripts discover Unreal 5.7 through its registered installation or Epic Launcher manifest. For a custom install, pass the engine root:

```powershell
./Play-TagArena.ps1 -Build -Paused -EnginePath 'C:/Epic/UE_5.7'
```

Alternatively, set `UE_ENGINE_ROOT`. The engine path option also works with the build, editor, packaging, and Unreal test scripts. First compilation can take several minutes. Close an open editor before rebuilding its modules.

If PowerShell blocks scripts, use a process-only policy for the terminal running this checkout, where permitted by your machine's policy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

The checkout includes the arena map, materials, native learner, and RLtools headers pinned to `b32d9985c65a5e098a6bbf190fd994962d288b99`. No dependency download is needed for the normal build. `Prepare-LearningDemo.ps1` validates the pinned dependency; its download path supports restoring a missing dependency directory.

## Demonstrate adaptation

1. Press **T** for a fresh model. Its LEFT/RIGHT probabilities begin at 50/50.
2. Press **2** for a runner that goes left 80% of the time. Let multiple rounds finish.
3. Press **3** to reverse the runner's preference without resetting the model.
4. Watch the actor's probabilities respond to the new outcomes. Use **F** to speed up automated rounds.

One catch gives reward +1, and one escape gives -1. The actor learns one LEFT/RIGHT choice per scored round. Movement to the interception point is scripted. The observation is constant [1], so the learner receives neither the current runner route nor its scripted mode.

| Key | Action |
| --- | --- |
| WASD | Move the runner in human mode |
| 1 / 2 / 3 / 4 | Human / 80%-left / 80%-right / balanced runner |
| Space | Pause or resume |
| F | Cycle 1x / 4x / 12x simulation speed |
| L | Freeze or resume training while inference continues |
| B | Select the labelled fixed 50/50 comparison policy, or return to the learner |
| R | Restart an unscored round |
| T | Reset weights, scores, and random streams |

The HUD shows probabilities, rewards, experiences, training counts, model versions, and runtime statistics. A fresh launch resets weights. See [Learning-Demo.md](Learning-Demo.md) for the learning loop and current limits.

## Build, test and package

Run from this directory. The native test is independent of Unreal, while arena tests require a successful editor build.

```powershell
./Build.ps1
./Test-Learning.ps1 -Mode Native
./Test-TagArena.ps1
./Test-Learning.ps1 -Mode Arena
./Package-LearningDemo.ps1
./Test-Learning.ps1 -Mode Packaged -Executable ./Saved/TagLearning/Package/Windows/CSC301Host.exe
./Play-TagArena.ps1 -Packaged -Paused
```

`Package-LearningDemo.ps1` builds both required targets and writes `Saved/TagLearning/Package/Windows`. Share the whole folder, including its Engine and CSC301Host subdirectories. Packaging includes the Unreal Windows prerequisite installer. This prototype has Windows build/test coverage; other platforms have not been validated.

Generated output belongs in ignored directories. Source control contains authored assets and source, not packaged executables. Learning reports and telemetry go under `Saved/TagLearning`, and the map/input smoke report goes under `Saved/Automation`. A packaged game's own logs are beneath `Windows/CSC301Host/Saved`.

## Source layout

| Path | Purpose |
| --- | --- |
| `CSC301Host.uproject`, `Config/` | Standalone Unreal project and startup map settings |
| `Content/TagArena/` | Tracked arena map and materials |
| `Source/CSC301Host/` | Game rules, input, collision and HUD |
| `Source/TagLearningRuntime/` | Numeric actor-critic and background training worker |
| `ThirdParty/RLtools/` | Unmodified pinned headers, revision and MIT license |
| `Tools/LearningTests/` | Four-seed learning and lifecycle checks |
| `Tools/CreateTagArena.py` | Optional editor authoring utility, unnecessary for build or play |
| `Evidence/FasterActor/` | Recorded 0.006-rate experiment results |

The project and native module names remain CSC301Host for compatibility with the authored map. No external directory named CSC301Host is needed.

## Scope and licensing

The categorical actor-critic is custom sandbox code built with RLtools networks, backpropagation, and Adam. It is not the library's built-in SAC or PPO implementation. Actor rate is 0.006, critic rate is 0.003, and entropy weight is 0.4. The policy controls every learning-mode route choice, including the first round.

This demonstrates native training during play in a small one-step task. Schola integration, learned navigation, saved models, larger workloads, and controlled performance comparisons remain future work.

Project code follows the repository's [MIT license](../../LICENSE.txt). RLtools retains its own [MIT notice](ThirdParty/RLtools/LICENSE), which the Unreal build stages with the game.
