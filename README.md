# Advanced NPCs for AMD Schola

## About the project

Schola is an AMD toolkit that connects Unreal Engine environments and agents to reinforcement learning tools. Our project explores advanced NPC behavior within that ecosystem.

The adaptive NPC work focuses on **continued learning during gameplay**. Instead of freezing an NPC's behavior after training, we want it to improve its decisions from new gameplay experiences and respond when player habits change.

For D2, the adaptive work demonstrates a **native Unreal Engine + RLtools prototype**: a tagger learns which route to favor while a player or scripted runner tries to escape. Training runs inside the game process without a Python service. Schola integration remains future work for this prototype.

### Other project features

| Subteam / feature | Description and value |
| --- | --- |
| | |
| | |

## Access the D2 product

| Item | Location |
| --- | --- |
| Course repository | [9-AMD-Advanced-NPCs](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs) |
| D2 release and download | |
| Adaptive NPC source | [Examples/AdaptiveTag](Examples/AdaptiveTag/README.md) |
| Demo recording | |
| Other feature access links | |

**Current availability:** the repository includes the standalone Unreal project, arena assets, and pinned RLtools headers under `Examples/AdaptiveTag`. You can build and run the simulation from a clone using the instructions below. A downloadable D2 package, independent-machine check, and final release URL still need confirmation.

### Clone and play locally

On Windows, install Unreal Engine **5.7** and Visual Studio 2022 with the **Game development with C++** workload and a Windows SDK. The prototype was tested with Unreal 5.7.4. No separate host repository, Python service, or downloaded model is needed.

```powershell
git clone https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs.git
cd 9-AMD-Advanced-NPCs/Examples/AdaptiveTag
./Play-TagArena.ps1 -Build -Paused
```

The script discovers Unreal, builds the project, and opens the game paused. Click the window and press **Space** to begin. The first build can take several minutes. For an Unreal installation that is not registered with Epic Launcher, pass `-EnginePath 'C:/Epic/UE_5.7'` or set `UE_ENGINE_ROOT` to your engine root.

See the [prototype README](Examples/AdaptiveTag/README.md) for troubleshooting, optional native tests, and packaging.

### Run the adaptive NPC demo

Once the Windows package is available from the D2 release:

1. Download and extract the **whole package**, keeping `CSC301Host`, `Engine`, and `CSC301Host.exe` together. Copying only the executable is insufficient.
2. Open `CSC301Host.exe` from the extracted `Windows` folder.
3. Click the game window. If the arena is paused, press **Space**.
4. Use **WASD** to move the cyan runner around the left or right side of the central cover and reach the escape area. Avoid the red tagger.
5. Watch the actor's LEFT/RIGHT probabilities, scored outcomes, and training update count in the game display.

The packaged demo does not require a Python server or a local Unreal Editor installation. We have tested it on the development machine. Fresh-machine graphics/runtime prerequisites still need verification before distributing it as the D2 build.

After packaging, the executable is under `Examples/AdaptiveTag/Saved/TagLearning/Package/Windows/`. From `Examples/AdaptiveTag`, this command opens the package paused:

```powershell
./Play-TagArena.ps1 -Packaged -Paused
```

### Observe learning and adaptation

1. Press **T** to start with fresh weights and scores. The actor begins at 50% LEFT and 50% RIGHT.
2. Press **2** to select a runner that chooses the left route 80% of the time. If paused, press **Space**.
3. Let multiple rounds complete. The learner should develop a preference for LEFT. Individual outcomes are random, so its probabilities can fluctuate.
4. Press **3** to make the runner favor RIGHT. **Do not press T between these phases:** the point is to observe the same model adapting to new behavior.
5. Watch the probability trend move toward RIGHT and compare the recent catch rate over multiple rounds.

Press **F** to cycle through 1x, 4x, and 12x simulation speed when using an automated runner. Faster simulation produces more real arena rounds per minute. It does not invent training outcomes. Press **1** to return to human control.

### Controls

| Key | Action |
| --- | --- |
| WASD | Move the runner in human mode |
| 1 | Human runner |
| 2 | Automated runner with an 80% left preference |
| 3 | Automated runner with an 80% right preference |
| 4 | Automated runner with a balanced preference |
| Space | Pause or resume the arena |
| F | Cycle simulation speed |
| L | Freeze or resume learning while the actor continues choosing actions |
| B | Switch between the learner and an explicitly labelled fixed 50/50 comparison policy |
| R | Restart the current round without scoring it |
| T | Reset model weights, scores, and random streams |

Changes to the runner mode preserve the learned weights. Fresh launches and **T** reset them. Models currently remain in memory only.

### Instructions for other features

| Feature | Access and steps | Expected result |
| --- | --- | --- |
| | | |
| | | |

## Key use cases

### Adaptive NPC: learn a player's route preference

**User story:** As a game developer, I want an NPC to adjust its decisions from gameplay outcomes so that it can respond to changing player habits.

**D2 result:** a working native sandbox proof of concept. The actor controls the tagger's LEFT/RIGHT decision from the first round. The game executes movement and detects catches and escapes.

The tagger commits to a route before the runner commits. A catch earns **+1**, and an escape earns **-1**. One scored round produces one training experience and one update. Aborted rounds and idle timeouts do not train the model.

The actor currently receives a constant observation, **`[1.0]`**. It cannot see the runner's position, chosen route, or scripted bias. Its weights learn a general route preference from rewards. This is a deliberately small learning task rather than learned navigation.

### Adaptive NPC: continue learning after behavior changes

The scripted runner can switch its preference without resetting the learner. Later decisions use updated actor weights. Learning happens during the same gameplay session, with training on a background worker.

This is online reinforcement learning: the policy learns from fresh interactions as play continues. A fixed policy can perform well against one consistent runner, so the useful demonstration is adaptation after the runner changes its behavior.

### Adaptive NPC: inspect and compare behavior

The display shows action probabilities, recent experiences and rewards, training counts, and runtime statistics. **L** lets the tester freeze learning, and **B** selects a fixed comparison policy. These controls support inspection; a controlled learning-versus-frozen performance study remains future work.

### Other subteam use case

**User story:**

**Description:**

**Implemented functionality:**

**Demo and validation:**

### Additional subteam use case

**User story:**

**Description:**

**Implemented functionality:**

**Demo and validation:**

### Connected product workflow

**Start-to-finish user journey:**

**Connections between subteam features:**

**Acceptance evidence:**

## Adaptive NPC implementation

The current learner is a small **custom categorical actor-critic** built with RLtools' C++ neural network, backpropagation, and Adam optimizer operations. It does not use RLtools' built-in SAC or PPO training loop.

| Component | Role in this demo |
| --- | --- |
| Actor | Produces LEFT/RIGHT probabilities and samples the tagger's route |
| Critic | Predicts expected reward and provides a reference for evaluating the actual outcome |
| Experience | Records the observation, actor action, chosen-action probability, reward, and policy version |
| Training worker | Updates the networks and publishes complete actor copies for later decisions |
| Unreal arena | Handles runner input, movement, collision, rewards, and the display |

The actor and critic each have two hidden layers of 16 neurons. The actor learning rate is **0.006**, the critic rate is **0.003**, and the entropy weight is **0.4**. Entropy regularization encourages the actor to retain some uncertainty. The actor's selected action passes directly to the game without a random-action override or random warmup.

The numeric learner owns its training data and networks. It does not access Unreal objects from the training thread. Gameplay uses complete actor snapshots, and shutdown joins the worker before destroying its owner.

### Validation results

These are recorded development test results, not performance guarantees for other environments or hardware.

A separate fresh checkout passed the native tests, Unreal editor and game builds,
packaging, 26 arena checks, and nine packaged learning checks without copying
project binaries, caches, or the old plugin junction. This used the same Windows
development machine and installed Unreal/C++ tools, rather than a fresh machine.

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Native learner across four seeds | All four adapt to both runner preferences | [Native test output](deliverables/D2/evidence/adaptive-native-tests.txt) |
| Packaged Unreal learning test | All 9 checks pass, with 512 scored experiences and 512 updates | [Packaged test report](deliverables/D2/evidence/adaptive-packaged-test.json) |
| Actor control | Every tagger route matches the actor's returned action in the packaged integration test | [Packaged test report](deliverables/D2/evidence/adaptive-packaged-test.json) |
| Arena input, movement, collision, and pause | All 26 checks pass | [Arena smoke report](deliverables/D2/evidence/adaptive-arena-smoke.json) |

In the packaged test, the same model completed **256 rounds against each runner bias**:

| Stage | Probability of choosing LEFT |
| --- | ---: |
| Fresh model | 50.0% |
| After the left-biased phase | 92.7% |
| After switching to the right-biased phase | 1.7% |

These values describe the actor's probabilities, not catch rates. The test preserved the weights across the switch. The final 64-round catch rate was **75%**.

The native and packaged learning results use the current actor rate of 0.006. The arena smoke test covers unchanged game behavior from before that learning-rate adjustment.

### Current limitations

- The prototype runs independently of Schola. Its API integration remains to be designed and implemented.
- It learns one route decision per round from a constant observation. Movement and interception execution are scripted.
- Learning depends on collecting enough scored rounds. Faster computation alone does not provide more experience.
- Models do not persist across launches.
- Testing establishes this small task. Larger agent counts, frame-time budgets, richer observations, and controlled comparisons need further evaluation.

### Other principal components

**User interface / developer experience:**

**Backend / API structure:**

**Tests and known limitations:**

## Development requirements and setup

### Adaptive NPC sandbox

The current development environment uses Windows, **Unreal Engine 5.7.4**, Visual Studio 2022 with the C++ toolchain and Windows SDK, and CMake 3.20+ for the standalone native tests. These tools are for development rather than playing the packaged demo.

RLtools is pinned to revision `b32d9985c65a5e098a6bbf190fd994962d288b99`, and its headers and license are included in the repository. `Prepare-LearningDemo.ps1` validates the dependency and can restore a missing dependency directory. Schola stays disabled in this prototype, and training needs no Python backend.

The following commands run from **`Examples/AdaptiveTag` inside this repository**:

```powershell
./Prepare-LearningDemo.ps1
./Build.ps1
./Test-Learning.ps1 -Mode Native
./Test-TagArena.ps1
./Test-Learning.ps1 -Mode Arena
./Package-LearningDemo.ps1
./Test-Learning.ps1 -Mode Packaged -Executable ./Saved/TagLearning/Package/Windows/CSC301Host.exe
./Play-TagArena.ps1 -Packaged -Paused
```

The build, launch, packaging, and Unreal test scripts discover the installed engine. They also accept `-EnginePath` for a custom location. `UE_ENGINE_ROOT` provides an alternative to passing the argument repeatedly.

| Location within `Examples/AdaptiveTag` | Contents |
| --- | --- |
| `Source/TagLearningRuntime` | Native numeric learner and training worker |
| `Source/CSC301Host/TagArena*` | Arena, input, and display |
| `ThirdParty/RLtools` | Pinned dependency and license notice |
| `Tools/LearningTests` | Standalone native learning checks |
| `Learning-Demo.md` | Detailed play, learning, build, and test guide |
| `Evidence/FasterActor` | Current learning reports and recorded transitions |
| `Saved/TagLearning/Package/Windows` | Locally generated standalone game |

### Installing Schola Into Your Project

For the existing Schola plugin's installation and Python tooling, use [the Schola setup guide](README-Schola.md). The adaptive tag prototype runs independently of that plugin setup.

Gameplay writes experience and telemetry CSV files beneath the packaged game's `CSC301Host/Saved/TagLearning` directory. New sessions overwrite those files, so copy them before starting another recording session if you need to retain them.

### Other subteam development setup

**Prerequisites:**

**Setup and run steps:**

**Validation commands:**

## Deployment and GitHub workflow

The adaptive prototype has repeatable build, test, and packaging scripts under `Examples/AdaptiveTag`. Its source, map, materials, and pinned dependency are versioned together in this repository. Generated binaries and caches remain ignored. The packaged game still needs a published D2 release location.

Existing Schola development conventions are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

**Team branch and pull request workflow:**

**Review and merge requirements:**

**Release and deployment process:**

**Fresh-machine validation and runtime prerequisites:**

## Licensing

The existing Schola codebase carries an MIT license in [LICENSE.txt](LICENSE.txt). RLtools also uses the MIT license. The adaptive sandbox retains the RLtools notice and stages it with the packaged demo.

**Team license choice and rationale:**

## Remaining work and priorities

Proposed next steps for the adaptive NPC experiment are to compare learning and frozen actors under the same runner behavior changes, introduce useful observations and richer decisions, and design the Schola integration after evaluating the native approach.

**Team MVP priorities:**

**Owners and milestones through D4:**

**Cross-subteam dependencies:**

## Iteration review and partner feedback

**What worked well:**

**What did not work well:**

**Changes for the next iteration:**

**Partner demo date and participants:**

**Partner feedback and resulting decisions:**

## Individual contributions

| Student | Role | Specific contributions and PR / commit links |
| --- | --- | --- |
| | | |
| | | |
| | | |
| | | |

## D2 submission checklist

The README supports the product demonstration. The iteration review and individual contribution document are separate required deliverables.

| Required item | Owner | Location / confirmation |
| --- | --- | --- |
| Root `README.md` with final access links and completed team sections | | |
| `deliverables/D2/iteration-02-review.md` | | |
| `deliverables/D2/Individual_Contributions_Deliverable2.md` | | |
| Stable deployed product and GitHub release tagged `D2` | | |
| Partner demonstration and feedback | | |
| One team submission linking the D2 release on Quercus | | |
