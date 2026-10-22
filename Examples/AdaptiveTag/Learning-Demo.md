# Native categorical actor tag demo

This sandbox uses Unreal Engine and RLtools, with Schola disabled. The actor's
actions are **LEFT** and **RIGHT**. Every learning-mode decision, including the
first round, is sampled directly from the actor's own categorical distribution.
There is no epsilon/random-action override, random warm-up, probability floor,
continuous probability-setting action, or second route-selection draw in the game.

This replaces the earlier continuous-action SAC experiment. The current algorithm
is a small **categorical actor-critic** for one-step episodes. Its policy-gradient
update is implemented in the sandbox using RLtools neural networks, backpropagation,
and Adam. It does not claim to be RLtools' built-in SAC or categorical PPO.

## Play

From `Examples/AdaptiveTag`, run `./Play-TagArena.ps1 -Build` after installing
Unreal 5.7 and its C++ build tools. Engine discovery and custom path options are
described in [README.md](README.md). For the standalone packaged
game, run `./Play-TagArena.ps1 -Packaged`. Click the viewport and use WASD to move
the cyan runner around either side of the cover. The actor chooses an interception
route before the runner commits. Movement to that route's interception point is
an automatic execution of its action; this demo does not train locomotion.

Add `-Paused` to either launch command to start in human mode with the arena
paused. Press **Space** when ready, then use **WASD**.

To observe adaptation, press **2** for an 80%-left runner and **F twice** for 12x
simulation. After the policy develops a preference, press **3** to reverse the
runner's bias. The actor, critic and optimizer are preserved across this switch.
Watch probability and catch-rate trends across many rounds, not a single result.
The actor can make mistakes because it samples its learned distribution; its
actions are never replaced by an external exploration rule.

| Key | Meaning |
| --- | --- |
| WASD | Move the runner in human mode |
| 1 / 2 / 3 / 4 | Human / 80%-left / 80%-right / balanced runner |
| F | Cycle 1x / 4x / 12x simulation speed |
| L | Freeze/resume training while continuing actor inference |
| B | Explicitly select the fixed 50/50 comparison baseline, or return to the actor |
| Space | Pause the arena; an in-progress worker update may finish |
| R | Restart an unscored round |
| T | Reset model weights, scores and random streams |

The fixed baseline is a user-selected comparison mode and is labelled as such.
It does not secretly intervene in learning mode. Fast-forward advances actual
arena physics substeps; it does not create synthetic training outcomes. Models
are kept in memory only, so every fresh launch starts with untrained weights.

## RL loop

1. The actor receives observation **[1.0]**, a constant bias input. It never receives
   the runner's position, current route, runner mode or bias probability.
2. Its RLtools neural network produces two logits. Softmax converts them to
   P(LEFT) and P(RIGHT), and the actor samples one action. The weights initially
   produce exactly 50/50. There are no hard minimum route probabilities.
3. Unreal assigns the tagger's route directly to that returned action. If the
   actor returns an invalid result, the game pauses instead of inventing a fallback.
4. A real catch yields **+1**; an escape yields **-1**. Aborts, idle timeouts and
   incomplete rounds produce no training experience.
5. An experience contains observation, chosen LEFT/RIGHT action, the probability
   under which it was chosen, reward, and policy version. The episode is terminal:
   next observation is [1], done=true. The worker never receives a target route label.
6. One fresh experience triggers one update. A value critic learns to predict
   reward; `advantage = reward - predicted value` determines whether to increase
   or decrease the chosen action's probability. The actor minimizes negative
   advantage-weighted log probability plus an entropy regularization term.
7. RLtools performs backpropagation and Adam parameter updates. A complete actor
   copy is published for subsequent decisions; gameplay never uses partially
   updated training weights.

The actor and value critic each use a small network with two 16-unit hidden
layers. The actor's Adam learning rate is **0.006**, doubled for more visible
per-round probability changes in recordings; the critic's rate remains 0.003.
Larger steps can also make probabilities fluctuate more. Entropy weight is 0.4: this is a term in the
actor's learning objective, not a chance of bypassing it. It discourages premature
certainty while still letting rewards determine the preferred route. The game
samples only the probabilities actually produced by the network.

There is **no replay buffer** in this version. Fresh outcomes enter a bounded
128-item worker queue and are consumed once. For an action made under a slightly
older snapshot, the policy-gradient estimator uses the ratio of its current
probability to the recorded behaviour probability. This corrects version lag;
it does not replace or change the chosen action. No Unreal object reaches the
numeric worker. Stop joins the worker before its owner is destroyed.

## HUD and data

The probability bar is the actor's actual LEFT/RIGHT distribution, not an average
over a separate exploration policy. Current route identity stays hidden until
the runner commits. The HUD shows actual experience and update counts, model
versions, queue/dropped samples, critic MSE, policy entropy, update and inference
timings, frame rate, recent catch rate, and recent terminal experience events.
Critic error is prediction fit, not a direct policy-quality score. The chart
covers approximately 30 wall-clock seconds regardless of simulation speed.

`Saved/TagLearning/Experiences.csv` records categorical actions and probabilities;
`Telemetry.csv` records statistics twice per wall-clock second. `Rounds.csv`
under `Saved/TagArena` records game outcomes. New sessions overwrite telemetry
and experience CSVs; the rounds file appends. Packaged gameplay writes beneath
the packaged project's `Saved` directory. Test reports are explicitly directed
back to the sandbox's `Saved/TagLearning` folder and checked for freshness.

## Build and verify

RLtools is pinned to `b32d9985c65a5e098a6bbf190fd994962d288b99` under
`ThirdParty/RLtools`. Run `./Prepare-LearningDemo.ps1` if it is missing. Its MIT
notice is retained and staged. No Python backend is required.

```powershell
./Build.ps1
./Test-Learning.ps1 -Mode Native
./Test-TagArena.ps1
./Test-Learning.ps1 -Mode Arena
./Package-LearningDemo.ps1
./Test-Learning.ps1 -Mode Packaged -Executable ./Saved/TagLearning/Package/Windows/CSC301Host.exe
./Play-TagArena.ps1 -Packaged
```

Native tests check four seeds, bias reversal, actor use and training from the
first round, one update per outcome, invalid actions, frozen weights, and whether
sampled route frequencies match the actor's reported distribution. Arena tests
exercise actual pawns, collision and rewards for 256 rounds per bias. Existing
26-check map/input tests remain separate. This is still a one-step learning
proof of concept, not a multi-step navigation or production-scale learner.

The faster actor's current 0.006-rate results are in **Evidence/FasterActor**.
All four native seeds and all nine packaged integration checks passed. The
packaged test completed 512 updates and adapted from P(LEFT)=0.927461 after the
left-biased phase to 0.017293 after the right-biased phase without resetting.

Source: `Source/TagLearningRuntime` owns the numeric learner;
`Source/CSC301Host/TagArena*` owns the game and HUD. No Schola API is used.
