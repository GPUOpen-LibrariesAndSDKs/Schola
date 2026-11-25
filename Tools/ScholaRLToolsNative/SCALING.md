# Scaling the plugin learner to many NPCs

`ScalingBenchmark.cpp` measures the plugin's own learner (`Schola::RLTools::FLearner`, the code behind
`URLToolsNPCPolicy`) at 1–400 NPCs, without Unreal. Observations and rewards are synthetic, because the
cost of a decision and of an update does not depend on their values.

```sh
Tools/ScholaRLToolsNative/run-scaling.sh                                  # 1,10,25,50,100,200 NPCs, all workers
Tools/ScholaRLToolsNative/run-scaling.sh Saved/S -- --workers 3 --npcs 1,50,100,200,400
```

Options: `--npcs`, `--rate` (decisions per NPC per second, default 10), `--seconds` (default 10),
`--workers` (training threads for independent learners, default hardware threads − 1),
`--gradient-steps` (updates per record, default 1) and `--csv`. Run it on an otherwise idle machine.

## Setup

- Profile `sac/scale-16x4` (`ScalingProfiles.inl`): SAC, 16 observations, 4 actions, 64-wide 3-layer actor
  and critics, batch 32, replay 50,000. This is a mid-sized NPC.
- 10 decisions per NPC per second. Each NPC tick submits the previous outcome and requests a new decision,
  the same work `UAdaptiveStepper` does per agent. 1 gradient step per record. Every learner is warmed up
  past `LearningStarts` before timing starts.
- **Shared:** one learner fed by every NPC (a `SharedGroup`). `FLearner::Train` is serialized, so one worker
  trains it.
- **Independent:** one learner per NPC. A fixed pool of workers trains the learners round-robin.
- "Needed" is NPCs × rate × gradient steps, the update rate required to train on every record.
  "Backlog" is the updates still owed when the run ends. A growing backlog means training falls behind,
  so the learner trains on older data and fewer updates per record than configured.

## Results (Apple M5 Pro, 18 threads, macOS, clang++ -O2, 2026-11-24)

Raw output: `Evidence/scaling/`.

**Memory.** 8.6 MB resident per independent learner (20 learners after warm-up). The 50k replay buffer is
most of it. A shared group costs this once.

**Game-thread cost.** Decide + Submit per NPC decision is 5–11 µs at the median and ≤ 38 µs at p99. The
single-NPC case is slower (~27 µs), probably because its caches go cold between 100 ms ticks. At 200 NPCs and 10 Hz
that is about 2 ms of game-thread time per second, or 0.2 ms per tick.

**All workers (17):**

| NPCs | Shared updates/s | Needed | Shared backlog | Independent updates/s | Independent backlog |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 0 | 10 | 0 |
| 10 | 100 | 100 | 0 | 100 | 0 |
| 25 | 250 | 250 | 0 | 250 | 0 |
| 50 | 500 | 500 | 0 | 500 | 0 |
| 100 | 908 | 1,000 | 911 | 999 | 0 |
| 200 | 915 | 2,000 | 10,847 | 2,000 | 0 |

**3 workers (about a 4-core machine with the game thread on its own core):**

| NPCs | Shared updates/s | Needed | Shared backlog | Independent updates/s | Independent backlog |
|---:|---:|---:|---:|---:|---:|
| 50 | 499 | 500 | 0 | 500 | 0 |
| 100 | 763 | 1,000 | 2,367 | 999 | 0 |
| 200 | 633 | 2,000 | 13,663 | 1,966 | 317 |
| 400 | 678 | 4,000 | 33,215 | 2,251 | 17,464 |

## What this means

- **One shared learner trains at roughly 650–900 updates/s** on one core, whatever the NPC count. At 10 Hz
  and 1 gradient step it keeps up with about **60–90 NPCs**. Above that, the backlog grows without bound.
  More gradient steps per record lower the limit proportionally: with 8 steps (the chase setting), one
  shared learner keeps up with only about 10 NPCs. Larger groups need fewer decisions per second, fewer
  gradient steps, or several smaller groups.
- **Independent learners scale with worker threads**, at about 750 updates/s per worker on this CPU. With
  3 workers they keep up to about 200 NPCs at 10 Hz. They cost 8.6 MB each, so 200 NPCs need about 1.7 GB
  for this profile. A smaller replay buffer reduces this almost linearly.
- **The game thread is not the bottleneck**: decisions cost microseconds per NPC. The limit is training
  throughput, which runs on workers and never blocks a frame. When it falls behind, the NPCs still act;
  they just adapt more slowly.

## Limits

- One machine and one profile. A wider network or a bigger batch costs more per update. Re-run on target
  hardware, especially lower-clocked consoles or laptops on battery.
- The shared-learner figure varies between runs on the same machine (633–915 updates/s here), probably
  because the scheduler moves the worker between performance and efficiency cores. Treat it as a range.
- This is the native learner without Unreal. The `URLToolsWorldSubsystem` executor adds scheduling overhead
  and caps concurrent workers (`MaximumConcurrentWorkers`, default 2), so in-engine throughput for
  independent learners is bounded by that setting, not by the core count.
