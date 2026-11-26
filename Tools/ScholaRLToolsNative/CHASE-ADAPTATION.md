# Chase adaptation on the plugin learner

`ChaseAdaptation.cpp` runs the D3 chase adaptation test on the plugin's own learner
(`Schola::RLTools::FLearner`, the code behind `URLToolsNPCPolicy`) instead of the example-only
`OnlineSACLearner` that `chase_sim` uses. It needs no Unreal. The world, runners, rewards and the 100
evaluation starts are chase_sim's own code (`Examples/AdaptiveTag/Source/OnlineSACRuntime`), with the same
seed families.

```sh
Tools/ScholaRLToolsNative/run-chase.sh                                    # replay 10k, seeds 1-3
SEEDS="1 2 3 4 5 6" JOBS=6 Tools/ScholaRLToolsNative/run-chase.sh Saved/C20 -- --replay 20000
```

Options: `--seed`, `--replay 10000|20000|60000` (compiled in `ChaseProfiles.inl`), `--learning-rate`,
`--updates-per-move`, `--pretrain-episodes`, `--chunk-episodes`, `--window-moves`, `--post-moves`,
`--eval-episodes`, `--csv`. Runs are deterministic: the learner trains in lockstep, so a seed gives the
same numbers every time. A second full run on the same machine reproduced every CSV row.

## Protocol and criteria

Train 150 episodes on `rightturn`, capture an immutable snapshot, switch to `corner`, and keep learning.
Every 5 episodes, evaluate the deterministic actor on chase_sim's 100 fixed starts (eval seed 4242).
The criteria are the D3 ones from `deliverables/D3/adaptive-npc-evidence.md`, unchanged:

1. Learner ≥ 0.50 on `rightturn` at the switch.
2. Learner ≤ 0.80 on `corner` right after the switch.
3. Learner back to ≥ 0.80 on `corner` within 20,000 moves of the switch.
4. At the first evaluation at or past 20,000 moves, the snapshot is **more than** 0.20 below the learner.
5. Criteria 1–4 hold on every seed.

Catch rates are compared in whole percentage points, so a gap of exactly 0.20 fails criterion 4.

Learner: SAC, 10 observations, 2 actions, 64-wide 3-layer actor and critics, batch 32, discount 0.99,
tau 0.005, adaptive entropy. Settings mirror chase_sim: learning rate 1e-3, 1,000 random warm-up moves,
8 gradient steps per move, one policy version published per move. Only the replay capacity varies.
After the 20,000-move window each run continues for another 20,000 moves to record stability. That part
is reported but not scored.

## Results (2026-11-24, Apple M5 Pro, clang++ -O2)

Raw output: `Evidence/chase-adaptation/replay-*/seed*.{txt,csv}`.

| Replay | Seed | 1. rightturn | 2. corner after | 3. ≥0.80 at move | 4. learner | snapshot | gap | 4. | After window: min | final |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 10k | 1 | 0.87 | 0.37 | 12,039 | 0.75 | 0.37 | 0.38 | PASS | 0.54 | 1.00 |
| 10k | 2 | 0.87 | 0.18 | 10,916 | 0.87 | 0.18 | 0.69 | PASS | 0.74 | 0.99 |
| 10k | 3 | 1.00 | 0.38 | 4,396 | 1.00 | 0.38 | 0.62 | PASS | 0.81 | 0.99 |
| 10k | 4 | 0.78 | 0.10 | 11,367 | 0.98 | 0.10 | 0.88 | PASS | 0.68 | 0.88 |
| 10k | 5 | 0.96 | 0.27 | 10,162 | 0.94 | 0.27 | 0.67 | PASS | 0.72 | 0.98 |
| 10k | 6 | 0.98 | 0.21 | 13,999 | 1.00 | 0.21 | 0.79 | PASS | 0.74 | 1.00 |
| 20k | 1 | 0.98 | 0.13 | 10,693 | 0.96 | 0.13 | 0.83 | PASS | 0.74 | 0.98 |
| 20k | 2 | 1.00 | 0.24 | 17,250 | 0.96 | 0.24 | 0.72 | PASS | 0.68 | 0.88 |
| 20k | 3 | 0.98 | 0.72 | 13,105 | 0.92 | 0.72 | 0.20 | **FAIL** | 0.66 | 1.00 |
| 20k | 4 | 0.77 | 0.34 | 17,473 | 0.86 | 0.34 | 0.52 | PASS | 0.82 | 0.98 |
| 20k | 5 | 0.94 | 0.23 | 7,668 | 0.92 | 0.23 | 0.69 | PASS | 0.59 | 0.78 |
| 20k | 6 | 0.90 | 0.50 | 7,972 | 0.99 | 0.50 | 0.49 | PASS | 0.80 | 0.86 |
| 60k | 1 | 0.98 | 0.13 | 12,469 | 0.84 | 0.13 | 0.71 | PASS | 0.77 | 0.99 |
| 60k | 2 | 1.00 | 0.24 | 19,602 | 0.73 | 0.24 | 0.49 | PASS | 0.36 | 0.98 |
| 60k | 3 | 0.98 | 0.72 | 17,994 | 0.77 | 0.72 | 0.05 | **FAIL** | 0.69 | 0.69 |

| Replay | Criterion 5 | Seeds passing 1–4 |
|---|---|---|
| **10k** | **PASS** | 6/6 (seeds 1–3 and 4–6) |
| 20k | FAIL | 5/6 |
| 60k (chase_sim's size) | FAIL | 2/3 |

## What this shows

- **The plugin learner adapts.** With a 10k replay it meets every D3 criterion on all six seeds,
  including seeds 4–6, which were never used to choose the setting. chase_sim's `OnlineSACLearner` did not
  meet criterion 5 in D3.
- **A large replay buffer slows adaptation.** With 60k, the buffer still holds about 18k pre-switch
  `rightturn` moves when the runner changes, and training keeps sampling them. Recovery takes 12–20k moves
  and two of three seeds are at 0.73–0.77 at the window. With 10k, the old data is gone about 10k moves
  after the switch, and recovery takes 4–14k moves.
- **Criterion 4 also depends on how good the snapshot already is on `corner`.** On seed 3 the 20k and 60k
  runs share the same pre-switch training: the buffer never filled before the switch, so the runs are
  identical until then. That snapshot already catches 0.72 on `corner`, so criterion 4 needs the learner
  to reach 0.93 or more. The 20k learner reached 0.92 and failed by one episode. The 10k run on that seed
  produced a different snapshot (0.38).
- **The default for `chase-sac` is therefore 10k** (see the NativeLearning host). This result comes from
  3 replay sizes; learning rate, update ratio and network size were not swept here.

## Open issue: stability after recovery

The D3 criteria end at 20,000 moves. In the following 20,000 moves, the 10k learner stays well above its
snapshot, but it is not steady. Seed 1 drops to 0.54 at one evaluation, and final values range from 0.88
to 1.00. 20k and 60k behave similarly (minimums 0.36–0.82). Defining a post-recovery stability criterion
before tuning against it is the next step. The D3 criteria should not be loosened to absorb this.

## Limits

- This is the native learner without Unreal. The same protocol through `UAdaptiveStepper` and
  `URLToolsNPCPolicy` is `NativeLearning.ChaseAdaptation` in the NativeLearning host, which has not been
  run yet.
- Wall times in the `.txt` files are not comparable: 15 runs shared the machine, and part of the run
  overlapped heavy throttling. Throughput is about 167 moves/s for a single run on an idle core.
