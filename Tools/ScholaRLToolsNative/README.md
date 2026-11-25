# ScholaRLToolsNative

Native, engine-free tests for checkpoint save/load in the Schola RLtools learner
(`Source/ScholaRLTools/Private/Learning/RLToolsLearner.{h,cpp}`). They run without Unreal on macOS and Linux.

The test lives here instead of under `Source/`. UnrealBuildTool compiles every `.cpp` under `Source/`,
so a `main()` there would break the Unreal build. The Unreal automation counterpart is
`Source/ScholaRLTools/Private/Test/RLToolsCheckpointTest.cpp` (`Schola.RLTools.Checkpoint.*`).

## Scaling benchmark

`run-scaling.sh` builds `ScalingBenchmark.cpp` and measures game-thread cost, shared-learner throughput and
per-learner memory at 1–200 NPCs. Results and guidance: [SCALING.md](SCALING.md).

## Run

```sh
Tools/ScholaRLToolsNative/build.sh            # builds into Saved/ScholaRLToolsNative (gitignored)
Tools/ScholaRLToolsNative/build.sh /tmp/out   # or a custom output directory
```

You need `clang++` or `g++` with C++20 (set `CXX` to pick one). The script compiles `CheckpointTests.cpp`
together with `RLToolsLearner.cpp` and runs the binary. It prints one `PASS`/`FAIL`/`SKIP <profile> <check>: <detail>`
line per check and a total, and exits non-zero if anything fails.

## Coverage (every compiled profile)

- Round trip for each mode (PolicyExport, TrainingState, FullResume): the fresh learner's deterministic outputs are bit-identical to the saver's.
- A learner restored from any mode keeps training and publishes a higher version.
- Restored state: replay size (FullResume vs TrainingState), update counters, and an empty PolicyExport.
- Version never goes backwards and Generation increments. Decisions made before a load are retired.
- Refusals leave the live learner unchanged: missing, empty, header-only, truncated or bit-flipped files, wrong magic, oversized payload declarations, other profile, other algorithm, other preprocessing, and a stopped learner.
- `ReadCheckpointInfo` metadata, atomic writes (no `.tmp` left behind, `IoError` on unwritable paths), and a concurrency smoke test.
