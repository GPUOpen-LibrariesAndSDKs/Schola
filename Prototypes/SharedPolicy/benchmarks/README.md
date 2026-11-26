# Shared policy architecture benchmarks

Status: A–E are implemented; the repeated comparison focuses on C/D.

This experiment compares the alternatives raised in AMD feedback while requiring
equivalent routed actions. Models use the same seeded weights and synthetic
observations; they do not train or evaluate NPC behaviour.

## D3 results

The portable matrix completed **144/144** trials without correctness failures.
On UE 5.8.4, macOS 26.6, Apple Silicon, and `NNERuntimeORTCpu`, all **48/48**
reports across 16 workloads passed correctness. C had lower request p50 in
**45/48** trials and all **16/16** workload medians, by **0.626–2.895
microseconds**. Three trials reversed direction, so this result does not establish
that C is universally faster. Memory, AMD hardware, trained policies, and gameplay
were not tested.

Reproduce the isolated Unreal matrix from the repository root after building
`Examples/BackgroundNPCs` and setting `UE_ENGINE_ROOT` to Unreal Engine 5.8.4:

```bash
Examples/BackgroundNPCs/Scripts/run_cd_isolated_matrix.sh
```

The runner writes ignored per-process reports and
`Saved/Automation/SharedPolicy/isolated/cd-isolated-summary.json`. It stops before
aggregation if Unreal fails, a report is missing, or correctness fails.

## Candidate definitions

Let `B` be the number of NPC rows and `K` the number of nonempty roles in a
request. The host knows one role label for every row before inference.
The synthetic roles are named `expert_0` and `expert_1` so the benchmark does
not assume a combat game or any particular gameplay meaning.

| Candidate | Execution for one mixed-role request | Model calls |
| --- | --- | --- |
| A: all heads | One backbone on all `B` rows; every head on all feature rows; host selects each row's action | 1 |
| B: runtime masks | Same backbone and head computation as A; dynamic role masks combine head outputs inside the graph | 1 |
| C: complete model per role | Host groups observations; each nonempty group runs its role's backbone-plus-head model | `K` |
| D: conditional graph | Host groups observations; the same ONNX graph runs once per group with one Boolean role condition and executes one head after its backbone | `K` |
| E: separate backbone and heads | One backbone on all `B` rows; host groups features; each nonempty group runs its head model | `1 + K` |
| F: mixed-row routing inside one graph | One graph groups features by role, executes heads on their subsets, and restores row order | Research extension; graph/runtime feasibility unproven |

A–E remain implemented feasibility evidence. Following AMD review, repeated
portable measurements focus on D against the current C baseline, with A retained
as an untimed correctness oracle. If Unreal NNE cannot execute D, the performance
fallback comparison is A/C/E. F is investigated only if the standard designs
cannot satisfy the target or if AMD specifically requires one mixed-role call.

C and D partition rows, so each NPC passes through a backbone once, although
separate role groups lose the efficiency of one larger backbone batch. C stores
one backbone copy per role; D stores one conditional model. E stores and executes
one backbone but adds intermediate tensor handoff and extra calls.

B is a negative control, not presumed optimization. Its mask must be a runtime
input so export-time constant folding cannot remove a head. Operator profiles
must establish which nodes actually execute. D is a required feasibility result:
successful selective execution and a precise unsupported-export/runtime result
are both reportable outcomes.

## Correctness contract

Every candidate returns float32 actions shaped `[B, 2]` in original NPC order.
For identical observations, role labels, and weights, selected actions must match
A within the documented numerical tolerance. Tests cover shuffled mixed roles,
homogeneous batches, singleton batches, invalid roles and shapes, empty-request
handling, exact call counts, and dynamic batch dimensions.

Correctness gates performance claims. A candidate that fails equivalence or
runtime validation is reported as unsupported, not benchmarked as a winner.

## Workload matrix

The initial controlled matrix uses two roles and sweeps:

- small, balanced, and expensive-head network profiles;
- batch sizes `1`, `16`, `64`, and `256`;
- homogeneous, approximately 50:50, and approximately 90:10 role mixtures;
- deterministic shuffled row order and identical observations per comparison.

Synthetic dimensions are stress profiles, not claims about AMD's production
network; real team dimensions should be added when known.

## Head-count scaling: A vs C

The two-role matrix above timed C and D, with A as an untimed oracle. The
scaling run (`scaling.py`) times **A and C** at `scaling_role_counts`
(2/4/8/16 heads) for `scaling_profiles` × `batch_sizes` × `scaling_mixes` ×
`trials`, one fresh worker per trial, with A as the correctness oracle.

- `equal` spreads rows across all K heads as evenly as integers allow, so every
  head is present once the batch has at least K rows.
- `homogeneous` sends every row to `expert_0`, so C makes one call while A still
  runs all K heads.
- Roles are `expert_0` … `expert_{K-1}` from the same seed. The backbone and the
  first heads keep identical weights as K grows, and K=2 matches the C/D models.
- Reported per cell: warmed complete-request p50 (median, min, and max over
  trials), model calls (A=1, C=heads present), and ONNX file bytes (A is one
  file, C is the sum of K files). File bytes are storage, not runtime memory.
- A cell says `C faster` or `A faster` only when every trial p50 of one
  architecture is below every trial p50 of the other; otherwise it is
  `inconclusive`. `c_faster_from_role_count` is the smallest K from which C is
  faster at that K and every larger K tested.

**Excluded:** D is a two-branch ONNX `If` (`ConditionalPolicyModel`) and does not
generalize past two roles, so it is not exported for scaling. B executes every
head like A and is not timed. E is exported in the 2/4/8-role tests to check that
it still matches A, but it is not timed.

```bash
uv run python -m benchmarks.scaling export
uv run python -m benchmarks.scaling run --output benchmarks/runs/scaling-results.json
uv run python -m benchmarks.scaling summarize --input benchmarks/runs/scaling-results.json
```

These results describe Python ONNX Runtime on the CPU of the machine that ran
them. They do not replace the Unreal C/D result above or measure Unreal NNE,
`USharedPolicyStepper`, gameplay, or memory.

## Measurement contract

All sessions are created once and reused. Initialization and first-run latency are
reported separately from warmed inference. Timed trials run without profiling,
use identical provider/thread/optimization settings, randomize candidate order
reproducibly, and retain raw samples.

Two latency views are required:

1. **Model/component time:** each `session.run` and model stage, exposing whether
   skipped head work reduces neural execution cost.
2. **Complete-request time:** grouping, copies, every model call, intermediate
   handoff, and restoration of original row order, exposing whether the saving
   survives routing overhead.

Required evidence includes correctness, call counts, cold/first-run latency,
warmed p50/p95 latency, rows per second, model artifact bytes, operator profiles,
and isolated process/session memory. Runtime memory is measured in fresh
inference-only workers without export or PyTorch imports and is reported
separately from file size.

The portable screen begins with ONNX Runtime `CPUExecutionProvider`, sequential
execution, fixed intra/inter-op thread counts, and recorded OS/CPU/package/model
metadata. Results apply only to measured environments. Python results do not
establish Unreal, GPU, AMD-hardware, or gameplay performance.

## Unreal NNE validation

`Examples/BackgroundNPCs` on Unreal Engine 5.8 is the integration host because it
already loads ONNX through NNE and exposes inference metrics. Unreal validation
is required after the portable screen:

1. Build and run the unchanged host to establish a working NNE baseline.
2. Probe import/model-creation/execution compatibility for C and D, recording
   unsupported control flow or opsets precisely.
3. If D is supported, compare it with C. Otherwise compare A/C/E under the same
   routing and output contract.
4. Measure isolated runs with persistent model instances, fixed workloads,
   warm-up, repeated trials, and raw evidence.

Required Unreal metrics are frame time/FPS, game-thread inference cost, model
execution time, decisions per second, action latency, calls, rows per call, and
process/session memory. The final recommendation must distinguish portable ORT
evidence from NNE evidence and name the platform/workload for every claim.

## Decision rule

No candidate wins merely because it returns the right output or zeros an unused
head. A recommendation requires:

- equivalent routed actions;
- profile evidence explaining what executed;
- repeated improvement larger than trial-to-trial variation in at least one
  declared target such as latency, throughput, or memory;
- routing/call/handoff costs included in the end-to-end result;
- Unreal NNE support and measurement for the proposed production path.

If no selective architecture beats A, the valid recommendation is to retain the
all-head model for the tested workload and document when larger heads or more
roles might change that decision.

## Files that exist now

| File | What it does |
| --- | --- |
| `README.md` | Explains what we will compare and how we will judge the results |
| `config.json` | Stores the network sizes, batch sizes, role mixtures, and timing settings |
| `models.py` | Defines the shared backbone and role-head pieces that A–E will reuse |
| `export_models.py` | Builds A–E and writes separate validated ONNX files |
| `runtime.py` | Reuses loaded ONNX sessions, groups roles where needed, and restores original row order |
| `benchmark.py` | Runs the correctness gate, warm-up, smoke timings, and raw JSON reporting |
| `matrix.py` | Plans the configured matrix and runs each C/D trial in a fresh sequential worker process |
| `scaling.py` | Exports A/C at 2/4/8/16 heads, times them in fresh workers, and summarizes the crossover |
| `profiling.py` | Captures ORT operator events and maps optimized nodes back to backbone/head stages |
| `tests/test_benchmark_models.py` | Checks that those shared pieces have the correct shapes and repeatable weights |
| `tests/test_benchmark_exports.py` | Checks the exported contracts and proves equivalent action values |
| `tests/test_benchmark_routing.py` | Sends the same mixed-role requests through A–E and checks actions and call counts |
| `tests/test_benchmark_measurement.py` | Checks workload generation, timing summaries, configuration failures, and smoke reports |
| `tests/test_benchmark_matrix.py` | Checks matrix expansion, worker isolation, failure handling, and aggregate reports |
| `tests/test_benchmark_profiling.py` | Proves A/B execute both heads while each D call executes only its selected branch |
| `tests/test_benchmark_scaling.py` | Checks that C and E match A at 2/4/8 heads, the call counts, the crossover rule, and a reduced scaling run |

The older model, exporter, demo, and their tests now live in `../legacy/`. We
keep them as a working baseline while building the benchmark.

## Files we will add in later increments

| Planned file | What it will do |
| --- | --- |
| `evidence/` | Keep the selected results, profiles, Unreal reports, and final analysis |

Generated ONNX files go in ignored `artifacts/`. Temporary results and profiles
go in ignored `runs/`. These generated files are not source code and will not be
committed.

From `Prototypes/SharedPolicy`, install the locked environment, run the tests,
export and validate all A–E artifacts, then validate the timing harness:

```bash
uv sync --python 3.12
uv run pytest -q
uv run python -m benchmarks.export_models
uv run python -m benchmarks.benchmark \
  --config benchmarks/config.json \
  --smoke \
  --output benchmarks/runs/smoke.json
```

Smoke results use only one short trial and must not be treated as the final
architecture comparison.

Operator profiling is a separate diagnostic run so its instrumentation does not
affect reported benchmark latency. Durations inside the profile report are
diagnostic metadata, not benchmark measurements:

```bash
uv run python -m benchmarks.benchmark \
  --config benchmarks/config.json \
  --profile \
  --output benchmarks/runs/profile-summary.json
```

The full portable comparison runs C and D under the same prepared workload in
one fresh process per profile/workload/trial. A runs once per worker as an
untimed correctness oracle. Workers execute sequentially so their measurements
do not compete for the CPU:

```bash
uv run python -m benchmarks.benchmark \
  --config benchmarks/config.json \
  --matrix \
  --architectures C D \
  --output benchmarks/runs/cpu-results.json
```

The aggregate report is written only after every worker passes. Generated raw
results remain ignored until selected evidence is deliberately curated.

## References

- [ONNX If operator](https://onnx.ai/onnx/operators/onnx__If.html): its condition
  contains one Boolean element, not one independent condition per row.
- [ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html):
  operator evidence is collected separately from reported timings.
- [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html):
  thread counts and execution mode affect comparisons.
- [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html): exported
  control flow and dynamic dimensions must be validated with installed versions.
