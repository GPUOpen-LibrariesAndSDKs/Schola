# Advanced NPCs for AMD Schola

## Iteration 02 - Review & Retrospect

- **When: 10/22/2026 8pm**
- **Where: Online**

The online learning entries cover our adaptive NPC prototype. The Background NPCs rows cover batched background inference (Max, the only member of that subteam), and then Event NPCs.

Format reference: [course D2 iteration review template](https://github.com/csc301-2026-f/deliverable-documents/blob/main/deliverables/D2/iteration-02-review.md).

## Process - Reflection

### Q1. What worked well

#### Online learning

1. **We isolated the technical uncertainty in a small playable experiment.** Building adaptive tag independently of Schola let us test native training inside Unreal before committing to a production integration. The two-route game made the actor's decisions, rewards, and changing probabilities easy to inspect and explain.
2. **We used repeatable tests to guide changes.** Tests across four random seeds checked that the same learner could adapt after the runner's bias reversed. A trial actor learning rate of 0.01 failed that reversal for one seed, so we rejected it and used 0.006, which passed all four. This kept a more dramatic demo setting from hiding a learning failure.
3. **We checked reproducibility from a fresh checkout.** After bringing the project into the repository, we built and packaged it without copying the old host's generated files or plugin junction. The fresh checkout passed the native tests, 26 arena checks, and nine packaged learning checks. This provides stronger setup evidence than testing only the original developer folder.
4. **We compared training options before committing.** We checked Unreal NNE, Learning Agents, LibTorch, ONNX Runtime training, mlpack and hand-written C++ against one requirement: learn inside Unreal with no Python. RLtools was chosen because it ships SAC as header-only C++ and a teammate's prototype showed it builds in Unreal.
5. **We fixed pass criteria before training.** Random and greedy-pursuit baselines set the thresholds (catch rate of at least 0.80 and at most 100 steps to catch) before any SAC run. All 3 seeds passed in Unreal.

Supporting artifacts: [prototype setup](../../Examples/AdaptiveTag/README.md), [learning-rate experiment](../../Examples/AdaptiveTag/Evidence/FasterActor/README.md), and [recorded test reports](README.md).

#### Other subteams

| Subteam | What worked well and supporting evidence |
| --- | --- |
| Background NPCs | **Decisions were driven by measurements and written success gates.** Before implementing, Max wrote a literature review, a plan, and pass/fail gates, then benchmarked the options. Batching a trained 3DBall policy at 64 rows gave 10.5× the throughput of single calls, so batching was adopted. No other dispatch strategy beat the shared queue by the required 10%, so it stayed the default. INT8, GPU execution, and deduplication were not built because the measurements never showed a need for them. **Correctness tests came before performance work.** Two pipelined-stepper defects were reproduced in failing tests before being fixed. The scheduler's fault-injection test sends 100,000 requests per dispatch strategy, and all 43 background and Mass automation tests pass. **The final comparison changes only one variable.** The same 20,000-agent Mass crowd ran at 10.4 FPS with per-agent synchronous inference and at the 60 FPS cap with the API, at the same ~200,000 decisions/s. Evidence: [PR #6](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/pull/6), [comparison example](../../Examples/BackgroundNPCs/README.md), [measurements](evidence/background-npc-comparison.md), [recording](evidence/background-npc-side-by-side.mp4). |
| Shared Policy NPCs | The subteam produced two adjacent but not yet connected layers: Jaela’s tested ONNX reference contract and Clementine’s Unreal configuration and Stepper work. The prototype gave the team concrete batching and routing behaviour to discuss. However, the team committed too early to a shared-backbone and specialized-head architecture. AMD asked the team to investigate other batching and inference options before finalizing the design. |
| Event NPCs| We successfully implemented and demonstrated the basic event-driven inference workflow. By reusing Scholas IStepper interface and agent/policy architecture, we could focus on triggering inference through gameplay events. A local Unreal Engine Blueprint prototype simulated a turn-ending event using a three-second delay before invoking TriggerInference(). Debug messages confirmed successful initialization and execution of the Observe(), Policy Think(), and Act() methods. This provided an initial proof of concept for event-triggered NPC decisions. **Evidence:** [Stepper Initialization](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/blob/e3bff40/deliverables/D2/evidence/event-npc-initialization.png), [Event-triggered Inference Execution](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/blob/e3bff40/deliverables/D2/evidence/event-npc-inference-execution.png)|

### Q2. What did not work well

#### Online learning

1. **We clarified the runtime goal too late.** Our earlier direction focused on an Unreal-to-Python request interface before we had settled whether continued learning should run inside the game. Revisiting that assumption changed the direction toward a native experiment. We need to confirm constraints such as a separate Python process and the expected learning behavior before expanding the integration layers around them.
2. **The first setup depended on local project structure.** The host project initially lived outside the repository and used a machine-specific Unreal path. A clone alone therefore did not contain the playable experiment. Moving the source and assets into `Examples/AdaptiveTag`, bundling the pinned dependency, and adding engine discovery addressed that gap. Validation on another person's machine still remains to be done.
3. **Our first chase task could not show adaptation.** The chaser was faster than the runner, so greedy pursuit caught the runner every time on every pattern. Chase v2 uses equal speeds, and greedy pursuit now catches 40% on circle. A circle-trained policy still catches every square start (one seed), so adaptation is not yet demonstrated.
4. **Results varied run to run when updating every 50 experiences.** The test harness raced the learner's snapshot swap. The v2 simulator waits for the learner's backlog, so it repeats exactly.
5. **Schola did not build unmodified for us on UE 5.7.** ScholaMass depends on MassCore, which UE 5.7 does not have. Our test host excludes it, and the shared-plugin fix is for the team.

#### Other subteams

| Subteam | What did not work well, its impact, and supporting evidence |
| --- | --- |
| Background NPCs | **Validation stayed on one machine.** Everything was built and measured on Linux with UE 5.8.3, while the team develops on Windows with UE 5.7. The benchmark scripts and research documents remain in a local commit rather than on `main`, so teammates cannot yet reproduce them. **Some integration problems only appeared at the end.** Packaging the comparison example showed that editor-only test code on `main` (`HasMetaData`) broke every game build using Schola. It also revealed that a Mass `MaxRowsPerFrame` above the scheduler's `MaxPendingRows` silently rejects every request. Both are fixed or documented in the [README limitations](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/blob/D2/README.md#background-npc-implementation). |
| Event NPCs| Due to compatibility issues, Event NPCs (and only this NPC) cannot run outside the local working environment. Mainly with Unreal having compilation issues with the MassCore module. Additionally, the implementation currently requires gameplay events to be connected manually to TriggerInference(). Automatic event subscriptions remain unimplemented.|

### Q3(a). Planned changes

#### Online learning

These are proposed process changes for the next iteration:

1. **Agree on the experiment's question and acceptance criteria before implementation.** Separate proving that native training runs from proving useful adaptation or Schola integration. For adaptation, compare learning and frozen actors against the same runner behavior changes rather than judging success from one demo recording.
2. **Keep experiments reproducible throughout development.** Store the required source, authored assets, dependency revision, and run instructions together. Check the documented setup from a fresh checkout before sharing a build, then ask another developer to verify it on their own machine.
3. **Review integration boundaries before expanding the prototype.** Identify what the learner owns and what Schola should supply before adding adapters. Keep experimental code changes separate from deliverable documentation so each review has a clear purpose.
4. **Check that a simple baseline cannot already solve the task before training, and agree a merge order for files two people edit.** The learner header is changed on two branches.

#### Other subteams

| Subteam | Planned process change and why it should help |
| --- | --- |
| Background NPCs | **Size measurement plans to the time budget from the start** and rank the gates, so the most important ones get full-length runs first instead of every gate getting reduced runs at the end. **Test on the team's platform early:** build and package on Windows with UE 5.7 and ask a teammate to run the example, so platform problems surface before the deliverable. **Package an end-to-end example early and keep using it as an integration test.** Problems in game builds and configuration mistakes that unit tests miss should then appear within days. The `MaxRowsPerFrame` mistake should also become a validation warning in the API. |
| Event NPCs| Establish a consistent Unreal Engine development environment. Coordinate with other subteams to agree on a supported Unreal version and verify compatibility before integrating new functionality.|

### Q3(b). Integration & Next steps

#### Online learning

Within the prototype, we connected the arena's outcomes to the native learner and returned updated actor snapshots to gameplay. Schola API integration is written (`UOnlineSACPolicy`) but not yet validated, and integration with the other subteams' features remains future work.

The learner core (`ChaseSession`) has no Unreal or printing code, so an Unreal actor can call it each frame and draw its state; the console is only a front end. `UOnlineSACPolicy` plugs the learner into the existing `UAdaptiveNPCPolicy` ticket API. Chloe's checkpoint save and load branch builds on the same learner and still needs merging. Proposed next steps: a harder pattern that really needs adaptation, an adaptive-versus-frozen comparison on 3 seeds, and running the learner in the tag arena.

#### Other subteams

| Subteam | Components integrated, division of work, and next steps |
| --- | --- |
| Background NPCs | **Integrated:** the batching scheduler sits behind Schola's existing policy interface (`UNPCInferencePolicy` with `Paradigm = Background`). It works with the NNE ORT/BasicCpu backends, Actor steppers (`UBackgroundStepper`), Blueprint submit/poll calls, StateTree's existing policy task, and Mass Entity (`ScholaMass`). **Division of work:** Max did the research, implementation, tests, benchmarks, comparison example, and documentation. **Next steps:** check the example on Windows/UE 5.7; fix Actor pool growth by preallocating the request pool, then re-measure; run the comparison with a trained policy. With the other subteams, explore whether the shared-policy stepper and the adaptive actor's decisions can be served through background batching. |
| Event NPCs| **Next steps:**. **Ensure** this is compatible with the remaining environments of the other NPCs for D3. Extend the prototype to support configurable subscriptions to one or more gameplay events, implement event unsubscriptions, and establish a reproducible Unreal demonstration. Investigate potential integration with Background NPC inference for non-blocking event-triggered decisions. |

## Product - Review

### Q4. How was your product demo?

#### Online learning

We shared a video demo of the adaptive tag game with the partner and are currently waiting to hear back.

**Partner feedback and change requests:** None received as of 23 October 2026.

**Follow-up actions:** Ask the partner which behaviors matter most for their research, and whether the adaptive tag game or the SAC chase simulator is the better base for the next iteration.

#### Other subteams

| Subteam | Demo, partner feedback, and follow-up |
| --- | --- |
| Background NPCs | The [side-by-side recording](evidence/background-npc-side-by-side.mp4) was shared with the partner along with the team's adaptive tag video. It shows the same 20,000-agent Mass crowd without and with the background API, with live frame-rate, frame-time, and inference statistics on screen. We are waiting for the partner's feedback. **Follow-up:** record the feedback and any change requests here. Ask the partner which agent counts, decision rates, and policies matter most for their research, so the next measurements match their use. |
| Event NPCs| Explained implementation and event-diagrams. Show the agent mesh execute upon policy. We are waiting for the partner's feedback.|
