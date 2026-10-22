# Advanced NPCs for AMD Schola

## Iteration 02 - Review & Retrospect

- **When:**
- **Where:**

The online learning entries cover our adaptive NPC prototype. The other subteam sections are open for their own reflections and integration updates.

Format reference: [course D2 iteration review template](https://github.com/csc301-2026-f/deliverable-documents/blob/main/deliverables/D2/iteration-02-review.md).

## Process - Reflection

### Q1. What worked well

#### Online learning

1. **We isolated the technical uncertainty in a small playable experiment.** Building adaptive tag independently of Schola let us test native training inside Unreal before committing to a production integration. The two-route game made the actor's decisions, rewards, and changing probabilities easy to inspect and explain.
2. **We used repeatable tests to guide changes.** Tests across four random seeds checked that the same learner could adapt after the runner's bias reversed. A trial actor learning rate of 0.01 failed that reversal for one seed, so we rejected it and used 0.006, which passed all four. This kept a more dramatic demo setting from hiding a learning failure.
3. **We checked reproducibility from a fresh checkout.** After bringing the project into the repository, we built and packaged it without copying the old host's generated files or plugin junction. The fresh checkout passed the native tests, 26 arena checks, and nine packaged learning checks. This provides stronger setup evidence than testing only the original developer folder.

Supporting artifacts: [prototype setup](../../Examples/AdaptiveTag/README.md), [learning-rate experiment](../../Examples/AdaptiveTag/Evidence/FasterActor/README.md), and [recorded test reports](README.md).

#### Other subteams

| Subteam | What worked well and supporting evidence |
| --- | --- |
| | |
| | |

### Q2. What did not work well

#### Online learning

1. **We clarified the runtime goal too late.** Our earlier direction focused on an Unreal-to-Python request interface before we had settled whether continued learning should run inside the game. Revisiting that assumption changed the direction toward a native experiment. We need to confirm constraints such as a separate Python process and the expected learning behavior before expanding the integration layers around them.
2. **The first setup depended on local project structure.** The host project initially lived outside the repository and used a machine-specific Unreal path. A clone alone therefore did not contain the playable experiment. Moving the source and assets into `Examples/AdaptiveTag`, bundling the pinned dependency, and adding engine discovery addressed that gap. Validation on another person's machine still remains to be done.

#### Other subteams

| Subteam | What did not work well, its impact, and supporting evidence |
| --- | --- |
| | |
| | |

### Q3(a). Planned changes

#### Online learning

These are proposed process changes for the next iteration:

1. **Agree on the experiment's question and acceptance criteria before implementation.** Separate proving that native training runs from proving useful adaptation or Schola integration. For adaptation, compare learning and frozen actors against the same runner behavior changes rather than judging success from one demo recording.
2. **Keep experiments reproducible throughout development.** Store the required source, authored assets, dependency revision, and run instructions together. Check the documented setup from a fresh checkout before sharing a build, then ask another developer to verify it on their own machine.
3. **Review integration boundaries before expanding the prototype.** Identify what the learner owns and what Schola should supply before adding adapters. Keep experimental code changes separate from deliverable documentation so each review has a clear purpose.

#### Other subteams

| Subteam | Planned process change and why it should help |
| --- | --- |
| | |
| | |

### Q3(b). Integration & Next steps

#### Online learning

Within the prototype, we connected the arena's outcomes to the native learner and returned updated actor snapshots to gameplay. Schola API integration and integration with the other subteams' features remain future work.

#### Other subteams

| Subteam | Components integrated, division of work, and next steps |
| --- | --- |
| | |
| | |

## Product - Review

### Q4. How was your product demo?

#### Online learning

We shared a video demo of the adaptive tag game with the partner and are currently waiting to hear back.

**Partner feedback and change requests:**

**Follow-up actions:**

#### Other subteams

| Subteam | Demo, partner feedback, and follow-up |
| --- | --- |
| | |
| | |
