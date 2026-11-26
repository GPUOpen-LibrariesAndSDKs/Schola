# Deliverable 2

The project description, adaptive tag and background NPC instructions, use
cases, developer setup, and team sections are in the [repository README at the D2 release](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/blob/D2/README.md).

Recorded adaptive NPC validation:

- [Native four-seed learner tests](evidence/adaptive-native-tests.txt)
- [Packaged Unreal learning report](evidence/adaptive-packaged-test.json)
- [Arena input and collision smoke report](evidence/adaptive-arena-smoke.json)
 [SAC learner evidence: baselines, thresholds, chase_sim transcripts and learning curves](../../Examples/AdaptiveTag/Evidence/SAC/README.md)

Recorded background NPC comparison (20,000 Mass agents, without and with the
background inference API):

- [Side-by-side recording](evidence/background-npc-side-by-side.mp4) and [still frame](evidence/background-npc-side-by-side.jpg)
- [Measurement summary](evidence/background-npc-comparison.md), with raw
  [synchronous](evidence/background-npc-sync.json) and
  [background](evidence/background-npc-background.json) reports

Recorded Shared Policy NPC evidence:

- [ONNX reference prototype](../../Prototypes/SharedPolicy/README.md) and [merged PR #5](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/pull/5)
- [Shared Policy project guide](https://docs.google.com/document/d/1QPLZo8vqCXEwkyKr5YaowDqcifTEvgNM/edit)
- [D2 presentation](https://docs.google.com/presentation/d/1oCS74S7zQehjbKYZ9ql0jUwtF_QFhtFMUqCipEH-tRk/edit)
- [PR #7: per-NPC head bindings](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/pull/7)
- [PR #13: Shared Policy Stepper](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/pull/13)
- [PR #16: RLtools SAC learner and interactive chase simulator](https://github.com/csc301-2026-f/9-AMD-Advanced-NPCs/pull/16)

The Python prototype verifies dynamic-batch ONNX export, one two-NPC ONNX
Runtime call, deterministic outputs, and ordered Combat/Support routing. The
Unreal-side work adds NPC-to-head configuration and a Stepper that calls the
Shared Policy interface with a batch and routes the returned actions. These
pieces are not yet connected to a real Unreal NNE policy, and the team has not
demonstrated a performance improvement.

Other required D2 documents:

- [Iteration 02 review and retrospective](iteration-02-review.md)
- [Individual contributions](Individual_Contributions_Deliverable2.md)

The review contains reflections from the online learning, background NPC, and
Shared Policy subteams. The individual-contributions document records the work
completed by each contributor represented in the current D2 snapshot. Final
partner feedback, release access, and any remaining team-member entries still
need confirmation.
