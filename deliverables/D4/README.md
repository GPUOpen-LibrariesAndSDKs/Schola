# Deliverable 4: Advanced NPCs for AMD Schola

- [Slideshow](index.html): open in a browser, offline, no server needed
- [Repository README](../../README.md): current workflows, NFRs, limitations
- [Maintainer guide](../../DEVELOPMENT.md)

## Present the slideshow

Open **index.html** in a modern browser and keep this folder in the repository so relative links work.

- Arrow keys, Page Up / Page Down or the on-screen buttons change slides. **O** opens the overview,
  **N** toggles speaker notes with each slide's time budget, **F** is fullscreen, **?** lists shortcuts.
- Notes appear in the same window. Close them before sharing the screen.
- Browser Print / Save as PDF exports one 16:9 page per slide (enable background graphics).

## Structure and timing

The limit is 20 minutes including questions. Use the section budgets below for rehearsal and reserve
about five minutes for questions. The Background NPC walkthrough combines design, demo, and access.

| Section (required length) | Slides | Budget | Owner |
| --- | --- | ---: | --- |
| Introduction (1–2 min) | [title](index.html#title), [problem](index.html#problem), [scope](index.html#scope) | 1:45 | Team |
| Demo · Adaptive NPC | [divider](index.html#demo), [before/after](index.html#demo-adaptive-change), [adaptive demo](index.html#demo-adaptive) | 1:45 | Adaptive |
| Background NPC | [team](index.html#background-team), [motivation](index.html#background-motivation), [recordings and benchmarks](index.html#demo-background), [ScholaMass architecture](index.html#tech-background), [CampusCrowd API calls](index.html#background-api), [Docker and development](index.html#background-development) | 2:30 | Max Xu |
| Demo · Shared Policy and Event | [shared policy](index.html#shared-policy-team), [event](index.html#demo-event) | 2:00 | Shared Policy + Event |
| Technical discussion (4–5 min) | [architecture](index.html#architecture), [adaptive design](index.html#tech-adaptive), [adaptive evidence](index.html#evidence-adaptive), [adaptive scaling](index.html#scaling-adaptive), [shared policy](index.html#tech-shared), [event](index.html#tech-event), [testing and CI](index.html#quality), [NFRs](index.html#nfr) | 4:00 | Team + subteams |
| Accessing the application (2–3 min, keep short) | [access and handoff](index.html#access) | 2:00 | Team |
| Key learnings (1–2 min) | [learnings](index.html#learnings) | 1:30 | Team |
| Questions | [questions](index.html#questions), backup: [adaptive per seed](index.html#backup-adaptive), [references](index.html#references) | ~5:00 | Team |

### Shared Policy live route

Present in deck order. Clementine introduces [Architecture A](index.html#shared-policy-nne-inference)
and the Lyra setup. Jaela presents the [A versus C benchmark](index.html#shared-policy-scaling),
the [merged Architecture C implementation](index.html#shared-policy-per-role),
[what the beacon demo verifies](index.html#shared-policy-beacon-walkthrough), and
the [stepper recording](index.html#demo-shared). Clementine then narrates the
[Lyra recording](index.html#shared-policy-statetree-demo). Close with the
[roadmap](index.html#shared-policy-roadmap).

For the five-minute team demo, budget about 1:15 for Shared Policy. Use one sentence for A
and jump to Jaela's benchmark. Use the detailed Lyra setup and StateTree slides for questions,
cue the beacon recording around 0:15 and pause on the log around 0:28. Show about 20 seconds of the Lyra clip and give the roadmap in one sentence.
Cover the architectural tradeoffs in the [technical section](index.html#tech-shared).

Serve this directory over HTTP for both recordings. To demonstrate the beacon live, open
`SharedPolicy.uproject`, load `LV_SharedPolicyD4`, and filter Unreal’s Output Log to `LogSharedPolicyD4`.

## What each subteam still has to add

Slides with a dashed **slot** are reserved. Each slot lists what the slide must show. Replace the slot with
content, keep the slide's `id` and its time budget, and update its speaker notes.

- **Shared Policy NPCs:** Check the shared-policy numbers on `nfr` and `quality`.
- **Event NPCs:** `demo-event`, `tech-event`, and the Event column on `architecture` if the design changed.
- **Team:** confirm release publication and add remaining subteam learnings and references.

## Rubric checklist

| Requirement | Where it is met |
| --- | --- |
| Separate what existed from what we built | `scope` table; the legend on `architecture` |
| Demo from the user's perspective, live, with a fallback | Each demo slide states the developer's steps; notes give the live plan and fallback |
| Architecture with diagrams, walked through; no code screenshots | `architecture`, `tech-adaptive`, `tech-background`, and `background-api` have diagrams and walkthrough notes |
| Tech stack and design justified | Key-decision takeaways on each design slide |
| Access and handoff, pointing to documents | `access` links the release, README and maintainer guide |
| Technical, multi-dimensional learnings | `learnings`: each card names its cause and effect chain |
| Finish on time | Use the section budgets and play only short excerpts from the Background NPC recordings |

## Adaptive NPC sources

The live adaptive demo is the chase demo in `Examples/NativeLearning` (see its README, "Chase demo",
for controls). It is shown live and is not a source for any number on the slides.

Every number on the adaptive slides comes from committed evidence:

- Adaptation: [CHASE-ADAPTATION.md](../../Tools/ScholaRLToolsNative/CHASE-ADAPTATION.md) and
  `Tools/ScholaRLToolsNative/Evidence/chase-adaptation/` (chart: replay 10k, seed 2).
- Scaling: [SCALING.md](../../Tools/ScholaRLToolsNative/SCALING.md) and
  `Tools/ScholaRLToolsNative/Evidence/scaling/` (chart: shared learner, 17-worker run).
- Design: [adaptive stepper guide](../../Docs/Sphinx/guides/adaptive_stepper.rst) and
  [native adaptive policy guide](../../Docs/Sphinx/guides/native_adaptive_policy.rst).

## Background NPC sources · Max Xu

- [ScholaMass](../../Source/ScholaMass) provides Unreal Mass processors and the background inference bridge.
- [CampusCrowd](../../Examples/CampusCrowd/README.md) covers the split-view campus day, learned turn policy,
  scaling preset, benchmark reports, and Docker build and package steps.
- The two 30-second recordings are embedded on the [CampusCrowd slide](index.html#demo-background).
  Play them side by side and narrate the four-student day and the 100,000-student first-class run.
