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

The limit is 20 minutes including questions. Slide budgets total about 15 minutes, leaving about 5 for
questions. Each section follows the D4 presentation format.

| Section (required length) | Slides | Budget | Owner |
| --- | --- | ---: | --- |
| Introduction (1–2 min) | [title](index.html#title), [problem](index.html#problem), [scope](index.html#scope) | 1:45 | Team |
| Demo (4–5 min) | [divider](index.html#demo), [adaptive](index.html#demo-adaptive), [background](index.html#demo-background), [shared policy](index.html#demo-shared), [event](index.html#demo-event) | 4:45 | Each subteam |
| Technical discussion (4–5 min) | [architecture](index.html#architecture), [adaptive design](index.html#tech-adaptive), [adaptive evidence](index.html#evidence-adaptive), [adaptive scaling](index.html#scaling-adaptive), [background](index.html#tech-background), [shared policy](index.html#tech-shared), [event](index.html#tech-event), [testing and CI](index.html#quality), [NFRs](index.html#nfr) | 5:00 | Team + subteams |
| Accessing the application (2–3 min, keep short) | [access and handoff](index.html#access) | 2:00 | Team |
| Key learnings (1–2 min) | [learnings](index.html#learnings) | 1:30 | Team |
| Questions | [questions](index.html#questions), backup: [adaptive per seed](index.html#backup-adaptive), [references](index.html#references) | ~5:00 | Team |

## What each subteam still has to add

Slides with a dashed **slot** are reserved. Each slot lists what the slide must show. Replace the slot with
content, keep the slide's `id` and its time budget, and update its speaker notes.

- **Background NPCs:** `demo-background`, `tech-background`. Check the background numbers on `nfr` and `quality`.
- **Shared Policy NPCs:** `demo-shared`, `tech-shared`. Check the shared-policy numbers on `nfr` and `quality`.
- **Event NPCs:** `demo-event`, `tech-event`, and the Event column on `architecture` if the design changed.
- **Team:** the deployment box on `quality` once the dev/release workflow is live; one learning per
  subteam on `learnings`; references on `references`.

## Rubric checklist

| Requirement | Where it is met |
| --- | --- |
| Separate what existed from what we built | `scope` table; the legend on `architecture` |
| Demo from the user's perspective, live, with a fallback | Each demo slide states the developer's steps; notes give the live plan and fallback |
| Architecture with diagrams, walked through; no code screenshots | `architecture` and `tech-adaptive` are diagrams; notes give the walk-through order |
| Tech stack and design justified | Key-decision takeaways on each design slide |
| Access and handoff, pointing to documents | `access` links the release, README and maintainer guide |
| Technical, multi-dimensional learnings | `learnings`: each card names its cause and effect chain |
| Finish on time | Budgets in the notes total about 15 minutes |

## Adaptive NPC sources

Every number on the adaptive slides comes from committed evidence:

- Adaptation: [CHASE-ADAPTATION.md](../../Tools/ScholaRLToolsNative/CHASE-ADAPTATION.md) and
  `Tools/ScholaRLToolsNative/Evidence/chase-adaptation/` (chart: replay 10k, seed 2).
- Scaling: [SCALING.md](../../Tools/ScholaRLToolsNative/SCALING.md) and
  `Tools/ScholaRLToolsNative/Evidence/scaling/` (chart: shared learner, 17-worker run).
- Design: [adaptive stepper guide](../../Docs/Sphinx/guides/adaptive_stepper.rst) and
  [native adaptive policy guide](../../Docs/Sphinx/guides/native_adaptive_policy.rst).
