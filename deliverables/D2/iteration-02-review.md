# Team 20: The Hard Workers - Schola for Godot

## Iteration 02 - Review & Retrospect

 * When: FILL IN THE DATE WHEN YOU ACTUALLY HAD YOUR REVIEW MEETING
 * Where: PHYSICAL LOCATION AND/OR ONLINE

## Process - Reflection

#### Q1. What worked well

 * **Defining every user story's API before implementing it.** Each owner opened a pull request with only the interfaces for their story: US1 (#11), US2 (#7), US3 (#12), US4 (#6), US5 (#9), and US7 (#2). Abstract types such as `Agent`, `Policy`, and `Space` let dependent stories start in parallel instead of waiting for each other's code. When AMD changed its policy state design upstream, only the US7 contract had to change (#14, #15), and no other story was affected.
 * **Building a shared foundation and conventions before feature work.** The first pull request (#1) set up the build, the runtime and training add-ons, pinned dependencies, and CI that checks formatting, builds both extensions, and runs Catch2 and GdUnit4 tests. The development contract and PR template gave everyone the same rules, such as no exceptions, errors returned as `Status`, and engine-independent `core`. As a result, pull requests from seven people stayed consistent and reviewable.
 * **Splitting work by user story with one owner each.** Every member owns one story end to end, which matches the course's recommended structure and makes responsibility clear. Non-coding work also has owners: meeting minutes (Sanjay), partner communication (Vansh), and project management (Guneev).
 * **Recording partner and TA decisions in minutes.** Decisions such as targeting the latest ONNX Runtime, avoiding Git LFS in favour of a setup script, and treating GPU inference as a stretch goal are written down in `deliverables/team/minutes/`. Members could act on them without asking again, and the ONNX Runtime setup followed them directly.

#### Q2. What did not work well

 * **Dependencies between user stories made implementations wait on each other.** The API pull requests merged late in the iteration (October 14 and 15). Several implementations depend on code that is not on `main` yet: US7's Godot classes need US1's agent adapter, US2's spaces and points, US5's space resources, and the Variant conversion. As a result, US7's runtime extension cannot link yet, and no end-to-end demo was possible by the deadline.
 * **Changing the implementation language cost time.** D1 planned a C# architecture, and the team switched to a C++ GDExtension at the start of this iteration. That was the right call for performance and packaging, but the planning and early prototype had to be redone.
 * **Waiting on upstream AMD changes delayed one API.** US7 waited for AMD's multi-agent and recurrent state commit before finalizing its API. That commit landed mid-iteration, so the API had to be synced and reworked in two extra pull requests (#14, #15).

#### Q3(a). Planned changes

 * **Implement shared dependencies first, with internal deadlines.** The stories other code depends on (US2 spaces and points, US1's agent adapter, and the Variant conversion) will be implemented and merged first. Each will have a deadline a few days before the stories that depend on it.
 * **Merge smaller pieces earlier.** Implementations will be merged in tested increments instead of one large pull request at the end, and feature branches will merge `main` regularly. Progress then stays visible on `main` and integration problems appear early.

#### Q3(b). Integration & Next steps

The work is split by user story (US1-US7), with shared build and CI infrastructure owned by Shahyar. Components integrate through the engine-independent interfaces in `Godot/src/core/`, so each story builds against merged contracts and CI checks every pull request. Next, the stories that others depend on merge their implementations, and then the end-to-end environment, training, and inference workflow is assembled into the demo scene.

## Product - Review

#### Q4. How was your product demo?
 * How did you prepare your demo?
 * What did you manage to demo to your partner?
 * Did your partner accept the features? And were there change requests?
 * What were your learnings through this process? This can be either from a process and/or product perspective.
 * *This section will be marked very leniently so keep it brief and just make sure the points are addressed*
