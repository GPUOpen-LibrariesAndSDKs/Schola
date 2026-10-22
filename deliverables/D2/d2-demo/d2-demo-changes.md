# D2 demo: what was changed and added

Branch: `Jimmy/d2-demo` (branched from `origin/main` at `75d4af3`). **Nothing is committed or pushed.**

There are two parts:

1. **Repo (`Godot/`)**: a real US2 implementation, covering core spaces/points, Godot value conversion, and the Inspector resources.
2. **Godot demo project** (`C:/Users/Bossy/OneDrive/Documents/d-2-demo`, outside the repo): a grid-world scene, an editor plugin and in-game UI, and a Python trainer. US2 is real there; everything else is a deliberately basic stand-in.

---

## 1. Repo changes (US2)

### Core: `src/core/spaces/` (new `.cpp` files behind the merged headers)

| File | What it does |
| --- | --- |
| `space.cpp` | `check_definition`, `validate`, `make_point`, `get_flattened_size`, `to_string`, `==`/`!=`, plus `DictSpace::find` and the Dict key sort in `Space(DictSpace)`. All five kinds are covered: Box, Discrete, MultiDiscrete, MultiBinary and Dict. |
| `point.cpp` | Point constructors, `get_kind`, `to_string`, `==`/`!=`, `DictPoint::find`. |

Behaviour, as the header comments describe it:

- `check_definition()` returns `INVALID_DATA` for:
  - an empty or non-positive Box shape
  - Box bounds whose size doesn't match the shape
  - a NaN bound
  - low > high
  - n ≤ 0, or an empty or non-positive nvec
  - a duplicate Dict key

  Nested Dict errors are prefixed with the key path, e.g. `at 'motor.speed': ...`.
- `validate()` returns `INVALID_ARGUMENT` and names the failing index, e.g. `Box value at index 1 is 7.5, outside [0, 5]` or `Discrete value 4 is outside [0, 4)`. It also rejects a wrong kind, a wrong size, NaN, a MultiBinary value other than 0 or 1, and Dict keys that are missing, extra, duplicated or unsorted.
- Unbounded Box sides are ±infinity, and `validate` accepts any value on them. `make_point()` clamps 0 into each Box element's bounds, so an unbounded side stays 0.
- `get_flattened_size()` follows the **merged header** (Unreal layout): Discrete = `n` (one-hot) and MultiDiscrete = `sum(nvec)`. The 1-vs-n question with US7 is still open.

### Interop: `src/interop/`

| File | What it does |
| --- | --- |
| `variant_conversion.cpp` (new, header already merged) | `variant_to_point` / `point_to_variant`. The mappings are `PackedFloat32Array` ↔ Box, `int` ↔ Discrete, `PackedInt32Array` ↔ MultiDiscrete, `PackedByteArray` ↔ MultiBinary and `Dictionary` ↔ Dict. Types are strict: a float for a Discrete space, or an `Array` for a Box, gives a clear error such as `expected int for a Discrete space but got float`. A wrong size is also reported here. Bounds are checked by `validate()`. |
| `space_dictionary.h/.cpp` (**new, not in the merged API**) | `space_to_dictionary(Space)` builds the Dictionary layout sent to Python, e.g. `{"type":"Box","low":[...],"high":[...],"shape":[4],"dtype":"float32"}`. An infinite bound is written as `null` because JSON has no infinity. This is a stand-in until US3 picks the wire format. |

### Bindings: `src/bindings/spaces/`

| File | What it does |
| --- | --- |
| `schola_space.cpp` (new) | Implements `get_validation_error(value)`, which checks the definition, then converts the value, then validates it. |
| `schola_space.h` (**modified**) | Adds five bound methods: `get_definition_error()` (from the mismatch table in `us2_d2_todo.md`; for configuration warnings), `to_dictionary()`, `get_flattened_size()`, `make_default_value()` and `describe()`. |
| `schola_box_space.cpp` (new) | Properties `low`, `high`, `shape` and `dtype` (an enum hint in `DType` order). An empty `shape` becomes `{size of whichever bound is set}`. An **empty `low` or `high` means unbounded on that side**. A non-empty bound with the wrong size is passed through unchanged, so `check_definition` reports it instead of it being silently padded. Setters call `emit_changed()`. |
| `schola_discrete_space.cpp` (new) | Property `n`. |
| `src/bindings/register_types.cpp` (modified) | Registers `ScholaSpace` (abstract), `ScholaBoxSpace` and `ScholaDiscreteSpace`. |

MultiDiscrete, MultiBinary and Dict **Inspector resources** were left alone (after D2 per the plan). The core and interop code already handles those kinds.

### Shared infra: `Godot/SConstruct` (one-line change)

- Added `src/interop` to the `schola` runtime sources. Nothing compiled `variant_conversion.cpp` before.
- The development contract says shared-infra changes go in their own commit/PR, so **commit this separately**.

### Tests: `tests/unit/spaces_test.cpp` (new)

There are 10 Catch2 test cases (92 assertions). They include the three the plan asked for:

- an omitted Box bound means unbounded, not zero
- an invalid point's error names the failing index
- a malformed definition is rejected

They also cover Dict key sorting and validation, `make_point`, flattened sizes, equality and `to_string`.

```
Godot\build\tests\schola_core_tests.exe   ->  All tests passed (92 assertions in 10 test cases)
```

`variant_conversion` and the bindings need Godot, so they have no Catch2 tests. The demo exercises them instead: see the **Test US2 validation** button below.

### Not done in the repo

- No `ScholaAgent` / `ScholaEnvironment` implementation. That is US5/US1 work, so the demo uses GDScript stand-ins.
- No protobuf conversion (`transport/grpc/space_conversion.h`). It depends on the US3 decision.
- No round-trip tests for interaction definitions or agent states.

---

## 2. The demo project (`OneDrive/Documents/d-2-demo`)

GDScript, Godot 4.7.2. It is not inside the repo, so copy it in if it should be committed.

```
d-2-demo/
├── project.godot                     main scene, 1280x760 window, schola_d2 plugin enabled
├── addons/
│   ├── schola/                       the GDExtension built from this branch (real US2)
│   │   ├── schola.gdextension
│   │   └── bin/windows/schola.windows.template_debug.x86_64.dll
│   └── schola_d2/                    Godot plugin: editor dock + in-game UI + basic stand-ins
│       ├── plugin.cfg, plugin.gd     EditorPlugin, adds the "Schola" dock
│       ├── editor/setup_dock.gd      environment setup + US2 space checks while editing
│       ├── nodes/schola_demo_environment.gd   ScholaDemoEnvironment: stand-in for US5 ScholaEnvironment + US1 callbacks (Node)
│       ├── nodes/schola_demo_agent.gd         ScholaDemoAgent: stand-in for US5 ScholaAgent (Node)
│       ├── icons/                    editor icons for the two Schola nodes
│       ├── runtime/demo_runner.gd    training client connection (US3 stand-in) + mode switching
│       ├── runtime/saved_policy.gd   saved-policy inference (US7 stand-in)
│       └── ui/schola_hud.gd, ui/reward_graph.gd   in-game panels
├── demo/                             3D arena, built from scene nodes
│   ├── main.tscn                     the scene (tree below)
│   ├── grid_environment.gd           GridEnvironment (extends ScholaDemoEnvironment): start cells, rewards,
│   │                                 _reset_episode / _get_reward / _is_terminated / _get_info
│   ├── grid_agent.gd                 GridAgent (extends ScholaDemoAgent): _observe() -> [x, y, goal dx, goal dy],
│   │                                 _act(int) -> player.move(direction)
│   ├── grid_arena.gd                 GridArena: the level (no RL logic); reads goal/hazard cells from GridMarker
│   │                                 children, banner + light effects
│   ├── grid_player.gd                GridPlayer: the body; knows its cell, hops and turns (no RL logic)
│   ├── grid_floor.gd                 "Floor" node: builds the checkerboard tiles from grid_size / cell_size (not saved)
│   ├── grid_marker.gd                GridMarker (kind Goal/Hazard): the cell comes from the node's position
│   └── goal.tscn, hazard.tscn        marker scenes (mesh nodes; the goal has a spinning gem via AnimationPlayer)
├── python/trainer.py                 local tabular Q-learning client (gymnasium + numpy)
└── policies/                         trainer writes q_policy.json here
```

### Scene tree (how the Schola nodes are used)

```
Main
├── Title, Help                       labels
├── ArenaView (SubViewportContainer)
│   └── SubViewport
│       ├── WorldEnvironment, Sun, Camera3D
│       └── ScholaEnvironment         GridEnvironment: rewards, step limit, start cells (Inspector)
│           └── Arena                 GridArena: the level
│               ├── Floor, Goal, Hazard1..4, Banner, OutcomeLight
│               └── Player            GridPlayer: the body (capsule + eyes)
│                   ├── Body, EyeLeft, EyeRight
│                   └── ScholaAgent   GridAgent: observation_space = ScholaBoxSpace, action_space = ScholaDiscreteSpace
├── Runner                            ScholaDemoRunner: environment = ScholaEnvironment
└── HUD                               ScholaHud: runner = Runner
```

- This follows the US5/US1 API:
  - `ScholaEnvironment` is an ancestor of the game objects.
  - Every agent node below it takes part in its episodes (`get_agents()` walks the subtree).
  - Each `ScholaAgent` is a config node attached to the object it controls.
- The game nodes (`Arena`, `Player`) have no RL logic. The two Schola nodes hold all of it, using the same callbacks as `ScholaEnvironment` (`_reset_episode`, `_get_reward`, `_is_terminated`, `_is_truncated`, `_get_info`) and `ScholaAgent` (`_observe`, `_act`).
- Step order follows US1: `_act`, then `_observe`, then the reward and end checks.
- The arena's banner and light react through scene signal connections (`episode_started` → `hide_outcome`, `episode_ended` → `show_outcome`).
- Both Schola nodes have their own editor icons and show up under those names in **Create New Node**.
- The HUD and the editor dock show the node path (`ScholaEnvironment › Arena/Player/ScholaAgent`). They read the environment's settings through `get_config()`, so a different environment subclass shows its own settings.

### What's real and what's a stand-in

| Piece | Status | Where |
| --- | --- | --- |
| Observation and action spaces, definition checks, value validation, Godot ↔ core conversion, space → Python Dictionary | **Real US2 C++** through the GDExtension | `ScholaBoxSpace`/`ScholaDiscreteSpace` in `main.tscn`; every `observe` and `step` goes through `get_validation_error` |
| Agent config node with `_observe` / `_act` and configuration warnings | Basic GDScript (US5 stand-in) | `ScholaDemoAgent` → `GridAgent` on `Player/ScholaAgent` (warnings use the real `get_definition_error`) |
| Environment node, reset/step lifecycle, rewards, terminated/truncated | Basic GDScript (US5/US1/US4 stand-in) | `ScholaDemoEnvironment` → `GridEnvironment` on `ScholaEnvironment` |
| Godot ↔ Python transport | JSON lines over a local TCP socket (US3 stand-in, **not gRPC**) | `demo_runner.gd` ↔ `trainer.py` |
| Trainer | Tabular Q-learning (**not SB3/RLlib**) | `trainer.py` |
| Saved-policy inference in Godot | Q-table JSON, greedy argmax in GDScript (US7 stand-in, **not ONNX**) | `saved_policy.gd` |

### Protocol (Python drives, like a gymnasium env)

```
Godot -> Python  {"type":"hello", "observation_space":{...}, "action_space":{...}, "config":{...}}
Python -> Godot  {"type":"reset", "seed":N}
Godot -> Python  {"type":"reset_result", "observation":[...]}
Python -> Godot  {"type":"step", "action":2}
Godot -> Python  {"type":"step_result", "observation":[...], "reward":r, "terminated":b, "truncated":b, "info":{"outcome":...}}
Python -> Godot  {"type":"progress", ...} / {"type":"saved", "path":...}
Godot -> Python  {"type":"error", "message":...}   when US2 rejects an action or observation
Godot -> Python  {"type":"stop"}                   Stop/Reset Run: trainer saves and exits
```

- Python rebuilds the spaces as `gymnasium.spaces.Box` / `Discrete` (`space_from_dict`).
- It checks every observation with `observation_space.contains()`, which gives a round-trip check on the Python side.
- The saved policy file stores the space Dictionaries. Godot refuses to load a policy whose spaces don't match the agent's.

### The UI (mapped to `what_I_want_for_d2.md`)

- **Scene (3D, built from nodes):**
  - a 6x6 floor with an agent, a goal and 4 hazards, seen from an angled `Camera3D`
  - the goal and hazards are instanced scenes; **drag them in the editor to change the level**
    - grid cell (x, y) is the floor square at world (x, z), with 1 m cells
    - `GridArena` warns if there isn't exactly one goal or a marker sits off the grid
  - an episode step limit, which ends the episode as a timeout
  - the agent hops between cells and turns to face each move ("up" is away from the camera)
  - a billboard `Label3D` banner and a coloured `OmniLight3D` flash for success, failure or timeout
  - a "pop" scale animation when the agent resets
- **Mode indicator:** a coloured box at the top of the panel:
  - MANUAL: keyboard
  - WAITING
  - TRAINING: "Training client controls agent"
  - STOPPING
  - INFERENCE: "Saved policy controls agent"
- **Training controls:**
  - Start Training, Stop / Reset Run, Load Saved Policy
  - connection state: disconnected / waiting / training / policy loaded
  - a speed slider (1 to 1000 steps/s)
  - **Test US2 validation**, which logs real US2 rejections for action 7, a float action, a 3-value observation, x = 9, NaN, and an `Array` instead of a `PackedFloat32Array`
- **Environment setup:** the observation and action spaces (from `describe()`), the goal reward, failure penalty, step penalty and episode step limit, plus any configuration errors.
- **Live telemetry:**
  - episode, step and cumulative reward
  - the last observation
  - the action and its source (Python trainer / saved policy / keyboard)
  - the terminal state
  - trainer progress (episode, epsilon, state count)
- **Training progress:** reward bars per episode, coloured by source, with a rolling-average line and a log next to it.
- **Editor dock "Schola"** (the plugin): each environment's reward settings, plus each agent space's ✓/✗ from `check_definition`, its flattened size, its default value and the Dictionary sent to Python. It refreshes every second while you edit the resources in the Inspector.

### How to run

1. Build the extension (already built and copied into the demo):
   ```
   scons -C Godot platform=windows target=template_debug
   copy Godot\build\windows\schola.windows.template_debug.x86_64.dll  d-2-demo\addons\schola\bin\windows\
   ```
   This machine built it with **TDM-GCC (MinGW)**, not MSVC: SCons picked g++ even though the build ran after `vcvars64.bat`. It loads fine in Godot.
2. Open `d-2-demo` in Godot 4.7.2 and press F5.
3. Press **Start Training**. Godot listens on `127.0.0.1:5757` and launches the trainer with `Runner.python_executable`.
   - That is set in `main.tscn` to `C:/Users/Bossy/20-AMD-Schola/.venv/Scripts/python.exe`, which has gymnasium and numpy. Change it on other machines.
   - Or untick `auto_launch_trainer` and run `python python/trainer.py --port 5757` yourself.
4. Watch it train (raise the speed slider to finish faster). The trainer saves `policies/q_policy.json` and disconnects.
5. Press **Load Saved Policy**. The agent now runs from the saved policy, with no Python process.

### Verified

- `schola_core_tests.exe`: all 92 assertions pass.
- A headless Godot import and editor load (with the plugin and dock) produce no script errors.
- A headless end-to-end run with the real Python trainer:
  - Training: 300 episodes, 36 states learned.
  - Reload: the saved policy reached the goal in **30/30** episodes.
- A windowed run with screenshots during training and inference: the 3D arena sits in the top-left, with the HUD panels on the right and bottom, at 1280x760.
- The arrow keys still move the agent in the 3D scene (an injected `ui_right` moved it one cell).
- A headless editor load of `main.tscn` (with the `@tool` floor, markers and config warnings) shows no errors.
- **Not checked by eye:** how the editor dock looks inside the editor. It loads without errors.

### Open points / follow-ups

- Confirm the Discrete flattened size (1 vs n) with US7 before ONNX. The current code uses n, per the merged header.
- Agree with US5 on adding `get_definition_error()` (and the other new `ScholaSpace` methods) to the API.
- Ask US3 whether `space_to_dictionary` (JSON layout) or protobuf conversion is the D2 boundary.
- Integer-dtype Box with a non-integral value (e.g. 3.5) is still accepted. This is still undecided.
- Put the `SConstruct` change in its own commit.
