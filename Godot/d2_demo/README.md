# D2 demo

A 3D grid world showing the D2 workflow end to end:
1. Godot sends observations and rewards to a local Python trainer.
2. Godot applies the trainer's actions to the agent.
3. The trained policy is reloaded and runs in Godot with Python closed.

The spaces are the real US2 resources (`ScholaBoxSpace`, `ScholaDiscreteSpace`) from the Schola GDExtension. The other stories are basic GDScript stand-ins in `addons/schola_d2`:

| Stand-in | For |
| --- | --- |
| `ScholaDemoEnvironment`, `ScholaDemoAgent` | US1, US4, US5 |
| JSON over TCP | US3 |
| saved Q-table | US7 |

## Requirements

- Godot 4.7.2
- Python with `gymnasium` and `numpy` (the repo's `.venv` has both)

## Run

1. Build the extension and copy it into the demo:
   ```sh
   scons -C Godot platform=windows target=template_debug
   mkdir -p Godot/d2_demo/addons/schola/bin/windows
   cp Godot/build/windows/schola.windows.template_debug.x86_64.dll Godot/d2_demo/addons/schola/bin/windows/
   ```
2. Open `Godot/d2_demo/project.godot` in Godot.
3. On the `Runner` node, set `python_executable` to a Python that has gymnasium, e.g. `.venv/Scripts/python.exe`.
4. Press F5.

## Controls

- **Arrow keys:** move the agent while in manual mode.
- **Start Training:** Godot listens on `127.0.0.1:5757` and launches `python/trainer.py`, which drives reset and step.
  - To run the trainer yourself instead, untick `auto_launch_trainer` on `Runner`.
- **Stop / Reset Run:** the trainer saves `policies/q_policy.json` and the scene resets.
- **Load Saved Policy:** the agent follows the saved policy inside Godot.
- **Test US2 validation:** logs the US2 errors for a set of invalid actions and observations.
