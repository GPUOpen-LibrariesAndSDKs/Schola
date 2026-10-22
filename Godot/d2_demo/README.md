# D2 demo

A 3D grid world showing the D2 workflow end to end:
1. Godot sends observations and rewards to a local Python trainer.
2. Godot applies the trainer's actions to the agent.
3. The trained policy is reloaded and runs in Godot with Python closed.

![demo2-image](screenshots/godot-demo2.png)

The spaces are the real US2 resources (`ScholaBoxSpace`, `ScholaDiscreteSpace`) from the Schola GDExtension. The other stories are basic GDScript stand-ins in `addons/schola_d2`:

| Stand-in | For |
| --- | --- |
| `ScholaDemoEnvironment`, `ScholaDemoAgent` | US1, US4, US5 |
| `ScholaLocalConnector`: JSON over localhost TCP, mirroring Schola's gRPC calls ([protocol](../docs/local_protocol.md)) | US3 |
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
3. On the `Runner` node, set `python_executable` to a Python that has gymnasium, e.g. `.venv/Scripts/python.exe`. Ensure the directories match for `python_executable`. Using `where python` helps.
4. Press F5 or the Play button.

## Controls

- **Arrow keys:** move the agent while in manual mode.
- **Start Training:** Godot listens on `127.0.0.1:8000` and launches `python/trainer.py`. The trainer connects and sends every request (start, definition, reset, step), and Godot answers each one.
  - The log panel shows the connection state: `waiting`, `connected`, `training`, `closed` or `error`. An error always comes with a message saying what went wrong.
  - If nothing connects within 45 seconds, the run ends with an error. Check `python_executable` on `Runner`, or untick `auto_launch_trainer` and run `python python/trainer.py --port 8000` yourself.
- **Stop / Reset Run:** the trainer saves `policies/q_policy.json` and the scene resets.
- **Load Saved Policy:** the agent follows the saved policy inside Godot as a jSON.
- **Test US2 validation:** logs the US2 errors for a set of invalid actions and observations.
