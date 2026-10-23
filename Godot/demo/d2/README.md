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
| Tabular Q-learning instead of SB3 or RLlib training | US3 |

US7 is real: the trained Q-table is exported to ONNX, and **Load ONNX Policy** runs it with the `ScholaOnnxPolicy` resource from the Schola GDExtension (ONNX Runtime 1.30). **Load Saved Policy** keeps the GDScript Q-table reader for comparison.

## Requirements

- Godot 4.7.2
- Python with `gymnasium`, `numpy`, `onnx`, and `onnxruntime`. Install them into a virtual environment with `python -m pip install gymnasium numpy onnx onnxruntime`.

Packaged builds for Windows x86_64, Linux x86_64, and macOS arm64 are attached to the [D2 release](https://github.com/csc301-2026-f/20-AMD-Schola/releases/tag/D2). They include a reference ONNX policy and do not require Godot, Python, or a source checkout for manual play, validation, or inference. Starting a new training session still requires this developer setup.

## Run

1. Download ONNX Runtime, build the extension, and copy both libraries into the demo:
   ```sh
   python Godot/tools/setup_onnxruntime.py
   scons -C Godot platform=windows target=template_debug
   mkdir -p Godot/demo/d2/addons/schola/bin/windows
   cp Godot/addons/schola/bin/windows/schola.windows.template_debug.x86_64.dll Godot/addons/schola/bin/windows/onnxruntime.dll Godot/demo/d2/addons/schola/bin/windows/
   ```
2. Open `Godot/demo/d2/project.godot` in Godot.
3. On the `Runner` node, set `python_executable` to a Python that has gymnasium, e.g. `.venv/Scripts/python.exe`. Ensure the directories match for `python_executable`. Using `where python` helps.
4. Press F5 or the Play button.

## US5 smoke check

After building and copying the runtime extension, verify the native US5 scene
configuration without starting Python. Import the Demo 2 project first so
Godot registers its GDScript global classes and GDExtension classes:

```sh
godot --headless --path Godot/demo/d2 --editor --quit
godot --headless --path Godot/demo/d2 -s res://tests/us5_smoke.gd
```

The check loads the demo scene and verifies the native environment and agent,
their valid space configuration, the single discovered agent, its `agent_0`
ID, and the 40-step limit.

## Controls

- **Arrow keys:** move the agent while in manual mode.
- **Start Training:** Godot listens on `127.0.0.1:8000` and launches `python/trainer.py`. The trainer connects and sends every request (start, definition, reset, step), and Godot answers each one.
  - The log panel shows the connection state: `waiting`, `connected`, `training`, `closed` or `error`. An error always comes with a message saying what went wrong.
  - If nothing connects within 45 seconds, the run ends with an error. Check `python_executable` on `Runner`, or untick `auto_launch_trainer` and run `python python/trainer.py --port 8000` yourself.
- **Stop / Reset Run:** the trainer saves `policies/q_policy.json`, exports it to `policies/q_policy.onnx`, and the scene resets.
- **Load Saved Policy:** the agent follows the saved Q-table (JSON) in GDScript.
- **Load ONNX Policy:** the agent follows the same policy exported to ONNX, run by Schola's ONNX Runtime inference (US7) with Python closed. To export a saved policy by hand, run `python python/export_onnx.py`.
- **Test US2 validation:** logs the US2 errors for a set of invalid actions and observations.
