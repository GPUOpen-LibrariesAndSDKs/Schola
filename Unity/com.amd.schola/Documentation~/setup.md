# Install, train, export, and adapt Schola (US6)

This walkthrough uses two Package Manager samples: **Setup Demo** checks installation without Python; **Training Demo** supplies separate training and ONNX inference scenes. All movement uses Schola observations and actions. The deterministic setup controller is a programmed rule, not a learned policy.

## Requirements

- Unity **6000.3.24f1** (Unity 6.3); a new **3D Built-In Render Pipeline** project is the baseline. URP/HDRP users must supply compatible materials.
- Git for cloning/Git installation. Python **3.10?3.12** for training/export only.
- Package Manager resolves `com.unity.ai.inference` **2.6.1** and the declared test framework dependency. No Unreal installation or native Unity gRPC library is needed.
- CPU inference is the default; no GPU is required. See [validation.md](validation.md) for tested results and remaining limits.

## Install and import

Clone the repository, then open **Window > Package Management > Package Manager > + > Install package from disk** and choose `Unity/com.amd.schola/package.json`. Keep the checkout in place: disk installs reference it directly. Do not also copy the package into Assets.

Alternatively, use **Install package from git URL**:

```text
https://github.com/csc301-2026-f/11-AMD-Schola.git?path=/Unity/com.amd.schola#US6-setup-integration
```

The completed sample is published on this branch. For repeatable installations, replace the branch revision with the tested commit `1b4e1e2053f9986eef7b44fbe825eeff1fa59a99`. The `?path=` must precede `#revision`; the repository root is not a UPM package.

Select **AMD Schola 0.1.0**, expand **Samples**, and import **Setup Demo** and **Training Demo**. They appear under `Assets/Samples/AMD Schola/0.1.0/`. Wait for compilation. Imported samples are editable copies; back up customizations before reimporting.

## 1. Check installation without Python

Open `Setup Demo/Scenes/ScholaSetupDemo.unity` and press Play. The cube moves toward the sphere and stops within 0.1 world units. The Game overlay shows `agent_position`, `target_position`, and **Target reached**. Move Target during Play Mode and the cube follows again. Stop Play Mode to restore the scene.

To rebuild this scene choose **Tools > AMD Schola > Create Setup Demo Scene** and save under Assets. The builder uses Unity primitives and the same wiring as the shipped scene.

The flow is `PositionSensor > ScholaAgent.Observe > SetupDemoController > ScholaAgent.Act > TranslateActuator`. The actuator consumes three world-space displacements per action, not velocity. The setup controller multiplies speed by delta time once; the actuator does not multiply again.

## 2. Train and export

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e "./Resources/python[sb3]"
```

Use your installed Python 3.10?3.12 executable if the Windows `py` launcher is unavailable. If activation is restricted, invoke `.venv/Scripts/python.exe` and `.venv/Scripts/schola.exe` directly.

Open `Training Demo/Scenes/ScholaTrainingDemo.unity`. Enable **Run In Background** in Player settings, wait for compilation to finish, then enter Play Mode. The scene waits for Python; no deterministic controller or inference runner is active. Run:

```powershell
schola unity train --unity-port 8001 --seed 0 --timesteps 10000 --checkpoint ckpt/unity_ppo --export-onnx
```

The cube receives bounded actions in `[-0.1, 0.1]` on each axis. Progress earns reward; arrival within 0.2 units terminates the episode; 128 steps truncate it. Reset restores the starting poses. The scene can move quickly because each training request advances a synchronous environment step, not a physics interval.

On normal completion expect `ckpt/unity_ppo.zip` (SB3 checkpoint) and `ckpt/unity_ppo.onnx` (Unity inference model), plus terminal save/export messages. PPO completes 128-step rollouts, so actual timesteps can exceed the requested total. A short run verifies the pipeline; learning quality is not guaranteed. Allow training to finish before stopping Play Mode. Ctrl+C interrupts training and is not a checkpoint-save guarantee.

See [training.md](training.md) for bridge lifecycle, reset semantics, ports, and supported spaces.

## 3. Run ONNX with Python closed

Stop training and Play Mode, then open `Training Demo/Scenes/ScholaInferenceDemo.unity`. It contains a **Policy Runner**, its required **Stepper Runner**, the agent, and a bundled model. It has no Training Server. Press Play: the CPU policy drives the cube toward the sphere; the status overlay and Console announce arrival and stop the runner. Stop/re-enter Play Mode to restart.

To use your own training output:

1. Stop Play Mode. Copy `ckpt/unity_ppo.onnx` into a project Assets folder and wait for Unity to import it.
2. Select **Schola Training Environment**, locate **Policy Runner**, and replace **Model** with the imported ONNX asset. Keep **Backend = CPU** and **Agents = Agent's ScholaAgent**.
3. Close the Python process and press Play. Confirm movement and no model/shape errors. A poorly trained policy may not reach the target; this is different from an import or execution error.

The `.zip` checkpoint cannot be assigned as a Unity model. Model inputs must match `agent_position` and `target_position` (three floats each), with a three-float output named `translate`. The inference scene uses the same finite action bounds as training. See the sample's `Models/README.md` for bundled-model provenance and regeneration.

Rebuild training/inference scenes with **Tools > Schola > Create Training Demo Scene** or **Create Inference Demo Scene**. The inference builder loads the model from the imported sample's Models folder.

## Adapt to your project

Duplicate the appropriate scene before editing. Replace Agent's visual or wire your own object:

| Component | Required wiring |
| --- | --- |
| Agent Position Sensor | Observation Key `agent_position`; reads world position |
| Target Position Sensor | Observation Key `target_position`; reads world position |
| Schola Agent | Sensors: agent then target; Actuators: agent's Translate Actuator |
| Translate Actuator | Action Key `translate`; Max Displacement `0.1` for training/inference |
| Schola Environment | Agent reference; Training Demo Task beneath the environment supplies reward and episode status |
| Training Demo Task | Agent and Target transforms; customize progress reward, arrival distance, and maximum steps here |
| Training Server | Training scene only; environment on the same object; port `8001` |
| Policy Runner | Inference scene only; agent list and matching ONNX model; CPU backend |

For the deterministic scene, reconnect Setup Demo Controller's agent and target sensor; keep its expected two-sensor/one-actuator wiring. Its default unbounded translation is intentional. It does not assign rewards or manage episodes.

Keep keys unique and preserve input/output shapes when reusing a model. Adding sensors, changing action dimensions, or changing the task requires a matching policy and usually retraining. The current agent lists accept PositionSensor and TranslateActuator specifically; arbitrary ISensor/IActuator implementations require extending those APIs.

Only one controller may send actions to an agent. Disable the setup controller and Training Server when using Policy Runner. Transform movement does not provide collision response or advance Rigidbody physics. Use a physics-aware stepper/actuator for physics tasks. Changing only the render mesh preserves the observation/action contract; changing units, bounds, coordinate conventions, rewards, or reset logic changes the learning problem.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Sample missing | Import both entries under AMD Schola's Samples; refresh a cached Git package revision if necessary. |
| Scripts from an older sample | Back up customizations and reimport; package updates do not update Assets copies. |
| Pink materials | Use Built-In for the baseline or assign pipeline-compatible materials. |
| Connection timeout | Enter Play Mode first, enable Run In Background, match `--unity-port` to Training Server, and check Console. |
| Port in use | Stop the other trainer/scene, or change the server port and matching CLI argument. |
| PPO rejects action bounds | Set Max Displacement above zero; the training sample uses 0.1. |
| No model / incompatible model | Assign an ONNX ModelAsset; check keys, shapes, bounds, and exported policy compatibility. |
| Double movement | Keep only one action driver active for the agent. |
| No arrival after training | Inspect reward/episode configuration and train longer; model execution alone does not prove convergence. |
| Changes disappear after Play | Expected Unity behavior; make persistent edits outside Play Mode. |

For package/sample test execution, scene regeneration, and verification evidence see [validation.md](validation.md).
