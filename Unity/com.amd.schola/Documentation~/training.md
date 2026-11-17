# Train Unity agents through Schola (US4)

US4 connects Unity observations, actions, rewards and episode lifecycle to the
existing Schola Python environment adapters. Users configure their environment
and reward logic; the package supplies the communication layer.

## Quick start

Use Unity **6000.3.24f1** and Python **3.10–3.12**. Install `com.amd.schola`
through Package Manager, then install the Python tools from this repository root:

```powershell
python -m pip install -e "./Resources/python[sb3]"
```

Import **Training Demo** in Package Manager. Select **Tools > Schola > Create
Training Demo Scene**, save the scene, and enter Play Mode. Enable **Run In
Background** in Player settings so the Editor continues processing requests when
another window has focus. In the installed Python environment, run:

```powershell
schola unity train --timesteps 10000 --checkpoint ckpt/unity_ppo --export-onnx
```

This starts the packaged adapter automatically, trains PPO using Schola's existing
`VecEnv` and `VecMergeDictActionWrapper`, and saves `ckpt/unity_ppo.zip`. The
trainer rounds the requested number of timesteps up to complete 128-step rollouts.
Ctrl+C interrupts training; a checkpoint is saved on normal completion. There is
no Unity-native gRPC library to install. A checkpoint is an SB3 model, not a Unity
inference asset; model export and loading belong to US5.

## Connect an existing scene

1. Configure a `ScholaAgent` with sensors and actuators (US1/US2).
2. Assign the agent to a `ScholaEnvironment` (US3). Provide `IReward` and
   `IEpisodeStatus` components beneath the environment, or bind them in code
   before training starts. Without them rewards remain zero and episodes do not end.
3. Add **Schola > Training Server** to the environment GameObject, or assign its
   Environment field explicitly. Default Port is **8001**.
4. For PPO with `TranslateActuator`, set **Max Displacement** above zero, e.g.
   **0.1**. This publishes and enforces per-axis finite bounds. Zero retains the
   existing unbounded action behavior, which the PPO convenience command rejects.
5. Disable any deterministic sample controller or inference stepper that also
   controls this agent. Enable Run In Background and enter Play Mode.
6. Run `schola unity train --unity-port 8001`, optionally setting `--seed`,
   `--timesteps`, `--checkpoint`, `--timeout` (seconds per Unity exchange),
   `--export-onnx` and `--onnx-path`.

Scene changes happen only on Unity's main thread. Each network step calls the
existing synchronous `ScholaEnvironment.Step` exactly once, applying the action
and immediately collecting reward, status and observations. **It does not advance
Unity physics by a fixed interval.** Use this release with synchronous,
Transform-based environments. Rigidbody simulation requires a separate stepper
that advances physics before reward and observation collection.

## Use the existing Python training tools

To manage the trainer separately, keep Unity in Play Mode and run:

```powershell
schola unity bridge --unity-port 8001 --port 8000
```

The adapter exposes the existing `GymService` protobuf contract at
`127.0.0.1:8000`, using gRPC local credentials. Connect existing Schola wrappers
with an `ExternalSimulator` (Unity is already running):

```python
from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
from schola.core.simulators.external_simulator import ExternalSimulator
from schola.sb3.env import VecEnv
from schola.sb3.utils import VecMergeDictActionWrapper

env = VecMergeDictActionWrapper(VecEnv(
    ExternalSimulator(), GrpcProtocol("127.0.0.1", port=8000)
))
try:
    observations = env.reset()
    # Pass env to SB3 PPO, or call env.step with batches of compatible actions.
finally:
    env.close()
```

The adapter uses the shared `.proto` messages without modifying generated Python
bindings. Internally it converts messages to versioned, length-prefixed UTF-8
JSON consumed by Unity's `JsonUtility`. Flat trees preserve dictionary keys,
shapes, optional bounds, FLOAT32 Box values and MultiBinary bools without
recursive serialization. This adapter is part of Schola; environment authors do
not implement it.

## Scope and lifecycle

- One environment, ID **0**, with one agent, ID **SingleAgent**, per endpoint.
  Run independent Unity instances/ports for independent endpoints. General
  multi-agent and in-process vector environments are outside this implementation.
- Supported values/spaces: **Dict**, **Box** with FLOAT32, **Discrete**, and
  **MultiBinary** (boolean vector; wire kind `multi_binary` with `shape` /
  `boolValues`). Unsupported types, mismatched action keys/shapes/lengths,
  NaN/infinity, and actions outside declared bounds fail explicitly before the
  environment is stepped.
- Startup initializes sensors/actuators and snapshots reset poses once. A reset
  is required before the first step and after an operation fails. Disconnecting
  and reconnecting preserves the original snapshot while requiring a fresh reset.
- **SAME_STEP** (SB3 default): return final observations/reward/flags and separate
  reset observations in the same response. **NEXT_STEP**: the next step ignores
  its action and returns the reset state with zero reward and cleared flags.
  **DISABLED**: reject further steps after episode end until explicit reset.
- Reset seeds/options are forwarded to `IEnvironment.Seed`/`SetOptions`. The
  current `ScholaEnvironment` implements those as no-ops; randomized seeded resets
  require environment support. The Python seed still configures PPO.
- Both listeners bind to loopback. One trainer owns an endpoint at a time. Pause,
  disconnection, an oversized frame or an exchange timeout produces an error;
  reconnect/start/reset to resume. Stopping Play Mode closes the listener and
  releases queued work. No background thread reads or mutates Unity scene state.
- The maximum frame size is 4 MiB. Match the component's Request Timeout Seconds
  to the Python `--timeout` if changing it. Optimizer time between requests has
  no idle timeout.

## Verification

Run Unity EditMode tests, including assembly `AMD.Schola.Tests.Training`, in Test
Runner. Python adapter tests are in `Test/unity/test_bridge.py`; the optional live
tests exercise the real Unity transport, Schola's existing gRPC client, episode
boundaries, PPO updates and checkpoint reload.

For a batch integration host, create a dedicated Unity project that installs this
package and lists `com.amd.schola` in the manifest's `testables`. Then run the
installed Editor with `-batchmode -nographics -projectPath <project>` and
`-executeMethod AMD.Schola.Tests.Training.TrainingIntegrationHost.Run`. Set
`SCHOLA_TEST_UNITY_PORT` to an unused port in both the Editor process and test
terminal. Once the Editor log contains `SCHOLA_US4_READY:<port>`, run:

```powershell
python -m pytest Test/unity --confcutdir=Test/unity -n 0
```

The integration host uses real agent, sensor, actuator, environment and training
server components. It pumps requests through Editor updates rather than Play
Mode and stays alive until its dedicated process is terminated. Without the
environment variable the live tests skip. This verifies the connection and a
short optimizer run; it does not measure task convergence or player/IL2CPP support.

### Local validation on 2026-10-27

- Unity 6000.3.24f1: **65 EditMode tests passed**, including action/observation,
  environment lifecycle, training sessions, main-thread dispatch and shutdown.
- Python: **17 adapter/live tests passed** against a real batch Unity Editor,
  including terminal observations, truncation, all reset modes, a 32-step PPO
  optimizer run, checkpoint save/reload and another action from the reloaded policy.
- The installed `schola unity train --unity-port 8017 --timesteps 128` command
  completed against the batch host and saved an SB3 checkpoint.
- Package manifest, metadata and C#/Python formatting checks passed. The 15 fast
  manifest/metadata validator tests passed. Two unrelated existing code-generation
  tests fail on this Windows host (mocked Windows platform detection and the
  Bash/Git baseline-deletion check); no code-generation implementation was changed.
