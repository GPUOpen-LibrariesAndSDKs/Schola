# Unity Training (US4)

This folder contains the Unity-side training connection. `TrainingServer` runs a
loopback TCP endpoint and dispatches requests to `TrainingSession` on Unity's main
thread. The session speaks `IMultiAgentEnvironment` (a single-agent
`ScholaEnvironment` is wrapped under the id `SingleAgent`). The packaged Python
adapter exposes Schola's existing gRPC `GymService` and translates messages using
`TrainingWire`'s versioned JSON format (version 2 keys every agent by id).

See [the training guide](../Documentation~/training.md) for setup and supported
behavior, including multi-agent scenes. `schola unity train` starts the adapter
and trains with existing Schola SB3 wrappers. `schola unity bridge` exposes the
adapter for a separate trainer such as RLlib. Unity listens on port 8001 by
default; the separate gRPC adapter defaults to 8000. Native gRPC and generated
C# messages are not required by this implementation.

## Optional direct-gRPC code generation tooling

From the repository root, run:

```sh
python .github/scripts/generate_csharp.py
python .github/scripts/generate_csharp.py --check
```

The generator downloads the checksum-verified **Grpc.Tools 2.67.0** NuGet package
and uses its bundled `protoc` and `grpc_csharp_plugin` to generate messages and
both client/server service bindings from every `Proto/**/*.proto`. It requires
Python 3.11 and network access, without Unity or .NET. Supported hosts are Linux
x64/arm64, Windows x64, and macOS x64 (Apple Silicon requires Rosetta 2).

Output belongs in `Unity/com.amd.schola/Training/Generated/`. This directory is
reserved for generated C#; regeneration removes stale `.cs` files, preserves
existing `.meta` files, and leaves `Proto/` and the Python/C++ outputs untouched.
Import the generated folder in Unity to create metadata, remove metadata for
deleted sources, and commit the `.cs` and `.meta` files together. The training
integration must supply compatible Google.Protobuf and gRPC runtime dependencies
before these sources can compile in Unity; the generator does not install them.

CI validates generation but reports an explicit comparison skip until the training
team commits its first C# baseline. After that, `codegen-check.yml` runs `--check`,
generating into a temporary directory and failing for changed, missing, or stale C# files, with a
diff and the regeneration command. `.meta` files are checked separately by
`checks.yml`. CI checks Git history so deleting even the entire baseline fails.
`--check` itself always fails when the baseline is missing. For isolated local
generation or comparison, use `--output /path/to/directory`.

## Existing player smoke-test contract (future direct-gRPC integration)

The reserved player smoke-test CI expects to start a future direct-gRPC player with:

```text
SmokeTest.x86_64 -batchmode -nographics -ScholaPort=50051 -logFile ...
```

See `.github/workflows/smoke-test.yml` (`SMOKE_PORT` / `run` job) and
`.github/scripts/unity_smoke_test.py`.

This direct-gRPC player contract is separate from the shipped TCP-plus-adapter
implementation. The player smoke project is not provided here. Its existing CI
jobs remain gated on that project, and must be adapted to start the Python bridge
before they can test this implementation. The implemented adapter is exercised
by Python adapter tests and the real-Editor tests in `Test/unity`.

**Original direct-gRPC contract (match Unreal):**

1. When the Unity gRPC gym connector / training server starts, parse the player
   command line for `-ScholaPort=<int>` (same spelling and `=` form as Unreal).
2. If the flag is present, bind the gRPC server to that port.
3. If the flag is missing, fall back to a documented default (Unreal uses
   `ServerSettings.Port` when parse fails).
4. Serve Schola's existing `GymService` protocol so
   `python .github/scripts/unity_smoke_test.py --port <port>` can connect with
   `GrpcProtocol(..., credential_mode="insecure")` and `ExternalSimulator`
   (no changes under `Resources/python`).

**Unreal reference (do not invent a different flag name):**

- `Source/ScholaProtobuf/Private/GymConnectors/gRPC/gRPCGymConnector.cpp`
  uses `FParse::Value(..., TEXT("ScholaPort="), Port)`.

**Do not commit `Unity/SmokeTestProject` until:**

1. This port parsing is implemented and tested locally against the smoke script, and
2. `.github/scripts/check_meta_files.py` skips project `ProjectSettings/` and
   `Packages/` (see the AGENT TODO in that file). Otherwise `checks.yml` fails.

Once both are done and the smoke-test project exists, the `build` and `run` jobs
in `smoke-test.yml` turn on with no workflow edits.
