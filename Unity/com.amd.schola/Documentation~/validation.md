# US6 validation record

## Environment and integration

Validated on **2026-11-11**, Windows, Unity **6000.3.24f1**, Python **3.12.14**, CPU inference. This implementation builds on merge commit `c6d9b32`, which integrates `origin/main` at `10fbe84` into `US6-setup-integration`. Recovery branch: `backup/us6-before-main-integration`. Both Package Manager samples and main's training/inference runtime are retained. No package runtime API was replaced.

A fresh, ignored consumer project under `enc_temp_folder/us6-unity` installed the package from disk and imported both samples. Scenes were generated and saved with Unity's serializer, then copied into the package with their generated metadata. The project, checkpoints, logs, screenshots, and caches are validation artifacts, not package content.

## Reproduce checks

From the repository root:

```powershell
dotnet tool restore
dotnet csharpier check Unity/
python -m unittest discover -s .github/scripts/tests -p test_checks.py
python .github/scripts/check_meta_files.py
python .github/scripts/check_package_json.py
python -m pytest Test/unity --confcutdir=Test/unity -n 0
git diff --check
```

Use a virtual environment with the repository's Python package, SB3, and test dependencies installed. On this host the executables were `.venv/Scripts/python.exe` and `C:/Program Files/dotnet/dotnet.exe`. The full tooling unittest suite has two pre-existing Windows code-generation failures (mocked platform detection and the Bash/Git deletion check); the 15 manifest/metadata validator tests pass.

For Unity tests, create a dedicated consumer project, install the package, add `com.amd.schola` to the project manifest's `testables`, and import both samples. Run Test Runner's EditMode suite, including the imported sample assemblies. The inference UnityTest enters Play Mode itself. Package-mode CI alone does not import `Samples~` and therefore does not execute sample tests.

```powershell
$unityEditor = 'C:\Program Files\Unity\Hub\Editor\6000.3.24f1\Editor\Unity.exe'
$demoProject = 'C:\Projects\ScholaSetupDemo'
& $unityEditor -batchmode -projectPath $demoProject -runTests -testPlatform EditMode `
    -testResults "$demoProject/editmode-results.xml" -logFile "$demoProject/editmode.log"
```

Close any Editor already using that project. Wait for the XML result; the Windows Unity launcher can return before the Editor exits. Normal expected-error logs in negative tests are not failed tests; inspect the result XML.

To regenerate the shipped assets, import both samples, use their scene creation menu commands, and save into each imported sample's Scenes folder. For training/inference maintainers, `AMD.Schola.Samples.TrainingDemo.Editor.SetupSceneAssets.Generate` generates both scenes in the imported Training Demo folder. Copy generated Scenes/Materials assets and metadata back to the package, retaining GUIDs. Do not hand-author scene YAML.

## Real training/export evidence

The actual Training Demo scene was opened in a dedicated batch Editor, with `TrainingServer.PumpRequests` driven by Editor updates on Unity's main thread. This uses the sample's real sensors, actions, rewards, episode hooks, and environment rather than the fake Python export fixture. Port **8018** was used to avoid conflicts; the shipped scene and user guide use **8001**.

```powershell
.venv/Scripts/schola.exe unity train --unity-port 8018 --seed 0 --timesteps 10000 `
    --checkpoint enc_temp_folder/us6-policy/unity_ppo --export-onnx
```

The successful run completed **10,112 timesteps**, saved an SB3 checkpoint, and exported the bundled ONNX. Final rollout means were **20.9 steps** and **2.64 reward**. The checkpoint's deterministic policy reached the fixed sample goal in **19 actions**, ending at distance **0.123**. This demonstrates the supplied fixed task, not general navigation performance. An earlier run was interrupted by an Editor assembly reload and was discarded; users should wait for compilation before training.

Bundled model SHA-256:

```text
048e76c6ff2e296efce4bf6b8a0d3d05833d530d7c063f21f395fdfd92cdc2e9
```

## Results and limits

Final Unity run: **87/87 passed**: 76 package tests, 8 imported-sample tests, and 3 temporary camera-render checks. The sample tests verify real deterministic observation/action movement, invalid configuration, scene save/reload, training reset, mutually exclusive action drivers, and Play Mode ONNX inference reaching the goal with no Training Server. The final test XML and logs are `enc_temp_folder/us6-editmode.xml` and `enc_temp_folder/us6-editmode.log`.

Rendered images of all three scenes were inspected. The initial white-on-white setup scene failed visual review; the final shipped scenes use persistent blue/orange materials with contrasting backgrounds. Their serialized GUID references resolve to packaged assets or Unity built-ins. All **123 package metadata GUIDs** are valid and unique, with no missing/orphan sample metadata; assembly references resolve. CSharpier **1.3.0** checked **58 C# files** successfully. Manifest, metadata, 15 fast validator tests, and whitespace checks pass.

The first test run exposed unsaved-scene assumptions in legacy Setup Demo tests and a frame-rate-dependent inference wait. Both were corrected and the full suite rerun successfully. The Python bridge/export suite passes **15 tests** with **4 live-host tests skipped** when no `SCHOLA_TEST_UNITY_PORT` host is configured. The real sample training run above independently exercises the actual connection and export path.

The verified baseline is Built-In rendering, one agent, Transform movement, and CPU inference. URP/HDRP, Rigidbody simulation, GPU backends, players/IL2CPP, and arbitrary new tasks are not claimed as tested. The Windows computer-use helper was unavailable; rendered camera images and automated Play Mode checks were used instead of interactive UI automation.

## Published Git installation

Pushed the completed branch through **`1b4e1e2053f9986eef7b44fbe825eeff1fa59a99`** and changed the consumer project's dependency from disk to:

```text
https://github.com/csc301-2026-f/11-AMD-Schola.git?path=/Unity/com.amd.schola#1b4e1e2053f9986eef7b44fbe825eeff1fa59a99
```

Unity Package Manager resolved source **git** and that exact hash in `packages-lock.json`; Editor compilation and a batch verification method passed. The verifier checks the resolved package source and the presence of all three scene files plus the ONNX model inside the Git package cache. Log: `enc_temp_folder/us6-git-install.log`, success marker `SCHOLA_US6_GIT_INSTALL_VERIFIED`. This reused the already-tested consumer project and imported samples; it was not a second from-empty project run. The ONNX model has an explicit exception to the repository's general model ignore rule so it is included in Git installs.

The latest fetched `main` remained `10fbe84` and is an ancestor of the completed US6 branch. The final documentation-only commit adds these results without changing the tested source/assets.
