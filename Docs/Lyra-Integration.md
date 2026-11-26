# Lyra + Schola integration (macOS, Unreal Engine 5.8.4)

How to build Epic's Lyra Starter Game together with this Schola checkout on Apple Silicon, and how a teammate reproduces the same setup on their own Mac. It builds on [macOS-Setup.md](macOS-Setup.md), which covers the gRPC dependencies and the general macOS toolchain.

## Status

- The `LyraEditor` target built on macOS arm64 with Unreal Engine 5.8.4 after two changes: the Schola UObject pointer fixes in PR #56, and three compatibility fixes in Lyra's `CommonUser` plugin. The `CommonUser` fixes are captured in `Tools/Lyra/patches/commonuser-ue58-mac.patch`.
- `Tools/Lyra/setup_mac.sh` and `Tools/Lyra/build_mac.sh` automate the preparation and the build command. They were exercised against stand-in engine and project directories, never against a clean machine. **A build from a clean machine using these scripts has not been run.** The first teammate to do so should report the result.
- The patch was generated from the locally modified Lyra files. Epic's pristine 5.8.4 sources were not available when it was written, so the unpatched side was reconstructed by reversing the three fixes, and the reconstructed lines were checked against Epic's published Lyra 5.5.4 sources, where they are byte-identical. If the patch does not apply to your pristine 5.8.4 Lyra, `setup_mac.sh` stops without changing anything; see [Troubleshooting](#troubleshooting).

## PR #56 must be in your Schola checkout

PR #56 ("Fix UObject pointer compatibility for Lyra UE 5.8") changes the `Connector` member in two Schola headers from a raw `UObject` pointer to `TObjectPtr`:

- `Source/ScholaTraining/Public/TrainingUtils/GymConnectorManager.h`
- `Source/ScholaImitation/Public/ImitationUtils/ImitationConnectorManager.h`

Both headers contain raw pointers on `main` today, and the Lyra build needs the change. Until PR #56 is merged into `main`, a clean Lyra + Schola build only works from a branch that contains it, for example the PR branch itself or your own branch merged with it. `setup_mac.sh` checks that both headers contain the fix and refuses to continue otherwise. After PR #56 merges, a normal checkout of `main` is enough and this section no longer applies.

`--allow-unfixed-schola` overrides the check. Use it only to inspect the rest of the setup; the Lyra build is expected to fail without the fix.

## What the CommonUser patch changes

`Tools/Lyra/patches/commonuser-ue58-mac.patch` is a unified diff, applied from the Lyra project root with `-p1`. It contains only these changes to Lyra's `Plugins/CommonUser/Source/CommonUser/Private`:

| File | Change |
| --- | --- |
| `CommonUserBasicPresence.cpp` | Remove the four `break;` statements after a `return` in `SessionStateToBackendKey`. |
| `CommonSessionSubsystem.cpp` | In `HandleQuickPlaySearchFinished`, replace the loop that returns on its first iteration with `JoinSession(..., SearchSettings->SearchRequest->Results[0])`. The existing `ResultCount > 0` guard stays. |
| `CommonSessionSubsystem.cpp` | In `OnFindSessionsComplete`, remove the unreachable `if (0)` fake-session test block. |

Clang rejects this unreachable code when Lyra is built against UE 5.8 on macOS. Lyra's source is not copied into this repository; only the diff is.

## Requirements

- An Apple Silicon Mac and a native arm64 shell. A shell running under Rosetta is rejected.
- Unreal Engine 5.8.4. `--ue-root` (or `UE_ENGINE_ROOT`) is the directory that contains `Engine/Build`.
- Lyra Starter Game for 5.8, as a normal project directory containing `LyraStarterGame.uproject`. It must live outside this repository, otherwise `Plugins/Schola` would link back into itself.
- Xcode with the command-line tools and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
- Git, CMake, and Ninja or Make for the one-time gRPC build, as in [macOS-Setup.md](macOS-Setup.md#requirements).
- A Schola checkout that contains PR #56.

The scripts take paths from flags or environment variables. Nothing is hardcoded:

```bash
export LYRA_PROJECT_DIR=/path/to/LyraStarterGame
export UE_ENGINE_ROOT=/path/to/UE_5.8
```

A flag always wins over the matching variable.

## Install

From the root of this repository:

```bash
# 1. gRPC and protobuf for macOS (once per checkout; skip if Source/ThirdParty/gRPC/Mac exists)
Resources/Build/mac_dependencies.sh

# 2. Preview what setup would do. Nothing is changed.
Tools/Lyra/setup_mac.sh --check-only

# 3. Run the setup
Tools/Lyra/setup_mac.sh
```

Setup first runs every check and changes nothing if any of them fails. Then it:

1. Creates `<Lyra>/Plugins/Schola` as a symlink to this checkout, if that path does not exist.
2. Applies the `CommonUser` patch to the files that are not patched yet.

It does not rebuild gRPC when usable archives are present. The check requires headers, the core gRPC and protobuf archives, arm64 code, and a macOS 14.0 deployment target, which is what `mac_dependencies.sh` produces. To let setup run the dependency script when the archives are missing or unusable, pass `--build-deps`.

### Safety behavior

| Situation | Behavior |
| --- | --- |
| `Plugins/Schola` is absent | Symlink to this checkout is created. |
| `Plugins/Schola` already links to this checkout | Nothing to do. |
| `Plugins/Schola` is a symlink elsewhere or dangling | Refused. `--replace-schola-link` removes only the symlink and creates the new one. |
| `Plugins/Schola` is a real directory or file | Refused. Setup never modifies it; move it aside yourself. |
| Patch not applied, sources match | Applied to the unpatched files, then verified. If anything fails, the files are restored. |
| Patch already applied | Detected and skipped, so repeated runs change nothing. |
| One file patched, the other not | Only the unpatched file is changed. |
| Sources modified or from another Lyra version | Reported as incompatible. Nothing is changed. Matching is strict, with no fuzz and no whitespace tolerance. |

`--no-patch` skips the patch. `--revert-patch` reverses it when both files are in the cleanly patched state.

The symlink target is an absolute path to your checkout, so it is specific to your machine. Do not commit it or copy it to another machine.

## Build

```bash
Tools/Lyra/build_mac.sh
```

The script first runs `setup_mac.sh --verify`, which fails if `Plugins/Schola` is not linked, the patch is not applied, or the gRPC archives are missing. Then it runs:

```bash
"$UE_ENGINE_ROOT/Engine/Build/BatchFiles/Mac/Build.sh" LyraEditor Mac Development \
    -Project="$LYRA_PROJECT_DIR/LyraStarterGame.uproject" \
    -NoHotReloadFromIDE -NoUBA -MaxParallelActions=2
```

`-NoUBA` matches [macOS-Setup.md](macOS-Setup.md#build). The parallel-action limit defaults to 2 so Macs with limited memory do not run out of RAM during the many large Lyra and Unreal translation units. Change it with `--max-parallel-actions N` or `LYRA_MAX_PARALLEL_ACTIONS=N`. Try 1 if the build is killed for memory, and raise it only if you have headroom. Extra arguments after `--` go to UnrealBuildTool unchanged.

To see the exact command without building:

```bash
Tools/Lyra/build_mac.sh --dry-run
```

Open the editor after the build:

```bash
open "$UE_ENGINE_ROOT/Engine/Binaries/Mac/UnrealEditor.app" --args "$LYRA_PROJECT_DIR/LyraStarterGame.uproject"
```

## Validate

Cheap checks, in order of cost:

1. `Tools/Lyra/setup_mac.sh --verify` exits 0 and prints `Verified`. This confirms the engine version, the PR #56 fixes, the gRPC archives, the `Plugins/Schola` link and the patch state.
2. `Tools/Lyra/build_mac.sh --dry-run` prints the build command with `-NoHotReloadFromIDE -NoUBA -MaxParallelActions=N`.
3. The full build finishes with `Result: Succeeded`.
4. The editor starts, Schola appears in Edit > Plugins, and the log shows the Schola modules loading without errors.

Only step 3 and step 4 prove the integration works. Steps 1 and 2 only prove the preparation is consistent.

To check a patch result by hand, compare the files with the expectations in the table above: no `break;` after the `return` statements in `SessionStateToBackendKey`, a single `JoinSession(... Results[0])` call in `HandleQuickPlaySearchFinished`, and no `if (0)` in `CommonSessionSubsystem.cpp`.

## Reproduce on your own machine

1. Install Unreal Engine 5.8.4, Xcode, the Metal toolchain, and the tools in [Requirements](#requirements).
2. Clone this repository and check out a branch that contains PR #56 (or `main` once it is merged).
3. Get Lyra Starter Game for UE 5.8 from Epic and place it anywhere outside the repository.
4. Set `LYRA_PROJECT_DIR` and `UE_ENGINE_ROOT`.
5. Run `Tools/Lyra/setup_mac.sh --check-only`, fix anything it reports, then run `Tools/Lyra/setup_mac.sh --build-deps`. `--build-deps` runs `Resources/Build/mac_dependencies.sh` only if the archives are missing.
6. Run `Tools/Lyra/build_mac.sh`.

Repeat steps 5 and 6 after pulling Schola changes. Setup is safe to re-run. If your Lyra copy differs from Epic's pristine 5.8.4 sources, say so when reporting a failed patch.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `PR #56 ... is not in this checkout` | The two connector headers still use raw pointers. Merge PR #56 into your branch. |
| `Mac gRPC archives are missing or unusable` | Run `Resources/Build/mac_dependencies.sh`, or setup with `--build-deps`. A `libz.a` built for a macOS newer than 14.0 is rejected because the engine links for 14.0. |
| `Plugins/Schola already exists and is not a symlink` | Setup will not touch real files. Move that directory aside, for example to `Schola.old`, and run setup again. |
| `Plugins/Schola is a symlink to ..., not to this checkout` | It points to another checkout. Use `--replace-schola-link` if you want to repoint it. |
| `... does not match the expected UE 5.8.4 Lyra source` | The file was edited, or your Lyra is not 5.8.4. Setup changed nothing. Compare your `CommonUser` files with the fixes in the patch and apply them by hand, or restore pristine files and rerun. Do not force the patch. |
| Clang: `'break' will never be executed`, or a warning that the loop runs at most once | The `CommonUser` patch is not applied. Run setup. |
| Build killed or machine freezing | Lower `--max-parallel-actions`, close other apps. |
| `Missing gRPC headers` from `gRPC.Build.cs` | The Mac archives are missing in the checkout that `Plugins/Schola` links to. Run the dependency script there. |
| Metal shader errors during the first build | Run `xcodebuild -downloadComponent MetalToolchain`. |
| `This shell reports x86_64` | Open a native arm64 terminal. |
| `The Lyra project ... is inside this Schola checkout` | Move Lyra outside the repository. |

## Files

- `Tools/Lyra/setup_mac.sh`: checks, links `Plugins/Schola`, applies the patch. Run with `--help` for all options.
- `Tools/Lyra/build_mac.sh`: verifies the setup and builds `LyraEditor Mac Development`.
- `Tools/Lyra/patches/commonuser-ue58-mac.patch`: the Lyra `CommonUser` compatibility fixes.
