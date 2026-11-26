# IREE third-party runtime (Schola)

Schola's `ScholaIREE` module links against the **IREE embedded C runtime** headers and static libraries under this directory. The headers are in git while, the libraries are checked in via git-lfs.

## Layout

```
Source/ThirdParty/IREE/
  Win64/
    include/   # e.g. iree/runtime/api.h, iree/vm/ref_cc.h
    lib/       # iree_runtime_unified.lib, iree_hal_local_elf_arch.lib, flatcc_runtime.lib, flatcc_parsing.lib
  Linux/
    include/
    lib/       # libiree_runtime_unified.a, libflatcc_runtime.a, libflatcc_parsing.a
```

`IREE.Build.cs` sets `SCHOLA_IREE_HAS_RUNTIME=1` when headers and libs are present. If they are missing, Schola still builds with `SCHOLA_IREE_HAS_RUNTIME=0` and `UIREEPolicy::Init` fails at runtime with a clear log until you populate this folder.

The runtime version must match the `iree-base-compiler` used by Python export (`save_model_as_iree`), otherwise `.vmfb` files may fail to load. Schola is currently tested against **IREE v3.14.0**.

## Building the IREE runtime from source

Build the runtime only (no compiler) with CMake. The Windows steps below are the ones Schola has been verified with; the Linux steps mirror them but have not been verified yet.

### Prerequisites

- CMake 3.21+ and Git.
- Windows: Visual Studio with the C++ workload (the examples use VS 2026, generator `"Visual Studio 18 2026"`).
- Linux: clang or gcc, and Ninja is recommended.
- Python 3 on `PATH` as `python3`. IREE's build runs `python3` directly in some generated steps; `-DPython3_EXECUTABLE` alone is not enough.

> **Windows:** `python3` often resolves to the Microsoft Store stub in `WindowsApps`, which breaks the build. Either disable the `python3.exe` app execution alias or put a shim earlier on `PATH`:
>
> ```powershell
> New-Item -ItemType Directory -Force C:\iree-build\pyshim | Out-Null
> Set-Content C:\iree-build\pyshim\python3.cmd '@"C:\path\to\python.exe" %*'
> $env:PATH = "C:\iree-build\pyshim;$env:PATH"
> ```

### 1. Get the source

```bash
git clone --depth 1 --branch v3.14.0 https://github.com/iree-org/iree.git iree-src
cd iree-src
# Only the runtime submodules are needed:
git submodule update --init --depth 1 $(cat build_tools/scripts/git/runtime_submodules.txt)
```

PowerShell equivalent for the submodule line:

```powershell
git submodule update --init --depth 1 (Get-Content build_tools\scripts\git\runtime_submodules.txt)
```

### 2. Configure

Windows (`/MD` CRT, to match Unreal):

```powershell
cmake -S C:\iree-src -B C:\iree-build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_INSTALL_PREFIX=C:\iree-install `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF `
  -DIREE_BUILD_BINDINGS_TFLITE=OFF -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF `
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON `
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF `
  -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON `
  -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF -DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON `
  -DPython3_EXECUTABLE=C:\path\to\python.exe
```

Linux:

```bash
cmake -S iree-src -B iree-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/iree-install" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_BINDINGS_TFLITE=OFF -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
  -DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=ON \
  -DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF -DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON
```

These settings enable only CPU execution (`local-sync` / `local-task` drivers, with embedded-ELF and VMVX executables). Export models with a matching CPU target (`llvm-cpu` or `vmvx`).

### 3. Build and install

```powershell
cmake --build C:\iree-build --config Release --target install -- /m /nodeReuse:false
cmake --install C:\iree-build --config Release --component IREEDevLibraries-Runtime
```

```bash
cmake --build iree-build --target install
cmake --install iree-build --component IREEDevLibraries-Runtime
```

The plain `install` target does not include the runtime headers and libraries; the `IREEDevLibraries-Runtime` component install provides them.

On Windows, `/nodeReuse:false` (or `$env:MSBUILDDISABLENODEREUSE = "1"`) stops idle MSBuild nodes from keeping files in the build tree locked after the build ends.

### 4. Populate this folder

The install still leaves out a few C++ headers (for example `iree/vm/ref_cc.h`) and the flatcc libraries the bytecode verifier needs. The build scripts in `Plugins/Schola/Resources/Build/iree/` add these from the source and build trees.

for Windows:
```powershell
<repo>\Plugins\Schola\Resources\Build\iree\build_iree_windows.ps1 -IreeInstallRoot C:\iree-install -IreeSourceRoot C:\iree-src -IreeBuildRoot C:\iree-build
```

for Linux:
```bash
<repo>/Plugins/Schola/Resources/Build/iree/build_iree_linux.sh "$PWD/iree-install" "$PWD/iree-src" "$PWD/iree-build"
```

To keep the folder small (about 6 MB instead of about 19 MB), the scripts:

- clear `include/` and `lib/` first, so stale files from earlier fetches are not kept or linked twice;
- copy only `iree_runtime_unified.lib` (one archive of the whole runtime, which the per-component libs duplicate), `iree_hal_local_elf_arch.lib` (it holds the Windows MASM ELF-call thunks the unified archive omits; on Linux it is copied only if present) and the flatcc libs;
- run `Resources/Build/iree/prune_iree.py`, which keeps only the headers reachable from the `iree/...` includes in `Source/ScholaIREE`. If you start including other IREE headers, re-run the build script.

> [!NOTE]
> the build script touches `IREE.Build.cs` so UnrealBuildTool re-detects the runtime on the next build (UBT caches its makefile otherwise).

Then rebuild the editor. The UBT output should print `Schola IREE: linking prebuilt runtime from ...` instead of `runtime not found`.
