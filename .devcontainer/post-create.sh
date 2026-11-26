#!/usr/bin/env bash
# Runs once after the dev container is created.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo chown -R ue4:ue4 /home/ue4/.venv /home/ue4/.cache/uv Prototypes/SharedPolicy/.venv

echo "==> Main venv (/home/ue4/.venv): schola[sb3] + test deps, CPU-only torch"
[ -x /home/ue4/.venv/bin/python ] || uv venv /home/ue4/.venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e "Resources/python[sb3]" \
    --group Resources/python/pyproject.toml:test \
    -r Examples/CampusCrowd/Tools/requirements.txt

# SharedPolicy's uv.lock pins CUDA torch (~several GB of nvidia wheels). Its deps
# (numpy/onnx/onnxruntime/onnxscript/torch/pytest) already live in the main CPU venv,
# so the locked sync is opt-in: SHAREDPOLICY_UV_SYNC=1 bash .devcontainer/post-create.sh
if [ "${SHAREDPOLICY_UV_SYNC:-0}" = 1 ]; then
    echo "==> SharedPolicy prototype (locked uv sync, Python 3.12)"
    (cd Prototypes/SharedPolicy && uv sync --python 3.12)
fi

# Example projects (e.g. Examples/CampusCrowd/Scripts/common.sh) symlink
# Plugins/Schola/{Binaries,Intermediate} to the repo root; make sure targets exist.
mkdir -p Binaries Intermediate

echo "==> Building ChaseSim"
(cd Examples/AdaptiveTag && sh Tools/ChaseSim/Build-ChaseSim.sh) || echo "ChaseSim build failed (non-fatal)"

echo "==> Fetching Git LFS assets (including Schola build libraries)"
if ! GIT_TERMINAL_PROMPT=0 git lfs pull; then
    echo "Git LFS pull failed. Set up your Git credentials and run 'git lfs pull' before building." >&2
fi

cat <<'MSG'

Dev container ready (Unreal Engine 5.8.4 at $UE_ROOT).
  UE build:       $UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh <Target>Editor Linux Development -Project=<abs .uproject>
  Headless run:   UnrealEditor-Cmd <abs .uproject> -nullrhi -unattended ...
  Schola tests:   pytest Test/core -o addopts=""   (engine_path preset via PYTEST_ADDOPTS)
  Example build:  bash Examples/CampusCrowd/Scripts/build.sh   (uses $UE_ENGINE_ROOT)
  SharedPolicy:   cd Prototypes/SharedPolicy && pytest -q   (uses main venv)
                  python -m benchmarks.export_models && python -m benchmarks.benchmark --config benchmarks/config.json --smoke --output benchmarks/runs/smoke.json
  ChaseSim:       cd Examples/AdaptiveTag && sh Tools/ChaseSim/ci-smoke.sh Saved/ChaseSim/chase_sim
MSG
