# Linux development and packaging image for 9-AMD-Advanced-NPCs.
# Base: Epic's official UE 5.8.4 development image (requires `docker login ghcr.io`
# with a GitHub account linked to Epic Games and a read:packages token).
#   - Unreal Engine 5.8.4 at /home/ue4/UnrealEngine ($UE_ENGINE_ROOT), user ue4
#   - Schola Python package + tests        (Resources/python, Test/)
#   - SharedPolicy prototype               (Prototypes/SharedPolicy)
#   - ChaseSim C++ build / smoke tests     (Examples/AdaptiveTag)
#   - CampusCrowd Python tools             (Examples/CampusCrowd/Tools)
FROM ghcr.io/epicgames/unreal-engine:dev-slim-5.8.4

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ clang clang-format clangd gdb cmake ninja-build \
        git git-lfs openssh-client pkg-config curl ca-certificates less \
        doxygen graphviz ripgrep ffmpeg xauth xdotool xvfb \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# uv / uvx (manages Python 3.12 and all venvs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UE_ROOT=/home/ue4/UnrealEngine \
    UE_ENGINE_ROOT=/home/ue4/UnrealEngine \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_CACHE_DIR=/home/ue4/.cache/uv \
    VIRTUAL_ENV=/home/ue4/.venv \
    PATH=/home/ue4/.venv/bin:/home/ue4/UnrealEngine/Engine/Binaries/Linux:${PATH}

USER ue4
RUN uv python install 3.12
