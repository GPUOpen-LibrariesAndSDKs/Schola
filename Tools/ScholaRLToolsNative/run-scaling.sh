#!/bin/sh
# Builds ScalingBenchmark (plugin FLearner at 1-200 NPCs, no Unreal) and runs it.
# Usage: Tools/ScholaRLToolsNative/run-scaling.sh [outdir] [-- ScalingBenchmark args]
#   e.g. Tools/ScholaRLToolsNative/run-scaling.sh Saved/Scaling -- --npcs 1,50,100 --seconds 5
# Run on an otherwise idle machine; other load changes every number.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT="$REPO/Saved/ScalingBenchmark"
if [ $# -gt 0 ] && [ "$1" != "--" ]; then OUT=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
CXX=${CXX:-}
if [ -z "$CXX" ]; then
    if command -v clang++ >/dev/null 2>&1; then CXX=clang++; else CXX=g++; fi
fi
LEARNING="$REPO/Source/ScholaRLTools/Private/Learning"
echo "Building with $CXX into $OUT"
"$CXX" -std=c++20 -O2 -fno-rtti -fno-exceptions -ffp-contract=off -DRL_TOOLS_DEBUG -pthread \
    "-DSCHOLA_RLTOOLS_PROJECT_PROFILES=\"$HERE/ScalingProfiles.inl\"" \
    -I "$LEARNING" -isystem "$REPO/Source/ThirdParty/RLTools/include" \
    "$HERE/ScalingBenchmark.cpp" "$LEARNING/RLToolsLearner.cpp" \
    -o "$OUT/ScalingBenchmark"
"$OUT/ScalingBenchmark" --csv "$OUT/scaling.csv" "$@"
