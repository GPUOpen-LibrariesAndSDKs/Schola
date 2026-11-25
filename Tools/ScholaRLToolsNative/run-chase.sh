#!/bin/sh
# Builds ChaseAdaptation (the D3 chase adaptation test on the plugin's FLearner, no Unreal) and runs it
# on several seeds in parallel. Criterion 5 passes only if every seed passes criteria 1-4.
# Usage: [SEEDS="1 2 3"] [JOBS=3] Tools/ScholaRLToolsNative/run-chase.sh [outdir] [-- ChaseAdaptation args]
#   e.g. SEEDS="4 5 6" Tools/ScholaRLToolsNative/run-chase.sh Saved/chase-r20k -- --replay 20000
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT="$REPO/Saved/ChaseAdaptation"
if [ $# -gt 0 ] && [ "$1" != "--" ]; then OUT=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SEEDS=${SEEDS:-"1 2 3"}
JOBS=${JOBS:-3}
CXX=${CXX:-}
if [ -z "$CXX" ]; then
    if command -v clang++ >/dev/null 2>&1; then CXX=clang++; else CXX=g++; fi
fi
LEARNING="$REPO/Source/ScholaRLTools/Private/Learning"
CHASE="$REPO/Examples/AdaptiveTag/Source/OnlineSACRuntime"
BIN="$OUT/ChaseAdaptation"
echo "Building with $CXX into $OUT"
"$CXX" -std=c++20 -O2 -fno-rtti -fno-exceptions -ffp-contract=off -DRL_TOOLS_DEBUG -pthread \
    "-DSCHOLA_RLTOOLS_PROJECT_PROFILES=\"$HERE/ChaseProfiles.inl\"" \
    -I "$LEARNING" -I "$CHASE/Public" -isystem "$REPO/Source/ThirdParty/RLTools/include" \
    "$HERE/ChaseAdaptation.cpp" "$LEARNING/RLToolsLearner.cpp" \
    "$CHASE/Private/ChaseWorld.cpp" "$CHASE/Private/RunnerPatterns.cpp" "$CHASE/Private/ChaseConfig.cpp" \
    -o "$BIN"

# Run seeds JOBS at a time; each writes seed<N>.txt and seed<N>.csv.
RUNNING=0
for SEED in $SEEDS; do
    "$BIN" --seed "$SEED" --csv "$OUT/seed$SEED.csv" "$@" > "$OUT/seed$SEED.txt" 2>&1 &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$JOBS" ]; then wait; RUNNING=0; fi
done
wait

FAILED=0
for SEED in $SEEDS; do
    sed -n '/^RESULT/,$p' "$OUT/seed$SEED.txt"
    grep -q '^FAIL\|^error' "$OUT/seed$SEED.txt" && FAILED=1
    grep -q '^RESULT' "$OUT/seed$SEED.txt" || FAILED=1
done
if [ "$FAILED" -eq 0 ]; then echo "PASS criterion5: criteria 1-4 hold on seeds $SEEDS"; else echo "FAIL criterion5: see above"; fi
exit "$FAILED"
