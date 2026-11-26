#!/usr/bin/env bash
# Measure both modes one after another with identical settings, then summarize.
# Usage: Scripts/compare.sh [output-dir] [extra game arguments...]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT="$(realpath -m "${1:-$PROJECT_DIR/Saved/Comparison}")"; shift || true
mkdir -p "$OUT"
for MODE in Sync Background; do
    echo "Measuring $MODE..."
    "$PROJECT_DIR/Scripts/run.sh" -InferenceBenchmark -ResX=1600 -ResY=900 -Mode="$MODE" -WarmupSeconds=10 -CaptureSeconds=30 \
        -Report="$OUT/${MODE,,}.json" -abslog="$OUT/${MODE,,}.log" "$@" > /dev/null || STATUS=$?
    # Exit 3: the report was written but the crowd missed its target decision rate.
    if [[ "${STATUS:-0}" == 3 ]]; then echo "$MODE missed the target decision rate; see $OUT/${MODE,,}.json" >&2; MISSED=3
    elif [[ "${STATUS:-0}" != 0 ]]; then exit "$STATUS"; fi
    unset STATUS
done
python3 "$PROJECT_DIR/Tools/summarize.py" "$OUT/sync.json" "$OUT/background.json" | tee "$OUT/summary.md"
exit "${MISSED:-0}"
