#!/usr/bin/env bash
# Record the pedestrian view. Requires X11, ffmpeg, xdotool, and rg.
# Usage: Scripts/record.sh [output-dir] [extra game arguments...]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT="$(realpath -m "${1:-$PROJECT_DIR/Saved/Recording}")"
if [[ $# -gt 0 ]]; then shift; fi
for TOOL in ffmpeg xdotool rg; do
    command -v "$TOOL" >/dev/null || { echo "Missing $TOOL (X11 recording required)." >&2; exit 1; }
done
mkdir -p "$OUT"
LOG="$OUT/demo.log"
: > "$LOG"
"$PROJECT_DIR/Scripts/run.sh" -InferenceBenchmark -Mode=Background -View=2 -Agents=2000 \
    -WarmupSeconds=12 -CaptureSeconds=12 -Report="$OUT/run.json" \
    -Screenshot="$OUT/pedestrians.png" -abslog="$LOG" -unattended "$@" > "$OUT/launcher.log" 2>&1 &
CAMPUS_DEMO_PID=$!
trap 'kill "$CAMPUS_DEMO_PID" 2>/dev/null || true' EXIT
READY=0
for ((I=0; I<90; I++)); do
    if [[ -f "$LOG" ]] && rg -q 'Tracing Screenshot.*pedestrians' "$LOG"; then READY=1; break; fi
    kill -0 "$CAMPUS_DEMO_PID" 2>/dev/null || { echo "Game exited; see $LOG" >&2; exit 1; }
    sleep 1
done
[[ "$READY" == 1 ]] || { echo "Timed out waiting for the scene; see $LOG" >&2; exit 1; }
CAMPUS_WINDOW_ID="$(xdotool search --onlyvisible --name '^CampusCrowd ' | head -n1)"
ffmpeg -hide_banner -loglevel error -y -f x11grab -framerate 30 -draw_mouse 0 \
    -window_id "$CAMPUS_WINDOW_ID" -i "$DISPLAY" -t 8 -vf scale=1280:720 \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUT/campus-crowd-demo.mp4"
wait "$CAMPUS_DEMO_PID"
trap - EXIT
echo "$OUT/campus-crowd-demo.mp4"
