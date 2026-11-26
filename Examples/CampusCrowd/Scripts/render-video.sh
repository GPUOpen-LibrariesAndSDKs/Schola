#!/usr/bin/env bash
# Record the top-down navigation demo at 1600 × 900 from an X11 display.
# Usage: Scripts/render-video.sh [output.mp4] [--agents N] [--seconds S]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for TOOL in python3 ffmpeg; do
    command -v "$TOOL" >/dev/null || { echo "Missing $TOOL" >&2; exit 1; }
done
OUTPUT="$(realpath -m "${1:-$PROJECT_DIR/Saved/Videos/campus-navigation-2d.mp4}")"
if [[ $# -gt 0 ]]; then shift; fi
exec python3 "$PROJECT_DIR/Tools/render_demo_video.py" \
    --project-dir "$PROJECT_DIR" --output "$OUTPUT" "$@"
