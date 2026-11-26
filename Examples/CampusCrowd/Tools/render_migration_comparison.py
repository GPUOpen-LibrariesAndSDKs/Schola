"""Render a synchronized six-panel replay from Unreal's migration trace files."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("direct", "Direct walking", "Centreline • no yielding",
     ("-MigrationPolicy=Rules", "-MigrationYield=0", "-MigrationLane=0")),
    ("rules", "Rule-based", "72 cm lane • default yielding", ("-MigrationPolicy=Rules",)),
    ("rules-wide", "Rule-based / wider", "100 cm lane • 1.35× yielding",
     ("-MigrationPolicy=Rules", "-MigrationYield=1.35", "-MigrationLane=100")),
    ("learned", "Learned", "72 cm lane • default yielding", ("-MigrationPolicy=Learned",)),
    ("learned-wide", "Learned / wider", "100 cm lane • 1.35× yielding",
     ("-MigrationPolicy=Learned", "-MigrationYield=1.35", "-MigrationLane=100")),
    ("learned-staggered", "Learned / staggered", "90 s release window",
     ("-MigrationPolicy=Learned", "-MigrationRelease=90")),
)
PHASES = ("Arrival from the four gateways", "First class change", "Second class change",
          "Third class change", "Departure to the four gateways")
COLORS = ("#FDE18B", "#F89274", "#ABD9FF", "#83E0B9", "#DDB9FF", "#FFCAEA",
          "#BCF388", "#F0F4FF", "#FFC88C", "#94EAE8", "#C5C7FF")
GATE_LABELS = ("Spadina & Bloor", "Avenue & Bloor", "Queen’s Park TTC", "Spadina & College")
FPS = 30
PANEL = (586, 445)
MAP = (408, 400)
PANEL_X = (66, 667, 1268)
PANEL_Y = (125, 582)
FG, MUTED, ACCENT = "#edf1f5", "#aebdc8", "#86e2c2"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def load_case(path):
    grouped = defaultdict(list)
    with path.open() as source:
        info = json.loads(next(source))
        for line in source:
            frame = json.loads(line)
            grouped[frame["wave"]].append(frame)
    if list(grouped) != list(range(1, 7)) or any(
        grouped[leg][-1]["arrived"] != info["students"] for leg in range(1, 7)
    ):
        raise ValueError(f"{path} has an incomplete migration day")
    for frames in grouped.values():
        for frame in frames:
            frame["p"] = {row[0]: row[1:] for row in frame["p"]}
    grouped = {leg: ([frame["t"] - frames[0]["t"] for frame in frames], frames) for leg, frames in grouped.items()}
    return info, grouped


def projector(bounds):
    x0, y0, x1, y1 = bounds
    scale = min(MAP[0] / (x1 - x0), MAP[1] / (y1 - y0))
    xpad = (MAP[0] - (x1 - x0) * scale) / 2
    ypad = (MAP[1] - (y1 - y0) * scale) / 2

    def screen(x, y):
        return round((x - x0) * scale + xpad), round((y - y0) * scale + ypad)

    return screen


def raster_map(sections, routes, bounds, stops, detailed):
    project = projector(bounds)
    image = Image.new("RGB", MAP, "#141e27")
    draw = ImageDraw.Draw(image)
    fills = {"Curbs": "#273747", "Parks": "#263e3c", "roofs": "#374654", "walls": "#374654"}
    for name in ("Curbs", "Parks", "walls", "roofs"):
        points, faces = sections[name]
        triangles = points[faces]
        relevant = (
            (triangles[:, :, 0].max(axis=1) >= bounds[0])
            & (triangles[:, :, 0].min(axis=1) <= bounds[2])
            & (triangles[:, :, 1].max(axis=1) >= bounds[1])
            & (triangles[:, :, 1].min(axis=1) <= bounds[3])
        )
        for a, b, c in triangles[relevant]:
            draw.polygon((project(*a), project(*b), project(*c)), fill=fills[name])
    for a, b in routes:
        draw.line((project(*a), project(*b)), fill="#4b6879", width=1)
    for i, stop in enumerate(stops):
        x, y = project(stop["x"], stop["y"])
        radius = 4 if i < 4 else 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS[i], width=1)
        if i < 4 and not detailed:
            dx, dy, anchor = (7, -8, "la") if i in (0, 3) else (-7, -8, "ra")
            draw.text((x + dx, y + dy), GATE_LABELS[i], fill=FG, font=font(12, True), anchor=anchor,
                      stroke_width=2, stroke_fill="#141e27")
    return image, project


def traces_at(series, elapsed):
    times, frames = series
    index = max(0, bisect_right(times, elapsed) - 1)
    frame = frames[index]
    next_frame = frames[min(index + 1, len(frames) - 1)]
    amount = np.clip((elapsed - times[index]) / max(0.01, next_frame["t"] - frame["t"]), 0, 1)
    return frame, next_frame["p"], amount


def panel_for(base, project, series, elapsed, count, name, settings, detailed):
    frame, following, amount = traces_at(series, elapsed)
    panel = Image.new("RGB", PANEL, "#111c26")
    panel.paste(base, (0, 45))
    draw = ImageDraw.Draw(panel)
    title, label, small = FONTS[0:3]
    draw.text((12, 12), name, fill=FG, font=title)
    draw.multiline_text((427, 53), settings.replace(" • ", "\n"), fill=MUTED, font=small, spacing=4)
    draw.text((427, 115), f"{len(frame['p']):3d}", fill=ACCENT, font=title)
    draw.text((427, 143), "walking", fill=MUTED, font=label)
    draw.text((427, 192), f"{frame['arrived']:3d} / {count}", fill=FG, font=title)
    draw.text((427, 220), "arrived this wave", fill=MUTED, font=label)
    draw.text((427, 270), f"{100 * frame['close_s'] / max(1, frame['walk_s']):.1f}%", fill=FG, font=title)
    draw.text((427, 298), "close-contact time", fill=MUTED, font=label)
    draw.rectangle((426, 367, 567, 370), fill="#22333e")
    draw.rectangle((426, 367, 426 + round(141 * frame["arrived"] / count), 370), fill=ACCENT)
    radius = 3 if detailed else 2
    for i, (x, y, stop) in frame["p"].items():
        end_x, end_y, _ = following.get(i, (x, y, stop))
        sx, sy = project(x + (end_x - x) * amount, y + (end_y - y) * amount)
        if not radius <= sx < MAP[0] - radius or not radius <= sy < MAP[1] - radius:
            continue
        sy += 45
        draw.ellipse((sx-radius, sy-radius, sx+radius, sy+radius), fill=COLORS[stop],
                     outline="#10181b" if detailed else None)
    return panel


def run_cases(directory, agents, seed, jobs):
    directory.mkdir(parents=True, exist_ok=True)

    def run_case(case):
        name, _, _, settings = case
        log = directory / f"{name}.log"
        with (directory / f"{name}.stdout").open("w") as output:
            try:
                subprocess.run(
                    [str(ROOT / "Scripts/run.sh"), "-Migrations", f"-Agents={agents}", f"-MigrationSeed={seed}",
                     "-MigrationFixedStep=0.1", "-ExitAfterMigrations", "-nullrhi", "-unattended", "-multiprocess",
                     "-MaxFPS=1000", f"-MigrationTrace={directory / (name + '.jsonl')}", f"-abslog={log}", *settings],
                    stdout=output, stderr=subprocess.STDOUT, check=True, timeout=420,
                    env={**os.environ, "CAMPUS_EDITOR": "1"},
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"{name} failed; see {log}") from error
        print(f"{name} complete", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for _ in pool.map(run_case, CASES):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wave-seconds", type=float, default=9)
    parser.add_argument("--detail-seconds", type=float, default=12)
    parser.add_argument("--simulate", action="store_true", help="Run the six Unreal configurations before rendering")
    parser.add_argument("--agents", type=int, default=240)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=3)
    args = parser.parse_args()
    args.traces = args.traces.resolve()
    if args.simulate:
        run_cases(args.traces, args.agents, args.seed, args.jobs)
    if args.wave_seconds <= 0 or args.detail_seconds < 0:
        parser.error("Choose a positive duration per wave and a non-negative detail duration")

    traces = [load_case(args.traces / f"{stem}.jsonl") for stem, _, _, _ in CASES]
    info = traces[0][0]
    for header, _ in traces[1:]:
        if any(header[field] != info[field] for field in ("seed", "students", "stops")):
            raise ValueError("Traces must share the seed, cohort size and route data")
    content = ROOT / "Content/Campus"
    campus = json.loads((content / "campus.json").read_text())
    routes_json = json.loads((content / "migration_routes.json").read_text())
    nodes = routes_json["nodes"]
    pairs = {tuple(sorted(pair)) for path in routes_json["paths"] for pair in zip(path, path[1:])}
    routes = [(nodes[a], nodes[b]) for a, b in sorted(pairs)]
    sections = {
        section["name"]: (np.asarray(section["vertices"])[:, :2], np.asarray(section["triangles"]).reshape(-1, 3))
        for section in campus["sections"] if section["name"] in {"Curbs", "Parks", "roofs", "walls"}
    }
    all_points = np.asarray(nodes)
    min_x, min_y = all_points.min(axis=0) - 2200
    max_x, max_y = all_points.max(axis=0) + 2800
    overview = raster_map(sections, routes, (min_x, min_y, max_x, max_y), info["stops"], detailed=False)
    spadina = info["stops"][0]
    x, y = spadina["x"], spadina["y"]
    detail = raster_map(sections, routes, (x - 3400, y - 1200, x + 4200, y + 6800), info["stops"], detailed=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-video_size", "1920x1080", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast",
              "-crf", "19", "-threads", "6", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    work = [(wave, max(frames[wave][0][-1] for _, frames in traces), args.wave_seconds, False)
            for wave in range(1, 6)]
    if args.detail_seconds:
        work.append((1, args.detail_seconds, args.detail_seconds, True))
    with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as process:
        try:
            for wave, max_time, seconds, detailed in work:
                count = max(2, round(seconds * FPS))
                base = Image.new("RGB", (1920, 1080), "#0b141c")
                draw = ImageDraw.Draw(base)
                draw.text((66, 27), "CampusCrowd  /  Five migrations", fill=FG, font=FONTS[3])
                phase = "Spadina & Bloor gate at walking speed" if detailed else f"Migration {wave} of 5  /  {PHASES[wave - 1]}"
                draw.text((67, 77), phase, fill=ACCENT, font=FONTS[0])
                draw.text((1854, 36), f"{info['students']} students per panel • same routes and seed",
                          fill=MUTED, font=FONTS[1], anchor="ra")
                draw.text((66, 1048), "Dots are coloured by destination. Close contact means another student was within 45 cm.",
                          fill=MUTED, font=FONTS[1])
                draw.text((1854, 1048), "Paths © OpenStreetMap contributors  •  Geometry © City of Toronto",
                          fill=MUTED, font=FONTS[1], anchor="ra")
                for number in range(count):
                    fraction = number / (count - 1)
                    elapsed = fraction * max_time + (23 if detailed else 0)
                    image = base.copy()
                    draw = ImageDraw.Draw(image)
                    draw.text((1854, 78), f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d} simulated within wave",
                              fill=FG, font=FONTS[0], anchor="ra")
                    for i, ((_, title, settings, _), (_, frames)) in enumerate(zip(CASES, traces)):
                        panel = panel_for(*(detail if detailed else overview), frames[wave], elapsed,
                                          info["students"], title, settings, detailed)
                        image.paste(panel, (PANEL_X[i % 3], PANEL_Y[i // 3]))
                    process.stdin.write(image.tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the comparison")

    summary = {}
    for (case, _, _, _), (_, grouped) in zip(CASES, traces):
        finished = grouped[6][1][-1]
        summary[case] = {
            "students": finished["arrived"],
            "day_seconds": finished["t"],
            "model_decisions": finished["decisions"],
            "close_contact_percent": round(100 * finished["close_s"] / finished["walk_s"], 2),
            "wave_last_arrival_seconds": [next(frame["t"] for frame in grouped[wave][1]
                if frame["arrived"] == finished["arrived"]) for wave in range(1, 6)],
        }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output)


FONTS = (font(21, True), font(16), font(12), font(36, True))


if __name__ == "__main__":
    main()
