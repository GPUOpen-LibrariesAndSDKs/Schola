"""Record CampusCrowd's top-down navigation view from an X11 display."""
import argparse
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.agents <= 100000 or args.seconds <= 0:
        parser.error("agents must be between 1 and 100,000 and seconds must be positive")
    if not os.environ.get("DISPLAY"):
        parser.error("an X11 display is required")

    project = args.project_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = project / "Saved/NavigationPresentation"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "recording-game.log"
    log.write_text("")
    with (work / "recording-launcher.log").open("w") as launcher_log:
        game = subprocess.Popen([
            str(project / "Scripts/run.sh"), "-NavigationRL", f"-Agents={args.agents}", "-MigrationTimeScale=3",
            "-ResX=1600", "-ResY=900", "-ForceRes", "-NoBorder", "-WinX=0", "-WinY=0",
            "-MaxFPS=60", f"-abslog={log}", "-unattended", "-NoSplash",
        ], stdout=launcher_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                game_log = log.read_text(errors="replace")
                if "Campus navigation:" in game_log or "navigation routes from" in game_log:
                    break
                if game.poll() is not None:
                    raise RuntimeError(f"Unreal exited with {game.returncode}. See {log}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out waiting for navigation. See {log}")
                time.sleep(.5)
            time.sleep(3)
            if game.poll() is not None:
                raise RuntimeError(f"Unreal exited before recording. See {log}")
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "x11grab", "-framerate", "30", "-draw_mouse", "0", "-video_size", "1600x900",
                "-i", os.environ["DISPLAY"], "-t", str(args.seconds), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-threads", "3", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ], check=True)
            print(output)
        finally:
            if game.poll() is None:
                game.terminate()
                try:
                    game.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    game.kill()
                    game.wait()


if __name__ == "__main__":
    main()
