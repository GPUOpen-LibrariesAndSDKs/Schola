# North Campus crowd demo

CampusCrowd shows a pedestrian map beside a fixed 3D overview. Both use the
same live simulation. A student's color follows their current destination,
and every building with students heading to it is highlighted in that color.
Entrances to the same building share a color.

| Preset | What it shows |
| --- | --- |
| **Four students** (`F1`, default) | One student from each entrance, four classes each, and an exit. The map shows each current leg's shortest route and the student's latest decision. |
| **Scaling test** (`F2`) | A configurable cohort with applied turns per second, rows per model call, and game-thread inference time. |

**Tab** switches between ScholaMass background batching and synchronous
inference. **Space** pauses the simulation. Switching presets starts a new
campus day.

Run these commands from `Examples/CampusCrowd` in the devcontainer.

```bash
Scripts/build.sh
CAMPUS_EDITOR=1 Scripts/run.sh -Demo=Four
CAMPUS_EDITOR=1 Scripts/run.sh -Demo=Scale -Students=1600
```

`-Students=N` sets the scaling cohort from 1 to 500,000 (default 650).
`-Turns -Agents=N` also selects the scaling preset. The nine mapped
destinations without matching CAD roofs use simplified footprint geometry.
[D4 CampusCrowd recordings](../../deliverables/D4/index.html#demo-background)

## Learned junction turns with ScholaMass

### Architecture

At a junction with multiple eligible exits, a student submits one
96-float row for up to eight exits. Each exit has 12 features covering
heading, turn angle, path length,
and progress toward the goal. Progress uses both straight-line distance
and a graph-distance estimate based on 96 fixed anchors. Unreal pads
unused slots and masks ineligible exits. [Features](Tools/train_turn_policy.py#L62)

A shared `12 → 32 → 1` ReLU network gives **one score per exit**.
The same weights serve every exit at every junction, so the model has
**449 parameters** (`12 × 32 + 32 = 416` in the hidden layer and
`32 × 1 + 1 = 33` in the output layer). The output slot identifies
the outgoing link. If left, straight, and right score `0.2`, `1.1`,
and `0.7`, Unreal takes straight. If straight leads to a visited
landmark, it takes right. [Model](Tools/train_turn_policy.py#L20) ·
[Turn selection](Source/CampusCrowd/CampusTurnDay.cpp#L330)

Students follow the mapped link polylines. They take a sole eligible
exit directly and retrace their last edge when they run out of
unvisited options. ScholaMass batches decisions across students with
multiple eligible exits. Students wait at the junction and retry stalled
requests after 250 ms by default (`-TurnRetryMs`).

### Training and route quality

The model learns to imitate Dijkstra. Each exit's training cost is its
length plus the shortest remaining graph distance from its neighbor to
the goal. The loss rewards all exits tied for the lowest cost within
1 mm and penalizes the expected extra distance of other choices.

The run used 42 AdamW epochs, with 24,576 sampled decisions per epoch.
Entire destination buildings were held out, yielding 149 training,
16 validation, and 32 test stops on the same graph. The checkpoint
was selected by p95 distance across 256 validation trips, with median
distance breaking ties. [Trainer](Tools/train_turn_policy.py#L149) ·
[Training report](Saved/NavigationRL/turns/report.json)

```bash
python Tools/train_turn_policy.py --export Content
```

| Held-out stop-to-stop metric | Result |
| --- | ---: |
| Trips completed | 1,200 / 1,200 |
| Trips that retraced at least one edge | 114 (9.5%) |
| Median distance relative to Dijkstra | 1.010× |
| p95 distance relative to Dijkstra | 1.307× |
| Maximum distance relative to Dijkstra | 18.489× |

Distances include retraced edges.

### Unreal performance and traces

Capture the two inference modes on the same seeded campus schedule.

```bash
for mode in Background Sync; do
  CAMPUS_EDITOR=1 Scripts/run.sh -Demo=Scale -Students=1600 -MigrationTimeScale=16 \
    -Mode="$mode" -nullrhi -unattended -WarmupSeconds=5 -CaptureSeconds=10 \
    -Report="$PWD/Saved/turn-$mode.json"
done
```

The reports include requests, applied turns, retraced edges, retries,
model rows and calls, p99 request-to-turn latency, and p99 game-thread
inference time. The latency clock starts at the first request and includes
time spent retrying. Reported latency uses 1 ms bins capped at 1,000 ms.

The 1,600-student headless run used two workers, a batch cap of 64, and a
60 FPS target. Both modes recorded zero inference retries or dropped rows.

| Metric | ScholaMass background | Synchronous |
| --- | ---: | ---: |
| Applied turns per second | 785 | 788 |
| Mean rows per model call | 13.1 | 1.0 |
| Mean game-thread inference per frame | 0.087 ms | 0.133 ms |
| p99 game-thread inference per frame | 0.141 ms | 0.212 ms |
| p99 request-to-applied-turn latency | 18 ms | 1 ms |

[Background capture](Evidence/d4-turn-benchmark/benchmark-Background.json) ·
[Synchronous capture](Evidence/d4-turn-benchmark/benchmark-Sync.json)

At 300,000 students in the packaged headless demo, ScholaMass reached
**54.09 FPS** and synchronous inference reached **38.30 FPS**.
Both modes applied about 9,650 turns per second with zero dropped rows.
[ScholaMass capture](Evidence/d4-turn-benchmark/matched-300000-Background.json) ·
[Synchronous capture](Evidence/d4-turn-benchmark/matched-300000-Sync.json)

```bash
for mode in Background Sync; do
  Saved/Package/Linux/CampusCrowd.sh -Demo=Scale -Students=300000 -Mode="$mode" \
    -MigrationSeed=301 -MigrationTimeScale=1 -Workers=2 -MaxBatch=64 -MaxFPS=60 \
    -nullrhi -nosound -unattended -NoSplash -WarmupSeconds=32 -CaptureSeconds=10 \
    -Report="$PWD/Saved/turn-300000-$mode.json"
done
```

Both inference modes completed the trace example with identical paths.
Each student retraced one edge. A separate run limited the background
queue and admission budget to one row and used a 100 ms retry interval.
It issued 34 retries and all 32 students arrived via the same paths.

Record the ordered landmark choices between two stop indices.

```bash
CAMPUS_EDITOR=1 Scripts/run.sh -Turns -Agents=32 -TurnStart=167 -TurnGoal=118 \
  -MigrationRelease=0 -MigrationTimeScale=16 -ExitAfterMigrations \
  -TurnTrace="$PWD/Saved/turn-sequences.jsonl"
```

Every consecutive landmark pair identifies an outgoing path. Each
completed leg records the walked sequence and the number of retraced
edges. A landmark may appear again during retracing.

## Earlier ETA navigation demo

Run from the project’s devcontainer after fetching the Git LFS assets.
Set `UE_ENGINE_ROOT` to the Unreal Engine 5.8 installation. The campus
geometry comes from the City of Toronto’s [3D massing data](https://opendata.toronto.ca/city.planning/3d-massing/2025/AutoCAD/50H_NORTH.dwg).

```bash
cd Examples/CampusCrowd
Scripts/build.sh
CAMPUS_EDITOR=1 Scripts/run.sh -NavigationRL
```

The default cohort is 24 students. Click the viewport to focus it. Use
**WASD** to pan, **E/Q** to zoom in/out, **N** to toggle the sampled route
overlay, and **Space** to pause. Colored lines show the planned routes of up
to four walking students. The highlights change as students depart and reach
stops. Press **1** to return to the map or
**2–4** for perspective camera presets. `-Agents=N` sets the cohort size,
and `-MigrationTimeScale=N` speeds up the campus day.

To compare live routing with the exact next-hop table on the same graph,
run these commands separately. The seed gives both runs the same student
itineraries.

```bash
CAMPUS_EDITOR=1 Scripts/run.sh -NavigationRL -Agents=4 -MigrationSeed=322 -MigrationRelease=68 -MigrationTimeScale=16
CAMPUS_EDITOR=1 Scripts/run.sh -NavigationRL -Agents=4 -MigrationSeed=322 -MigrationRelease=68 -MigrationTimeScale=16 \
  -NavigationRoutes="$PWD/Content/Campus/navigation_waypoint_routes.json"
```

The earlier comparison recording used the original 78-landmark graph. [Four-student comparison](Saved/Videos/campus-live-vs-precomputed-4-entrances.mp4)

Record the top-down view from an X11 display that accommodates a 1600 × 900
window. The recorder uses a 3× simulation setting so progress is visible in
a short clip.

```bash
CAMPUS_EDITOR=1 Scripts/render-video.sh
```

Generated recordings are in the local [Saved/Videos](Saved/Videos) archive,
including previous navigation experiments and the
[large-crowd scaling video](Saved/Videos/campus-navigation-scaling-optimized.mp4).

## Five campus migrations

After building, start the scheduled crowd from the devcontainer.

```bash
CAMPUS_EDITOR=1 Scripts/run.sh -Migrations -Agents=650 -MigrationTimeScale=8
```

Students enter through Spadina & Bloor, Avenue & Bloor, the northwest stairs at Queen’s Park station, or Spadina & College. Each student visits four randomly selected teaching buildings and then exits through one of the four gateways. Consecutive classes use different buildings. Each wave ends when everyone reaches the destination, and students wait indoors before the next wave. Sidewalks and crossings come from [OpenStreetMap contributors, ODbL 1.0](https://www.openstreetmap.org/copyright). The route generator checks the network against the imported CAD footprints and uses outdoor approaches where a mapped entrance lies inside the simplified shell.

The learned policy sees the route-relative position and velocity, upcoming turn, remaining distance, preferred lane and the four nearest pedestrians' relative positions and velocities. It returns desired forward and lateral speed. The day schedule and route table handle destinations. The ONNX model was trained by imitating a rule-based walking controller on synthetic encounters. Use **Tab** to switch between Schola background inference and the synchronous baseline while comparing game-thread inference time, decisions per second and rows per call in the HUD. Only walking students hold background inference slots.

| Option | Default | Effect |
| --- | --- | --- |
| `-MigrationPolicy` | `Learned` | `Rules` runs the controller directly |
| `-MigrationSeed` | 301 | Replays the same choice of gates, buildings and student paces |
| `-MigrationYield` | 1.0 | Scales slowing and sidestepping for nearby students |
| `-MigrationLane` | 72 | Preferred rightward offset in centimetres, subject to route clearance |
| `-MigrationRelease` | 30 | Release window within each wave in simulated seconds |
| `-MigrationDwell` | 18 | Time indoors between waves in simulated seconds |
| `-MigrationTimeScale` | 1 | Scales the simulation clock. In interactive play, model decisions are scheduled by wall time |

Render a synchronized comparison of six configurations. The capture records actual positions from six fixed-step Unreal runs using the same routes and cohort, then renders them as an overhead replay. Fixed-step learned runs use synchronous inference to make playback repeatable. The final shot shows the Spadina & Bloor arrival area at walking speed. The command requires NumPy, Pillow and FFmpeg.

```bash
python Tools/render_migration_comparison.py Saved/MigrationComparison \
  Saved/Videos/campus-migrations.mp4 --simulate
```

For inference performance, use a wall-clock capture. This measures the game-thread costs of the two inference paths for the same seeded cohort, while accounting for the number of students walking at each frame.

```bash
mkdir -p Saved/MigrationComparison
for mode in Background Sync; do
  CAMPUS_EDITOR=1 Scripts/run.sh -Migrations -Agents=1200 -MigrationTimeScale=16 \
    -Mode="$mode" -nullrhi -unattended -WarmupSeconds=14 -CaptureSeconds=15 \
    -Report="$PWD/Saved/MigrationComparison/perf-$mode.json"
done
```

A 15-second headless capture at 10 decisions/s per walking student measured about 1,155 walking students in background mode and 1,163 in sync mode. Both held the 60 FPS cap. Background delivered 99.95% of its active decision target, discarded 14 rows, and had zero rejected submissions.

| Inference mode | Mean game-thread inference | p99 game-thread inference | Mean rows per model call |
| --- | ---: | ---: | ---: |
| Schola background | 0.151 ms/frame | 0.208 ms/frame | 51.5 |
| Synchronous | 0.517 ms/frame | 0.671 ms/frame | 1.0 |

[Route export](Tools/build_migration_routes.py) requires an OpenStreetMap extract, pyproj and Shapely. [Policy training](Tools/train_migration_policy.py) requires NumPy, PyTorch and ONNX.

## Performance

On 1,200 trips to held-out buildings in the expanded graph, the neural
estimates reduced average A* expansions from 258.4 to 232.0 compared with
a straight-line heuristic. End-to-end routing time against straight-line A*
remains unmeasured. [Held-out routing results](Tools/NAVIGATION_RL.md#held-out-route-quality)

On the original 78-landmark graph, a rendered 10,000-student run with Schola background inference averaged
1.37 ms/frame of game-thread planning versus 2.59 ms/frame with
`-Mode=Sync`, which runs the same model one row at a time on the game thread.
Both modes ran at around 60 FPS. [Runtime benchmark](Tools/NAVIGATION_RL.md#historical-inference-benchmark-on-the-78-landmark-graph)

The D3 background-inference benchmark remains available through
`Scripts/compare.sh` or the explicit `-InferenceBenchmark` option.
[Historical benchmark and validation](Evidence/d3-validation-2026-11-05/README.md).

<a id="local-validation-and-deployment-d3"></a>

## Validation and packaging

Use the [Docker devcontainer](../../README.md#docker-devcontainer) from this directory. The Python
suite covers CAD import, navigation, migration routes, and learned junction turns.

```bash
python -m pytest Tools/tests -q
Scripts/build.sh
"$UE_ENGINE_ROOT/Engine/Binaries/Linux/UnrealEditor-Cmd" "$PWD/CampusCrowd.uproject" \
  -nullrhi -unattended -nosound \
  "-ExecCmds=Automation RunTests Schola.Inference.Background+Schola.Mass" \
  "-TestExit=Automation Test Queue Empty" "-ReportExportPath=$PWD/Saved/Automation"
Scripts/package.sh
Saved/Package/Linux/CampusCrowd.sh -Demo=Scale -Students=32 -MigrationTimeScale=16 -nullrhi -unattended -nosound \
  -WarmupSeconds=1 -CaptureSeconds=3 -Report="$PWD/Saved/package-smoke.json"
```

Check `Saved/Automation/index.json` for failed or missing tests. The smoke report should contain
applied turns and no dropped rows. On a graphics-capable run, start `Saved/Package/Linux/CampusCrowd.sh
-Demo=Four` and check both views, destination colors, preset and mode switching, pause, and normal
close. Keep the full package directory together when distributing it.

On UE 5.8.4 in Docker, all 43 background and Mass tests completed with zero failures.
The packaged smoke run applied 78 turns with zero dropped rows. The Python suite passed 19 tests
and 96 subtests.

The [D3 validation record](Evidence/d3-validation-2026-11-05/README.md) covers UE 5.8.3/Linux, including
the 43 background and Mass automation tests and the earlier packaged inference benchmark.

Optional full CAD validation: place the original `50H_NORTH.dwg` at the repository root, install [Tools/requirements.txt](Tools/requirements.txt), then run `python Tools/validate_campus.py`. It checks the source hash, finite vertices, triangle indices, connected reciprocal routes, and three-metre building clearance. [Recorded geometry validation](Evidence/geometry-validation.json) is available without that source file.

Attach devcontainer validation results to the PR for review. Package the selected release commit and retain the test report, smoke report, build log, engine version, and commit ID with the release.

## Non-functional requirements

- **Performance:** the navigation HUD shows frame rate, active walkers, completions, and timeouts. Timed captures record frame and movement costs. The inference benchmark also reports throughput and action latency.
- **Input validation and errors:** crowd size, decision rate, worker count, and camera preset are clamped. Submissions are clamped to queue capacity with a warning. Missing route data or model failures produce an explicit error. Missed decision targets are reported in text as well as color.
- **Usability:** the default map view supports pan, zoom, pause, and an overlay of selected routes. A full accessibility audit has not been completed.
- **Local security:** the unused Android file server and its network access are disabled, and its generated token was removed from the current config. This does not remove it from Git history. The demo needs no remote service credentials; only use trusted models and CAD files.

D2 reflection identified single-machine testing and late packaging as weaknesses. D3 adds a repeatable campus workflow, CAD regression tests, geometry checks, and explicit overload reporting. Independent-machine validation remains open. D4 prioritizes clearer setup and controls and less integration boilerplate.
