# Two-stage campus navigation

The `-Turns` demo uses the separate [learned junction-turn scorer](../README.md#learned-junction-turns-with-scholamass)
to choose exits on mapped paths. This page documents the ETA planner and
the earlier ordered-route encoder/decoder experiment.

Each student has a five-leg itinerary consisting of a trip from a gateway to
class, three class changes, and a trip to an exit. Unreal selects a route on
each leg and advances the itinerary on arrival.

## Architecture

1. **Plan each leg live.** Entrances, gateways, junctions, path endpoints, and
   intermediate vertices along longer paths form a 2,556-landmark graph.
   Links retain sidewalk geometry and include straight connections up to
   45 m verified against the walkability grid. The neural network takes the landmark and
   destination coordinates, straight-line distance, and features of the
   outgoing links. Dijkstra supplies training targets for the extra time a
   walk takes beyond the straight-line estimate. At runtime the model adds
   its predicted detour time to that estimate.

   Unreal groups departures by destination and submits one row per landmark
   through Schola's background inference scheduler the first time a
   destination is requested. Students wait at the stop while workers run
   ONNX inference. Unreal reuses the estimates for later trips to that stop. A* prioritizes landmarks by the sum of the walk time already
   incurred and the model's estimate of the remaining time. Unreal
   reconstructs the route from the graph's actual links. Students who depart
   together on the same leg share the planned route. If inference fails or
   expires, Dijkstra supplies a route from the same graph.
2. **Steer around corners.** The planner pulls bends toward straight lines
   while checking neighboring segments against the grid. The C++ walking
   controller caches the fastest visible suffix from each route point and
   scores shortcuts every 100 ms of simulated time. Short rollouts check
   whether the current velocity can make the turn without hitting a wall.
   Collision-checked movement runs every frame. Unreal processes independent
   walkers in parallel batches.

## Build the campus graph

The exporter uses the OpenStreetMap St. George campus boundary and building
outlines to discover public, ground-level entrances. It also matches
stand-alone mapped door points to a building when one outline is clearly
closer than the others. The exported graph has
193 building entrances across 96 buildings, plus four gateways. Each door
record has a stable OpenStreetMap node ID, a parent building ID, and the
original mapped coordinates. The navigation map assigns distinct, reachable
outdoor arrival points. The recorded approach offset shows where simplified
CAD footprints required moving those points away from a mapped door.

Student schedules choose a building uniformly and then one of its entrances.
Consecutive classes use different buildings. Additional exits, restricted
doors, and entrances missing from the source map require separately verified
data before they can become student destinations.

Use the campus OpenStreetMap extract described in
[the pedestrian route exporter](build_migration_routes.py). In the
devcontainer, regenerate the network and arrival points with:

```bash
python Tools/build_navigation_network.py /path/to/campus.osm
python Tools/build_navigation_map.py
```

## Train, evaluate, and run

Run from `Examples/CampusCrowd` in the devcontainer. Training uses PyTorch,
NumPy, pyproj, Shapely, ONNX, and a C++17 compiler. The shipped graph and
model let the example run immediately after building. Set `UE_ENGINE_ROOT`
to the Unreal installation in the devcontainer.

```bash
base=Saved/NavigationRL/expanded
python Tools/train_waypoint_navigation.py --output "$base" --export-live Content \
  --export-routes Content/Campus/navigation_waypoint_routes.json
Scripts/build.sh
CAMPUS_EDITOR=1 Scripts/run.sh -NavigationRL
```

The trainer writes the checkpoint and evaluation report under `--output`.
The live export writes `Content/Campus/navigation_landmarks.json` with the
graph and feature normalization scale. Unreal builds features from the graph
on demand. The export also writes `Content/Models/landmark_eta.onnx` with the
selected model. Add `-Mode=Sync` to run the same model one row at a time on
the game thread. `-Workers` and `-MaxBatch` set the background worker count
and maximum rows per call. Pass `-NavigationRoutes=/path/to/routes.json`
to use the exact next-hop table for comparison. The trainer exports this
compact format with `--export-routes` or `--export-exact`. The optional
`--export-learned` writes explicit routes selected by the learned heuristic.

## Train the ordered-route model

See the [ordered-route architecture and performance report](ORDERED_ROUTE_MODEL.md)
for tensor shapes, training details, and latency measurements.

The route model uses three message-passing layers to encode the campus graph
as 128-dimensional landmark embeddings. Cache the embeddings during decoding.
A GRU starts with the source and destination embeddings and scores outgoing
links at each hop. A mask excludes links outside the graph and previously
visited landmarks. The decoder returns an ordered landmark list and falls
back to Dijkstra if it runs out of eligible neighbors or reaches its hop limit.

The training labels are shortest-path next hops, including ties. Training
unrolls several hops at a time and sometimes follows the model's own choice
before asking the teacher for the best eligible next hop. Validation and
testing reserve whole destination buildings, and checkpoint selection uses
end-to-end greedy rollouts. The model sees the full graph topology at both
training and evaluation time.

From the devcontainer, train, refine, and decode a sample route:

```bash
base=Saved/NavigationRL/ordered
python Tools/train_ordered_routes.py --output "$base"
python Tools/train_ordered_routes.py --warm-start "$base/ordered_routes.pt" \
  --output "$base/refinement" --epochs 36 --window 8 --max-steps 128 \
  --learning-rate .0009 --regret-weight .45
python Tools/train_ordered_routes.py --output "$base/refinement" --route 2541 7
```

The decode command takes a start **landmark index** and a destination
**stop index** from the exported graph. It prints the landmark sequence and whether
the route required fallback. Each run writes a PyTorch checkpoint, a training report, and the actual
landmark sequences for both held-out test sets under its output directory.
These are offline training and evaluation tools. The Unreal demo uses the
ETA planner described above.

## Top-down view

`-NavigationRL` opens the earlier navigation mode in an orthographic map. For the 24-student cohort, the map displays up to four active
routes and their student markers. Use **WASD** to pan, **Q/E** to zoom,
**N** to toggle routes, **Space** to pause, and **1** to return to the map.
`Scripts/render-video.sh` records the map to
`Saved/Videos/campus-navigation-2d.mp4` on X11.

## Held-out route quality

All entrances of a building stay in the same split. The expanded-graph
evaluation used 60,000 training pairs, 6,000 validation pairs, and 1,200 test
pairs, sampled as landmark-to-stop trips. The selected model had a validation
mean absolute ETA error of 29.28 seconds.

| Planner | Mean expansions on 1,200 held-out trips | Routes matching shortest graph cost within 0.1% |
| --- | ---: | ---: |
| Dijkstra | 1,132.1 | 1,200 |
| Straight-line A* | 258.4 | 1,200 |
| Learned ETA | 232.0 | 740 |

The maximum learned route cost increase was 9.79%. The comparison measures
search work and graph cost. End-to-end timing against straight-line A*
remains unmeasured.

Run the navigation regression tests with
`python -m unittest discover -s Tools/tests -p test_navigation.py -v`.

## Shared environment

The map exporter samples the imported CAD building footprints on a 0.5 m
grid. A 0.30 m margin gives the student body clearance, and an additional
half-cell diagonal keeps raster cells clear at their corners. Stops come
from the mapped gateways and outdoor approaches and are moved to their
nearest walkable cell when necessary. The campus boundary is impassable.
Walkers may overlap so each can concentrate on navigation.

The same C++ header powers the vectorized headless environment and the
Unreal level. It sweeps each movement segment through grid cells, including
both sides of a diagonal corner. A blocked step reaches the wall and
can slide along the unobstructed axis.
An arrival is within 0.65 m of the stop with a clear final segment.
The training and runtime grid permit movement across open roads and plazas
as well as paths.

## Historical inference benchmark on the 78-landmark graph

Both modes use the same learned estimates to plan routes. The benchmark
compares batched inference on Schola's background workers with one
synchronous model call per row on the game thread.

The commands below document the earlier capture. Current runs use the
expanded graph and reuse destination estimates, so they measure a different
workload.

```bash
mkdir -p Saved/NavigationRL/live
for mode in Background Sync; do
  CAMPUS_EDITOR=1 Scripts/run.sh -NavigationRL -Agents=10000 -Mode="$mode" -MigrationRelease=10 \
    -MaxFPS=60 -ResX=1600 -ResY=900 -ForceRes -WarmupSeconds=0 -CaptureSeconds=11 \
    -Report="$PWD/Saved/NavigationRL/live/rendered-$mode.json"
done
```

The report records frame time, game-thread planning time, model rows and
calls, inference-group latency, and student legs routed, including fallback
counts. Game-thread planning includes request handling, graph search, and
route refinement. The HUD displays frame rate and live planner costs.

A local 1600 × 900 run on a Ryzen 9 9950X and RTX PRO 6000 Blackwell used two
Schola workers and a batch cap of 64. Both modes planned all 10,000 first
legs during capture without falling back to Dijkstra.

| Mode | FPS | Mean game-thread planning | p99 game-thread planning | Mean rows per model call |
| --- | ---: | ---: | ---: | ---: |
| Schola background | 59.86 | 1.37 ms/frame | 2.57 ms/frame | 59.45 |
| Synchronous | 59.85 | 2.59 ms/frame | 3.98 ms/frame | 1.00 |

The four-student video used one student per entrance. The live run planned
20 legs using 1,560 ETA predictions across 40 model calls, with zero
fallback routes. All four students reached their final exits in both modes.
[Local comparison video](../Saved/Videos/campus-live-vs-precomputed-4-entrances.mp4)

A separate fixed-step Unreal run completed all 120 legs in 24 five-leg
itineraries in each mode, with zero timeouts and zero fallback routes.

The earlier precomputed-route benchmark used the same display settings and
hardware. It reached 60 FPS in the top-down view with 100,000 walkers and
a p99 frame time of 16.77 ms. Pass
`-NavigationRoutes="$PWD/Content/Campus/navigation_waypoint_routes.json"`
to test the precomputed mode.
