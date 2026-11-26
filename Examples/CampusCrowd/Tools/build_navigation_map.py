"""Export the common navigation grid from the campus CAD footprints and outdoor stops."""
import json
from pathlib import Path

import numpy as np
import shapely

from build_migration_routes import building_footprints
from navigation_env import NativeNavigation, NavigationMap, CELL_SIZE

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "Content/Campus"
BODY_RADIUS = .30


def main():
    campus = json.loads((CONTENT / "campus.json").read_text())
    network = CONTENT / "navigation_network.json"
    routes = json.loads((network if network.exists() else CONTENT / "migration_routes.json").read_text())
    ground = np.asarray(next(part["vertices"] for part in campus["sections"] if part["name"] == "ground"))
    lower = ground[:, :2].min(axis=0) / 100
    upper = ground[:, :2].max(axis=0) / 100
    dimensions = np.ceil((upper - lower) / CELL_SIZE).astype(int)
    # Buffer for the walking body and the furthest corner of a sampled grid cell.
    obstacles = shapely.buffer(building_footprints(campus), BODY_RADIUS + CELL_SIZE / np.sqrt(2))
    shapely.prepare(obstacles)
    axes = [start + (np.arange(n) + .5) * CELL_SIZE for start, n in zip(lower, dimensions)]
    walls = shapely.intersects_xy(obstacles, axes[0][None, :], axes[1][:, None])
    nav = NavigationMap(walls, tuple(lower))
    stops = []
    occupied = set()
    reachable = ~walls
    for stop in routes["stops"]:
        target = np.asarray(routes["nodes"][stop["node"]]) / 100
        ix, iy = np.floor((target - lower) / CELL_SIZE).astype(int)
        max_offset = 25 if "building_id" in stop else 5
        radius = round(max_offset / CELL_SIZE)
        xs = np.arange(max(0, ix-radius), min(dimensions[0], ix+radius+1))
        ys = np.arange(max(0, iy-radius), min(dimensions[1], iy+radius+1))
        cy, cx = np.where(reachable[np.ix_(ys, xs)])
        if not len(cx):
            raise ValueError(f"No accessible placement for {stop['name']}")
        candidates = np.column_stack((axes[0][xs[cx]], axes[1][ys[cy]]))
        distances = np.linalg.norm(candidates - target, axis=1)
        order = np.argsort(distances, kind="stable")
        for index in order[distances[order] <= max_offset]:
            candidate = candidates[index]
            position = tuple(candidate)
            if position not in occupied:
                break
        else:
            raise ValueError(f"No distinct arrival point for {stop['name']}")
        occupied.add(position)
        arrival = {**{key: value for key, value in stop.items() if key != "node"}, "position": position}
        if "building_id" in stop:
            arrival.update(door_position=target.tolist(), approach_offset_m=round(float(distances[index]), 2))
        stops.append(arrival)
        if len(stops) == 1:
            native = NativeNavigation(nav, ROOT / "Saved/NavigationRL/cache")
            reachable = np.isfinite(native.field(position))
    nav.save(CONTENT / "navigation_grid.bin")
    (CONTENT / "navigation_stops.json").write_text(
        json.dumps({"gate_count": routes["gate_count"], "stops": stops}, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {nav.width} × {nav.height} cells and {len(stops)} stops")


if __name__ == "__main__":
    main()
