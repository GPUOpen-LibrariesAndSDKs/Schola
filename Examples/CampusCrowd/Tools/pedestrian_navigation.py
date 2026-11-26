"""Plan campus walks on the mapped pedestrian network and follow their geometry."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from build_migration_routes import shortest_paths
from navigation_env import ROOT

@dataclass
class PedestrianRoute:
    points: np.ndarray
    decision_indices: list[int]
    edges: np.ndarray = field(init=False)
    lengths: np.ndarray = field(init=False)
    distances: np.ndarray = field(init=False)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, np.float32)
        self.edges = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.edges, axis=1)
        if not len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("A pedestrian route requires nonzero segments")
        self.distances = np.r_[0., np.cumsum(self.lengths, dtype=np.float64)]


def trace_grid_connector(native, distance_field, start, end):
    """Follow an existing stop distance field to connect a nearby path to the stop."""
    nav = native.map
    x, y = nav.cell_indices(start)
    end_cell = tuple(nav.cell_indices(end))
    origin = np.asarray(nav.origin, np.float32)
    route = [start]
    while (x, y) != end_cell:
        candidates = []
        current = distance_field[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < nav.width and 0 <= ny < nav.height):
                    continue
                remaining = distance_field[ny, nx]
                if remaining < current and not (nav.walls[y, nx] or nav.walls[ny, x]):
                    candidates.append((remaining + nav.cell * np.hypot(dx, dy), nx, ny))
        if not candidates:
            raise ValueError("No walkable grid connector to campus stop")
        _, x, y = min(candidates)
        route.append(origin + (np.asarray([x, y], np.float32) + .5) * nav.cell)
    if (route[-1] != end).any():
        route.append(end)
    # Keep the furthest visible cell centre at each bend.
    route = np.asarray(route)
    simplified = [route[0]]
    index = 0
    while index < len(route) - 1:
        candidates = route[index + 1:]
        starts = np.broadcast_to(route[index], candidates.shape)
        visible = np.flatnonzero(native.clear_segments(starts, candidates))
        if not len(visible):
            raise ValueError("Grid connector leaves the walkable region")
        index += int(visible[-1]) + 1
        if (simplified[-1] != route[index]).any():
            simplified.append(route[index])
    return np.asarray(simplified, np.float32)


class PedestrianNetwork:
    def __init__(self, native, goals, fields=None, source=None):
        if source is None:
            directory = ROOT / "Content/Campus"
            source = directory / "navigation_network.json"
            if not source.exists():
                source = directory / "migration_routes.json"
        data = json.loads(Path(source).read_text())
        self.attribution = data["source"]
        self.points = np.asarray(data["nodes"], np.float32) / 100
        self.stops = [stop["node"] for stop in data["stops"]]
        self.points[self.stops] = goals
        links = np.asarray(data["corridors"], dtype=int)[:, :2]
        starts, ends = self.points[links[:, 0]], self.points[links[:, 1]]
        valid = np.any(starts != ends, axis=1) & native.clear_segments(starts, ends)
        self.rejected_links = int((~valid).sum())
        self.edges = [{} for _ in self.points]
        self.geometry = {}
        for a, b in links[valid]:
            self.add_link(a, b, self.points[[a, b]])

        connected = sorted({self.stops[0], *shortest_paths(self.stops[0], self.edges)})
        x, y = native.map.cell_indices(self.points[connected])
        self.connectors = []
        anchors = self.points[connected]
        for stop_index, node in enumerate(self.stops):
            if node in connected:
                continue
            goal = self.points[node]
            # Most doors can join the path network directly from the outdoor
            # arrival point. Use a grid route if walls block the local approaches.
            if "building_id" in data["stops"][stop_index]:
                distances = np.linalg.norm(anchors - goal, axis=1)
                nearby = np.flatnonzero((distances > 0) & (distances <= 50))
                nearby = nearby[np.argsort(distances[nearby], kind="stable")]
                clear = native.clear_segments(anchors[nearby], np.broadcast_to(goal, (len(nearby), 2)))
                if clear.any():
                    anchor = connected[nearby[np.argmax(clear)]]
                    self.add_link(anchor, node, self.points[[anchor, node]])
                    continue
            field = native.field(goal) if fields is None else fields.fields[stop_index]
            distances = field[y, x]
            distinct = np.any(anchors != goal, axis=1)
            distances = np.where(distinct, distances, np.inf)
            anchor = connected[int(np.argmin(distances))]
            if not np.isfinite(distances).any():
                raise ValueError(f"No grid connector for {data['stops'][stop_index]['name']}")
            connector = trace_grid_connector(native, field, self.points[anchor], self.points[node])
            self.add_link(anchor, node, connector)
            self.connectors.append({"stop": data["stops"][stop_index]["name"],
                                    "length_m": self.edges[anchor][node]})

    def add_link(self, a, b, points):
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        self.edges[a][b] = self.edges[b][a] = length
        self.geometry[a, b] = points
        self.geometry[b, a] = points[::-1]

    def route(self, start, end):
        source, destination = self.stops[start], self.stops[end]
        previous = shortest_paths(source, self.edges)
        nodes = [destination]
        while nodes[-1] != source:
            nodes.append(previous[nodes[-1]])
        nodes.reverse()
        points = [self.points[source]]
        decisions = []
        for a, b in zip(nodes, nodes[1:]):
            points.extend(self.geometry[a, b][1:])
            if b == destination or len(self.edges[b]) > 2:
                decisions.append(len(points) - 1)
        return PedestrianRoute(np.asarray(points), decisions)
