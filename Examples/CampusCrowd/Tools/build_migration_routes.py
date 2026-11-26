"""Build the migration path table from a local OpenStreetMap campus extract.

Download https://api.openstreetmap.org/api/0.6/map?bbox=-79.4060,43.6559,-79.3870,43.6715
and pass the resulting XML with `python Tools/build_migration_routes.py campus.osm`.
The exported path data is derived from OpenStreetMap contributors (ODbL 1.0).
"""
import argparse
from collections import defaultdict
import heapq
import json
from math import dist
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# Sidewalk anchors on the campus side of the intersections, then mapped building doors.
GATES = {
    "Spadina & Bloor": 9531251002,
    "Avenue & Bloor": 8699776875,
    "Queen’s Park Station": 2129327570,
    "Spadina & College": 9170559897,
}
BUILDINGS = {
    "Bahen Centre": 733206245,
    "Sidney Smith Hall": 703098879,
    "University College": 703096144,
    "Medical Sciences": 14209599299,
    "Myhal Centre": 13076878683,
    "McLennan Physical Laboratories": 703098888,
    "Lash Miller": 13647896605,
}

ROOT = Path(__file__).resolve().parents[1]
WALKABLE = {"footway", "path", "pedestrian", "steps", "living_street"}


def load_graph(source, campus):
    from pyproj import Transformer

    osm = ET.parse(source).getroot()
    origin_x, origin_y = campus["cad_origin_m"]
    # The DWG uses legacy MTM. Register the OSM footprints to the CAD roofs
    # with a constant local correction of 16 m west and 7 m south.
    project = Transformer.from_crs(4326, 2019, always_xy=True)
    ground = np.asarray(next(section["vertices"] for section in campus["sections"] if section["name"] == "ground"))
    min_x, min_y = ground[:, :2].min(axis=0) / 100
    max_x, max_y = ground[:, :2].max(axis=0) / 100
    points = {}
    for node in osm.findall("node"):
        x, y = project.transform(float(node.attrib["lon"]), float(node.attrib["lat"]))
        x, y = x - origin_x - 16, origin_y - y + 7
        if min_x <= x <= max_x and min_y <= y <= max_y:
            points[int(node.attrib["id"])] = (round(100 * x), round(100 * y))
    edges = defaultdict(dict)
    for way in osm.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if (
            tags.get("highway") not in WALKABLE
            or tags.get("access") in {"no", "private"}
            or tags.get("foot") in {"no", "private"}
            or tags.get("indoor") == "yes"
            or tags.get("tunnel") not in {None, "no"}
            or tags.get("bridge") not in {None, "no"}
            or tags.get("covered") == "yes"
            or tags.get("level", "0") not in {"0", "0;1"}
        ):
            continue
        nodes = [int(node.attrib["ref"]) for node in way.findall("nd")]
        for a, b in zip(nodes, nodes[1:]):
            if a in points and b in points and points[a] != points[b]:
                cost = dist(points[a], points[b]) * (1.7 if tags["highway"] == "steps" else 1.0)
                edges[a][b] = edges[b][a] = min(cost, edges[a].get(b, float("inf")))
    return points, edges


def shortest_paths(start, edges):
    queue = [(0, start)]
    costs, previous = {start: 0}, {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        for neighbor, edge_cost in edges[node].items():
            new_cost = cost + edge_cost
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                previous[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))
    return previous


def building_footprints(campus):
    import shapely

    polygons = []
    for section in campus["sections"]:
        if section["name"] not in {"roofs", "walls"}:
            continue
        faces = np.asarray(section["triangles"]).reshape(-1, 3)
        triangles = np.asarray(section["vertices"])[faces]
        outlines = shapely.polygons(triangles[:, :, :2] / 100)
        polygons.append(outlines[(triangles[:, :, 2].max(axis=1) > 200) & (shapely.area(outlines) > .02)])
    return shapely.union_all(np.concatenate(polygons))


def remove_blocked_edges(points, edges, obstacles):
    import shapely

    links = sorted((a, b) for a, links in edges.items() for b in links if a < b)
    lines = shapely.linestrings(np.array([[points[a], points[b]] for a, b in links]) / 100)
    blocked_links = shapely.intersects(lines, obstacles)
    for (a, b), blocked in zip(links, blocked_links):
        if blocked:
            del edges[a][b]
            del edges[b][a]


def build_routes(source, campus):
    import shapely

    points, edges = load_graph(source, campus)
    # Ignore the outer 0.45 m of the roof projection to allow for eaves and map
    # registration error. A walking centreline may never enter the remaining footprint.
    obstacles = shapely.buffer(building_footprints(campus), -.45)
    remove_blocked_edges(points, edges, obstacles)
    first_gate = next(iter(GATES.values()))
    connected = set(shortest_paths(first_gate, edges)) | {first_gate}
    for gate in GATES.values():
        if gate not in connected:
            raise ValueError(f"Gateway {gate} is unreachable after CAD clearance checks")
    stops = dict(GATES)
    for building, target in BUILDINGS.items():
        # Use the walkable approach when the mapped door lies inside the simplified CAD shell.
        distance, anchor = min((dist(points[target], points[node]), node) for node in connected)
        if distance > 2500:
            raise ValueError(f"No outdoor approach within 25 m of {building} ({distance / 100:.1f} m)")
        stops[building] = anchor
    names = list(stops)
    routes = []
    for start in stops.values():
        if start not in edges:
            raise ValueError(f"Stop {start} is outside the connected pedestrian features")
        previous = shortest_paths(start, edges)
        for end in stops.values():
            if end == start:
                routes.append([])
                continue
            if end not in previous:
                raise ValueError(f"No walkable path from {start} to {end}")
            route = [end]
            while route[-1] != start:
                route.append(previous[route[-1]])
            routes.append(list(reversed(route)))
    nodes = sorted({node for route in routes for node in route})
    index = {node: i for i, node in enumerate(nodes)}
    used_links = sorted({tuple(sorted(pair)) for route in routes for pair in zip(route, route[1:])})
    used_lines = shapely.linestrings(np.array([[points[a], points[b]] for a, b in used_links]) / 100)
    clearance = np.clip(np.floor(shapely.distance(used_lines, obstacles) * 100 - 20), 0, 115).astype(int)
    return {
        "source": "© OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "gate_count": len(GATES),
        "stops": [{"name": name, "node": index[stops[name]]} for name in names],
        "nodes": [points[node] for node in nodes],
        "paths": [[index[node] for node in route] for route in routes],
        "corridors": [[index[a], index[b], int(width)] for (a, b), width in zip(used_links, clearance)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osm", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "Content/Campus/migration_routes.json")
    args = parser.parse_args()
    campus = json.loads((ROOT / "Content/Campus/campus.json").read_text())
    args.output.write_text(json.dumps(build_routes(args.osm, campus), ensure_ascii=False, separators=(",", ":")) + "\n")
