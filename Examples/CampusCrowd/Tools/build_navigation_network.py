"""Export the campus pedestrian network and mapped outdoor entrances."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pyproj import Transformer
import shapely
from shapely.ops import polygonize, unary_union

from build_migration_routes import BUILDINGS, GATES, ROOT, building_footprints, load_graph, remove_blocked_edges, shortest_paths


CONTENT = ROOT / "Content/Campus"
# OpenStreetMap multipolygon for the University of Toronto St. George campus.
CAMPUS_RELATION = "18447148"


def tags(element):
    return {tag.get("k"): tag.get("v") for tag in element.findall("tag")}


def mapped_entrances(osm, points):
    ways = {way.get("id"): [int(node.get("ref")) for node in way.findall("nd")] for way in osm.findall("way")}
    relations = {relation.get("id"): relation for relation in osm.findall("relation")}
    campus = relations[CAMPUS_RELATION]

    def outlines(relation, role):
        return [ways[member.get("ref")] for member in relation.findall("member")
                if member.get("type") == "way" and member.get("role") == role and member.get("ref") in ways]

    # Campus boundary segments may extend outside the CAD scene. Use the full OSM
    # coordinates for building membership and the projected scene for routing.
    nodes = {int(node.get("id")): node for node in osm.findall("node")}

    def polygon(rings):
        lines = [shapely.LineString([(float(nodes[node].get("lon")), float(nodes[node].get("lat"))) for node in ring])
                 for ring in rings if len(ring) >= 2 and all(node in nodes for node in ring)]
        return unary_union(list(polygonize(lines)))

    boundary = polygon(outlines(campus, "outer")).difference(polygon(outlines(campus, "inner")))
    if boundary.is_empty:
        raise ValueError("The OSM extract has no complete St. George campus boundary")

    selected_names = {node: name for name, node in BUILDINGS.items()}
    buildings = []
    building_members = set()
    for relation in relations.values():
        properties = tags(relation)
        if properties.get("type") == "multipolygon" and "building" in properties:
            outer = outlines(relation, "outer")
            buildings.append((f"osm:relation:{relation.get('id')}", properties, outer))
            building_members.update(member.get("ref") for member in relation.findall("member")
                                    if member.get("type") == "way" and member.get("role") == "outer")
    for way in osm.findall("way"):
        if way.get("id") in building_members:
            continue
        properties = tags(way)
        if "building" in properties:
            buildings.append((f"osm:way:{way.get('id')}", properties, [ways[way.get("id")]]))

    def eligible(properties):
        return (properties.get("entrance") in {"yes", "main", "secondary", "entrance"} and
                properties.get("access") not in {"private", "no"} and properties.get("foot") not in {"private", "no"} and
                properties.get("indoor") != "yes" and "0" in properties.get("level", "0").split(";") and
                properties.get("min_height", "0") == "0" and properties.get("railway") != "subway_entrance" and
                properties.get("barrier") != "gate")

    candidates = {node: tags(nodes[node]) for node in points}
    candidates = {node: properties for node, properties in candidates.items() if eligible(properties)}
    stops = {}

    def add_stop(node, building_id, name):
        stops.setdefault(node, {
            "name": candidates[node].get("name") or name,
            "id": f"osm:node:{node}",
            "building_id": building_id,
            "building_name": name,
        })

    project = Transformer.from_crs(4326, 2019, always_xy=True).transform
    boundaries = []
    for building_id, properties, rings in buildings:
        members = set().union(*rings)
        if not (members & points.keys()):
            continue
        footprint = polygon(rings)
        known = sorted(members & selected_names.keys())
        if not known and (footprint.is_empty or not boundary.covers(footprint.representative_point())):
            continue
        name = selected_names[known[0]] if known else properties.get("name") or properties.get("addr:housenumber")
        name = name or f"Building {building_id.rsplit(':', 1)[-1]}"
        for node in sorted(members & candidates.keys()):
            add_stop(node, building_id, name)
        if not footprint.is_empty:
            edge = shapely.transform(footprint.boundary, project, interleaved=False)
            boundaries.append((building_id, name, edge))

    # Some surveyed doors are stand-alone OSM points. Match only a nearby,
    # unambiguous boundary and prefer explicit outline membership above.
    for node in sorted(candidates.keys() - stops.keys()):
        position = shapely.Point(float(nodes[node].get("lon")), float(nodes[node].get("lat")))
        if not boundary.covers(position):
            continue
        position = shapely.transform(position, project, interleaved=False)
        matches = sorted((position.distance(edge), building_id, name) for building_id, name, edge in boundaries)
        if matches and matches[0][0] <= 3 and (len(matches) == 1 or matches[1][0] - matches[0][0] >= .75):
            _, building_id, name = matches[0]
            add_stop(node, building_id, name)

    missing = selected_names.keys() - stops.keys()
    if missing:
        raise ValueError(f"Known campus doors missing from eligible building outlines: {sorted(missing)}")
    return stops


def build_network(source, campus):
    points, edges = load_graph(source, campus)
    obstacles = shapely.buffer(building_footprints(campus), -.45)
    remove_blocked_edges(points, edges, obstacles)
    first_gate = next(iter(GATES.values()))
    connected = set(shortest_paths(first_gate, edges)) | {first_gate}
    if not set(GATES.values()) <= connected:
        raise ValueError("Campus gateways are disconnected after building clearance checks")
    entrances = mapped_entrances(ET.parse(source).getroot(), points)
    nodes = sorted(connected)
    index = {node: i for i, node in enumerate(nodes)}
    stops = [{"name": name, "node": index[node]} for name, node in GATES.items()]
    # Keep sidewalk vertices even if they are tagged as doors. The separate
    # entrance node can move outside the CAD shell without changing a sidewalk.
    for node, stop in sorted(entrances.items()):
        stops.append({**stop, "node": len(nodes)})
        nodes.append(node)
    corridors = [[index[a], index[b]] for a in sorted(connected) for b in sorted(edges[a])
                 if a < b and b in connected]
    return {
        "source": "© OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "gate_count": len(GATES),
        "stops": stops,
        "nodes": [points[node] for node in nodes],
        "corridors": corridors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osm", type=Path)
    args = parser.parse_args()
    campus = json.loads((CONTENT / "campus.json").read_text())
    network = build_network(args.osm, campus)
    (CONTENT / "navigation_network.json").write_text(json.dumps(network, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Exported {len(network['nodes'])} sidewalk and entrance nodes, {len(network['stops'])} stops")
