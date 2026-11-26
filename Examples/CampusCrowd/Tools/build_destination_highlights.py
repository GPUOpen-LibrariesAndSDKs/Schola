"""Associate campus destinations with CAD surfaces and export highlight geometry."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
import shapely
from shapely.ops import polygonize

from navigation_env import NavigationMap, ROOT

CONTENT = ROOT / "Content/Campus"


def footprints(source, campus, ids):
    root = ET.parse(source).getroot()
    coordinates = {int(node.get("id")): (float(node.get("lon")), float(node.get("lat")))
                   for node in root.findall("node")}
    ways = {way.get("id"): [int(node.get("ref")) for node in way.findall("nd")] for way in root.findall("way")}
    relations = {relation.get("id"): relation for relation in root.findall("relation")}
    project = Transformer.from_crs(4326, 2019, always_xy=True)
    origin_x, origin_y = campus["cad_origin_m"]

    def line(way):
        lon, lat = np.asarray([coordinates[n] for n in ways[way]]).T
        x, y = project.transform(lon, lat)
        return shapely.LineString(np.column_stack((x - origin_x - 16, origin_y - y + 7)))

    result = []
    for key in ids:
        _, kind, osm_id = key.split(":")
        if kind == "way":
            polygon = shapely.union_all(list(polygonize([line(osm_id)])))
        else:
            relation = relations[osm_id]
            rings = {role: [line(m.get("ref")) for m in relation.findall("member")
                            if m.get("type") == "way" and m.get("role") == role and m.get("ref") in ways]
                     for role in ("outer", "inner")}
            polygon = shapely.union_all(list(polygonize(rings["outer"]))).difference(
                shapely.union_all(list(polygonize(rings["inner"]))))
        if polygon.is_empty:
            raise ValueError(f"No mapped building outline for {key}")
        result.append(polygon)
    return result


def matched_faces(campus, outlines):
    outline_array = np.asarray(outlines)
    nearby = shapely.STRtree(shapely.buffer(outline_array, 2.5))
    rows = [[] for _ in outlines]
    roofs = np.zeros(len(outlines), dtype=int)
    for section in campus["sections"]:
        if section["name"] not in ("roofs", "walls"):
            continue
        xyz = np.asarray(section["vertices"], dtype=np.int32)[np.asarray(section["triangles"]).reshape(-1, 3)]
        points = shapely.points(xyz[:, :, :2].mean(axis=1) / 100)
        face, building = nearby.query(points, predicate="intersects")
        distance = shapely.distance(points[face], outline_array[building])
        order = np.lexsort((building, distance, face))
        face, building = face[order], building[order]
        if not len(face):
            continue
        unique = np.r_[True, face[1:] != face[:-1]]
        for index in np.unique(building[unique]):
            faces = xyz[face[unique & (building == index)]]
            rows[index].append(faces)
            if section["name"] == "roofs":
                roofs[index] += len(faces)
    return rows, roofs


def proxy_building(footprint, triangles, height):
    faces = [[(x * 100, y * 100, height) for x, y in triangle] for triangle in triangles]
    for part in shapely.get_parts(footprint):
        for ring in (part.exterior, *part.interiors):
            for (ax, ay), (bx, by) in zip(ring.coords, list(ring.coords)[1:]):
                a, b = (ax * 100, ay * 100, 0), (bx * 100, by * 100, 0)
                c, d = (bx * 100, by * 100, height), (ax * 100, ay * 100, height)
                faces.extend(([a, b, c], [a, c, d]))
    return np.rint(faces).astype(np.int32)


def export_map():
    from PIL import Image, ImageDraw

    nav = NavigationMap.load(CONTENT / "navigation_grid.bin")
    walls = nav.walls[::2, ::2]
    pixels = np.empty((*walls.shape, 3), dtype=np.uint8)
    pixels[:] = [22, 38, 46]
    pixels[walls] = [48, 66, 76]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    graph = json.loads((CONTENT / "turn_paths.json").read_text())
    points = graph["positions"]
    origin = np.asarray(nav.origin)
    for a, b, inside in graph["links"]:
        row = (np.asarray([points[a], *inside, points[b]]) - origin) / (2 * nav.cell)
        draw.line(list(map(tuple, row)), fill=(74, 100, 108), width=1)
    image.save(CONTENT / "turn_campus_map.png")
    return [*nav.origin, nav.width * nav.cell, nav.height * nav.cell]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osm", type=Path)
    args = parser.parse_args()
    campus = json.loads((CONTENT / "campus.json").read_text())
    stops = json.loads((CONTENT / "navigation_stops.json").read_text())["stops"]
    ids = sorted({stop["building_id"] for stop in stops if "building_id" in stop})
    outlines = footprints(args.osm, campus, ids)
    geometry, roof_count = matched_faces(campus, outlines)
    buildings = []
    inferred = 0
    for osm_id, footprint, parts, roofs in zip(ids, outlines, geometry, roof_count):
        triangles = [list(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(footprint).geoms]
        if not roofs:
            # A mapped destination whose roof is missing still gets a visible footprint.
            parts.append(proxy_building(footprint, triangles, 480))
            inferred += 1
        vertices, indices = np.unique(np.concatenate(parts).reshape(-1, 3), axis=0, return_inverse=True)
        buildings.append({"id": osm_id, "vertices": vertices.tolist(), "indices": indices.reshape(-1).tolist(),
                          "footprint": [[[round(x, 2), round(y, 2)] for x, y in t] for t in triangles]})
    out = CONTENT / "destination_highlights.json"
    out.write_text(json.dumps({"map": export_map(), "buildings": buildings}, separators=(",", ":")) + "\n")
    print(f"{len(buildings)} buildings, {inferred} use mapped footprint geometry, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
