import json
from pathlib import Path
import sys
import unittest

import numpy as np
import shapely

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from build_migration_routes import GATES, building_footprints


class MigrationRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        content = TOOLS.parent / "Content/Campus"
        cls.campus = json.loads((content / "campus.json").read_text())
        cls.data = json.loads((content / "migration_routes.json").read_text())

    def test_all_gateways_and_buildings_can_route_to_each_other(self):
        data = self.data
        gates = [stop["name"] for stop in data["stops"][:data["gate_count"]]]
        self.assertEqual(gates, list(GATES))
        stops = [stop["node"] for stop in data["stops"]]
        self.assertEqual(len(stops), len(set(stops)))
        corridors = {(a, b) for a, b, _ in data["corridors"]}
        self.assertEqual(len(data["paths"]), len(stops) ** 2)
        for index, path in enumerate(data["paths"]):
            start, end = divmod(index, len(stops))
            if start == end:
                self.assertEqual(path, [])
            else:
                self.assertEqual((path[0], path[-1]), (stops[start], stops[end]))
                self.assertTrue(all(tuple(sorted(pair)) in corridors for pair in zip(path, path[1:])))

    def test_corridors_respect_cad_building_footprints(self):
        gap = 0.20
        nodes = np.asarray(self.data["nodes"]) / 100
        corridors = np.asarray(self.data["corridors"])
        lines = shapely.linestrings(nodes[corridors[:, :2]])
        obstacles = shapely.buffer(building_footprints(self.campus), -.45)
        self.assertFalse(shapely.intersects(lines, obstacles).any())
        widths = corridors[:, 2] / 100
        self.assertTrue((shapely.distance(lines, obstacles)[widths > 0] >= widths[widths > 0] + gap - 1e-6).all())


if __name__ == "__main__":
    unittest.main()
