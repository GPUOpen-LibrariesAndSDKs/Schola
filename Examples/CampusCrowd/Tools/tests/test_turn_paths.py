import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from navigation_env import NativeNavigation, campus_map


class TurnPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = TOOLS.parent / "Content/Campus/turn_paths.json"
        cls.asset = json.loads(path.read_text())
        cls.positions = np.asarray(cls.asset["positions"])
        cls.stops = [row["node"] for row in cls.asset["stops"]]
        cls.neighbors = [[] for _ in cls.positions]
        for a, b, _ in cls.asset["links"]:
            cls.neighbors[a].append(b)
            cls.neighbors[b].append(a)
        for row in cls.neighbors:
            row.sort()

    def test_path_polylines_clear_the_walkability_grid(self):
        nav, goals, _ = campus_map()
        np.testing.assert_allclose(self.positions[self.stops], goals)
        polylines = [np.asarray([self.positions[a], *ends, self.positions[b]])
                     for a, b, ends in self.asset["links"]]
        with tempfile.TemporaryDirectory() as path:
            native = NativeNavigation(nav, Path(path))
            starts = np.concatenate([line[:-1] for line in polylines])
            ends = np.concatenate([line[1:] for line in polylines])
            self.assertTrue(np.any(starts != ends, axis=1).all())
            self.assertTrue(native.clear_segments(starts, ends).all())

    def test_all_destination_buildings_have_highlights_in_both_views(self):
        content = TOOLS.parent / "Content/Campus"
        highlights = json.loads((content / "destination_highlights.json").read_text())
        expected = {stop["building_id"] for stop in self.asset["stops"] if "building_id" in stop}
        ids = [row["id"] for row in highlights["buildings"]]
        self.assertEqual(len(ids), len(expected))
        self.assertEqual(set(ids), expected)
        self.assertGreater(min(highlights["map"][2:]), 0)
        self.assertGreater((content / "turn_campus_map.png").stat().st_size, 0)
        for row in highlights["buildings"]:
            with self.subTest(building=row["id"]):
                self.assertTrue(row["footprint"])
                self.assertTrue(row["indices"])
                self.assertEqual(len(row["indices"]) % 3, 0)
                self.assertTrue(all(0 <= index < len(row["vertices"]) for index in row["indices"]))

    def test_every_node_can_reach_each_stop_through_allowed_exits(self):
        self.assertNotIn("next_hop_by_stop", self.asset)
        allowed = self.asset["allowed_exit_bits_by_stop"]
        self.assertEqual(len(allowed), len(self.stops))
        incoming = [[] for _ in self.positions]
        for start, exits in enumerate(self.neighbors):
            for slot, end in enumerate(exits):
                incoming[end].append((start, 1 << slot))
        for goal, masks in zip(self.stops, allowed):
            self.assertEqual(len(masks), len(self.positions))
            self.assertEqual(masks[goal], 0)
            reached = {goal}
            queue = [goal]
            for node in queue:
                for predecessor, bit in incoming[node]:
                    if predecessor not in reached and masks[predecessor] & bit:
                        reached.add(predecessor)
                        queue.append(predecessor)
            self.assertEqual(len(reached), len(self.positions))


if __name__ == "__main__":
    unittest.main()
