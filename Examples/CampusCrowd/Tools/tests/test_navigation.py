import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation_landmarks import LandmarkGraph, build_landmarks, shortest_path_tables, split_destinations
from navigation_env import GoalDistances, NativeNavigation, campus_map, toy_map
from pedestrian_navigation import PedestrianRoute
from train_waypoint_navigation import refine_route, search


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.cache.name)
        cls.map, cls.goals, _ = toy_map()
        cls.native = NativeNavigation(cls.map, cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def test_destination_split_keeps_all_entrances_of_a_building_together(self):
        stops = [{}] + [{"building_id": str(building)} for building in range(12) for _ in range(2)]
        training, validation, test = split_destinations(stops, 317)
        self.assertIn(0, training)
        self.assertEqual(sorted([*training, *validation, *test]), list(range(len(stops))))
        for split in (training, validation, test):
            self.assertTrue(len(split))
            for index in split:
                if index > 0:
                    self.assertIn(index + 1 if index % 2 else index - 1, split)

    def test_landmark_spacing_retains_intermediate_path_nodes_and_stops(self):
        points = np.column_stack((np.arange(2.25, 14, 2), np.full(6, 3.25)))
        graph = LandmarkGraph(points, np.array([0, 2, 5]), [{} for _ in points], {})
        for node in range(len(points) - 1):
            graph.add_edge(node, node + 1, points[node:node + 2], 2.)
        network = SimpleNamespace(points=points, stops=graph.stops, edges=graph.edges, geometry=graph.geometry)
        dense, _ = build_landmarks(self.native, network, shortcut_range=0., landmark_spacing=2.)
        sparse, _ = build_landmarks(self.native, network, shortcut_range=0., landmark_spacing=np.inf)
        self.assertGreater(len(dense.positions), len(sparse.positions))
        for result in (dense, sparse):
            np.testing.assert_array_equal(result.positions[result.stops], points[graph.stops])
        costs, hops = shortest_path_tables(dense)
        np.testing.assert_array_equal(costs[dense.stops[0]], [0., 4., 10.])
        self.assertEqual(hops[dense.stops[0], 2], 1)

    def test_wall_stays_impassable_and_detour_counts_as_progress(self):
        goal = self.goals[[1]]
        body = np.array([[2.75, 7.25, 0, 0]], np.float32)
        direct = np.array([[1, 0]], np.float32)
        for _ in range(80):
            result = self.native.step(body, goal, direct)
        self.assertTrue(result[0] & 1)
        self.assertFalse(result[0] & 2)
        self.assertLess(float(body[0, 0]), 7.5)
        fields = GoalDistances(self.native, self.goals, self.directory)
        samples = np.array([[7.25, 7.25], [7.25, 8.25]], np.float32)
        distance = fields.at(samples, np.array([1, 1]))
        self.assertLess(distance[1], distance[0])
        self.assertGreater(np.linalg.norm(samples[1] - goal[0]), np.linalg.norm(samples[0] - goal[0]))

    def test_oblique_command_slides_along_wall(self):
        body = np.array([[7.498, 7.25, 0, 0]], np.float32)
        result = self.native.step(body, self.goals[[1]], np.array([[1, .6]], np.float32))
        self.assertTrue(result[0] & 1)
        self.assertLess(float(body[0, 0]), 7.5)
        self.assertGreater(float(body[0, 1]), 7.25)

    def test_shortcut_skips_a_junction_and_finishes_at_destination(self):
        route = PedestrianRoute([[2.25, 3.25], [3.25, 4.25], [7.25, 3.25]], [1, 2])
        offsets = np.array([0, 3], np.int32)
        route_ids = np.zeros(1, np.int32)
        points = route.points
        times = self.native.shortcut_times(points, offsets)
        body = np.array([[2.25, 3.25, 0, 0]], np.float32)
        cursor = np.ones((1, 2), np.int32)
        replan = np.zeros(1, np.float32)

        for _ in range(50):
            result = self.native.shortcut_step(body, points, times, offsets, route_ids, cursor, replan)
            self.assertEqual(cursor[0, 1], 2)
            self.assertFalse(result[0] & 1)
            if result[0] & 2:
                break
        self.assertTrue(result[0] & 2)
        self.assertAlmostEqual(float(body[0, 1]), 3.25)

    def test_shortcut_rejoins_the_route_after_a_building_detour(self):
        route = PedestrianRoute([[2.75, 7.25], [7.25, 4.25], [12.75, 4.25], [17.25, 7.25]], [3])
        original_length = route.distances[-1]
        route = refine_route(self.native, route)
        self.assertLessEqual(route.distances[-1], original_length)
        self.assertTrue(self.native.clear_segments(route.points[:-1], route.points[1:]).all())
        offsets = np.array([0, 4], np.int32)
        route_ids = np.zeros(1, np.int32)
        points = route.points
        times = self.native.shortcut_times(points, offsets)
        body = np.array([[2.75, 7.25, 0, 0]], np.float32)
        cursor = np.ones((1, 2), np.int32)
        replan = np.zeros(1, np.float32)

        for _ in range(1200):
            result = self.native.shortcut_step(body, points, times, offsets, route_ids, cursor, replan, dt=1/60)
            self.assertFalse(result[0] & 1)
            if result[0] & 2:
                break
        self.assertTrue(result[0] & 2)
        self.assertGreater(cursor[0, 0], 1)

    def test_blocked_cached_shortcut_replans_on_the_next_frame(self):
        route = PedestrianRoute([[2.75, 7.25], [7.25, 4.25], [12.75, 4.25], [17.25, 7.25]], [3])
        offsets = np.array([0, len(route.points)], np.int32)
        route_ids = np.zeros(1, np.int32)
        times = self.native.shortcut_times(route.points, offsets)
        body = np.array([[7.49, 7.25, 1.4, 0]], np.float32)
        cursor = np.array([[1, 3]], np.int32)
        replan = np.array([.08], np.float32)

        result = self.native.shortcut_step(body, route.points, times, offsets, route_ids, cursor, replan, dt=1/60)
        self.assertTrue(result[0] & 1)
        self.assertLess(body[0, 0], 7.5)
        self.assertEqual(replan[0], 0)

        result = self.native.shortcut_step(body, route.points, times, offsets, route_ids, cursor, replan, dt=1/60)
        self.assertFalse(result[0] & 1)
        self.assertLess(cursor[0, 1], 3)
        self.assertGreater(replan[0], 0)

    def test_learned_search_recovers_from_an_attractive_dead_end(self):
        edges = [{1: 1., 2: 3.}, {0: 1.}, {0: 3., 3: 1.}, {2: 1.}]
        graph = LandmarkGraph(np.zeros((4, 2)), np.array([0, 3]), edges, {})
        estimate = np.array([0., 0., 10., 0.])
        self.assertEqual(search(graph, 0, 3)[0], [0, 2, 3])
        self.assertEqual(search(graph, 0, 3, estimate)[0], [0, 2, 3])

    def test_campus_stops_share_walkable_region(self):
        nav, goals, gateways = campus_map()
        self.assertEqual(gateways, 4)
        content = Path(__file__).resolve().parents[2] / "Content/Campus"
        metadata = json.loads((content / "navigation_stops.json").read_text())["stops"]
        graph = json.loads((content / "navigation_landmarks.json").read_text())
        entrances = metadata[gateways:]
        self.assertGreater(len(graph["positions"]), 78)
        self.assertGreater(len(entrances), 7)
        by_id = {stop["id"]: stop for stop in entrances}
        self.assertEqual(len(by_id), len(entrances))
        # The separately mapped Myhal door belongs to the same building as its outline vertex.
        self.assertEqual(by_id["osm:node:13076878683"]["building_id"],
                         by_id["osm:node:13076878684"]["building_id"])
        self.assertEqual(len(np.unique(goals, axis=0)), len(goals))
        self.assertLess(len({stop["building_id"] for stop in entrances}), len(entrances))
        np.testing.assert_allclose(np.asarray(graph["positions"])[graph["stops"]], goals)
        x, y = nav.cell_indices(goals)
        self.assertFalse(nav.walls[y, x].any())
        field = NativeNavigation(nav, self.directory).field(goals[0])
        self.assertTrue(np.isfinite(field[y, x]).all())


if __name__ == "__main__":
    unittest.main()
