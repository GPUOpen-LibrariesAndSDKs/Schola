"""Landmark graph and evaluation splits for campus navigation."""
from dataclasses import dataclass
import heapq

import numpy as np

from pedestrian_navigation import PedestrianRoute


@dataclass
class LandmarkGraph:
    positions: np.ndarray
    stops: np.ndarray
    edges: list[dict]
    geometry: dict

    def add_edge(self, start, end, points, cost):
        if start == end or cost >= self.edges[start].get(end, np.inf):
            return
        self.edges[start][end] = self.edges[end][start] = float(cost)
        self.geometry[start, end] = points
        self.geometry[end, start] = points[::-1]

    def polyline(self, path):
        points = [self.positions[path[0]]]
        decisions = []
        for start, end in zip(path, path[1:]):
            points.extend(self.geometry[start, end][1:])
            decisions.append(len(points) - 1)
        return PedestrianRoute(np.asarray(points), decisions)


def build_landmarks(native, network, shortcut_range, landmark_spacing=20.):
    connected = {network.stops[0]}
    todo = [network.stops[0]]
    while todo:
        new = network.edges[todo.pop()].keys() - connected
        todo.extend(sorted(new))
        connected.update(new)

    # Preserve every junction and entrance, plus intermediate path vertices
    # along long chains of degree-two sidewalk nodes.
    required = {node for node in connected if len(network.edges[node]) != 2} | set(network.stops)
    retained = set(required)
    traversed = set()
    for first in sorted(required):
        for second in sorted(network.edges[first]):
            if (first, second) in traversed:
                continue
            previous, current = first, second
            distance = network.edges[first][second]
            while current not in required:
                if distance >= landmark_spacing:
                    retained.add(current)
                    distance = 0.
                following = next(node for node in network.edges[current] if node != previous)
                distance += network.edges[current][following]
                previous, current = current, following
            traversed.add((first, second))
            traversed.add((current, previous))

    originals = sorted(retained)
    ids = {original: i for i, original in enumerate(originals)}
    points = network.points[originals]
    graph = LandmarkGraph(points, np.asarray([ids[i] for i in network.stops]),
                          [{} for _ in originals], {})
    for start, first in enumerate(originals):
        for second in network.edges[first]:
            previous, current = first, second
            shape = [network.points[first], *network.geometry[first, second][1:]]
            distance = network.edges[first][second]
            while current not in ids:
                following = next(node for node in network.edges[current] if node != previous)
                shape.extend(network.geometry[current, following][1:])
                distance += network.edges[current][following]
                previous, current = current, following
            graph.add_edge(start, ids[current], np.asarray(shape), distance)

    sidewalk_links = sum(len(row) for row in graph.edges) // 2
    # Process one landmark at a time to avoid allocating a dense pair array.
    for a, point in enumerate(points[:-1]):
        length = np.linalg.norm(points[a + 1:] - point, axis=1)
        candidates = np.flatnonzero((length <= shortcut_range) & (length > 0))
        ends = points[a + 1 + candidates]
        clear = native.clear_segments(np.broadcast_to(point, ends.shape), ends)
        for b in candidates[clear]:
            neighbor = a + 1 + b
            graph.add_edge(a, neighbor, points[[a, neighbor]], length[b])
    return graph, sidewalk_links


def shortest_path_tables(graph):
    """Each column holds distances and the next hop toward one destination."""
    count = len(graph.edges)
    distances = np.full((count, len(graph.stops)), np.inf)
    next_hops = np.full(distances.shape, -1, np.int32)
    for column, goal in enumerate(graph.stops):
        row = distances[:, column]
        row[goal] = 0
        queue = [(0., goal)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != row[node]:
                continue
            for neighbor, cost in graph.edges[node].items():
                candidate = distance + cost
                if candidate < row[neighbor]:
                    row[neighbor] = candidate
                    next_hops[neighbor, column] = node
                    heapq.heappush(queue, (candidate, neighbor))
    if not np.isfinite(distances).all():
        raise ValueError("The landmark graph is disconnected")
    return distances, next_hops


def split_destinations(stops, seed):
    """Keep all doors of a building in the same split. Train on the gateways."""
    buildings = list(dict.fromkeys(stop["building_id"] for stop in stops if "building_id" in stop))
    if len(buildings) < 3:
        raise ValueError("At least three mapped buildings are needed to split destinations")
    shuffled = np.random.default_rng(seed).permutation(buildings)
    test = set(shuffled[:max(1, len(buildings) // 5)])
    validation = set(shuffled[len(test):len(test) + max(1, len(buildings) // 10)])
    return tuple(np.flatnonzero([classify(stop.get("building_id")) for stop in stops])
                 for classify in (lambda b: b not in test and b not in validation,
                                  validation.__contains__, test.__contains__))
