"""Train trip-level landmark search and benchmark the shared CampusCrowd shortcut walker."""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np

from navigation_env import PACE, ROOT, NativeNavigation, campus_map
from navigation_landmarks import build_landmarks, shortest_path_tables, split_destinations
from pedestrian_navigation import PedestrianNetwork, PedestrianRoute


def search(graph, start, goal, estimate=None):
    """Search feasible links with exact edge costs and an optional learned queue priority."""
    def remaining(node):
        return 0. if estimate is None or node == goal else estimate[node]

    best = {start: 0.}
    parent = {}
    queue = [(remaining(start), 0., start)]
    expanded = 0
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        expanded += 1
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            seconds = sum(graph.edges[a][b] / PACE for a, b in zip(path, path[1:]))
            return path, seconds, expanded
        for successor, length in graph.edges[node].items():
            candidate = cost + length / PACE
            if candidate >= best.get(successor, np.inf):
                continue
            best[successor] = candidate
            parent[successor] = node
            heapq.heappush(queue, (candidate + remaining(successor), candidate, successor))
    raise ValueError(f"No route from landmark {start} to {goal}")


def route_features(graph, scale):
    points = graph.positions / scale
    shape = (len(points), len(graph.stops), 2)
    origin = np.broadcast_to(points[:, None, :], shape)
    destination = np.broadcast_to(points[graph.stops][None, :, :], shape)
    straight = np.linalg.norm(origin - destination, axis=2)
    neighborhood = np.empty((*straight.shape, 4), np.float32)
    for node, links in enumerate(graph.edges):
        neighbors = list(links)
        costs = np.asarray(list(links.values()))[:, None] / scale
        next_estimates = np.sort(costs + straight[neighbors], axis=0)
        neighborhood[node, :, :3] = next_estimates[np.minimum(np.arange(3), len(neighbors) - 1)].T
        neighborhood[node, :, 3] = len(neighbors) / (len(points) - 1)
    return np.concatenate((origin, destination, straight[..., None], neighborhood), axis=2).astype(np.float32), straight


def sample_pairs(graph, destinations, count, rng):
    pairs = np.column_stack((np.repeat(np.arange(len(graph.positions)), len(destinations)),
                             np.tile(destinations, len(graph.positions))))
    pairs = pairs[pairs[:, 0] != graph.stops[pairs[:, 1]]]
    return tuple(pairs[rng.choice(len(pairs), min(count, len(pairs)), replace=False)].T)


def eta_model(width):
    from torch import nn

    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))


def fit(features, straight, reference, training, validation, time_scale, seed, epochs):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = eta_model(features.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    x_train = torch.from_numpy(features[training])
    y_train = torch.from_numpy((reference[training] / time_scale - straight[training]).astype(np.float32))
    x_valid = torch.from_numpy(features[validation])
    y_valid = torch.from_numpy((reference[validation] / time_scale - straight[validation]).astype(np.float32))
    best_weights = {}
    best_error = np.inf
    for epoch in range(1, epochs + 1):
        for batch in torch.randperm(len(x_train)).split(256):
            predicted = model(x_train[batch]).squeeze(-1)
            loss = nn.functional.smooth_l1_loss(predicted, y_train[batch], beta=.02)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            residual = model(x_valid).squeeze(-1).clamp_min(0)
            error = (residual - y_valid).abs().mean().item() * time_scale
        if error < best_error:
            best_error, best_epoch = error, epoch
            best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    with torch.no_grad():
        detour = model(torch.from_numpy(features)).squeeze(-1).clamp_min(0).numpy()
    estimate = (straight + detour) * time_scale
    return estimate, best_weights, {"seed": seed, "epoch": best_epoch, "validation_mae_seconds": best_error}


def export_live_planner(directory, graph, features, weights, scale):
    """Export walkable links and the ETA model. Unreal builds observations from the links."""
    import torch
    from torch import nn

    class Seconds(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = eta_model(features.shape[-1])
            self.model.load_state_dict(weights)

        def forward(self, obs):
            return (obs[:, 4:5] + self.model(obs).clamp_min(0)) * (scale / PACE)

    campus = directory / "Campus"
    models = directory / "Models"
    campus.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    graph_asset = {
        "positions": graph.positions.tolist(),
        "stops": graph.stops.tolist(),
        "links": [[a, int(b), length / PACE, graph.geometry[a, b][1:-1].tolist()]
                  for a, edges in enumerate(graph.edges) for b, length in sorted(edges.items()) if a < b],
        "scale": scale,
    }
    (campus / "navigation_landmarks.json").write_text(json.dumps(graph_asset, separators=(",", ":")) + "\n")
    torch.onnx.export(Seconds().eval(), torch.from_numpy(features[:2, 0].copy()),
                      models / "landmark_eta.onnx", input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)


def routing_metrics(graph, reference, pairs, estimates):
    results = [search(graph, a, graph.stops[b], None if estimates is None else estimates[:, b]) for a, b in pairs]
    ratio = np.array([seconds / reference[a, b] for (_, seconds, _), (a, b) in zip(results, pairs)])
    return {
        "trips": len(ratio),
        "optimal_routes": int(np.count_nonzero(np.isclose(ratio, 1, atol=.001, rtol=0))),
        "median_cost_ratio": float(np.median(ratio)),
        "max_cost_ratio": float(ratio.max()),
        "mean_expanded_nodes": float(np.mean([expanded for _, _, expanded in results])),
    }


def plan_stop_routes(graph, estimate):
    return {(a, b): graph.polyline(search(graph, start, end, estimate[:, b])[0])
            for a, start in enumerate(graph.stops) for b, end in enumerate(graph.stops) if a != b}


def refine_route(native, route):
    """Pull each bend toward a straight chord while keeping both legs walkable."""
    points = route.points.copy()
    cell = native.map.cell
    # Update each bend from both ends so it can use an already refined neighbor on either side.
    bends = range(1, len(points) - 1)
    for sweep in (bends, reversed(bends)):
        for index in sweep:
            previous, bend, following = points[index - 1:index + 2]
            chord = following - previous
            squared = float(chord @ chord)
            if squared == 0:
                continue
            projection = previous + chord * np.clip((bend - previous) @ chord / squared, 0, 1)
            delta = projection - bend
            travel = np.linalg.norm(delta)
            if travel <= cell:
                continue
            lo, hi = 0., 1.
            while (hi - lo) * travel > cell:
                t = (lo + hi) / 2
                candidate = bend + delta * t
                starts = [previous, bend, candidate]
                ends = [candidate, candidate, following]
                if native.clear_segments(starts, ends).all():
                    lo = t
                else:
                    hi = t
            if lo:
                points[index] = bend + delta * lo
    return PedestrianRoute(points, route.decision_indices)


def export_routes(path, routes, source, planner):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"start": a, "end": b, "decisions": route.decision_indices, "points": route.points.tolist()}
            for (a, b), route in routes.items()]
    path.write_text(json.dumps({"source": source, "planner": planner, "routes": rows}, separators=(",", ":")) + "\n")


def export_next_hops(path, graph, table, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"source": source, "planner": "Dijkstra", "stop_nodes": graph.stops.tolist(),
            "next_hop_by_stop": table.T.tolist()}
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--export-live", type=Path, help="Write the live graph and ONNX model under this Content directory")
    parser.add_argument("--export-routes", type=Path, help="Write the compact exact next-hop route table")
    parser.add_argument("--export-exact", type=Path, help="Write routes from Dijkstra")
    parser.add_argument("--export-learned", type=Path, help="Write learned routes before per-trip selection")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--training-pairs", type=int, default=60000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 41, 83])
    args = parser.parse_args()
    if min(args.epochs, args.training_pairs) < 1:
        parser.error("--epochs and --training-pairs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    nav, goals, gates = campus_map()
    cache = ROOT / "Saved/NavigationRL/cache"
    native = NativeNavigation(nav, cache)
    network = PedestrianNetwork(native, goals)
    graph, sidewalk_links = build_landmarks(native, network, 45.)
    metadata = json.loads((ROOT / "Content/Campus/navigation_stops.json").read_text())["stops"]
    for index, stop in enumerate(metadata[gates:], gates):
        stop.setdefault("building_id", f"legacy:{index}")
    train_goals, validation_goals, test_goals = split_destinations(metadata, 317)
    rng = np.random.default_rng(317)
    training = sample_pairs(graph, train_goals, args.training_pairs, rng)
    validation = sample_pairs(graph, validation_goals, 6000, rng)
    test = sample_pairs(graph, test_goals, 1200, rng)
    reference, next_hops = shortest_path_tables(graph)
    reference /= PACE
    features, straight = route_features(graph, nav.scale)
    runs = []
    for seed in args.seeds:
        estimate, weights, metrics = fit(features, straight, reference, training, validation,
                                         nav.scale / PACE, seed, args.epochs)
        runs.append((metrics, estimate, weights))
        print(f"Seed {seed}, epoch {metrics['epoch']}, validation MAE {metrics['validation_mae_seconds']:.2f}s", flush=True)
    chosen, estimate, weights = min(runs, key=lambda run: run[0]["validation_mae_seconds"])
    torch.save({"model": weights, "metrics": chosen}, args.output / "landmark_eta.pt")
    if args.export_live:
        export_live_planner(args.export_live, graph, features, weights, nav.scale)
    routing = {name: routing_metrics(graph, reference, list(zip(*test)), heuristic)
               for name, heuristic in (("exact", None), ("straight_line", straight * nav.scale / PACE),
                                       ("learned", estimate))}
    for name, metrics in routing.items():
        print(name, metrics, flush=True)
    for path in {args.export_exact, args.export_routes} - {None}:
        export_next_hops(path, graph, next_hops, network.attribution)
    if args.export_learned:
        learned = {pair: refine_route(native, route) for pair, route in plan_stop_routes(graph, estimate).items()}
        export_routes(args.export_learned, learned, network.attribution, "learned ETA search")
    report = {"landmarks": len(graph.positions), "links": sum(map(len, graph.edges)) // 2,
              "sidewalk_links": sidewalk_links, "training_pairs": len(training[0]),
              "validation_pairs": len(validation[0]), "test_pairs": len(test[0]),
              "train_destinations": len(train_goals), "validation_destinations": len(validation_goals),
              "test_destinations": len(test_goals), "training": [run[0] for run in runs],
              "chosen_model": chosen, "routing": routing}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
