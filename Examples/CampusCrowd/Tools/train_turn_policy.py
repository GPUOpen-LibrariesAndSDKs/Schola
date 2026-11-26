"""Train a shared MLP to choose outgoing paths at campus junctions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from navigation_env import ROOT, NativeNavigation, campus_map
from navigation_landmarks import build_landmarks, shortest_path_tables, split_destinations
from pedestrian_navigation import PedestrianNetwork

EXITS = 8
FEATURES = 12
ANCHOR_COUNT = 96


class TurnPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, observation):
        return self.scorer(observation.reshape(-1, EXITS, FEATURES)).squeeze(-1)


def unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1.e-6)


class TrainingGraph:
    def __init__(self):
        nav, goals, self.gates = campus_map()
        native = NativeNavigation(nav, ROOT / "Saved/NavigationRL/cache")
        source = PedestrianNetwork(native, goals)
        # Keep path junctions and stops. Degree-two bends remain in edge geometry.
        graph, _ = build_landmarks(native, source, 0., np.inf)
        self.graph = graph
        self.stops = json.loads((ROOT / "Content/Campus/navigation_stops.json").read_text())["stops"]
        self.distance, _ = shortest_path_tables(graph)
        self.allowed = reachable_exits(graph)
        self.anchors = anchor_distances(graph)
        self.lower_bound = np.maximum.reduce([np.abs(row[:, None] - row[graph.stops]) for row in self.anchors])
        self.neighbors = np.zeros((len(graph.edges), EXITS), dtype=np.int64)
        self.valid = np.zeros_like(self.neighbors, dtype=bool)
        self.costs = np.zeros(self.neighbors.shape, np.float32)
        self.outgoing = np.zeros((*self.neighbors.shape, 2), np.float32)
        self.incoming = np.zeros_like(self.outgoing)
        for at, edges in enumerate(graph.edges):
            if len(edges) > EXITS:
                raise ValueError(f"Landmark {at} exceeds the model's {EXITS} exit slots")
            for index, (to, length) in enumerate(sorted(edges.items())):
                points = graph.geometry[at, to]
                self.neighbors[at, index] = to
                self.costs[at, index] = length
                self.valid[at, index] = True
                self.outgoing[at, index] = unit(points[1] - points[0])
                self.incoming[at, index] = unit(points[0] - points[1])

    def features(self, nodes, destinations, previous):
        xy = self.graph.positions
        neighbor = self.neighbors[nodes]
        goal = self.graph.stops[destinations]
        delta = xy[goal] - xy[nodes]
        left = np.linalg.norm(xy[goal, None] - xy[neighbor], axis=-1)
        distance = np.linalg.norm(delta, axis=-1)[:, None]
        heading = self.outgoing[nodes]
        incoming_matches = self.valid[nodes] & (neighbor == previous[:, None])
        incoming = (self.incoming[nodes] * incoming_matches[..., None]).sum(axis=1)
        dot = (heading * incoming[:, None, :]).sum(axis=-1)
        cross = incoming[:, :1] * heading[..., 1] - incoming[:, 1:] * heading[..., 0]
        row = np.empty((len(nodes), EXITS, FEATURES), np.float32)
        bound = self.lower_bound[neighbor, destinations[:, None]]
        row[..., 0] = bound / 100
        row[..., 1] = (self.lower_bound[nodes, destinations, None] - bound) / 100
        row[..., 2:4] = delta[:, None] / 100
        row[..., 4:6] = heading
        row[..., 6], row[..., 7] = dot, cross
        row[..., 8] = self.costs[nodes] / 100
        row[..., 9] = left / 100
        row[..., 10] = (distance - left) / 100
        row[..., 11] = neighbor == goal[:, None]
        return row.reshape(len(nodes), -1)

    def export(self, path):
        graph = self.graph
        links = []
        for at, edges in enumerate(graph.edges):
            for to in sorted(edges):
                if at < to:
                    links.append([at, int(to), graph.geometry[at, to][1:-1].tolist()])
        data = {"positions": graph.positions.tolist(), "links": links, "gate_count": self.gates,
                "stops": [{**stop, "node": int(node)} for stop, node in zip(self.stops, graph.stops)],
                "exits": EXITS, "features": FEATURES,
                "anchor_distance_cm": np.rint(self.anchors * 100).astype(np.int32).tolist(),
                "allowed_exit_bits_by_stop": self.allowed.T.tolist()}
        save_json(path, data)


def reachable_exits(graph):
    """An exit is legal only if it can reach the goal without returning here."""
    result = np.zeros((len(graph.edges), len(graph.stops)), dtype=np.uint8)
    adjacency = [sorted(edges) for edges in graph.edges]
    for at, neighbors in enumerate(adjacency):
        component = [-1] * len(adjacency)
        component[at] = at
        for exit_node in neighbors:
            if component[exit_node] >= 0:
                continue
            component[exit_node] = exit_node
            queue = [exit_node]
            for node in queue:
                for other in adjacency[node]:
                    if component[other] < 0:
                        component[other] = exit_node
                        queue.append(other)
        goals = np.asarray(component)[graph.stops]
        for slot, node in enumerate(neighbors):
            result[at] |= (goals == component[node]).astype(np.uint8) << slot
    return result


def legal_exits(campus, nodes, goals):
    bits = 1 << np.arange(EXITS)
    return (campus.allowed[nodes, goals, None] & bits) != 0


def anchor_distances(graph):
    """Landmark distances give a lower bound on the walk between two nodes."""
    nearest = np.full(len(graph.positions), np.inf)
    node = graph.stops[0]
    rows = []
    for _ in range(min(ANCHOR_COUNT, len(nearest))):
        distances, _ = shortest_path_tables(replace(graph, stops=np.array([node])))
        distances = distances[:, 0]
        rows.append(np.rint(distances * 100) / 100)
        np.minimum(nearest, distances, out=nearest)
        node = nearest.argmax()
    return np.asarray(rows, dtype=np.float32)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")


def train_epoch(policy, optimizer, campus, destinations, rng):
    nodes, goals = np.nonzero(np.ones((len(campus.graph.edges), len(destinations)), dtype=bool))
    goals = destinations[goals]
    options = campus.allowed[nodes, goals].astype(np.int32)
    keep = (options & (options - 1)) != 0
    nodes, goals = nodes[keep], goals[keep]
    selected = rng.choice(len(nodes), 24576, replace=len(nodes) < 24576)
    losses = []
    for indices in np.array_split(selected, 48):
        current, goal = nodes[indices], goals[indices]
        neighbors = campus.neighbors[current]
        legal = legal_exits(campus, current, goal)
        options = campus.costs[current] + campus.distance[neighbors, goal[:, None]]
        # Vary approach direction and occasionally remove a non-optimal exit,
        # matching the way the visited mask changes during an actual rollout.
        slots = (rng.random(legal.shape) * campus.valid[current]).argmax(axis=1)
        previous = neighbors[np.arange(len(current)), slots].copy()
        previous[rng.random(len(current)) < .2] = -1
        blocked = (neighbors == previous[:, None]) & (options > np.where(legal, options, np.inf).min(axis=1, keepdims=True) + .001)
        legal &= ~blocked
        excess = np.where(legal, options, np.inf)
        excess -= excess.min(axis=1, keepdims=True)
        logits = policy(torch.from_numpy(campus.features(current, goal, previous)))
        logits = logits.masked_fill(~torch.from_numpy(legal), -torch.inf)
        best = torch.from_numpy(excess <= .001)
        regret = (logits.softmax(1) * torch.from_numpy(excess.clip(0, 40).astype(np.float32)) / 10).sum(1)
        loss = (logits.logsumexp(1) - logits.masked_fill(~best, -torch.inf).logsumexp(1) + .7 * regret).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def pairs(campus, destinations, count, seed):
    rng = np.random.default_rng(seed)
    options = [(source, goal) for goal in destinations for source, stop in enumerate(campus.stops)
               if source != goal and stop.get("building_id") != campus.stops[goal].get("building_id")]
    selected = rng.choice(len(options), min(count, len(options)), replace=False)
    start, goal = np.asarray(options)[selected].T
    return campus.graph.stops[start], goal


@torch.inference_mode()
def evaluate(policy, campus, requests):
    sources, destinations = requests
    goals = campus.graph.stops[destinations]
    current = sources.copy()
    visited = np.zeros((len(sources), len(campus.graph.edges)), dtype=bool)
    previous = np.full(len(sources), -1, dtype=int)
    distances = np.zeros(len(sources))
    decisions = np.zeros(len(sources), dtype=int)
    retraced = np.zeros(len(sources), dtype=int)
    trails = [[] for _ in sources]
    rows = np.arange(len(sources))
    visited[rows, current] = True
    # Each forward edge discovers a node and each retrace pops one such edge.
    for _ in range(2 * (len(campus.graph.edges) - 1)):
        active = current != goals
        if not active.any():
            break
        neighbors = campus.neighbors[current]
        legal = legal_exits(campus, current, destinations) & ~visited[rows[:, None], neighbors]
        counts = legal.sum(axis=1)
        branch = active & (counts > 1)
        slot = legal.argmax(axis=1)
        if branch.any():
            scores = policy(torch.from_numpy(campus.features(current[branch], destinations[branch], previous[branch]))).numpy()
            slot[branch] = np.where(legal[branch], scores, -np.inf).argmax(axis=1)
        decisions += branch
        chosen = neighbors[rows, slot]
        for i in np.flatnonzero(active):
            if counts[i]:
                assert legal[i, slot[i]]
                trails[i].append(current[i])
            else:
                assert trails[i], "Reachable goal must be found before exhausting the source"
                chosen[i] = trails[i].pop()
                retraced[i] += 1
            distances[i] += campus.graph.edges[current[i]][chosen[i]]
        previous[active] = current[active]
        current[active] = chosen[active]
        visited[rows[active], current[active]] = True
    assert (current == goals).all()
    ratio = distances / campus.distance[sources, destinations]
    return {"trips": len(sources), "completed": int((current == goals).sum()),
            "trips_without_retracing": int((retraced == 0).sum()),
            "trips_with_retracing": int((retraced > 0).sum()),
            "retraced_edges": int(retraced.sum()), "junction_inferences": int(decisions.sum()),
            "median_route_ratio": float(np.median(ratio)),
            "p95_route_ratio": float(np.quantile(ratio, .95)),
            "max_route_ratio": float(np.max(ratio))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "Saved/NavigationRL/turns")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    campus = TrainingGraph()
    train, validation, test = split_destinations(campus.stops, 317)
    validation_pairs = pairs(campus, validation, 256, 18)
    policy = TurnPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0003)
    best = (np.inf, np.inf)
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(policy, optimizer, campus, train, rng)
        scheduler.step()
        metrics = evaluate(policy, campus, validation_pairs)
        quality = (metrics["p95_route_ratio"], metrics["median_route_ratio"])
        history.append({"epoch": epoch, "loss": loss, "validation": metrics})
        if quality < best:
            best = quality
            selected = epoch
            torch.save(policy.state_dict(), args.output / "turn_policy.pt")
        print(epoch, round(loss, 4), metrics["trips_without_retracing"], round(metrics["p95_route_ratio"], 3), flush=True)
    policy.load_state_dict(torch.load(args.output / "turn_policy.pt", weights_only=True))
    policy.eval()
    report = {"landmarks": len(campus.graph.positions), "stops": len(campus.stops),
              "links": sum(map(len, campus.graph.edges)) // 2, "selected_epoch": selected,
              "split_stops": [len(train), len(validation), len(test)], "history": history,
              "held_out": evaluate(policy, campus, pairs(campus, test, 1200, 317))}
    save_json(args.output / "report.json", report)
    print(report["held_out"], flush=True)
    if args.export:
        campus.export(args.export / "Campus/turn_paths.json")
        output = args.export / "Models/turn_policy.onnx"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(policy, torch.zeros(1, EXITS * FEATURES), output, input_names=["observation"],
                          output_names=["turn_scores"], dynamic_axes={"observation": {0: "batch"},
                          "turn_scores": {0: "batch"}}, opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()
