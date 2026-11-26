"""Train and evaluate an encoder/decoder that produces ordered campus routes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from navigation_env import PACE, ROOT
from navigation_landmarks import LandmarkGraph, shortest_path_tables, split_destinations
from train_waypoint_navigation import sample_pairs, search

CONTENT = ROOT / "Content/Campus"
WIDTH = 128


class CampusGraph:
    def __init__(self):
        raw = (CONTENT / "navigation_landmarks.json").read_bytes()
        asset = json.loads(raw)
        metadata = json.loads((CONTENT / "navigation_stops.json").read_text())
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.stops = metadata["stops"]
        positions = np.asarray(asset["positions"], np.float32)
        stops = np.asarray(asset["stops"])
        edges = [{} for _ in positions]
        for a, b, seconds, _ in asset["links"]:
            edges[a][b] = edges[b][a] = seconds * PACE
        self.graph = LandmarkGraph(positions, stops, edges, {})

        # Express geometry in units of 100 m. Use the same features for graphs
        # of different sizes and share the weights across all landmarks.
        self.xy = torch.tensor(positions / 100)
        self.goals = torch.tensor(stops)
        node_type = torch.zeros(len(positions), 3)
        node_type[:, 0] = 1
        node_type[self.goals] = 0
        gate_count = metadata["gate_count"]
        node_type[self.goals[:gate_count], 1] = 1
        node_type[self.goals[gate_count:], 2] = 1
        self.nodes = torch.cat((self.xy, node_type), dim=1)
        degree = max(map(len, edges))
        self.neighbors = torch.zeros((len(edges), degree), dtype=torch.long)
        self.costs = torch.zeros(self.neighbors.shape)
        self.connected = torch.zeros_like(self.neighbors, dtype=torch.bool)
        source, target, weights = [], [], []
        for node, links in enumerate(edges):
            neighbors = sorted(links)
            costs = torch.tensor([links[neighbor] for neighbor in neighbors])
            self.neighbors[node, :len(neighbors)] = torch.tensor(neighbors)
            self.costs[node, :len(neighbors)] = costs
            self.connected[node, :len(neighbors)] = True
            weight = 1 / (1 + costs / 30)
            source.extend([node] * len(neighbors))
            target.extend(neighbors)
            weights.extend((weight / weight.sum()).tolist())
        self.adjacency = torch.sparse_coo_tensor([source, target], weights, (len(edges), len(edges))).coalesce().to_sparse_csr()

    def options(self, current, visited):
        neighbors = self.neighbors[current]
        return neighbors, self.costs[current], self.connected[current] & ~visited.gather(1, neighbors)


class RouteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embedding = nn.Linear(5, WIDTH)
        self.message_layers = nn.ModuleList(nn.Linear(2 * WIDTH, WIDTH) for _ in range(3))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(3))
        self.initial = nn.Linear(2 * WIDTH, WIDTH)
        self.decoder = nn.GRUCell(2 * WIDTH, WIDTH)
        self.key = nn.Linear(WIDTH, 64)
        self.query = nn.Linear(3 * WIDTH, 64)
        self.geometry = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, campus):
        embedding = self.node_embedding(campus.nodes)
        for layer, norm in zip(self.message_layers, self.norms):
            messages = torch.sparse.mm(campus.adjacency, embedding)
            embedding = norm(embedding + torch.nn.functional.silu(layer(torch.cat((embedding, messages), dim=1))))
        return embedding, self.key(embedding)

    def start(self, embedding, source, goal):
        return torch.tanh(self.initial(torch.cat((embedding[source], embedding[goal]), dim=1)))

    def step(self, campus, encoded, current, goal, hidden, neighbors, costs):
        embedding, keys = encoded
        at, destination = embedding[current], embedding[goal]
        hidden = self.decoder(torch.cat((at, destination), dim=1), hidden)
        query = self.query(torch.cat((hidden, at, destination), dim=1))
        pointer = (query[:, None, :] * keys[neighbors]).sum(dim=2) / 8

        offset = campus.xy[goal] - campus.xy[current]
        edge = campus.xy[neighbors] - campus.xy[current, None, :]
        remaining = offset[:, None, :] - edge
        direct = torch.linalg.vector_norm(offset, dim=1, keepdim=True).expand_as(costs)
        left = torch.linalg.vector_norm(remaining, dim=2)
        features = torch.cat((offset[:, None, :].expand_as(edge), edge, direct[..., None], left[..., None],
                              (direct - left)[..., None], costs[..., None] / 100), dim=2)
        return hidden, pointer + self.geometry(features).squeeze(2)


def imitation_loss(model, campus, source, destination, remaining, steps, roll_in, regret_weight):
    encoded = model.encode(campus)
    goal = campus.goals[destination]
    hidden = model.start(encoded[0], source, goal)
    current = source
    rows = torch.arange(len(source))
    visited = torch.zeros((len(source), len(campus.xy)), dtype=torch.bool)
    active = current != goal
    loss = 0.
    decisions = 0
    for _ in range(steps):
        visited[rows, current] = True
        neighbors, costs, eligible = campus.options(current, visited)
        active = active & eligible.any(dim=1)
        if not active.any():
            break
        hidden, logits = model.step(campus, encoded, current, goal, hidden, neighbors, costs)
        allowed = eligible[active]
        logits = logits[active].masked_fill(~allowed, -torch.inf)
        # All equal-cost shortest hops are good labels. If a sampled model
        # action has blocked the shortest hop, supervise the best allowed hop.
        value = costs[active] + remaining[neighbors[active], destination[active, None]]
        value = value.masked_fill(~allowed, torch.inf)
        excess = value - value.amin(dim=1, keepdim=True)
        optimal = excess <= .001
        regret = (logits.softmax(dim=1) * excess.clamp(max=50.) / 10.).sum(dim=1)
        loss += (logits.logsumexp(dim=1) - logits.masked_fill(~optimal, -torch.inf).logsumexp(dim=1)
                 + regret_weight * regret).sum()
        decisions += int(active.sum())
        teacher = value.argmin(dim=1)
        chosen = torch.where(torch.rand(len(teacher)) < roll_in, logits.detach().argmax(dim=1), teacher)
        next_node = current.clone()
        next_node[active] = neighbors[active].gather(1, chosen[:, None]).squeeze(1)
        current = next_node
        active = active & (current != goal)
    return loss / decisions


@torch.inference_mode()
def decode(model, campus, encoded, source, destination, max_steps):
    """Return each greedy path and whether it reached the destination."""
    goal = campus.goals[destination]
    current = source
    hidden = model.start(encoded[0], source, goal)
    visited = torch.zeros((len(source), len(campus.xy)), dtype=torch.bool)
    rows = torch.arange(len(source))
    path = torch.full((len(source), max_steps + 1), -1, dtype=torch.long)
    path[:, 0] = current
    active = current != goal
    for hop in range(1, max_steps + 1):
        visited[rows, current] = True
        neighbors, costs, eligible = campus.options(current, visited)
        active = active & eligible.any(dim=1)
        if not active.any():
            break
        hidden, logits = model.step(campus, encoded, current, goal, hidden, neighbors, costs)
        chosen = logits.masked_fill(~eligible, -torch.inf).argmax(dim=1)
        current = torch.where(active, neighbors[rows, chosen], current)
        path[active, hop] = current[active]
        active = active & (current != goal)
    return path.numpy(), (current == goal).numpy()


def route_cost(graph, path):
    return sum(graph.edges[a][b] for a, b in zip(path, path[1:]))


def shortest_route(graph, next_hops, source, destination):
    goal = graph.stops[destination]
    path = [source]
    while path[-1] != goal:
        path.append(int(next_hops[path[-1], destination]))
    return path


@torch.inference_mode()
def evaluate(model, campus, pairs, reference, next_hops, batch_size, max_steps):
    model.eval()
    started = time.perf_counter()
    encoded = model.encode(campus)
    encode_ms = (time.perf_counter() - started) * 1000
    sources, destinations = pairs
    ratios, examples, planned = [], [], []
    started = time.perf_counter()
    for offset in range(0, len(sources), batch_size):
        group = slice(offset, offset + batch_size)
        paths, completed = decode(model, campus, encoded, torch.from_numpy(sources[group]),
                                  torch.from_numpy(destinations[group]), max_steps)
        planned.extend(zip(paths, completed))
    decode_ms = (time.perf_counter() - started) * 1000
    for source, destination, (nodes, completed) in zip(sources, destinations, planned):
        raw = nodes[nodes >= 0].tolist()
        path = raw if completed else shortest_route(campus.graph, next_hops, source, destination)
        cost = route_cost(campus.graph, path)
        baseline = reference[source, destination]
        assert len(set(raw)) == len(raw) and raw[0] == source
        route_cost(campus.graph, raw)  # Every neural hop must be a real edge, including failed attempts.
        assert path[-1] == campus.graph.stops[destination]
        if completed:
            ratios.append(cost / baseline)
        examples.append({"start": int(source), "destination_stop": int(destination),
                         "destination_landmark": int(campus.graph.stops[destination]),
                         "landmarks": [int(node) for node in path], "fallback": not bool(completed),
                         "predicted_landmarks": raw, "graph_meters": cost, "shortest_graph_meters": baseline})

    return {
        "requests": len(sources),
        "model_completed": len(ratios),
        "fallbacks": len(sources) - len(ratios),
        "model_completion_rate": len(ratios) / len(sources),
        "model_routes_within_1_percent_of_optimal": int(np.count_nonzero(np.asarray(ratios) <= 1.01)),
        "model_median_cost_ratio": float(np.median(ratios)) if ratios else None,
        "model_p95_cost_ratio": float(np.quantile(ratios, .95)) if ratios else None,
        "model_max_cost_ratio": float(max(ratios)) if ratios else None,
        "encoder_ms": encode_ms, "batch_size": batch_size, "total_batched_decode_ms": decode_ms,
        "batched_decode_ms_per_request": decode_ms / len(sources),
    }, examples


def stop_pairs(campus, destinations, count, rng):
    pairs = [(start, destination) for destination in destinations for start, stop in enumerate(campus.stops)
             if campus.stops[destination].get("building_id") != stop.get("building_id") and start != destination]
    selected = rng.choice(len(pairs), min(count, len(pairs)), replace=False)
    starts, goals = np.asarray(pairs, dtype=np.int64)[selected].T
    return campus.graph.stops[starts], goals


def train(args, campus, model, reference, next_hops):
    train_goals, validation_goals, test_goals = split_destinations(campus.stops, 317)
    rng = np.random.default_rng(args.seed)
    validation_pairs = stop_pairs(campus, validation_goals, 256, rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate * .15)
    remaining = torch.tensor(reference, dtype=torch.float32)
    best = None
    history = []
    if args.warm_start:
        initial = load_weights(campus, model, args.warm_start)
        validation, _ = evaluate(model, campus, validation_pairs, reference, next_hops, args.batch_size, args.max_steps)
        best = validation_quality(validation)
        print(f"warm start validation {validation['model_completed']}/256 "
              f"p95 cost ratio {validation['model_p95_cost_ratio']}", flush=True)
        torch.save({**initial, "epoch": 0, "max_steps": args.max_steps}, args.output / "ordered_routes.pt")
        history.append({"epoch": 0, "validation": validation})
    for epoch in range(1, args.epochs + 1):
        model.train()
        sources, destinations = sample_pairs(campus.graph, train_goals, args.batch_size * args.updates, rng)
        losses = []
        for source, destination in zip(torch.from_numpy(sources).split(args.batch_size),
                                       torch.from_numpy(destinations).split(args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            roll_in = min(.2, (epoch - 1) * .025) if args.warm_start is None else .2
            loss = imitation_loss(model, campus, source, destination, remaining, args.window, roll_in, args.regret_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        validation, _ = evaluate(model, campus, validation_pairs, reference, next_hops, args.batch_size, args.max_steps)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": validation}
        history.append(row)
        # Select on complete-route behavior. Penalize both fallback and detours.
        quality = validation_quality(validation)
        if best is None or quality > best:
            best = quality
            torch.save({"state_dict": model.state_dict(), "graph_sha256": campus.fingerprint, "epoch": epoch,
                        "max_steps": args.max_steps}, args.output / "ordered_routes.pt")
        print(f"epoch {epoch:02d} loss {row['training_loss']:.4f} "
              f"validation {validation['model_completed']}/256 "
              f"p95 cost ratio {validation['model_p95_cost_ratio']}", flush=True)
    checkpoint = load_weights(campus, model, args.output / "ordered_routes.pt")
    reports = evaluate_tests(args, campus, model, reference, next_hops, test_goals)
    return {
        "graph_sha256": campus.fingerprint, "landmarks": len(campus.xy), "stops": len(campus.stops),
        "split_destinations": {"train": len(train_goals), "validation": len(validation_goals), "test": len(test_goals)},
        "settings": {"seed": args.seed, "epochs": args.epochs, "updates_per_epoch": args.updates,
                     "batch_size": args.batch_size, "window": args.window, "max_steps": args.max_steps,
                     "learning_rate": args.learning_rate, "regret_weight": args.regret_weight,
                     "warm_start": str(args.warm_start) if args.warm_start else None},
        "checkpoint_epoch": checkpoint["epoch"], "history": history, "test": reports,
    }


def validation_quality(validation):
    return (validation["model_completion_rate"] / (validation["model_p95_cost_ratio"] or 100),
            validation["model_routes_within_1_percent_of_optimal"])


def evaluate_tests(args, campus, model, reference, next_hops, test_goals):
    rng = np.random.default_rng(317)
    reports = {}
    for name, pairs in (
        ("held_out_landmark_to_stop", sample_pairs(campus.graph, test_goals, 1200, rng)),
        ("held_out_stop_to_stop", stop_pairs(campus, test_goals, 1200, rng)),
    ):
        metrics, routes = evaluate(model, campus, pairs, reference, next_hops, args.batch_size, args.max_steps)
        reports[name] = metrics
        save_json(args.output / f"{name}_routes.json", routes)
        print(name, metrics, flush=True)
    return reports


def load_weights(campus, model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["graph_sha256"] != campus.fingerprint:
        raise ValueError("Checkpoint was trained on a different campus graph")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Saved/NavigationRL/ordered")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--updates", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--regret-weight", type=float, default=0.)
    parser.add_argument("--warm-start", type=Path, help="Initialize training from a compatible checkpoint")
    parser.add_argument("--route", nargs=2, type=int, metavar=("START_LANDMARK", "DESTINATION_STOP"),
                        help="Decode a route from the checkpoint under --output")
    args = parser.parse_args()
    if min(args.epochs, args.updates, args.batch_size, args.window, args.max_steps, args.threads) < 1:
        parser.error("training and decoding limits must be positive")
    if args.learning_rate <= 0 or args.regret_weight < 0:
        parser.error("learning rate must be positive and regret weight cannot be negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    campus, model = CampusGraph(), RouteModel()
    if args.route:
        start, destination = args.route
        if not (0 <= start < len(campus.xy) and 0 <= destination < len(campus.stops)):
            parser.error("start landmark or destination stop index out of range")
        checkpoint = load_weights(campus, model, args.output / "ordered_routes.pt")
        with torch.inference_mode():
            encoded = model.encode(campus)
            path, completed = decode(model, campus, encoded, torch.tensor([start]), torch.tensor([destination]),
                                     checkpoint["max_steps"])
        route = path[0][path[0] >= 0].tolist()
        if not completed[0]:
            route, _, _ = search(campus.graph, start, campus.graph.stops[destination])
            route = [int(node) for node in route]
        print(json.dumps({"landmarks": route, "fallback": not bool(completed[0])}))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    reference, next_hops = shortest_path_tables(campus.graph)
    save_json(args.output / "report.json", train(args, campus, model, reference, next_hops))


if __name__ == "__main__":
    main()
