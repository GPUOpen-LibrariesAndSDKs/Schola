"""Train a local walking policy by imitating the migration rules on synthetic encounters."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SEED = 301
OBSERVATIONS = 64
NEIGHBORS = 4


def synthetic_encounters(seed, count):
    rng = np.random.default_rng(seed)
    rows = np.zeros((count, OBSERVATIONS), np.float32)
    rows[:, 0] = rng.uniform(-1, 1, count)
    rows[:, 1] = rng.uniform(0, 1, count)
    rows[:, 2] = rng.uniform(-0.65, 0.65, count)
    rows[:, 3] = np.minimum(1, rng.uniform(0, 5, count))
    rows[:, 4:6] = rng.uniform([-1, 0], [1, 1], (count, 2))
    rows[:, 6] = rng.uniform(0, 1, count)

    neighbors = rng.normal(0, [0.37, 0.29, 0.63, 0.24, 0], (count, NEIGHBORS, 5))
    radius = np.linalg.norm(neighbors[:, :, :2], axis=2)
    neighbors = np.take_along_axis(neighbors, np.argsort(radius, axis=1)[:, :, None], axis=1)
    present = (np.linalg.norm(neighbors[:, :, :2], axis=2) < 1) & (
        np.arange(NEIGHBORS) < rng.integers(0, NEIGHBORS + 1, count)[:, None]
    )
    neighbors[:, :, 2:4] = np.clip(neighbors[:, :, 2:4], -2, 2)
    neighbors[:, :, 4] = present
    neighbors *= present[:, :, None]
    rows[:, 8:28] = neighbors.reshape(count, -1)
    return rows


def teacher(rows):
    """Vectorized counterpart of FCampusMigrationDay::Rules at yield 1."""
    n = rows[:, 8:28].reshape(-1, NEIGHBORS, 5)
    front, side, relative_speed, _, present = n.transpose(2, 0, 1)
    ahead = np.clip((0.60 - front) * 2.5, 0, 1) * present * (front > 0)
    aligned = np.clip(1 - np.abs(side) * 5, 0, 1)
    slowdown = (ahead * aligned * np.clip(0.55 - relative_speed, 0, 1)).max(axis=1)
    dodge = (0.32 * ahead * aligned * np.where(side <= 0.03, 1, -1)).sum(axis=1)
    base = np.clip(rows[:, 3] * 2.5, 0.30, 1) * (1 - 0.22 * np.abs(rows[:, 4]) * rows[:, 5])
    speed = np.maximum(0.13, base * (1 - 0.75 * slowdown))
    lateral = np.clip((rows[:, 6] - rows[:, 0]) * 0.8 + dodge, -0.65, 0.65)
    return np.column_stack((speed, lateral)).astype(np.float32)


def export(model, path):
    nodes = []
    weights = []
    source = "observation"
    for index, layer in enumerate(model):
        target = "velocity" if index == len(model) - 1 else f"hidden_{index}"
        if isinstance(layer, nn.Linear):
            weight, bias = f"w{index}", f"b{index}"
            weights.extend([
                numpy_helper.from_array(layer.weight.detach().numpy(), weight),
                numpy_helper.from_array(layer.bias.detach().numpy(), bias),
            ])
            nodes.append(helper.make_node("Gemm", [source, weight, bias], [target], transB=1))
        else:
            nodes.append(helper.make_node("Relu", [source], [target]))
        source = target
    graph = helper.make_graph(
        nodes, "campus_migration",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", OBSERVATIONS])],
        [helper.make_tensor_value_info("velocity", TensorProto.FLOAT, ["batch", 2])],
        weights,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 8
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, path)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    observations = synthetic_encounters(SEED, 140_000)
    x = torch.from_numpy(observations)
    y = torch.from_numpy(teacher(observations))
    test_rows = synthetic_encounters(SEED + 1, 12_000)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for epoch in range(24):
        if epoch == 16:
            optimizer.param_groups[0]["lr"] *= 0.4
        for indices in torch.randperm(len(x)).split(1024):
            loss = (model(x[indices]) - y[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        error = np.abs(model(torch.from_numpy(test_rows)).numpy() - teacher(test_rows)).mean(axis=0)
    print(f"Held-out mean absolute error: forward {error[0]:.3f}, lateral {error[1]:.3f}")
    export(model, ROOT / "Content/Models/migration_policy.onnx")


if __name__ == "__main__":
    main()
