#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Create the deterministic grid policy bundled with packaged D2 demos."""

import argparse
import json
from collections import deque
from pathlib import Path


GRID_SIZE = (6, 6)
GOAL = (5, 0)
HAZARDS = {(2, 1), (2, 2), (4, 3), (1, 4)}
MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))


def add(left, right):
    return left[0] + right[0], left[1] + right[1]


def is_free(cell):
    return (
        0 <= cell[0] < GRID_SIZE[0]
        and 0 <= cell[1] < GRID_SIZE[1]
        and cell not in HAZARDS
    )


def shortest_distances():
    distances = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        cell = queue.popleft()
        for move in MOVES:
            neighbor = add(cell, move)
            if is_free(neighbor) and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def create_policy():
    distances = shortest_distances()
    expected_cells = GRID_SIZE[0] * GRID_SIZE[1] - len(HAZARDS)
    if len(distances) != expected_cells:
        raise RuntimeError("not every non-hazard grid cell can reach the goal")

    q_table = {}
    for cell, distance in sorted(distances.items()):
        values = [0.0] * len(MOVES)
        if cell != GOAL:
            action = next(
                index
                for index, move in enumerate(MOVES)
                if distances.get(add(cell, move)) == distance - 1
            )
            values[action] = 1.0
        observation = (cell[0], cell[1], GOAL[0] - cell[0], GOAL[1] - cell[1])
        q_table[",".join(str(value) for value in observation)] = values

    return {
        "format": "schola-d2-qtable",
        "observation_space": {
            "type": "Box",
            "low": [0, 0, -5, -5],
            "high": [5, 5, 5, 5],
            "shape": [4],
            "dtype": "float32",
        },
        "action_space": {"type": "Discrete", "n": 4},
        "episodes": 0,
        "q": q_table,
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=here.parent / "policies" / "q_policy.json"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy = create_policy()
    args.out.write_text(json.dumps(policy, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(policy['q'])} states to {args.out}")


if __name__ == "__main__":
    main()
