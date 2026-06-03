# The below code is adapted from https://github.com/Farama-Foundation/Gymnasium/blob/main/tests/spaces/utils.py
"""
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np

from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Sequence,
    Space,
    Text,
    Tuple,
)

TESTING_FUNDAMENTAL_SPACES = [
    Discrete(3),
    # Discrete(3, start=-1),
    Box(low=0.0, high=1.0),
    # Box(low=0.0, high=np.inf, shape=(2, 2)),
    Box(low=np.array([-10.0, 0.0]), high=np.array([10.0, 10.0]), dtype=np.float64),
    # Box(low=-np.inf, high=0.0, shape=(2, 1)),
    # Box(low=0.0, high=np.inf, shape=(2, 1)),
    Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8),
    Box(low=np.array([0, 0, 1]), high=np.array([1, 0, 1]), dtype=np.bool_),
    # Box(
    #    low=np.array([-np.inf, -np.inf, 0, -10]),
    #    high=np.array([np.inf, 0, np.inf, 10]),
    #    dtype=np.int32,
    # ),
    MultiDiscrete([2, 2]),
    # MultiDiscrete([[2, 3], [3, 2]]),
    # MultiDiscrete([2, 2], start=[10, 10]),
    # MultiDiscrete([[2, 3], [3, 2]], start=[[10, 20], [30, 40]]),
    MultiBinary(8),
    # MultiBinary([2, 3]),
    Text(6),
    Text(min_length=3, max_length=6),
    Text(6, charset="abcdef"),
]
TESTING_FUNDAMENTAL_SPACES_IDS = [f"{space}" for space in TESTING_FUNDAMENTAL_SPACES]


TESTING_COMPOSITE_SPACES = [
    # Tuple spaces
    # Tuple([Discrete(5), Discrete(4)]),
    # Tuple(
    #   (
    #        Discrete(5),
    #        Box(
    #            low=np.array([0.0, 0.0]),
    #            high=np.array([1.0, 5.0]),
    #            dtype=np.float64,
    #        ),
    #    )
    # ),
    # Tuple((Discrete(5), Tuple((Box(low=0.0, high=1.0, shape=(3,)), Discrete(2))))),
    # Tuple((Discrete(3), Dict(position=Box(low=0.0, high=1.0), velocity=Discrete(2)))),
    # Tuple((Graph(node_space=Box(-1, 1, shape=(2, 1)), edge_space=None), Discrete(2))),
    # Dict spaces
    Dict(
        {
            "position": Discrete(5),
            "velocity": Box(
                low=np.array([0.0, 0.0]),
                high=np.array([1.0, 5.0]),
                dtype=np.float64,
            ),
        }
    ),
    Dict(
        position=Discrete(6),
        velocity=Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 5.0]),
            dtype=np.float64,
        ),
    ),
    Dict(
        {
            "a": Box(low=0, high=1, shape=(3, 3)),
            "b": Dict(
                {
                    "b_1": Box(low=-100, high=100, shape=(2,)),
                    "b_2": Box(low=-1, high=1, shape=(2,)),
                }
            ),
            "c": Discrete(4),
        }
    ),
    # Dict(
    #    a=Dict(
    #        a=Graph(node_space=Box(-100, 100, shape=(2, 2)), edge_space=None),
    #        b=Box(-100, 100, shape=(2, 2)),
    #    ),
    #    b=Tuple((Box(-100, 100, shape=(2,)), Box(-100, 100, shape=(2,)))),
    # ),
    # Graph spaces not supported yet by Schola
    # Graph(node_space=Box(-1, 1, shape=(2,)), edge_space=None),
    # Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5)),
    # Graph(node_space=Discrete(5), edge_space=Box(low=-100, high=100, shape=(3, 4))),
    # Graph(node_space=Discrete(3), edge_space=Discrete(4)),
    # Sequence spaces not supported yet by Schola
    # Sequence(Discrete(4)),
    # Sequence(Dict({"feature": Box(0, 1, (3,))})),
    # Sequence(Graph(node_space=Box(-100, 100, shape=(2, 2)), edge_space=Discrete(4))),
    # Sequence(Box(low=0.0, high=1.0), stack=True),
    # Sequence(Dict({"a": Box(0, 1, (3,)), "b": Discrete(5)}), stack=True),
    # OneOf spaces not supported yet by Schola
    # OneOf([Discrete(3), Box(low=0.0, high=1.0)]),
    # OneOf([MultiBinary(2), MultiDiscrete([2, 2])]),
]
TESTING_COMPOSITE_SPACES_IDS = [f"{space}" for space in TESTING_COMPOSITE_SPACES]

TESTING_SPACES: list[Space] = TESTING_FUNDAMENTAL_SPACES + TESTING_COMPOSITE_SPACES
TESTING_SPACES_IDS = TESTING_FUNDAMENTAL_SPACES_IDS + TESTING_COMPOSITE_SPACES_IDS


class CustomSpace(Space):
    def __eq__(self, o: object) -> bool:
        return isinstance(o, CustomSpace)


TESTING_CUSTOM_SPACE = CustomSpace()
