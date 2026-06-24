# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
RLlib connector factories for Schola env runners.
"""


def schola_env_to_module_flatten_connector(env, spaces=None, device=None):
    """Return ``FlattenObservations`` for Schola multi-agent env runners.

    RLlib may invoke this with a live ``env`` or with ``spaces`` only before the env
    exists. In the ``spaces`` path, ``spaces`` must contain ``__env_single__`` mapping
    to ``(observation_space, action_space)``.
    """
    from ray.rllib.connectors.env_to_module import FlattenObservations

    if env is not None:
        return FlattenObservations(
            input_observation_space=env.single_observation_space,
            input_action_space=env.single_action_space,
            multi_agent=True,
        )
    if spaces is None or "__env_single__" not in spaces:
        raise ValueError(
            "Schola env_to_module connector requires a constructed env or "
            "`spaces` containing '__env_single__' (observation_space, action_space). "
            f"Got env={env!r}, spaces_keys={None if spaces is None else list(spaces)}."
        )
    obs_space, action_space = spaces["__env_single__"]
    return FlattenObservations(
        input_observation_space=obs_space,
        input_action_space=action_space,
        multi_agent=True,
    )
