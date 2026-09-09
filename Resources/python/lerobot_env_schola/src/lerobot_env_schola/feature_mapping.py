# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Infer LeRobot feature metadata from a Schola observation mapping."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten_space
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_CHANNELS = (1, 3, 4)


def is_image_policy_key(policy_key: str) -> bool:
    """Return whether a policy key names a visual observation."""
    return policy_key == OBS_IMAGE or policy_key.startswith(f"{OBS_IMAGES}.")


def infer_features(
    policy_sources: Mapping[str, tuple[str, ...]],
    source_spaces: Mapping[str, gym.Space],
    action_space: Box,
) -> tuple[dict[str, PolicyFeature], dict[str, str]]:
    """Infer LeRobot features from policy sources and live source spaces.

    Returns ``(features, features_map)``. ``features_map`` is always the
    identity ``{key: key for key in features}`` because ``features`` is
    already keyed by policy names. LeRobot still requires the map.
    """
    features: dict[str, PolicyFeature] = {}
    claimed_sources: set[str] = set()

    for policy_key, sources in policy_sources.items():
        unknown = set(sources) - source_spaces.keys()
        if unknown:
            raise ValueError(
                f"Policy feature {policy_key!r} references unknown Schola "
                f"observations {sorted(unknown)}"
            )

        reused = claimed_sources.intersection(sources)
        if reused:
            logger.warning(
                "Schola observations %s are reused by policy feature %r.",
                sorted(reused),
                policy_key,
            )
        claimed_sources.update(sources)

        if is_image_policy_key(policy_key):
            _add_image_feature(
                policy_key,
                sources,
                source_spaces,
                features,
            )
        else:
            _add_value_feature(
                policy_key,
                sources,
                source_spaces,
                features,
            )

    missing = source_spaces.keys() - claimed_sources
    if missing:
        logger.warning(
            "Schola observations %s are not mapped to policy inputs and will "
            "be ignored.",
            sorted(missing),
        )

    features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=action_space.shape,
    )
    # Identity: Schola features already use policy keys.
    return features, {key: key for key in features}


def as_source_tuple(
    policy_key: str, configured_sources: str | Sequence[str]
) -> tuple[str, ...]:
    """Normalize a YAML source string or list into an ordered source tuple."""
    if isinstance(configured_sources, str):
        sources = (configured_sources,)
    elif isinstance(configured_sources, Sequence) and all(
        isinstance(source, str) for source in configured_sources
    ):
        sources = tuple(configured_sources)
    else:
        raise TypeError(
            f"Policy feature {policy_key!r} must map to a source string "
            "or list of source strings"
        )
    if not sources or any(not source for source in sources):
        raise ValueError(f"Policy feature {policy_key!r} has an empty source")
    return sources


def _add_image_feature(
    policy_key: str,
    sources: tuple[str, ...],
    source_spaces: Mapping[str, gym.Space],
    features: dict[str, PolicyFeature],
) -> None:
    if len(sources) != 1:
        raise ValueError(f"Image feature {policy_key!r} must map to exactly one source")
    space = source_spaces[sources[0]]
    if not isinstance(space, Box) or len(space.shape) != 3:
        raise TypeError(
            f"Image observation {sources[0]!r} must be a three-dimensional Box"
        )
    channels, height, width = space.shape
    if channels not in SUPPORTED_IMAGE_CHANNELS:
        raise ValueError(
            f"Image observation {sources[0]!r} must have 1, 3, or 4 channels"
        )
    is_float = np.issubdtype(space.dtype, np.floating)
    is_uint8 = space.dtype == np.dtype(np.uint8)
    if not (is_float or is_uint8):
        raise TypeError(
            f"Image observation {sources[0]!r} must use float or uint8 values"
        )
    if is_float and not (np.all(space.low >= 0) and np.all(space.high <= 1)):
        raise ValueError(
            f"Floating-point image observation {sources[0]!r} must be "
            "bounded within [0, 1]"
        )

    features[policy_key] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(height, width, channels),
    )


def _add_value_feature(
    policy_key: str,
    sources: tuple[str, ...],
    source_spaces: Mapping[str, gym.Space],
    features: dict[str, PolicyFeature],
) -> None:
    size = 0
    for source in sources:
        flattened = flatten_space(source_spaces[source])
        if not isinstance(flattened, Box):
            raise TypeError(
                f"Schola observation {source!r} cannot be flattened to a Box"
            )
        size += flattened.shape[0]
    feature_type = FeatureType.ENV if policy_key == OBS_ENV_STATE else FeatureType.STATE
    features[policy_key] = PolicyFeature(type=feature_type, shape=(size,))
