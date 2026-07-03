# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Schola RLlib Algorithm extensions for checkpointing policy mapping metadata.
"""

from __future__ import annotations

from typing import Any, Collection, List, Optional, Tuple, Type, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.typing import StateDict

from schola.rllib.policy_mapping import (
    SCHOLA_POLICY_MAPPING_COMPONENT,
    make_policy_mapping_checkpoint_from_config,
)


def schola_algorithm_subclass(base_algo_class: Type[Algorithm]) -> Type[Algorithm]:
    """Return an Algorithm subclass that checkpoints Schola policy mapping metadata.

    The frozen agent-to-policy record (stashed in ``config.env_config`` by the
    training script) is exposed as a ``Checkpointable`` subcomponent so it is
    saved and restored by RLlib's own checkpoint machinery, landing under
    ``<algorithm_checkpoint>/schola_policy_mapping/``.
    """

    class ScholaAlgorithm(base_algo_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._schola_policy_mapping = make_policy_mapping_checkpoint_from_config(
                self.config
            )

        @override(Checkpointable)
        def get_state(
            self,
            components: Optional[Union[str, Collection[str]]] = None,
            *,
            not_components: Optional[Union[str, Collection[str]]] = None,
            **kwargs: Any,
        ) -> StateDict:
            state = super().get_state(
                components=components,
                not_components=not_components,
                **kwargs,
            )
            # RLlib's ``save_to_path`` pulls each subcomponent's state from the
            # parent via ``get_state(components=<name>)``, so the component must
            # be represented here for it to be written to disk.
            if self._check_component(
                SCHOLA_POLICY_MAPPING_COMPONENT, components, not_components
            ):
                state[SCHOLA_POLICY_MAPPING_COMPONENT] = (
                    self._schola_policy_mapping.get_state()
                )
            return state

        @override(Checkpointable)
        def set_state(self, state: StateDict) -> None:
            super().set_state(state)
            if SCHOLA_POLICY_MAPPING_COMPONENT in state:
                self._schola_policy_mapping.set_state(
                    state[SCHOLA_POLICY_MAPPING_COMPONENT]
                )

        @override(Checkpointable)
        def get_checkpointable_components(self) -> List[Tuple[str, Any]]:
            components = super().get_checkpointable_components()
            components.append(
                (SCHOLA_POLICY_MAPPING_COMPONENT, self._schola_policy_mapping)
            )
            return components

    ScholaAlgorithm.__name__ = f"Schola{base_algo_class.__name__}"
    ScholaAlgorithm.__qualname__ = ScholaAlgorithm.__name__
    return ScholaAlgorithm
