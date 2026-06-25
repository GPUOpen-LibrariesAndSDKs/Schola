# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Utility Functions and Classes for managing environment and agent ids.
"""

from functools import cached_property, singledispatchmethod
from typing import List, Optional, Tuple, TypeVar, Dict, Iterable, Union, cast

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
AgentTypes = Union[Dict[int, Dict[str, str]], List[Dict[str, str]]]


# A generic recursive dictionary type
NestedDict = Dict[K, Union[V, "NestedDict[K, V]"]]


def nested_get(dct: NestedDict[K, V], keys: Iterable[K], default: V) -> V:
    """
    Get a value from a nested dictionary, returning a default value if the key is not found.

    Parameters
    ----------
    dct : NestedDict[K,V]
        The dictionary to search.
    keys : Iterable[K]
        The keys to search for in the dictionary.
    default : V
        The value to return if the key is not found.

    Returns
    -------
    V
        The value found in the dictionary, or the default value if the key is not found.
    """
    curr_dct: NestedDict[K, V] | V = dct
    for key in keys:
        if not isinstance(curr_dct, dict):
            return default
        if key not in curr_dct:
            return default
        curr_dct = curr_dct[key]
    return cast(V, curr_dct)


class IdManager:
    """
    A class to manage the mapping between nested and flattened ids.

    Parameters
    ----------
    ids : List[List[int]]
        A nested list of lists of ids to manage, index in the list is first id, second id is stored in the second list.
    agent_types : dict or list of dict, optional
            Optional per-agent type metadata, either ``{env_id: {agent_id: type}}``
            or a list aligned with ``ids`` indices.

    Attributes
    ----------
    ids : List[List[int]]
        The nested list of lists of ids to manage.

    """

    def __init__(
        self,
        ids: List[List[str]],
        agent_types: Optional[AgentTypes] = None,
    ):
        self.ids = ids
        self._agent_types = self._normalize_agent_types(agent_types or {})

    def _get_agent_types_for_env(
        self, agent_types: AgentTypes, env_id: int
    ) -> Dict[str, str]:
        """
        Read one environment's agent types from either protocol metadata shape.
        """
        if isinstance(agent_types, list):
            if env_id < len(agent_types):
                return agent_types[env_id]
            return {}
        return agent_types.get(env_id, {})

    def _normalize_agent_types(
        self, agent_types: AgentTypes
    ) -> Dict[int, Dict[str, str]]:
        """
        Normalize optional per-agent metadata to the managed environment/agent IDs.

        Missing agent types are stored as empty strings, matching Schola's
        "no grouping type" convention.
        """
        return {
            env_id: {
                agent_id: self._get_agent_types_for_env(agent_types, env_id).get(
                    agent_id, ""
                )
                for agent_id in agent_ids
            }
            for env_id, agent_ids in enumerate(self.ids)
        }

    @property
    def agent_types(self) -> Dict[int, Dict[str, str]]:
        """
        Nested mapping of environment IDs to agent IDs to optional agent types.
        """
        return {
            env_id: dict(agent_types)
            for env_id, agent_types in self._agent_types.items()
        }

    def agent_types_for_env(self, env_id: int) -> Dict[str, str]:
        """
        Get the agent type mapping for one managed environment.
        """
        return dict(self._agent_types.get(env_id, {}))

    def get_agent_type(self, env_id: int, agent_id: str) -> str:
        """
        Get one agent's type, or an empty string when it has no type metadata.
        """
        return self._agent_types.get(env_id, {}).get(agent_id, "")

    def flatten_dict_of_dicts(
        self, nested_id_dict: Dict[int, Dict[str, T]], default: Optional[T] = None
    ) -> List[T]:
        """
        Flatten a dictionary of nested ids into a list of values.

        Parameters
        ----------
        nested_id_dict : Dict[int, Dict[int, T]]
            The dictionary to flatten.
        default : Optional[T], optional
            The default value to use if a key is not found, by default None.

        Returns
        -------
        List[T]
            A flattened list of the values found in the dictionary.
        """
        output_list = [default for i in range(0, self.num_ids)]
        for first_id, nested_ids in nested_id_dict.items():
            for second_id, value in nested_ids.items():
                output_list[self.id_map[first_id][second_id]] = value
        return cast(List[T], output_list)

    def flatten_list_of_dicts(
        self, nested_id_list: List[Dict[str, T]], default: Optional[T] = None
    ):
        """
        Flatten a list of dictionaries with nested ids into a single list.

        Parameters
        ----------
        nested_id_list : List[Dict[str, T]]
            A list of dictionaries to flatten, where the list index represents
            the first id and dictionary keys represent the second ids.
        default : Optional[T], optional
            The default value to use if a key is not found, by default None.

        Returns
        -------
        List[T]
            A flattened list of the values found in the nested structure. Ordered by UID.
        """
        output_list = [default for i in range(0, self.num_ids)]
        for first_id, nested_ids in enumerate(nested_id_list):
            for second_id, value in nested_ids.items():
                output_list[self.id_map[first_id][second_id]] = value
        return output_list

    def nest_list_to_dict_of_dicts(
        self, id_list: Iterable[T], default: Optional[T] = None
    ) -> Dict[int, Dict[str, Optional[T]]]:
        """
        Nest a list of values, indexed by flattened id, into a dictionary of nested ids.

        Parameters
        ----------
        id_list : List[T]
            The list of values to convert into a nested dictionary.
        default : Optional[T], optional
            The default value to use if a key is not found, by default None.

        Returns
        -------
        Dict[int, Dict[int, T]]
            A nested dictionary of the values in `id_list` or `default` if values are missing.
        """
        output_dict = {
            first_id: {second_id: default for second_id in nested_ids}
            for first_id, nested_ids in enumerate(self.ids)
        }
        for flat_id, body in enumerate(id_list):
            first_id, second_id = self[flat_id]
            output_dict[first_id][second_id] = body
        return output_dict

    @singledispatchmethod
    def __getitem__(self, key):
        """
        Convert a key into a nested or flattened id, from a flattened or nested id respectively.

        Parameters
        ----------
        key : Union[int, Tuple[int,int]]
            The key to convert.

        Returns
        -------
        Union[Tuple[int,int], int]
            The converted key.

        Raises
        ------
        NotImplementedError
            If the key is not of type int or Tuple[int,int].
        """
        raise NotImplementedError(
            "get item not supported for keys that aren't int or Tuple[int,int]"
        )

    @__getitem__.register
    def _(self, key: int) -> Tuple[int, str]:
        return self.id_list[key]

    @__getitem__.register(tuple)
    def _(self, key: tuple[int, str]) -> int:
        assert len(key) == 2, "if supplying tuple key must supply a key of length 2"
        return self.id_map[key[0]][key[1]]

    def get_nested_id(self, flat_id: int) -> Tuple[int, str]:
        """
        Get the nested id from a flattened id.

        Parameters
        ----------
        flat_id : int
            The flattened id to convert.

        Returns
        -------
        Tuple[int,int]
            The nested id.
        """
        return self[flat_id]

    def get_flattened_id(self, first_id: int, second_id: str) -> int:
        """
        Get the flattened id from a nested id.

        Parameters
        ----------
        first_id : int
            The first id.
        second_id : int
            The second id.

        Returns
        -------
        int
            The flattened id.
        """
        return self[first_id, second_id]

    @cached_property
    def id_list(self) -> List[Tuple[int, str]]:
        """
        List of nested ids, for lookups from flattened id to nested ids.

        Returns
        -------
        List[Tuple[int, str]]
            List of nested ids.
        """
        id_list = []
        for first_id, nested_ids in enumerate(self.ids):
            for second_id in nested_ids:
                id_list.append((first_id, second_id))
        return id_list

    @cached_property
    def id_map(self) -> List[Dict[str, int]]:
        """
        List of dictionaries mapping nested ids to flattened ids.

        Returns
        -------
        List[Dict[int,str]]
            List of dictionaries mapping nested ids to flattened ids.
        """
        id_map = [{} for first_id in self.ids]
        uid = 0
        for first_id, nested_ids in enumerate(self.ids):
            for second_id in nested_ids:
                id_map[first_id][second_id] = uid
                uid += 1
        return id_map

    def partial_get(self, first_id: int) -> List[str]:
        """
        Get the second ids for a given first id.

        Parameters
        ----------
        first_id : int
            The first id to get the second ids for.

        Returns
        -------
        List[int]
            The second ids for the given first id.
        """
        return self.ids[first_id]

    @cached_property
    def num_ids(self) -> int:
        """
        The number of ids managed by the IdManager.

        Returns
        -------
        int
            The number of ids.
        """
        return sum(map(len, self.ids))

    @property
    def num_envs(self) -> int:
        """
        Number of top-level environments (length of ``ids``).

        Returns
        -------
        int
            Count of environment slots managed by this instance.
        """
        return len(self.ids)
