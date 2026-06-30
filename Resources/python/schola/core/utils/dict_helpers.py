# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Utility Functions and Classes for manipulating dictionaries.
"""

import itertools
from collections.abc import Callable, Iterator
from typing import TypeVar, cast


V = TypeVar("V")
Y = TypeVar("Y")
K = TypeVar("K")
Z = TypeVar("Z")

# Define a recursive type for a nested dictionary and nested dictionary iterator
NestedIterator = Iterator[tuple[K, V | "NestedIterator[K, V]"]]
NestedDict = dict[K, V | "NestedDict[K, V]"]
_IteratorState = Iterator[tuple[object, object]]


def _flatten(
    _iterator: NestedIterator[str, V], prefix: str = ""
) -> Iterator[tuple[str, V]]:
    for key, value in _iterator:
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Iterator):
            yield from _flatten(cast(NestedIterator[str, V], value), new_prefix)
        else:
            yield (new_prefix, value)


def _no_prefix_flatten(_iterator: NestedIterator[str, V]) -> Iterator[tuple[str, V]]:
    for key, value in _iterator:
        if isinstance(value, Iterator):
            yield from _no_prefix_flatten(cast(NestedIterator[str, V], value))
        else:
            yield (key, value)


def _kfilter(
    func: Callable[[K], bool], _iterator: NestedIterator[K, V], any_key: bool = True
) -> NestedIterator[K, V]:

    for key, value in _iterator:
        if isinstance(value, Iterator):
            if func(key) and any_key:  # some key was true which is good enough
                yield (key, value)
            elif func(key) or any_key:  # need to keep checking keys
                # evaluate here so that we don't output empty dictionaries
                output_ = [
                    (k, v)
                    for k, v in _kfilter(
                        func, cast(NestedIterator[K, V], value), any_key
                    )
                ]
                if len(output_) > 0:
                    yield (key, iter(output_))
        else:
            if func(key):
                yield (key, value)


def _map(
    func: Callable[[V], Y], _iterator: NestedIterator[K, V]
) -> NestedIterator[K, Y]:
    yield from (
        (
            key,
            (
                _map(func, cast(NestedIterator[K, V], value))
                if isinstance(value, Iterator)
                else func(value)
            ),
        )
        for key, value in _iterator
    )


def _kmap(
    func: Callable[[K], Y], _iterator: NestedIterator[K, V]
) -> NestedIterator[Y, V]:
    yield from (
        (
            func(key),
            (
                _kmap(func, cast(NestedIterator[K, V], value))
                if isinstance(value, Iterator)
                else value
            ),
        )
        for key, value in _iterator
    )


def _unflatten(
    _iterator: NestedIterator[str, V], flat_dict: dict[str, Z], prefix: str = ""
) -> NestedIterator[str, Z]:
    for key, value in _iterator:
        flat_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Iterator):
            yield (
                key,
                _unflatten(cast(NestedIterator[str, V], value), flat_dict, flat_prefix),
            )
        else:
            yield (key, flat_dict[flat_prefix])


def _leaves(_iterator: NestedIterator[K, V]) -> Iterator[tuple[K, V]]:
    for key, value in _iterator:
        if isinstance(value, Iterator):
            yield from _leaves(cast(NestedIterator[K, V], value))
        else:
            yield (key, value)


class DIterator(NestedIterator[K, V]):
    """
    Fluent wrapper for iterating and transforming nested dictionaries.

    Operations mutate the internal iterator chain in place and return ``self``
    (or a cast sibling) for chaining.

    Parameters
    ----------
    source_dict : dict
        Root nested mapping to traverse.
    """

    source_dict: NestedDict[K, V]
    _base_iterator: NestedIterator[K, V]
    _iterator: _IteratorState

    def __init__(self, source_dict: NestedDict[K, V]):
        self.source_dict = source_dict
        self._base_iterator = self._make_iterator(self.source_dict)
        self._iterator = self._base_iterator

    @staticmethod
    def _make_iterator(_dict: NestedDict[K, V]) -> NestedIterator[K, V]:
        for key, value in _dict.items():
            if isinstance(value, dict):
                yield (key, DIterator._make_iterator(cast(NestedDict[K, V], value)))
            else:
                yield (key, value)

    def map(self, func: Callable[[V], Y]) -> "DIterator[K,Y]":
        """
        Apply ``func`` to each leaf value.

        Parameters
        ----------
        func : callable
            Unary function applied to leaves.

        Returns
        -------
        DIterator
            ``self`` with leaf type updated to ``Y`` (fluent).
        """
        self._iterator = _map(func, cast(NestedIterator[K, V], self._iterator))
        return cast("DIterator[K,Y]", self)

    def kmap(self, func: Callable[[K], Y]) -> "DIterator[Y,V]":
        """
        Rename keys at every level using ``func``.

        Parameters
        ----------
        func : callable
            Unary function applied to each key ``K``.

        Returns
        -------
        DIterator
            ``self`` with key type updated to ``Y`` (fluent).
        """
        self._iterator = _kmap(func, cast(NestedIterator[K, V], self._iterator))
        return cast("DIterator[Y,V]", self)

    def kfilter(
        self, func: Callable[[K], bool], any_key: bool = True
    ) -> "DIterator[K,V]":
        """
        Keep subtrees whose keys satisfy ``func``.

        Parameters
        ----------
        func : callable
            Predicate on keys at the current level.
        any_key : bool, default=True
            When True, retain a branch if any descendant key matches; when False,
            use inclusive-or semantics for nested branches (see implementation).

        Returns
        -------
        DIterator
            ``self`` filtered in place (fluent).
        """
        self._iterator = _kfilter(
            func, cast(NestedIterator[K, V], self._iterator), any_key
        )
        return self

    def flatten(self, prefix: str = "") -> "DIterator[str,V]":
        """
        Collapse nested keys into underscore-separated leaf keys.

        Parameters
        ----------
        prefix : str, optional
            Prefix prepended to the first path segment (useful when chaining).

        Returns
        -------
        DIterator
            Iterator over ``(str, V)`` leaves with flattened string keys.
        """
        self._iterator = _flatten(cast(NestedIterator[str, V], self._iterator), prefix)
        return cast("DIterator[str,V]", self)

    def no_prefix_flatten(self) -> "DIterator[str,V]":
        """
        Flatten to leaf keys without accumulating parent prefixes.

        Returns
        -------
        DIterator
            Iterator over ``(str, V)`` leaves using only the immediate key names.
        """
        self._iterator = _no_prefix_flatten(
            cast(NestedIterator[str, V], self._iterator)
        )
        return cast("DIterator[str,V]", self)

    def unflatten(
        self, flat_dict: dict[str, Z], prefix: str = ""
    ) -> "DIterator[str,Z]":
        """
        Replace leaf values by looking them up in ``flat_dict`` using flattened paths.

        Parameters
        ----------
        flat_dict : dict[str, Z]
            Mapping from flattened key strings to replacement values.
        prefix : str, optional
            Prefix used when resolving keys (must match prior ``flatten`` usage).

        Returns
        -------
        DIterator
            ``self`` with leaf values of type ``Z`` (fluent).
        """
        self._iterator = _unflatten(
            cast(NestedIterator[str, V], self._iterator), flat_dict, prefix
        )
        return cast("DIterator[str,Z]", self)

    def chain(self, other: "DIterator[K,V]") -> "DIterator[K,V]":
        """
        Chain two DIterators together. Note this function is not robust to reused intermediate keys e.g. dict(a=dict(b=2)) and dict(a=dict(c=3)) will cause errors
        """
        self._iterator = cast(
            _IteratorState, itertools.chain(self._iterator, other._iterator)
        )
        return self

    @staticmethod
    def _to_dict(_iterator: _IteratorState, prune: bool = False) -> NestedDict[K, V]:
        out: NestedDict[K, V] = {}
        for key, value in _iterator:
            if isinstance(value, Iterator):
                processed_value: NestedDict[K, V] = DIterator._to_dict(
                    cast(_IteratorState, value), prune
                )
                if processed_value or not prune:
                    out[cast(K, key)] = processed_value
            else:
                out[cast(K, key)] = cast(V, value)
        return out

    def to_dict(self, prune: bool = True) -> NestedDict[K, V]:
        """
        Materialize the current iterator chain back into a nested dict.

        Parameters
        ----------
        prune : bool, default=True
            If True, drop empty child dicts produced while nesting.

        Returns
        -------
        dict
            Reconstructed nested mapping.
        """
        return self._to_dict(self._iterator, prune)

    def __iter__(self) -> NestedIterator[K, V]:
        return self.leaves()

    def __next__(self) -> tuple[K, V | "NestedIterator[K, V]"]:
        key, value = next(self._iterator)
        return cast(tuple[K, V | NestedIterator[K, V]], (key, value))

    def leaves(self) -> Iterator[tuple[K, V]]:
        """
        Yield ``(key, value)`` pairs for every leaf under the current iterator.

        Yields
        ------
        tuple
            Leaf key and value pairs.
        """
        yield from _leaves(cast(NestedIterator[K, V], self._iterator))

    def keys(self) -> Iterator[K]:
        """
        Yield leaf keys in depth-first order.

        Yields
        ------
        key
            Leaf key ``K``.
        """
        yield from (
            key for key, _ in _leaves(cast(NestedIterator[K, V], self._iterator))
        )

    def values(self) -> Iterator[V]:
        """
        Yield leaf values in depth-first order.

        Yields
        ------
        value
            Leaf value ``V``.
        """
        yield from (
            value for _, value in _leaves(cast(NestedIterator[K, V], self._iterator))
        )


def map_dict(func: Callable[[V], Y], input_dict: NestedDict[K, V]) -> NestedDict[K, Y]:
    """
    Map a function over every leaf value while preserving nesting.

    Parameters
    ----------
    func : callable
        Applied to each leaf value ``V``.
    input_dict : dict
        Possibly nested mapping.

    Returns
    -------
    dict
        New nested dict with the same keys and transformed leaves.

    See Also
    --------
    DIterator.map
    """
    return DIterator(input_dict).map(func).to_dict()


def kfilter_dict(
    func: Callable[[K], bool], input_dict: NestedDict[K, V], any_key: bool = True
) -> NestedDict[K, V]:
    """
    Filter a nested dict by key predicate.

    Parameters
    ----------
    func : callable
        Predicate on keys at each nesting level.
    input_dict : dict
        Possibly nested mapping.
    any_key : bool, default=True
        Forwarded to ``DIterator.kfilter``.

    Returns
    -------
    dict
        Key-filtered nested dict.

    See Also
    --------
    DIterator.kfilter
    """
    return DIterator(input_dict).kfilter(func, any_key).to_dict()


def flatten_dict(input_dict: NestedDict[K, V], prefix: str = "") -> dict[str, V]:
    """
    Flatten nested keys into a single string key per leaf.

    Parameters
    ----------
    input_dict : dict
        Possibly nested mapping.
    prefix : str, optional
        Prepended to each flattened key (underscore-separated path segments).

    Returns
    -------
    dict of str to V
        Mapping from underscore-joined key paths (with optional ``prefix``) to leaves.

    See Also
    --------
    flatten_dict_no_prefix
    """
    return cast(dict[str, V], DIterator(input_dict).flatten(prefix).to_dict())


def flatten_dict_no_prefix(input_dict: NestedDict[K, V]) -> dict[str, V]:
    """
    Flatten using only the leaf key name at each branch (no hierarchical prefix).

    Parameters
    ----------
    input_dict : dict
        Possibly nested mapping.

    Returns
    -------
    dict of str to V
        Single-level dict; collisions at the same nesting depth are the caller's
        responsibility to avoid.

    See Also
    --------
    flatten_dict
    """
    return cast(dict[str, V], DIterator(input_dict).no_prefix_flatten().to_dict())


def unflatten_dict(
    flat_dict: dict[str, Y], input_dict: NestedDict[str, V], prefix: str = ""
) -> NestedDict[str, Y]:
    """
    Inverse of ``flatten_dict`` for a given template shape.

    Parameters
    ----------
    flat_dict : dict of str to Y
        Flat source values.
    input_dict : dict
        Template nested structure whose key paths index into ``flat_dict``.
    prefix : str, optional
        Same prefix convention as ``flatten_dict``.

    Returns
    -------
    dict
        Nested dict with values taken from ``flat_dict``.
    """
    return DIterator(input_dict).unflatten(flat_dict, prefix).to_dict()


def flattened_key_iterator(
    input_dict: NestedDict[K, V], prefix: str = ""
) -> Iterator[str]:
    """
    Iterator over flattened string keys.

    Parameters
    ----------
    input_dict : dict
        Possibly nested mapping.
    prefix : str, optional
        Prefix passed to ``flatten``.

    Yields
    ------
    str
        Keys in depth-first order.
    """
    return DIterator(input_dict).flatten(prefix).keys()


def flattened_value_iterator(input_dict: NestedDict[K, V]) -> Iterator[V]:
    """
    Iterator over leaf values in depth-first order.

    Parameters
    ----------
    input_dict : dict
        Possibly nested mapping.

    Yields
    ------
    V
        Leaf values.
    """
    return DIterator(input_dict).values()
