# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for ``IdManager`` agent and environment ID bookkeeping."""

from schola.core.utils.id_manager import IdManager


def test_agent_types_are_normalized_to_known_ids():
    """Agent type metadata is kept alongside the IDs it describes."""
    id_manager = IdManager(
        [["AgentA", "AgentB"], ["AgentC"]],
        {
            0: {"AgentA": "TeamA"},
            1: {"AgentC": "TeamC", "UnknownAgent": "Ignored"},
            2: {"OtherEnvAgent": "Ignored"},
        },
    )

    assert id_manager.agent_types == {
        0: {"AgentA": "TeamA", "AgentB": ""},
        1: {"AgentC": "TeamC"},
    }
    assert id_manager.agent_types_for_env(0) == {
        "AgentA": "TeamA",
        "AgentB": "",
    }
    assert id_manager.get_agent_type(1, "AgentC") == "TeamC"
    assert id_manager.get_agent_type(1, "MissingAgent") == ""


def test_agent_types_accept_list_metadata_shape():
    """Protocol metadata can also arrive as a list indexed by environment ID."""
    id_manager = IdManager(
        [["AgentA"], ["AgentB"]],
        [{"AgentA": "TeamA"}, {"AgentB": "TeamB"}],
    )

    assert id_manager.agent_types == {
        0: {"AgentA": "TeamA"},
        1: {"AgentB": "TeamB"},
    }


def test_agent_type_accessors_return_copies():
    """Callers should not mutate IdManager's normalized metadata by accident."""
    id_manager = IdManager([["AgentA"]], {0: {"AgentA": "TeamA"}})

    agent_types = id_manager.agent_types
    agent_types[0]["AgentA"] = "Changed"
    env_agent_types = id_manager.agent_types_for_env(0)
    env_agent_types["AgentA"] = "Changed"

    assert id_manager.get_agent_type(0, "AgentA") == "TeamA"


def test_nest_list_to_dict_of_dicts_preserves_all_agents_per_env():
    """Each environment must retain every agent when nesting a flat list."""
    id_manager = IdManager([["AgentA", "AgentB"], ["AgentC"]])
    values = ["action_a", "action_b", "action_c"]

    nested = id_manager.nest_list_to_dict_of_dicts(values)

    assert nested == {
        0: {"AgentA": "action_a", "AgentB": "action_b"},
        1: {"AgentC": "action_c"},
    }


def test_nest_and_flatten_list_of_dicts_round_trip():
    """Nesting and flattening should be inverse operations for complete data."""
    id_manager = IdManager([["AgentA", "AgentB"], ["AgentC"]])
    values = ["action_a", "action_b", "action_c"]

    nested = id_manager.nest_list_to_dict_of_dicts(values)
    flattened = id_manager.flatten_list_of_dicts(list(nested.values()))

    assert flattened == values


def test_flatten_list_of_dicts_uses_canonical_agent_order():
    """Flattening must follow IdManager order, not input mapping key order."""
    id_manager = IdManager([["AgentA", "AgentB"], ["AgentC"]])
    # Simulate protobuf map deserialization where key order is arbitrary.
    nested = [
        {"AgentB": "action_b", "AgentA": "action_a"},
        {"AgentC": "action_c"},
    ]

    assert id_manager.flatten_list_of_dicts(nested) == [
        "action_a",
        "action_b",
        "action_c",
    ]
