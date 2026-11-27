# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# pyright: reportAny=false, reportUnknownMemberType=false
"""
Tests for schola.core.utils.plugins module
"""

from unittest.mock import Mock, patch
from schola.core.utils.plugins import get_plugins


class TestGetPlugins:
    """Tests for get_plugins function"""

    def test_get_plugins_loads_entry_points_for_group(self):
        """Test get_plugins queries entry_points by group and loads each result"""
        mock_ep1 = Mock()
        mock_ep1.load.return_value = "plugin1"

        mock_ep2 = Mock()
        mock_ep2.load.return_value = "plugin2"

        with patch(
            "importlib.metadata.entry_points", return_value=[mock_ep1, mock_ep2]
        ) as mock_entry_points:
            result = get_plugins("test.group")

            mock_entry_points.assert_called_once_with(group="test.group")

            assert result == ["plugin1", "plugin2"]
            assert mock_ep1.load.called
            assert mock_ep2.load.called

    def test_get_plugins_empty_group(self):
        """Test get_plugins with non-existent group"""
        with patch("importlib.metadata.entry_points", return_value=[]):
            result = get_plugins("nonexistent.group")

            assert result == []

    def test_get_plugins_loads_multiple(self):
        """Test that get_plugins loads multiple plugins correctly"""
        mock_plugins = []
        for i in range(5):
            mock_ep = Mock()
            mock_ep.load.return_value = f"plugin{i}"
            mock_plugins.append(mock_ep)

        with patch("importlib.metadata.entry_points", return_value=mock_plugins):
            result = get_plugins("test.group")

            assert len(result) == 5
            assert result == [f"plugin{i}" for i in range(5)]

    def test_get_plugins_with_different_group_names(self):
        """Test get_plugins with various group names"""
        group_names = [
            "schola.envs",
            "schola.trainers",
            "my.custom.group",
            "test_group",
        ]

        for group_name in group_names:
            with patch(
                "importlib.metadata.entry_points", return_value=[]
            ) as mock_entry_points:
                _ = get_plugins(group_name)
                mock_entry_points.assert_called_with(group=group_name)

    def test_get_plugins_returns_loaded_objects(self):
        """Test that get_plugins returns the loaded plugin objects"""

        # Create mock plugin classes
        class Plugin1:
            pass

        class Plugin2:
            pass

        mock_ep1 = Mock()
        mock_ep1.load.return_value = Plugin1

        mock_ep2 = Mock()
        mock_ep2.load.return_value = Plugin2

        with patch(
            "importlib.metadata.entry_points", return_value=[mock_ep1, mock_ep2]
        ):
            result = get_plugins("test.group")

            assert Plugin1 in result
            assert Plugin2 in result
            assert len(result) == 2
