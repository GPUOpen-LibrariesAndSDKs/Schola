# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for schola.core.utils.ubt module
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from schola.core.utils.ubt import (
    get_unreal_platform,
    UBTCommand,
    get_ue_version,
    get_project_file,
    get_engine_path_from_sln,
    get_sln_file_from_project,
    get_editor_executable_path,
    resolve_editor_executable,
)


class TestGetUnrealPlatform:
    """Tests for get_unreal_platform function"""

    @patch("platform.system")
    def test_windows_platform(self, mock_system: MagicMock) -> None:
        """Test that Windows returns Win64"""
        mock_system.return_value = "Windows"
        assert get_unreal_platform() == "Win64"

    @patch("platform.system")
    def test_linux_platform(self, mock_system: MagicMock) -> None:
        """Test that Linux returns Linux"""
        mock_system.return_value = "Linux"
        assert get_unreal_platform() == "Linux"

    @patch("platform.system")
    def test_unsupported_platform(self, mock_system: MagicMock) -> None:
        """Test that unsupported platforms raise ValueError"""
        mock_system.return_value = "Darwin"  # macOS
        with pytest.raises(ValueError, match="Unsupported platform"):
            _ = get_unreal_platform()


class TestUBTCommand:
    """Tests for UBTCommand dataclass"""

    def test_basic_initialization(self):
        """Test basic UBTCommand initialization"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt", project_file="/path/to/project.uproject"
        )

        assert cmd.ubt_path == "/path/to/ubt"
        assert cmd.project_file == "/path/to/project.uproject"

    def test_build_args_basic(self):
        """Test build_args with default settings"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt", project_file="/path/to/project.uproject"
        )

        args = cmd.build_args()

        assert "/path/to/ubt" in args
        assert "BuildCookRun" in args
        assert "-project=/path/to/project.uproject" in args

    def test_build_args_with_maps(self):
        """Test build_args with specific maps"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            maps=["Map1", "Map2"],
            all_maps=False,
        )
        assert not cmd.all_maps, "all_maps should be False when maps are specified"
        args = cmd.build_args()

        assert "-map=Map1+Map2" in args
        assert "-AllMaps" not in args

    def test_build_args_all_maps(self):
        """Test build_args with all maps"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            all_maps=True,
        )

        args = cmd.build_args()

        assert "-AllMaps" in args

    def test_build_args_with_staging_dir(self):
        """Test build_args with staging directory"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            staging_dir="/path/to/staging",
        )

        args = cmd.build_args()

        assert "-stage" in args
        assert "-stagingdirectory=/path/to/staging" in args

    def test_build_args_shipping_configuration(self):
        """Test build_args with Shipping configuration"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            configuration="Shipping",
        )

        args = cmd.build_args()

        assert "-configuration=Shipping" in args

    def test_build_args_disable_options(self):
        """Test build_args with various options disabled"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            should_package=False,
            should_clean=False,
            should_cook=False,
            should_build=False,
            no_p4=False,
            prereqs=False,
            no_compile=False,
            no_compile_uat=False,
            stdout=False,
            no_debug_info=False,
            unattended=False,
            force_monolithic=False,
        )

        args = cmd.build_args()

        assert "-package" not in args
        assert "-clean" not in args
        assert "-cook" not in args
        assert "-build" not in args
        assert "-NoP4" not in args
        assert "-prereqs" not in args
        assert "-nocompile" not in args
        assert "-nocompileuat" not in args
        assert "-stdout" not in args
        assert "-nodebuginfo" not in args
        assert "-unattended" not in args
        assert "-ForceMonolithic" not in args

    def test_build_args_fast_cook(self):
        """Test build_args with FastCook enabled"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            should_cook=True,
            fast_cook=True,
        )

        args = cmd.build_args()

        assert "-cook" in args
        assert "-FastCook" in args

    def test_build_args_no_fast_cook(self):
        """Test build_args with FastCook disabled"""
        cmd = UBTCommand(
            ubt_path="/path/to/ubt",
            project_file="/path/to/project.uproject",
            should_cook=True,
            fast_cook=False,
        )

        args = cmd.build_args()

        assert "-cook" in args
        assert "-FastCook" not in args


class TestGetUEVersion:
    """Tests for get_ue_version function"""

    def test_get_version_from_uproject(self):
        """Test extracting UE version from .uproject file"""
        project_data = {"EngineAssociation": "5.5"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".uproject", delete=False
        ) as f:
            json.dump(project_data, f)
            temp_path = f.name

        try:
            version = get_ue_version(Path(temp_path))
            assert version == "5.5"
        finally:
            os.remove(temp_path)

    def test_get_version_missing_association(self):
        """Test handling of missing EngineAssociation"""
        project_data = {"FileVersion": 3}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".uproject", delete=False
        ) as f:
            json.dump(project_data, f)
            temp_path = f.name

        try:
            version = get_ue_version(Path(temp_path))
            assert version is None
        finally:
            os.remove(temp_path)


class TestGetProjectFile:
    """Tests for get_project_file function"""

    def test_find_uproject_file(self):
        """Test finding .uproject file in directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .uproject file
            uproject_path = Path(tmpdir) / "TestProject.uproject"
            uproject_path.touch()

            found_file = get_project_file(Path(tmpdir))

            assert found_file is not None
            assert found_file.name == "TestProject.uproject"

    def test_no_uproject_file(self):
        """Test when no .uproject file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            found_file = get_project_file(Path(tmpdir))

            assert found_file is None


class TestGetEnginePathFromSln:
    """Tests for get_engine_path_from_sln function"""

    def test_extract_engine_path(self):
        """Test extracting engine path from .sln file"""
        sln_content = """Microsoft Visual Studio Solution File, Format Version 12.00
Project("{2150E333-8FDC-42A3-9474-1A3956D46DE8}") = "Engine", "Engine", "{12345}"
EndProject
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "UnrealBuildTool", "..\\..\\Engine\\Source\\Programs\\UnrealBuildTool\\UnrealBuildTool.csproj", "{67890}"""

        with tempfile.TemporaryDirectory() as tmpdir:
            sln_path = Path(tmpdir) / "Test.sln"
            with open(sln_path, "w") as f:
                _ = f.write(sln_content)

            engine_path = get_engine_path_from_sln(sln_path)

            # Should extract path up to Engine
            assert engine_path is not None


class TestGetSlnFileFromProject:
    """Tests for get_sln_file_from_project function"""

    def test_find_sln_file(self):
        """Test finding .sln file in directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sln_path = Path(tmpdir) / "TestProject.sln"
            sln_path.touch()

            found_file = get_sln_file_from_project(Path(tmpdir))

            assert found_file is not None
            assert found_file.name == "TestProject.sln"

    def test_no_sln_file(self):
        """Test when no .sln file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            found_file = get_sln_file_from_project(Path(tmpdir))

            assert found_file is None


class TestGetEditorExecutablePath:
    """Tests for get_editor_executable_path function"""

    @patch("platform.system")
    def test_windows_editor_path(self, mock_system: MagicMock) -> None:
        """Test editor path on Windows"""
        mock_system.return_value = "Windows"

        engine_path = Path("C:/Program Files/Epic Games/UE_5.5")
        editor_path = get_editor_executable_path(engine_path)

        assert "UnrealEditor-Cmd.exe" in str(editor_path)
        assert "Win64" in str(editor_path)

    @patch("platform.system")
    def test_linux_editor_path(self, mock_system: MagicMock) -> None:
        """Test editor path on Linux"""
        mock_system.return_value = "Linux"

        engine_path = Path("/opt/UnrealEngine")
        editor_path = get_editor_executable_path(engine_path)

        assert "UnrealEditor-Cmd" in str(editor_path)
        assert "Linux" in str(editor_path)
        assert not str(editor_path).endswith(".exe")


class TestResolveEditorExecutable:
    """Tests for resolve_editor_executable function"""

    def test_resolve_from_uproject(self):
        """Test resolving editor executable from a .uproject file"""
        sln_content = """Microsoft Visual Studio Solution File, Format Version 12.00
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "UnrealBuildTool", "..\\..\\Engine\\Source\\Programs\\UnrealBuildTool\\UnrealBuildTool.csproj", "{67890}"""

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            sln_path = project_dir / "TestProject.sln"
            uproject_path = project_dir / "TestProject.uproject"
            with open(sln_path, "w") as f:
                f.write(sln_content)
            uproject_path.touch()

            with patch("platform.system", return_value="Windows"):
                editor_path = resolve_editor_executable(uproject_path)

            assert "UnrealEditor-Cmd.exe" in str(editor_path)
            assert "Win64" in str(editor_path)

    def test_missing_sln_raises(self):
        """Test that missing .sln file raises FileNotFoundError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            uproject_path = Path(tmpdir) / "TestProject.uproject"
            uproject_path.touch()

            with pytest.raises(FileNotFoundError, match="Could not locate a .sln"):
                resolve_editor_executable(uproject_path)
