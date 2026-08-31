# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from itertools import islice
import json
import subprocess
from typing import Callable, Dict, List, Literal, Optional, Any, Sequence, Tuple, TypedDict
import sys
import time

import pytest
import logging
logger = logging.getLogger(__name__)
import pathlib
from functools import cache

from unreal_test_classes import UnrealTestBatch, UnrealTestEvent, UnrealTestResult, UnrealTestFile, UnrealTestItem
from schola.core.utils.ubt import (
    UBTCommand,
    get_project_file,
    get_editor_executable_path,
    get_ue_version,
    get_ubt_path,
    build_executable,
)
from opencppcoverage import (
    OPENCPPCOVERAGE_ENV,
    OpenCppCoverageRuntime,
    UnrealCoverageTestBatch,
    find_opencppcoverage_exe,
    merge_opencpp_batches,
    resolve_schola_plugin_root,
    set_measure_cpp_coverage_env,
)

# Constants
CPP_EXTENSIONS = {".cpp", ".cc"}

def batched(iterable, size):
    it = iter(iterable)
    batch = list(islice(it, size))
    while batch:
        yield batch
        batch = list(islice(it, size))


def run_unreal_tests(
    unreal_test_batches: Sequence[UnrealTestBatch],
    editor_path: pathlib.Path,
    uproject_path: pathlib.Path,
    global_timeout: int,
):
    """Run Unreal automation tests using a simple blocking subprocess call."""
    try:
        for test_batch in unreal_test_batches:
            test_batch.prepare()
            
        remaining_test_batches = [test_batch for test_batch in unreal_test_batches]
        start_time = time.time()

        # There is a weird edge case, where batch_0 times-out, and batch_n failes to retry a critical failure because we were waiting for batch_0.
        # We handle this by never waiting more than half the remaining timeout for a single batch 
        # This can still fail but would require several threads to timeout while also having critical failures.
        while remaining_test_batches:
            failed_test_batches = []
            
            # Check if we've already exceeded the global timeout, and there are no active processes, since we are between rounds of retries
            elapsed_time = time.time() - start_time
            if elapsed_time >= global_timeout:
                logger.warning(
                    "Global timeout of %s seconds reached. Killing all remaining processes.",
                    global_timeout,
                )
                for test_batch in remaining_test_batches:
                    test_batch.clean_up()
                break
            
            for test_batch in remaining_test_batches:
                test_batch.run(editor_path, uproject_path)

            for test_batch in remaining_test_batches:
                # Calculate remaining time for this batch
                elapsed_time = time.time() - start_time
                remaining_time = max(0, global_timeout - elapsed_time)
                
                if remaining_time <= 0:
                    logger.warning(
                        "Global timeout of %s seconds reached. Killing batch %s.",
                        global_timeout,
                        test_batch.batch_index,
                    )
                    test_batch.clean_up()
                    continue
                
                try:
                    stdout_str, stderr_str = test_batch.process.communicate(timeout=remaining_time/2)
                    # 0 is success, 255 is some test came back as failed but the overall test run was successful
                    if test_batch.process.returncode != 0 and test_batch.process.returncode != 255:
                        logger.warning(
                            "Unreal Automation Test batch %s failed with error %s. Retrying...",
                            test_batch.batch_index,
                            test_batch.process.returncode,
                        )
                        test_batch.retries += 1
                        failed_test_batches.append(test_batch)
                    else:
                        logger.info(
                            "Unreal Automation Test batch %s completed with no errors",
                            test_batch.batch_index,
                        )

                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Unreal Automation Test batch %s timed out (local timeout of %s seconds reached), killing process.",
                        test_batch.batch_index,
                        remaining_time / 2,
                    )
                    test_batch.clean_up()

            # retry all the failed test batches
            remaining_test_batches = failed_test_batches
    except:
        # cleanup all the test batches if something unexpected occurs
        for test_batch in unreal_test_batches:
            test_batch.clean_up()
        raise Exception("Unreal Automation Test failed with unknown error, killing all batches..")


def get_uproject_file(project_dir: pathlib.Path) -> Optional[pathlib.Path]:
    uproject_file = get_project_file(project_dir)
    return uproject_file.resolve() if uproject_file else None

def get_engine_path_from_config(config) -> pathlib.Path:
    engine_path = config.getoption("--engine-path") or config.getini("engine_path")
    return pathlib.Path(engine_path)

def get_build_unreal_from_config(config) -> bool:
    # --no-build-unreal takes precedence
    if config.getoption("--no-build-unreal"):
        return False
    build_option = config.getoption("--build-unreal")
    if build_option is not None:
        return build_option
    return config.getini("build_unreal")

def get_unreal_timeout_from_config(config) -> int:
    timeout_option = config.getoption("--unreal-timeout")
    if timeout_option is not None:
        return int(timeout_option)
    return int(config.getini("unreal_timeout"))

def get_null_rhi_from_config(config) -> bool:
    # --no-null-rhi takes precedence
    if config.getoption("--no-null-rhi"):
        return False
    null_rhi_option = config.getoption("--null-rhi")
    if null_rhi_option is not None:
        return null_rhi_option
    return config.getini("null_rhi")

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--engine-path",
            action="store",
            default=None,
            help="The absolute path to the Unreal Engine installation",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addini(
            "engine_path",
            help="The path to the Unreal Engine installation",
            default="C:/Program Files/Epic Games/UE_5.6",
        )
    except ValueError:
        pass  # Exists in .ini

    try:
        parser.addoption(
            "--build-unreal",
            action="store_true",
            default=None,
            help="Build Unreal project before running tests",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addoption(
            "--no-build-unreal",
            action="store_true",
            default=False,
            help="Skip building Unreal project before running tests",
        )
    except ValueError:
        pass  # Option already exists

    try:
        parser.addini(
            "build_unreal",
            type="bool",
            help="Build Unreal project before running tests",
            default=False,
        )
    except ValueError:
        pass  # Exists in .ini

    try:
        parser.addoption(
            "--schola-cpp-coverage-html",
            action="store_true",
            default=None,
            help="Generate HTML coverage report for Schola C++ tests.",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--schola-cpp-coverage-cobertura",
            action="store_true",
            default=None,
            help="Generate Cobertura (XML) coverage report for Schola C++ tests.",
        )
    except ValueError:
        pass

    try:
        parser.addini(
            "schola_cpp_coverage_html",
            type="bool",
            default=False,
            help="Generate HTML coverage report for Schola C++ tests.",
        )
    except ValueError:
        pass

    try:
        parser.addini(
            "schola_cpp_coverage_cobertura",
            type="bool",
            default=False,
            help="Generate Cobertura (XML) coverage report for Schola C++ tests.",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--unreal-timeout",
            action="store",
            default=None,
            type=int,
            metavar="SECONDS",
            help="Global timeout in seconds for all Unreal test batches (overrides unreal_timeout ini).",
        )
    except ValueError:
        pass

    try:
        parser.addini(
            "unreal_timeout",
            help="Global timeout in seconds for all Unreal test batches",
            default="240",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--null-rhi",
            action="store_true",
            default=None,
            help="Run Unreal automation tests with NullRHI (default unless null_rhi ini is false)",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--no-null-rhi",
            action="store_true",
            default=False,
            help="Run Unreal with a real RHI (required for NonNullRHI automation tests)",
        )
    except ValueError:
        pass

    try:
        parser.addini(
            "null_rhi",
            type="bool",
            help="Run Unreal automation tests with NullRHI",
            default=True,
        )
    except ValueError:
        pass

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "non_null_rhi: Unreal automation test that requires a real RHI (pass --no-null-rhi)",
    )

def pytest_collection_modifyitems(config, items):
    if not get_null_rhi_from_config(config):
        return
    skip_marker = pytest.mark.skip(reason="Requires real RHI (pass --no-null-rhi)")
    for item in items:
        if isinstance(item, UnrealTestItem) and "requires_rhi" in item.flags:
            item.add_marker(skip_marker)

@pytest.fixture(scope="session")
def unreal_path(request) -> pathlib.Path:
    return get_engine_path_from_config(request.config)

def is_cpp_coverage_html_enabled(config) -> bool:
    coverage_option = config.getoption("--schola-cpp-coverage-html")
    if coverage_option is not None:
        return coverage_option
    return config.getini("schola_cpp_coverage_html")

def is_cpp_coverage_cobertura_enabled(config) -> bool:
    coverage_option = config.getoption("--schola-cpp-coverage-cobertura")
    if coverage_option is not None:
        return coverage_option
    return config.getini("schola_cpp_coverage_cobertura")

def is_cpp_coverage_enabled(config) -> bool:
    return is_cpp_coverage_html_enabled(config) or is_cpp_coverage_cobertura_enabled(config)

@pytest.fixture(scope="session")
def unreal_report_dir(tmp_path_factory) -> pathlib.Path:
    """Create a session-scoped temporary directory for Unreal test reports."""
    report_dir = tmp_path_factory.mktemp("unreal_test_reports")
    return report_dir

@pytest.fixture(scope="session")
def unreal_command_line_file_dir(tmp_path_factory) -> pathlib.Path:
    """Create a session-scoped temporary directory for the Unreal Command Line Args written to a file."""
    command_line_file_dir = tmp_path_factory.mktemp("unreal_command_line_args")
    return command_line_file_dir


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Set up session-level test report directory."""
    # Use pytest's temp path factory to create a session temp directory
    tmp_path_factory = session.config._tmp_path_factory
    report_dir = tmp_path_factory.mktemp("unreal_test_reports")
    session.config._unreal_report_dir = report_dir



def is_cpp_test_file(file_path: pathlib.Path) -> bool:
    if file_path.suffix.lower() not in CPP_EXTENSIONS:
        return False
    return file_path.name.endswith("Test" + file_path.suffix)

def pytest_collect_file(parent, file_path : pathlib.Path):
    # Skip C++ test collection on Linux
    if sys.platform != "win32":
        return None
        
    # Only custom-collect C++ automation tests; let pytest's normal Python collectors
    # handle .py tests in `Test/` and doctests in `Resources/python/schola`.
    if file_path.suffix.lower() in CPP_EXTENSIONS:
        return UnrealTestFile.from_parent(parent=parent, path=file_path)
    return None

def pytest_ignore_collect(collection_path: pathlib.Path, config):
    # Skip C++ test files on Linux
    if sys.platform != "win32" and is_cpp_test_file(collection_path):
        return True  # Ignore C++ test files on Linux
        
    if is_cpp_test_file(collection_path):
        return False # Forcibly do not ignore C++ test files
    return None # Punt on everything else

class UnrealTestRunner:

    def __init__(self, session, unreal_items: List[UnrealTestItem], report_dir: pathlib.Path, command_line_file_dir: pathlib.Path):
        self.session = session
        self.unreal_items = unreal_items
        self.unreal_path = get_engine_path_from_config(session.config)
        self.should_build = get_build_unreal_from_config(session.config)
        self.generate_cpp_coverage_report = is_cpp_coverage_enabled(session.config)
        self.use_null_rhi = get_null_rhi_from_config(session.config)
        self.report_dir = report_dir
        self.command_line_file_dir = command_line_file_dir

    def run_tests(self) -> bool:
        # Setup the report directory
        
        editor_path = get_editor_executable_path(self.unreal_path)
        max_tests_per_batch = 100
        batches_list = list(batched(self.unreal_items, max_tests_per_batch))
        # handle if we are in a few common subdirectories of the project/using common pytest.ini file locations
        uproject_path = (get_uproject_file(self.session.config.rootpath)
        or get_uproject_file(self.session.config.rootpath.parent.parent)
        or get_uproject_file(self.session.config.inipath.parent)
        or get_uproject_file(self.session.config.inipath.parent.parent.parent))
        
        if not uproject_path:
            logger.error(
                "Error Building Unreal Project: No .uproject file found in directory"
            )
            unreal_test_batches = [
                UnrealTestBatch(
                    batch, self.report_dir, i, self.command_line_file_dir, use_null_rhi=self.use_null_rhi
                )
                for i, batch in enumerate(batches_list)
            ]
            for test_batch in unreal_test_batches:
                test_batch.mark_all_unreal_tests_failed("Unreal Engine failed to build and run tests")
            return False

        # Try and generate the OpenCppCoverage runtime if we are generating a coverage report
        opencpp: Optional[OpenCppCoverageRuntime] = None
        if self.generate_cpp_coverage_report:
            occ_exe = find_opencppcoverage_exe()
            if not occ_exe:
                logger.warning(
                    "Schola C++ coverage is enabled but OpenCppCoverage was not found on PATH "
                    "(set %s to its executable). Skipping C++ code coverage; Unreal tests will run without OpenCppCoverage.",
                    OPENCPPCOVERAGE_ENV,
                )
            else:
                plugin_root = resolve_schola_plugin_root(uproject_path, self.unreal_items[0].cpp_file)
                if not plugin_root:
                    logger.warning(
                        "Could not locate Schola.uplugin for this project; skipping OpenCppCoverage.",
                    )
                else:
                    opencpp = OpenCppCoverageRuntime(
                        occ_exe=occ_exe,
                        plugin_root=plugin_root,
                        engine_path=self.unreal_path,
                    )
        # if opencpp is None, we weren't able to load the OpenCppCoverage runtime, 
        # so we will just run the tests without coverage

        if opencpp is not None:
            unreal_test_batches = [
                UnrealCoverageTestBatch(
                    batch,
                    self.report_dir,
                    i,
                    self.command_line_file_dir,
                    use_null_rhi=self.use_null_rhi,
                    opencpp=opencpp,
                )
                for i, batch in enumerate(batches_list)
            ]
        else:
            unreal_test_batches = [
                UnrealTestBatch(
                    batch, self.report_dir, i, self.command_line_file_dir, use_null_rhi=self.use_null_rhi
                )
                for i, batch in enumerate(batches_list)
            ]

        if self.should_build:
            set_measure_cpp_coverage_env(self.generate_cpp_coverage_report)
            try:
                self._build_unreal_project(uproject_path)
                set_measure_cpp_coverage_env(False)
            except Exception as e:
                set_measure_cpp_coverage_env(False)
                logger.error("Error building Unreal project: %s", e)
                for test_batch in unreal_test_batches: 
                    test_batch.mark_all_unreal_tests_failed("Unreal Engine failed to build and run tests")
                return False
        
        try:
            run_unreal_tests(
                editor_path=editor_path,
                uproject_path=uproject_path,
                unreal_test_batches=unreal_test_batches,
                global_timeout=get_unreal_timeout_from_config(self.session.config),
            )
        except Exception as e:
            logger.error("Error running Unreal tests: %s", e)
            for test_batch in unreal_test_batches: 
                test_batch.mark_all_unreal_tests_failed("Error Running Tests")
            return False

        if opencpp is not None:

            merge_opencpp_batches(
                opencpp,
                unreal_test_batches, # type: ignore
                self.report_dir,
                export_html=is_cpp_coverage_html_enabled(self.session.config),
                export_cobertura=is_cpp_coverage_cobertura_enabled(self.session.config),
            )

        for test_batch in unreal_test_batches:
            test_batch.load_test_results()

        return True

    def _build_unreal_project(self, uproject_path: pathlib.Path):
        """Build the Unreal project before running tests."""
        project_folder = uproject_path.parent
        
        # Get UE version and UBT path
        ue_version = get_ue_version(uproject_path)
        if not ue_version:
            raise Exception("Could not determine Unreal Engine version from .uproject file")
        
        ubt_path = get_ubt_path(project_folder, ue_version)

        # Try to use the engine path if we couldn't find UBT via the project files
        if not ubt_path and self.unreal_path:
            import platform
            script_name = "RunUAT.bat" if platform.system() == "Windows" else "RunUAT.sh"
            possible_path = self.unreal_path / "Engine" / "Build" / "BatchFiles" / script_name
            if possible_path.exists():
                ubt_path = possible_path

        if not ubt_path:
            raise Exception("Could not find Unreal Build Tool (UBT) path")
        
        # Setup build directory
        #build_dir = project_folder / "Build" / "Staged"
        #build_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Building Unreal project: %s", uproject_path)
        logger.info("Using UBT at: %s", ubt_path)
        
        command = UBTCommand(
            ubt_path=ubt_path,
            project_file=uproject_path,
            should_package=False,
            should_cook=False, # No need to cook content since tests should not have visibility on content
            force_monolithic=False, # Tests run from editor build so no need for monolithic build
            all_maps=False, # No need to cook any maps since tests should not have visibility on content
            should_clean=False, # No need to clean since we are not building a new project
        )
        
        result = command.run()

        logger.info("=" * 10 + " Unreal Build Output " + "=" * 10)
        logger.info(result.stdout)
        logger.info("=" * 10 + " End of Unreal Build Output " + "=" * 10)
        if result.returncode != 0:
            if result.stderr:
                logger.error("=" * 10 + " Unreal Build Error " + "=" * 10)
                logger.error(result.stderr)
                logger.error("=" * 10 + " Unreal Build Error " + "=" * 10)
            raise Exception(f"Unreal build failed with return code {result.returncode}")
             
@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if session.testsfailed and not session.config.option.continue_on_collection_errors:
        raise session.Interrupted(
            f"{session.testsfailed} error{'s' if session.testsfailed != 1 else ''} during collection"
        )

    if session.config.option.collectonly:
        return True

    unreal_items = [
        item
        for item in session.items
        if isinstance(item, UnrealTestItem) and not item.get_closest_marker("skip")
    ]

    # Run Unreal tests in batch if we have any (only on Windows, since linux can hang)
    if unreal_items and sys.platform == "win32":
        # Ensure report directory exists (create if pytest_sessionstart wasn't called)
        if not hasattr(session.config, '_unreal_report_dir'):
            tmp_path_factory = session.config._tmp_path_factory
            session.config._unreal_report_dir = tmp_path_factory.mktemp("unreal_test_reports")
        if not hasattr(session.config, '_unreal_command_line_file_dir'):
            tmp_path_factory = session.config._tmp_path_factory
            session.config._unreal_command_line_file_dir = tmp_path_factory.mktemp("unreal_command_line_args")
        report_dir = session.config._unreal_report_dir
        command_line_file_dir = session.config._unreal_command_line_file_dir

        runner = UnrealTestRunner(session, unreal_items, report_dir, command_line_file_dir)
        runner.run_tests()
    
    # Run all tests (including the now-populated Unreal tests)
    _run_all_tests(session)

    return True


def _run_all_tests(session):
    for i, item in enumerate(session.items):
        nextitem = session.items[i + 1] if i + 1 < len(session.items) else None
        item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)

        if session.shouldfail:
            raise session.Failed(session.shouldfail)
        if session.shouldstop:
            raise session.Interrupted(session.shouldstop)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    set_measure_cpp_coverage_env(False)
