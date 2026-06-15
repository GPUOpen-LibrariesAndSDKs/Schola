# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""Unreal automation test batch types shared by pytest conftest.

.. note::
    Unreal's session report JSON can omit a test when its full path is a strict prefix of another
    registered test (for example ``Schola.Foo.ToString`` next to ``Schola.Foo.ToString.Empty``). Pytest
    then raises "Test not found in Unreal Engine results". Prefer sibling leaf names that are not
    prefixes of one another (e.g. ``...ToString.WithValues`` and ``...ToString.Empty``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import pathlib
import subprocess
from typing import Any, List, Literal, Optional
import pytest
import sys
import re

# Match IMPLEMENT_SIMPLE_AUTOMATION_TEST(...); `\s` spans newlines so split arguments are found.
# matches One Word Struct Name, Test Name in quotes, and then arbitrary flags and closing parenthesis
# , and ) need to be on the same line as the preceding content and no space between them
AUTOMATION_TEST_PATTERN = r'^IMPLEMENT_SIMPLE_AUTOMATION_TEST\(\s*[^,\s]+,\s*"([^"]+)",\s*[^,]+\)'

def iter_automation_test_paths_with_lines(source: str):
    """Yield (test_path, lineno) for each automation test macro in C++ source.

    lineno is 0-based (consistent with enumerate over readlines) and points to the
    line where the macro invocation starts.
    """
    pattern = re.compile(AUTOMATION_TEST_PATTERN, re.MULTILINE)
    prev_start = 0
    lineno = 0
    for match in pattern.finditer(source):
        lineno += source.count("\n", prev_start, match.start())
        yield match.group(1), lineno
        prev_start = match.start()

logger = logging.getLogger(__name__)

class UnrealTestItem(pytest.Item):

    def __init__(self, name:str, test_path:str, cpp_file:pathlib.Path, line_number:int=0, **kwargs):
        name = name.strip()
        super().__init__(name=name,**kwargs)
        self.unreal_result : Optional[UnrealTestResult] = None  # Will be populated by pytest_runtestloop
        self.cpp_file = cpp_file
        self.lineno = line_number 
        self.test_path = test_path
        try:
            self.add_marker(pytest.mark.xdist_group("unreal_automation_tests"))
        except:
            # wasn't able to mark, probably because pytest-xdist isn't installed which is OK
            pass

    @property
    def sanitized_test_path(self) -> str:
        return self.test_path.strip()

    def runtest(self):
        if self.unreal_result is None:
            pytest.skip(
                "Unreal Engine did not execute this test. "
                "Verify --engine-path, --test-hierarchy-path, and any test filters."
            )

        else:
            self._handle_test_outcome(self.unreal_result)

    # Handles unreal test results depending on success or failure of test execution
    def _handle_test_outcome(self, result: UnrealTestResult):
        if result.skipped:
            pytest.skip("Test not executed in this batch")
        elif result.passed:
            pass
        elif result.failed:
            # Format error messages from Unreal entries
            message_parts = [f"Test failed with {result.num_errors} errors and {result.num_warnings} warnings"]
            
            # Add error details
            for entry in result.errors:
                message_parts.append(f"\n  Error")
                if entry.filename is not None and entry.line_number is not None:
                    message_parts.append(f" at {entry.filename}:{entry.line_number}")
                message_parts.append(f":\n")
                message_parts.append(f"    {entry.message}")
            
            # Add warning details
            for entry in result.warnings:
                message_parts.append(f"\n  Warning")
                if entry.filename is not None and entry.line_number is not None:
                    message_parts.append(f" at {entry.filename}:{entry.line_number}")
                message_parts.append(f":\n")
                message_parts.append(f"    {entry.message}")
            
            message = "".join(message_parts)
            raise UnrealTestException(message)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, UnrealTestException):
            return excinfo.value.args[0]
        return super().repr_failure(excinfo, style=None)

    def reportinfo(self):
        if self.cpp_file:
            return self.cpp_file, self.lineno, f"usecase: {self.name}"
        return self.fspath, 0, f"usecase: {self.name}"

class UnrealTestFile(pytest.File):

    def collect(self):
        # Skip C++ test collection on Linux
        if sys.platform != "win32":
            return
            
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        test_details = list(iter_automation_test_paths_with_lines(source))

        for test_path, line_number in test_details:
            name = test_path.split(".")[-1] # Use the last part of the full path
            yield UnrealTestItem.from_parent(self, name=name, test_path=test_path, cpp_file=self.path, line_number=line_number)

# Use this to have custom exceptions
class UnrealTestException(Exception):
    pass


@dataclass
class UnrealTestEvent:
    message: str
    filename: Optional[pathlib.Path]
    line_number: Optional[int]
    timestamp: str
    context: Optional[str] = None  # Only known usage is for log entries which use "log"

    @classmethod
    def make(cls, event_entry: Any):
        filename = None if event_entry.get("filename", "") == "" else pathlib.Path(event_entry["filename"])
        line_number = None if event_entry.get("lineNumber", -1) == -1 else event_entry.get("lineNumber", None)
        context = None if event_entry["event"].get("context", "") == "" else event_entry["event"].get("context", "")
        return cls(
            message=event_entry["event"].get("message", ""),
            filename=filename,
            line_number=line_number,
            timestamp=event_entry.get("timestamp", datetime.now().isoformat()),
            context=context,
        )

    @classmethod
    def make_from_item(cls, item: Any, msg: str):
        return cls(
            message=msg,
            filename=item.cpp_file,
            line_number=item.lineno,
            timestamp=datetime.now().isoformat(),
        )


@dataclass
class UnrealTestResult:
    name: str
    test_path: str
    outcome: Literal["passed", "failed", "skipped"]
    duration: float
    errors: List[UnrealTestEvent]
    warnings: List[UnrealTestEvent]

    @property
    def num_warnings(self) -> int:
        return len(self.warnings)

    @property
    def num_errors(self) -> int:
        return len(self.errors)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"

    @property
    def skipped(self) -> bool:
        return self.outcome == "skipped"

    @property
    def sanitized_test_path(self) -> str:
        return self.test_path.strip()

    @classmethod
    def make(cls, test_results: Any):
        entries = [x for x in test_results.get("entries", []) if "event" in x]
        errors = filter(lambda x: x["event"].get("type", "").lower() == "error", entries)
        warnings = filter(lambda x: x["event"].get("type", "").lower() == "warning", entries)

        duration = float(test_results.get("duration", 0))

        outcome = test_results.get("state", "").strip().lower()
        if outcome == "success":
            outcome = "passed"
        elif outcome == "fail":
            outcome = "failed"
        else:
            outcome = "skipped"
        if not test_results["testDisplayName"] or not test_results["fullTestPath"]:
            raise Exception(
                f"Invalid test results: {test_results}. Test display name or full test path is missing."
            )

        return cls(
            name=test_results["testDisplayName"],
            test_path=test_results["fullTestPath"],
            outcome=outcome,
            duration=duration,
            errors=[UnrealTestEvent.make(entry) for entry in errors],
            warnings=[UnrealTestEvent.make(entry) for entry in warnings],
        )


@dataclass
class UnrealTestBatch:
    unreal_test_items: List[UnrealTestItem]
    report_dir: pathlib.Path
    batch_index: int
    command_line_args_folder: pathlib.Path
    _process: Optional[subprocess.Popen] = None
    retries: int = 0

    @property
    def process(self) -> subprocess.Popen:
        if self._process is None:
            raise Exception("Process is not initialized")
        return self._process

    @property
    def test_string(self) -> str:
        return "+".join((item.test_path for item in self.unreal_test_items))

    @property
    def report_path(self) -> pathlib.Path:
        return self.report_dir / f"batch_{self.batch_index}"

    @property
    def report_file(self) -> pathlib.Path:
        return self.report_path / "index.json"

    @property
    def command_line_args_file(self) -> pathlib.Path:
        return self.command_line_args_folder / f"batch_{self.batch_index}.txt"

    @property
    def command_line_arg_string(self) -> str:
        return " ".join(
            [
                "-unattended",
                "-nullrhi",
                "-NoTrace",
                f'-ReportExportPath="{self.report_path}"',
                f'-ExecCmds="Automation RunTest {self.test_string};Quit"',
            ]
        )

    def write_command_line_args_file(self) -> None:
        with open(self.command_line_args_file, "w") as f:
            f.write(self.command_line_arg_string)

    def prepare(self) -> None:
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.command_line_args_folder.mkdir(parents=True, exist_ok=True)
        self.write_command_line_args_file()

    def run(self, editor_path: pathlib.Path, uproject_path: pathlib.Path) -> subprocess.Popen:
        argv = [
            str(editor_path),
            f"-project={uproject_path}",
            "-ResumeRunTest",
            f"-CmdLineFile={self.command_line_args_file.resolve()}",
        ]
        print("Running Unreal Test Command: " + " ".join(argv))
        self._process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,  # type: ignore
            stderr=subprocess.PIPE,  # type: ignore
            text=True,
        )
        return self.process

    def load_test_results(self) -> None:
        test_results: dict = {}
        if self.report_file.resolve().exists():
            with open(self.report_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                for test in data.get("tests", []):
                    test_result = UnrealTestResult.make(test)
                    test_results[test_result.sanitized_test_path] = test_result
        else:
            logger.warning(
                "Expected Report file for batch %s but no file found. This can sometimes mean Unreal failed to build correctly.",
                self.batch_index,
            )
        for item in self.unreal_test_items:
            if item.sanitized_test_path in test_results:
                item.unreal_result = test_results[item.sanitized_test_path]
            else:
                item.unreal_result = UnrealTestResult(
                    name=item.name,
                    test_path=item.test_path,
                    outcome="failed",
                    duration=0.0,
                    errors=[
                        UnrealTestEvent(
                            message="Test not found in Unreal Engine results",
                            filename=item.cpp_file,
                            line_number=item.lineno,
                            timestamp="",
                        )
                    ],
                    warnings=[],
                )

    def mark_all_unreal_tests_failed(self, msg: str) -> None:
        for item in self.unreal_test_items:
            item.unreal_result = UnrealTestResult(
                name=item.name,
                test_path=item.test_path,
                outcome="failed",
                duration=0.0,
                errors=[UnrealTestEvent.make_from_item(item, msg)],
                warnings=[],
            )

    def clean_up(self) -> None:
        if self._process:
            self.process.kill()
            self.process.communicate()
