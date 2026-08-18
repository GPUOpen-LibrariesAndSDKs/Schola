# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Unit tests for ``command_template`` using lightweight fake script/algorithm dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Dict, Type, Union
from unittest.mock import MagicMock

import pytest
import yaml
from cyclopts import App, Parameter, validators

from schola.scripts.common.settings import (
    AllSimulatorConfigs,
    BaseSimulatorConfig,
    EnvironmentSettings,
    ExternalSimulatorConfig,
    GrpcProtocolConfig,
    IgnoreParameter,
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
)
from schola.scripts.common.command_template import AlgorithmSpec, ScholaCommandTemplate

# --- Fake script / algorithm types (minimal stand-ins for SB3/RLlib settings) ---


@dataclass
class FakeAlgoAlpha:
    """First fake algorithm; a few CLI-overridable fields."""

    alpha: Annotated[float, Parameter(validator=validators.Number(gt=0.0, lte=1.0))] = (
        0.5
    )
    "Fake learning-rate-like scalar."

    extra: int = 7


@dataclass
class FakeAlgoBeta:
    """Second fake algorithm for multi-algorithm routing tests."""

    beta_steps: Annotated[int, Parameter(validator=validators.Number(gte=1))] = 11

@dataclass
class FakeEnvironmentSettings(EnvironmentSettings[AllSimulatorConfigs]):
    """Fake environment settings for testing."""
    simulator_settings: Annotated[AllSimulatorConfigs, IgnoreParameter] = field(default_factory=ExternalSimulatorConfig)

@dataclass
class FakeScriptSettings:
    """Minimal script container compatible with ``ScholaCommandTemplate`` wiring."""

    environment_settings: FakeEnvironmentSettings = field(
        default_factory=FakeEnvironmentSettings
    )

    algorithm_settings: Annotated[
        Union[FakeAlgoAlpha, FakeAlgoBeta], Parameter(show=False, parse=False)
    ] = field(default_factory=FakeAlgoAlpha)

    base_level_parameter: int = 1

FULL_ALGORITHM_TABLE: Dict[str, Type[Any]] = {
    "alpha": FakeAlgoAlpha,
    "beta": FakeAlgoBeta,
}
FULL_SIMULATOR_TABLE: Dict[str, Type[Any]] = {
    "executable": UnrealExecutableSimulatorConfig,
    "project": UnrealProjectSimulatorConfig,
    "external": ExternalSimulatorConfig,
}
FULL_SIMULATOR_KEYS: tuple[str, ...] = tuple(FULL_SIMULATOR_TABLE.keys())


def _make_meta_alg_command_class(
    algorithm_keys: tuple[str, ...],
    simulator_keys: tuple[str, ...],
    mock_main: MagicMock,
) -> Type[ScholaCommandTemplate[FakeScriptSettings]]:
    """Build a ``ScholaCommandTemplate`` subclass with a chosen number of algorithms / simulators."""

    alg_table = {k: FULL_ALGORITHM_TABLE[k] for k in algorithm_keys}
    sim_table = {k: FULL_SIMULATOR_TABLE[k] for k in simulator_keys}

    class _DynamicScholaCommandTemplate(ScholaCommandTemplate[FakeScriptSettings]):
        @property
        def algorithm_table(self) -> Dict[str, Type[Any]]:
            return alg_table

        @property
        def algorithm_help(self) -> Dict[str, str]:
            return {k: f"Test help for {k}." for k in algorithm_keys}

        @property
        def simulator_table(self) -> Dict[str, Type[Any]]:
            return sim_table

        @property
        def script_args_type(self) -> Type[FakeScriptSettings]:
            return FakeScriptSettings

        @property
        def main_func(self):
            return mock_main

    return _DynamicScholaCommandTemplate


def _build_meta_alg_app(
    mock_main: MagicMock,
    algorithm_keys: tuple[str, ...],
    simulator_keys: tuple[str, ...],
    *,
    app_name: str = "train-param",
) -> Any:
    app = App(name=app_name, help="Parameterized ScholaCommandTemplate tests")
    logger = logging.getLogger(f"test_command_template.{app_name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    cls = _make_meta_alg_command_class(algorithm_keys, simulator_keys, mock_main)
    return cls(app, logger).make().meta


@pytest.fixture
def mock_main() -> MagicMock:
    return MagicMock()


@pytest.fixture
def meta_app(mock_main: MagicMock):
    """Return ``app.meta.meta`` — the parse entry that runs ``train_command_config_handler`` first."""
    app = App(name="train-fake", help="Fake train CLI for template tests")
    logger = logging.getLogger("test_command_template")
    logger.addHandler(logging.NullHandler())
    cls = _make_meta_alg_command_class(
        ("alpha", "beta"),
        FULL_SIMULATOR_KEYS,
        mock_main,
    )
    built = cls(app, logger).make()
    # ``make()`` returns ``app.meta``; config YAML handling lives on ``app.meta.meta.default``.
    return built.meta


@pytest.fixture
def no_alg_meta_app(mock_main: MagicMock):
    """``ScholaCommandTemplate`` with no algorithms: entry is ``app.meta.meta`` (config handler outermost)."""
    app = App(name="train-no-alg-fake", help="Fake no-algorithm train CLI")
    logger = logging.getLogger("test_command_template_no_alg")
    logger.addHandler(logging.NullHandler())
    cls = _make_meta_alg_command_class(
        (),
        ("executable", "external"),
        mock_main,
    )
    built = cls(app, logger).make()
    return built.meta

def test_yaml_split_meta_no_alg_config_handler():
    """Like ``make_train_config_handler``: only ``environment.simulator`` is removed; ``algorithm`` stays."""
    raw = """
algorithm:
  reserved: 1
environment:
  simulator:
    external: {}
"""
    config_dict = yaml.safe_load(raw) or {}
    sim_config_dict = (config_dict.get("environment") or {}).pop("simulator", {})
    assert sim_config_dict == {"external": {}}
    assert config_dict.get("algorithm") == {"reserved": 1}
    assert "simulator" not in (config_dict.get("environment") or {})


def test_no_alg_cli_default_external_simulator(no_alg_meta_app, mock_main: MagicMock):
    """Default route invokes external simulator and ``main`` with script args."""
    no_alg_meta_app([], result_action="return_value", exit_on_error=False)

    mock_main.assert_called_once()
    args = mock_main.call_args[0][0]
    assert isinstance(args, FakeScriptSettings)
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_no_alg_cli_explicit_external(no_alg_meta_app, mock_main: MagicMock):
    no_alg_meta_app(["external"], result_action="return_value", exit_on_error=False)

    mock_main.assert_called_once()
    assert isinstance(
        mock_main.call_args[0][0].environment_settings.simulator_settings,
        ExternalSimulatorConfig,
    )


def test_no_alg_cli_executable(no_alg_meta_app, mock_main: MagicMock, tmp_path: Path):
    exe = tmp_path / "FakeNoAlg.exe"
    exe.write_bytes(b"")
    no_alg_meta_app(
        ["executable", "--executable-path", str(exe)],
        result_action="return_value",
        exit_on_error=False,
    )
    sim = mock_main.call_args[0][0].environment_settings.simulator_settings
    assert isinstance(sim, UnrealExecutableSimulatorConfig)
    assert sim.executable_path == exe


def test_no_alg_config_file_yaml(no_alg_meta_app, mock_main: MagicMock, tmp_path: Path):
    """``--config-file`` loads YAML; algorithm key stays in flat config (no pop)."""
    cfg = tmp_path / "no_alg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "environment": {
                    "simulator": {
                        "external": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    no_alg_meta_app(
        ["--config-file", str(cfg), "external"],
        result_action="return_value",
        exit_on_error=False,
    )

    mock_main.assert_called_once()
    assert isinstance(
        mock_main.call_args[0][0].environment_settings.simulator_settings,
        ExternalSimulatorConfig,
    )


def test_cli_default_algorithm_selects_alpha_and_external_simulator(
    meta_app, mock_main: MagicMock
):
    """``<algorithm>`` alone should default to the external simulator and invoke ``main`` once."""
    meta_app(["alpha"], result_action="return_value", exit_on_error=False)

    mock_main.assert_called_once()
    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert isinstance(args, FakeScriptSettings)
    assert isinstance(args.algorithm_settings, FakeAlgoAlpha)
    assert args.algorithm_settings.alpha == 0.5
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_cli_overrides_algorithm_fields(meta_app, mock_main: MagicMock):
    """CLI tokens should override fake algorithm defaults."""
    meta_app(
        ["alpha", "--alpha", "0.25", "--extra", "99"],
        result_action="return_value",
        exit_on_error=False,
    )

    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, FakeAlgoAlpha)
    assert args.algorithm_settings.alpha == 0.25
    assert args.algorithm_settings.extra == 99


def test_cli_second_algorithm_route(meta_app, mock_main: MagicMock):
    meta_app(
        ["beta", "--beta-steps", "42"],
        result_action="return_value",
        exit_on_error=False,
    )

    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, FakeAlgoBeta)
    assert args.algorithm_settings.beta_steps == 42


def test_cli_executable_simulator(meta_app, mock_main: MagicMock, tmp_path: Path):
    """Non-default simulator subcommand should attach ``UnrealExecutableSimulatorConfig``."""
    exe = tmp_path / "FakeGame.exe"
    exe.write_bytes(b"")

    meta_app(
        [
            "alpha",
            "executable",
            "--executable-path",
            str(exe),
        ],
        result_action="return_value",
        exit_on_error=False,
    )

    args: FakeScriptSettings = mock_main.call_args[0][0]
    sim = args.environment_settings.simulator_settings
    assert isinstance(sim, UnrealExecutableSimulatorConfig)
    assert sim.executable_path == exe


def test_config_file_yaml_merges_algorithm_environment_and_simulator(
    meta_app, mock_main: MagicMock, tmp_path: Path
):
    """YAML ``algorithm`` / ``environment.simulator`` split; algorithm block merges into CLI."""
    # Omit nested ``environment.protocol_settings`` here: ``flatten_dict_no_prefix`` yields
    # leaf keys like ``port`` that do not match cyclopts option names for ``FakeScriptSettings``
    # (that path is covered by integration tests with full script settings types).
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "algorithm": {
                    "alpha": {
                        "alpha": 0.11,
                        "extra": 3,
                    }
                },
                "environment": {
                    "simulator": {
                        "external": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    meta_app(
        [
            "--config-file",
            str(cfg),
            "alpha",
            "external",
        ],
        result_action="return_value",
        exit_on_error=False,
    )

    mock_main.assert_called_once()
    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, FakeAlgoAlpha)
    assert args.algorithm_settings.alpha == 0.11
    assert args.algorithm_settings.extra == 3
    assert isinstance(
        args.environment_settings.simulator_settings, ExternalSimulatorConfig
    )


def test_config_file_without_optional_simulator_key(
    meta_app, mock_main: MagicMock, tmp_path: Path
):
    """Missing ``environment.simulator`` still runs; inner command returns ``None`` (``main`` stub)."""
    cfg = tmp_path / "minimal.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "algorithm": {"alpha": {}},
                "environment": {},
            }
        ),
        encoding="utf-8",
    )

    meta_app(
        ["--config-file", str(cfg), "alpha"],
        result_action="return_value",
        exit_on_error=False,
    )

    mock_main.assert_called_once()
    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, FakeAlgoAlpha)


_EXE_PLACEHOLDER = "__EXE_PATH__"


@dataclass(frozen=True, slots=True)
class MetaAlgCliTestParameters:
    """Row for ``test_meta_alg_cli_algorithm_simulator_count_matrix``."""

    case_id: str
    algorithm_keys: tuple[str, ...]
    simulator_keys: tuple[str, ...]
    cli_tokens: list[str]
    expected_sim_settings: FakeScriptSettings


@dataclass(frozen=True, slots=True)
class MetaAlgConfigTestParameters:
    """Row for ``test_meta_alg_config_algorithm_simulator_count_matrix``."""

    case_id: str
    algorithm_keys: tuple[str, ...]
    simulator_keys: tuple[str, ...]
    config_doc: Dict[str, Any]
    cli_tokens: list[str]
    expected_sim_settings: FakeScriptSettings


_META_ALG_CLI_ALGORITHM_SIMULATOR_COUNT_MATRIX: tuple[MetaAlgCliTestParameters, ...] = (
    MetaAlgCliTestParameters(
        case_id="0_alg_0_sim",
        algorithm_keys=(),
        simulator_keys=(),
        cli_tokens=["--base-level-parameter", "2"],
        expected_sim_settings=FakeScriptSettings(base_level_parameter=2),
    ),
    MetaAlgCliTestParameters(
        case_id="0_alg_1_sim",
        algorithm_keys=(),
        simulator_keys=("external",),
        cli_tokens=[],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="0_alg_multi_sim_default_external",
        algorithm_keys=(),
        simulator_keys=FULL_SIMULATOR_KEYS,
        cli_tokens=[],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="1_alg_0_sim",
        algorithm_keys=("alpha",),
        simulator_keys=(),
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="1_alg_1_sim",
        algorithm_keys=("alpha",),
        simulator_keys=("external",),
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="1_alg_multi_sim_default_external",
        algorithm_keys=("alpha",),
        simulator_keys=FULL_SIMULATOR_KEYS,
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="multi_alg_0_sim",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=(),
        cli_tokens=["beta", "--beta-steps", "33"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoBeta(beta_steps=33),
        ),
    ),
    MetaAlgCliTestParameters(
        case_id="multi_alg_1_sim",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=("external",),
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgCliTestParameters(
        case_id="multi_alg_multi_sim_executable",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=FULL_SIMULATOR_KEYS,
        cli_tokens=[
            "alpha",
            "executable",
            "--executable-path",
            _EXE_PLACEHOLDER,
        ],
        expected_sim_settings=FakeScriptSettings(
            environment_settings=FakeEnvironmentSettings(
                simulator_settings=UnrealExecutableSimulatorConfig(
                    executable_path=Path(_EXE_PLACEHOLDER),
                ),
            ),
            algorithm_settings=FakeAlgoAlpha(),
        ),
    ),
)


_META_ALG_CONFIG_ALGORITHM_SIMULATOR_COUNT_MATRIX: tuple[
    MetaAlgConfigTestParameters, ...
] = (
    MetaAlgConfigTestParameters(
        case_id="cfg_0_alg_0_sim",
        algorithm_keys=(),
        simulator_keys=(),
        config_doc={},
        cli_tokens=[],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_0_alg_1_sim",
        algorithm_keys=(),
        simulator_keys=("external",),
        config_doc={"environment": {"simulator": {"external": {}}}},
        cli_tokens=[],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_0_alg_multi_sim",
        algorithm_keys=(),
        simulator_keys=FULL_SIMULATOR_KEYS,
        config_doc={"environment": {"simulator": {"external": {}}}},
        cli_tokens=["external"],
        expected_sim_settings=FakeScriptSettings(),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_1_alg_0_sim",
        algorithm_keys=("alpha",),
        simulator_keys=(),
        config_doc={"algorithm": {"alpha": {"alpha": 0.61, "extra": 4}}},
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoAlpha(alpha=0.61, extra=4),
        ),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_1_alg_1_sim",
        algorithm_keys=("alpha",),
        simulator_keys=("external",),
        config_doc={
            "algorithm": {"alpha": {"alpha": 0.62, "extra": 5}},
            "environment": {"simulator": {"external": {}}},
        },
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoAlpha(alpha=0.62, extra=5),
        ),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_1_alg_multi_sim",
        algorithm_keys=("alpha",),
        simulator_keys=FULL_SIMULATOR_KEYS,
        config_doc={
            "algorithm": {"alpha": {"alpha": 0.63, "extra": 6}},
            "environment": {"simulator": {"external": {}}},
        },
        cli_tokens=["alpha", "external"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoAlpha(alpha=0.63, extra=6),
        ),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_multi_alg_0_sim",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=(),
        config_doc={"algorithm": {"alpha": {"alpha": 0.64, "extra": 8}}},
        cli_tokens=["alpha"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoAlpha(alpha=0.64, extra=8),
        ),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_multi_alg_1_sim",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=("external",),
        config_doc={
            "algorithm": {"beta": {"beta_steps": 55}},
        },
        cli_tokens=["beta"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoBeta(beta_steps=55),
        ),
    ),
    MetaAlgConfigTestParameters(
        case_id="cfg_multi_alg_multi_sim",
        algorithm_keys=("alpha", "beta"),
        simulator_keys=FULL_SIMULATOR_KEYS,
        config_doc={
            "algorithm": {"alpha": {"alpha": 0.66, "extra": 9}},
            "environment": {"simulator": {"external": {}}},
        },
        cli_tokens=["alpha", "external"],
        expected_sim_settings=FakeScriptSettings(
            algorithm_settings=FakeAlgoAlpha(alpha=0.66, extra=9),
        ),
    ),
)


@pytest.mark.parametrize(
    "case",
    _META_ALG_CLI_ALGORITHM_SIMULATOR_COUNT_MATRIX,
    ids=lambda c: c.case_id,
)
def test_meta_alg_cli_algorithm_simulator_count_matrix(
    mock_main: MagicMock,
    tmp_path: Path,
    case: MetaAlgCliTestParameters,
) -> None:
    """Every count of algorithms (0 / 1 / 2+) × simulators (0 / 1 / 3) routes to ``main`` as expected."""
    exe = tmp_path / "FakeGame.exe"
    exe.write_bytes(b"")
    resolved_cli = [
        str(exe) if t == _EXE_PLACEHOLDER else t for t in case.cli_tokens
    ]
    meta = _build_meta_alg_app(
        mock_main,
        case.algorithm_keys,
        case.simulator_keys,
        app_name=f"cli-{case.algorithm_keys}-{case.simulator_keys}",
    )
    meta(resolved_cli, result_action="return_value", exit_on_error=False)

    mock_main.assert_called_once()
    args: FakeScriptSettings = mock_main.call_args[0][0]
    expected = case.expected_sim_settings
    sim = expected.environment_settings.simulator_settings
    if isinstance(sim, UnrealExecutableSimulatorConfig) and sim.executable_path == Path(
        _EXE_PLACEHOLDER
    ):
        expected = replace(
            expected,
            environment_settings=replace(
                expected.environment_settings,
                simulator_settings=UnrealExecutableSimulatorConfig(
                    executable_path=exe,
                ),
            ),
        )
    assert args == expected


@pytest.mark.parametrize(
    "case",
    _META_ALG_CONFIG_ALGORITHM_SIMULATOR_COUNT_MATRIX,
    ids=lambda c: c.case_id,
)
def test_meta_alg_config_algorithm_simulator_count_matrix(
    mock_main: MagicMock,
    tmp_path: Path,
    case: MetaAlgConfigTestParameters,
) -> None:
    """``--config-file`` merges correctly for every algorithm × simulator count combination."""
    cfg = tmp_path / "matrix.yaml"
    cfg.write_text(yaml.safe_dump(case.config_doc), encoding="utf-8")
    meta = _build_meta_alg_app(
        mock_main,
        case.algorithm_keys,
        case.simulator_keys,
        app_name=f"cfg-{case.algorithm_keys}-{case.simulator_keys}",
    )
    cli = ["--config-file", str(cfg), *case.cli_tokens]
    meta(cli, result_action="return_value", exit_on_error=False)

    mock_main.assert_called_once()
    args: FakeScriptSettings = mock_main.call_args[0][0]
    assert args == case.expected_sim_settings


def _write_invalid_yaml(path: Path) -> None:
    """YAML that ``yaml.safe_load`` rejects (nested mapping on one line)."""
    path.write_text("foo: bar: baz\n", encoding="utf-8")


def test_invalid_yaml_config_file_logs_error_with_details(
    meta_app, mock_main: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Invalid ``--config-file`` triggers ``logger.error`` with path and parser context."""
    cfg = tmp_path / "bad.yaml"
    _write_invalid_yaml(cfg)

    with caplog.at_level(logging.ERROR):
        meta_app(
            ["--config-file", str(cfg), "alpha"],
            result_action="return_value",
            exit_on_error=False,
        )

    error_messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
    ]
    assert error_messages, "expected at least one ERROR log for invalid YAML"
    combined = " ".join(error_messages)
    assert "Error loading config file" in combined
    assert str(cfg) in combined
    # ``load_yaml_file`` appends ``MarkedYAMLError.problem_mark`` when present
    assert " at " in combined


def test_invalid_yaml_no_alg_config_file_logs_error_with_details(
    no_alg_meta_app,
    mock_main: MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """``MetalgCommand`` with no algorithms: invalid YAML still logs a detailed error before empty config."""
    cfg = tmp_path / "bad_no_alg.yaml"
    _write_invalid_yaml(cfg)

    with caplog.at_level(logging.ERROR):
        no_alg_meta_app(
            ["--config-file", str(cfg), "external"],
            result_action="return_value",
            exit_on_error=False,
        )

    error_messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
    ]
    assert error_messages
    combined = " ".join(error_messages)
    assert "Error loading config file" in combined
    assert str(cfg) in combined
    assert " at " in combined


def test_algorithm_specs_support_mixed_simulator_topologies(
    mock_main: MagicMock,
):
    """An explicit empty simulator table creates a leaf beside online commands."""
    class MixedTopologyCommand(ScholaCommandTemplate[FakeScriptSettings]):
        @property
        def algorithm_specs(self):
            return {
                "online": AlgorithmSpec(FakeAlgoAlpha, "online"),
                "offline": AlgorithmSpec(FakeAlgoBeta, "offline", simulator_table={}),
            }

        @property
        def simulator_table(self) -> Dict[str, Type[BaseSimulatorConfig[Any]]]:
            return {"external": ExternalSimulatorConfig}

        @property
        def script_args_type(self):
            return FakeScriptSettings

        @property
        def main_func(self):
            return mock_main

    app = App(name="mixed")
    command = MixedTopologyCommand(app, logging.getLogger(__name__)).make()
    command.meta(["offline"], result_action="return_value", exit_on_error=False)

    args = mock_main.call_args[0][0]
    assert isinstance(args.algorithm_settings, FakeAlgoBeta)
    with pytest.raises(Exception):
        command.meta(
            ["offline", "external"], result_action="return_value", exit_on_error=False
        )
