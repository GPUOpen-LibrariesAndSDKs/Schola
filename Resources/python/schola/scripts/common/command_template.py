# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Cyclopts template for generating Schola subcommands (multi-simulator and algorithm dispatch).
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Generic,
    NewType,
    Protocol,
    TypeVar,
    cast,
)

from cyclopts import App, ArgumentCollection, Group, Parameter
import cyclopts
from cyclopts.argument import update_argument_collection
from schola.core.utils.dict_helpers import flatten_dict_no_prefix

from schola.scripts.common.settings import (
    BaseSimulatorConfig,
    UnrealExecutableSimulatorConfig,
    UnrealProjectSimulatorConfig,
    ExternalSimulatorConfig,
    GymSimulatorConfig,
)

ScriptArgsType = TypeVar("ScriptArgsType", bound="_ScriptArgsProtocol")


class _ScriptArgsProtocol(Protocol):
    environment_settings: Any
    algorithm_settings: Any


@dataclass(frozen=True)
class AlgorithmSpec:
    """Declarative registration for one algorithm command."""

    settings_type: type[Any]
    help: str = ""
    runner: Callable[[Any], Any] | None = None


def load_yaml_file(file_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to an existing YAML file.
    logger : logging.Logger
        Receives parse errors (including marked locations when available).

    Returns
    -------
    dict
        Parsed mapping, or ``{}`` if parsing fails.
    """
    import yaml

    # assume path exists
    try:
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        # print the location of the yaml error
        error_message = f"Error loading config file {file_path}: {e}"
        if isinstance(e, yaml.MarkedYAMLError) and e.problem_mark is not None:
            error_message += f" at {e.problem_mark}"
        logger.error(error_message)
        return {}
    return config_dict


class _ScholaConfig(cyclopts.config.Dict):
    """
    Cyclopts dict config source tailored for Schola meta-CLI merging.

    Notes
    -----
    Behaves like ``cyclopts.config.Dict`` but passes ``None`` for the app stack when
    calling ``update_argument_collection``, which Schola's nested commands require.
    """

    def __call__(
        self, app: "App", commands: tuple[str, ...], arguments: ArgumentCollection
    ):
        config: dict[str, Any] = self.config.copy()

        try:
            if self.use_commands_as_keys and len(commands) > 0:
                config = config[commands[-1]]
        except KeyError:
            return

        update_argument_collection(
            config,
            self.source,
            arguments,
            None,  # passing app stack breaks Schola meta-app config parsing
            root_keys=self.root_keys,
            allow_unknown=self.allow_unknown,
        )


class ScholaCommandTemplate(Generic[ScriptArgsType]):
    """
    Template for creating Schola commands, that have a variable number of algorithms and simulators. These are remapped into subcommands to
    allow for proper parsing without dumping all the arguments into the main function.

    Parameters
    ----------
    app : App
        The main app to add subcommands to.
    logger : logging.Logger
        The logger to use for logging.

    Notes
    -----
    If any algorithms are set in the algorithm table, then the template will create a subcommand for each algorithm first before repeating for simulators.
    """

    def __init__(
        self,
        app: App,
        logger: logging.Logger,
    ):
        self.app = app
        self._logger = logger

    @property
    def simulator_table(self) -> dict[str, type[BaseSimulatorConfig[Any]]]:
        # Ignore the type here as it is difficult to resolve with current typing tooling
        return {
            "gym": GymSimulatorConfig,
            "executable": UnrealExecutableSimulatorConfig,
            "project": UnrealProjectSimulatorConfig,
            "external": ExternalSimulatorConfig,
        }

    @property
    def algorithm_specs(self) -> dict[str, AlgorithmSpec]:
        """Return the algorithm commands supported by this template.

        Subclasses should override this property. The legacy table/help hooks are
        retained as a compatibility adapter for existing commands and third-party
        templates.
        """
        return {
            name: AlgorithmSpec(settings_type, self.algorithm_help.get(name, ""))
            for name, settings_type in self.algorithm_table.items()
        }

    @property
    def simulator_help(self) -> dict[str, str]:
        return {
            "gym": "Run a standard Gymnasium environment in-process via the Schola gym connector.",
            "executable": "Run Unreal from a pre-built executable.",
            "project": "Build and Run Unreal from a UProject File.",
            "external": "Connect to an externally managed UE process (e.g. Unreal Editor, Kubernetes pod, remote host). Default if no simulator is provided.",
        }

    @property
    def simulator_aliases(self) -> dict[str, str | Iterable[str] | None]:
        return {
            "external": "editor",
            "gym": None,
            "project": None,
            "executable": None,
        }

    @property
    def default_simulator_name(self) -> str:
        return "external"

    @property
    def bind_default_simulator(self) -> bool:
        """Whether a simulator subcommand is selected when the user omits one.

        When True (the default), ``external`` is the implicit simulator. When
        False, omitting a simulator runs the command with no simulator bound,
        which is required for data-only training that must not connect to Unreal.
        """
        return True

    @property
    def script_args_type(self) -> type[ScriptArgsType]:
        raise NotImplementedError(
            "script_args_type must be implemented in the subclass"
        )

    @property
    def main_func(self) -> Callable[[ScriptArgsType], Any]:
        raise NotImplementedError("main_func must be implemented in the subclass")

    def make_no_simulator_command(self, runner: Callable[[ScriptArgsType], Any]):
        """Default command for algorithms with no simulator sub-apps.

        Without this, ``parse_args(())`` has nothing to resolve to and the algorithm
        command cannot run on its own.
        """

        def no_simulator_command(
            *,
            hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
        ):
            self._logger.debug("Arguments: %s", hidden_script_args)
            return runner(hidden_script_args)

        return no_simulator_command

    def make_simulator_command(
        self,
        simulator_type: type[BaseSimulatorConfig[Any]],
        runner: Callable[[ScriptArgsType], Any],
    ):
        # Cyclopts needs a distinct annotation per generated command. NewType is
        # applied to a runtime class object, which is not a valid static NewType argument.
        SimulatorType = NewType(
            "SimulatorType", simulator_type
        )  # pyright: ignore[reportGeneralTypeIssues]

        try:
            _sim_default = simulator_type()
        except TypeError:
            _sim_default = None

        if _sim_default is not None:
            # The constructed dataclass is a simulator_type instance, not SimulatorType.
            simulator_default = cast(SimulatorType, _sim_default)

            def default_simulator_command(
                simulator_args: Annotated[
                    SimulatorType, Parameter(name="*")
                ] = simulator_default,
                *,
                hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            ):
                hidden_script_args.environment_settings.simulator_settings = (
                    simulator_args
                )
                self._logger.debug("Arguments: %s", hidden_script_args)
                return runner(hidden_script_args)

            return default_simulator_command
        else:

            def non_default_simulator_command(
                simulator_args: Annotated[SimulatorType, Parameter(name="*")],
                *,
                hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            ):
                hidden_script_args.environment_settings.simulator_settings = (
                    simulator_args
                )
                self._logger.debug("Arguments: %s", hidden_script_args)
                return runner(hidden_script_args)

            return non_default_simulator_command

    def make_algorithm_command(
        self,
        algorithm_app: App,
        algorithm_spec: AlgorithmSpec,
        algorithm_name: str | None = None,
    ):
        algorithm_type = algorithm_spec.settings_type
        # Cyclopts needs a distinct annotation per generated command. NewType is
        # applied to a runtime class object, which is not a valid static NewType argument.
        AlgorithmType = NewType(
            "AlgorithmType", algorithm_type
        )  # pyright: ignore[reportGeneralTypeIssues]

        def meta_algorithm_command(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            algorithm_args: Annotated[
                AlgorithmType | None,  # pyright: ignore[reportInvalidTypeForm]
                Parameter(name="*"),
            ] = None,
            hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            hidden_sim_config_dict: Annotated[
                dict[str, Any] | None, Parameter(parse=False)
            ] = None,
        ):  # type: ignore

            additional_kwargs = {
                "hidden_script_args": hidden_script_args,
            }
            if algorithm_args is None:
                try:
                    algorithm_args = algorithm_type()
                except TypeError as exc:
                    raise ValueError(
                        f"{algorithm_name} requires its mandatory algorithm "
                        "arguments. See `--help` for details."
                    ) from exc
            hidden_script_args.algorithm_settings = algorithm_args

            if self.no_simulator:
                # Nothing below this command reads a config block: the algorithm
                # settings were bound when this command was parsed, and there are no
                # simulator settings. Clear the config inherited from the root app so
                # that the algorithm block is not offered a second time and rejected.
                algorithm_app.config = []
            elif hidden_sim_config_dict is not None:
                algorithm_app.config = [
                    _ScholaConfig(
                        hidden_sim_config_dict,
                        use_commands_as_keys=self.multiple_simulators,
                        allow_unknown=False,
                        source=f"config:environment:simulator",
                    ),
                ]

            command, bound, ignored = algorithm_app.parse_args(tokens)
            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        return meta_algorithm_command

    def make_train_meta_command(self):
        # Cyclopts needs a distinct annotation per generated command. NewType is
        # applied to a runtime class object, which is not a valid static NewType argument.
        ResolvedScriptArgsType = NewType(
            "ResolvedScriptArgsType", self.script_args_type
        )  # pyright: ignore[reportGeneralTypeIssues]
        _main_func = self.main_func
        _logger = self._logger

        def train_meta_command(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            script_args: Annotated[
                ResolvedScriptArgsType,  # pyright: ignore[reportInvalidTypeForm]
                Parameter(name="*"),
            ] = self.script_args_type(),
            hidden_sim_config_dict: Annotated[
                dict[str, Any] | None, Parameter(parse=False)
            ] = None,
        ):
            # we can naively forward the hidden_sim_config_dict as we will handle the None checking when we go to actually
            # apply it to the app
            additional_kwargs = {
                "hidden_script_args": script_args,
            }
            hidden_sim_config_dict = (
                {} if hidden_sim_config_dict is None else hidden_sim_config_dict
            )

            if self.no_algorithm:
                if self.no_simulator:
                    if tokens:
                        command, bound, ignored = self.app.parse_args(tokens)
                        if command == self.app.help_print:
                            return command(*bound.args, **bound.kwargs, **{})
                        return command(*bound.args, **bound.kwargs, **additional_kwargs)
                    _logger.debug("Arguments: %s", script_args)
                    return _main_func(script_args)
                else:
                    # if there is no algorithm, it's time to apply the config to the app.
                    self.app.config = [
                        _ScholaConfig(
                            hidden_sim_config_dict,
                            use_commands_as_keys=(
                                True if self.multiple_simulators else False
                            ),
                            allow_unknown=False,
                            source=f"config:environment:simulator",
                        ),
                    ]
            else:
                # forward to the algorithm command so that we use the correct subcommands as keys
                additional_kwargs["hidden_sim_config_dict"] = hidden_sim_config_dict
            # continue parsing
            command, bound, ignored = self.app.parse_args(tokens)

            # need to drain the additional kwargs here so that the help command doesn't raise an error
            if command == self.app.help_print:
                return command(*bound.args, **bound.kwargs, **{})
            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        return train_meta_command

    def make_train_config_handler(self):
        def train_command_config_handler(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            config_file: Annotated[
                cyclopts.types.ExistingYamlPath | None,
                Parameter(
                    parse=True, show=True, help="Path to a YAML configuration file."
                ),
            ] = None,
        ):
            additional_kwargs: dict[str, Any] = {}
            config_dict = {}
            if config_file is not None:
                config_dict = load_yaml_file(config_file, self._logger)

            # if we have a simulator, extract ``environment.simulator`` and forward it as a hidden argument
            # this lets us put it at the root level of the app instead of nested in the algorithm command
            sim_config_dict: dict[str, Any] | None = None
            if self.has_simulator:
                sim_config_dict = self._create_simulator_config_dict(config_dict)

            # we can naively forward the sim_config_dict as we will handle the None checking when we go to actually
            # apply it to the app
            if sim_config_dict:
                additional_kwargs["hidden_sim_config_dict"] = sim_config_dict

            # if we have an algorithm, we need to extract the keys here
            if self.has_algorithm:
                alg_config_dict = config_dict.pop("algorithm", {})
                self.app.config = [
                    _ScholaConfig(
                        alg_config_dict,
                        use_commands_as_keys=True,
                        allow_unknown=False,
                        source=f"config:algorithm",
                    ),
                ]

            flat_config = flatten_dict_no_prefix(config_dict)
            self.app.meta.config = [
                _ScholaConfig(
                    flat_config,
                    use_commands_as_keys=False,
                    allow_unknown=False,
                    source=f"config",
                ),
            ]

            command, bound, ignored = self.app.meta.parse_args(tokens)

            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        return train_command_config_handler

    def make(self):
        # setup the default meta func on the base app to parse the Script Args
        self.app.meta.default(self.make_train_meta_command())
        # This takes the config file and adds it to the meta app to allow for the config to be aligned with the script args
        self.app.meta.meta.default(self.make_train_config_handler())

        # Propagate help to meta apps so parent command listings show descriptions.
        if self.app.help:
            self.app.meta.help = self.app.help
            self.app.meta.meta.help = self.app.help

        # setup the algorithm commands (e.g. PPO, SAC, etc.)
        # inline the algorithm command if there is only one algorithm type
        self.maybe_make_algorithm_commands(self.app)
        return self.app.meta

    def maybe_make_algorithm_commands(self, root_app: App):
        if self.no_algorithm:
            self.maybe_make_simulator_commands(root_app)
        else:
            self.make_algorithm_commands(root_app)

    def make_algorithm_commands(self, root_app: App):
        alg_group = Group("Algorithm (Choose One)", sort_key=0)
        for algorithm, algorithm_spec in self.algorithm_specs.items():
            algorithm_app = App(
                name=algorithm,
                group_commands=Group("Simulator (Choose One)", sort_key=0),
                help=algorithm_spec.help,
            )
            algorithm_app.meta.default(
                self.make_algorithm_command(algorithm_app, algorithm_spec, algorithm)
            )
            self.maybe_make_simulator_commands(
                algorithm_app,
                algorithm,
                algorithm_spec.runner or self.main_func,
            )

            root_app.command(algorithm_app.meta, name=algorithm)
            root_app[algorithm].group = alg_group

    def maybe_make_simulator_commands(
        self,
        root_app: App,
        algorithm: str | None = None,
        runner: Callable[[ScriptArgsType], Any] | None = None,
    ):
        runner = runner or self.main_func
        if self.no_simulator:
            # An algorithm that opts out of simulators still needs a default command,
            # otherwise its sub-app has nothing to resolve to. The no-algorithm case
            # is handled directly in ``make_train_meta_command`` and needs no default.
            if algorithm is not None:
                root_app.default(self.make_no_simulator_command(runner))
        elif self.single_simulator and self.bind_default_simulator:
            self.collapse_simulator_command(root_app, algorithm, runner)
        else:
            self.make_simulator_commands(root_app, algorithm, runner)

    def collapse_simulator_command(
        self,
        algorithm_app: App,
        algorithm: str | None = None,
        runner: Callable[[ScriptArgsType], Any] | None = None,
    ):
        runner = runner or self.main_func
        simulator_name, simulator_type = list(self.simulator_table.items())[0]
        sim_command = self.make_simulator_command(simulator_type, runner)
        algorithm_app.default(sim_command)
        algorithm_app.command(
            sim_command,
            name=simulator_name,
        )

    def make_simulator_commands(
        self,
        algorithm_app: App,
        algorithm: str | None = None,
        runner: Callable[[ScriptArgsType], Any] | None = None,
    ):
        runner = runner or self.main_func
        if not self.bind_default_simulator:
            algorithm_app.default(self.make_no_simulator_command(runner))
        for simulator_name, simulator_type in self.simulator_table.items():
            sim_command = self.make_simulator_command(simulator_type, runner)
            if (
                self.bind_default_simulator
                and simulator_name == self.default_simulator_name
            ):
                algorithm_app.default(sim_command)
            algorithm_app.command(
                sim_command,
                name=simulator_name,
                alias=self.simulator_aliases[simulator_name],
            )
            algorithm_app[simulator_name].help = self.simulator_help[simulator_name]

    @property
    def algorithm_table(self) -> dict[str, type[Any]]:
        raise NotImplementedError("algorithm_table must be implemented in the subclass")

    @property
    def algorithm_help(self) -> dict[str, str]:
        return defaultdict(str)

    @property
    def no_algorithm(self) -> bool:
        return len(self.algorithm_specs) == 0

    @property
    def no_simulator(self) -> bool:
        return len(self.simulator_table) == 0

    @property
    def single_simulator(self) -> bool:
        return len(self.simulator_table) == 1

    @property
    def multiple_simulators(self) -> bool:
        return len(self.simulator_table) > 1

    @property
    def has_simulator(self) -> bool:
        return len(self.simulator_table) > 0

    def _create_simulator_config_dict(
        self, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract ``environment.simulator`` from *config_dict* and shape it for cyclopts.

        Removes the ``environment.simulator`` block from *config_dict* in place. Config
        files nest settings under the simulator name (e.g. ``external``). When only one
        simulator is registered, ``collapse_simulator_command`` inlines that subcommand,
        so the returned mapping omits the name key.
        """
        sim_config_dict = config_dict.pop("environment", {}).pop("simulator", {})
        if not sim_config_dict or not self.single_simulator:
            return sim_config_dict

        # single simulator case so we can just return the first and only key from the simulator table
        simulator_name = next(iter(self.simulator_table))
        if simulator_name in sim_config_dict:
            nested = sim_config_dict[simulator_name]
            return nested if isinstance(nested, dict) else {}
        # single simulator case where the user inlined the simulator parameters, rather than adding a subcommand key
        return sim_config_dict

    @property
    def has_algorithm(self) -> bool:
        return len(self.algorithm_specs) > 0
