# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Cyclopts template for generating Schola subcommands (multi-simulator and algorithm dispatch).
"""

from collections import defaultdict
from itertools import chain
import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    NewType,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
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

ScriptArgsType = TypeVar("ScriptArgsType")


def load_yaml_file(file_path: Path, logger: logging.Logger) -> Dict[str, Any]:
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
        self, app: "App", commands: Tuple[str, ...], arguments: ArgumentCollection
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
    def simulator_table(self) -> Dict[str, Type[BaseSimulatorConfig[Any]]]:
        # Ignore the type here as it is difficult to resolve with current typing tooling
        return {
            "gym": GymSimulatorConfig,
            "executable": UnrealExecutableSimulatorConfig,
            "project": UnrealProjectSimulatorConfig,
            "external": ExternalSimulatorConfig,
        }

    @property
    def simulator_help(self) -> Dict[str, str]:
        return {
            "gym": "Run a standard Gymnasium environment in-process via the Schola gym connector.",
            "executable": "Run Unreal from a pre-built executable.",
            "project": "Build and Run Unreal from a UProject File.",
            "external": "Connect to an externally managed UE process (e.g. Unreal Editor, Kubernetes pod, remote host). Default if no simulator is provided.",
        }

    @property
    def simulator_aliases(self) -> Dict[str, str | Iterable[str] | None]:
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
    def script_args_type(self) -> Type[ScriptArgsType]:
        raise NotImplementedError(
            "script_args_type must be implemented in the subclass"
        )

    @property
    def main_func(self) -> Callable[[ScriptArgsType], Any]:
        raise NotImplementedError("main_func must be implemented in the subclass")

    def make_simulator_command(self, simulator_type: Type[BaseSimulatorConfig[Any]]):
        SimulatorType = NewType("SimulatorType", simulator_type)  # type: ignore
        _main_func = self.main_func
        try:
            _sim_default = simulator_type()
        except TypeError:
            _sim_default = None

        if _sim_default is not None:

            def default_simulator_command(
                simulator_args: Annotated[SimulatorType, Parameter(name="*")] = _sim_default,  # type: ignore
                *,
                hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            ):
                hidden_script_args.environment_settings.simulator_settings = simulator_args  # type: ignore
                self._logger.debug("Arguments: %s", hidden_script_args)
                return _main_func(hidden_script_args)

            return default_simulator_command
        else:

            def non_default_simulator_command(
                simulator_args: Annotated[SimulatorType, Parameter(name="*")],
                *,
                hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            ):
                hidden_script_args.environment_settings.simulator_settings = simulator_args  # type: ignore
                self._logger.debug("Arguments: %s", hidden_script_args)
                return _main_func(hidden_script_args)

            return non_default_simulator_command

    def make_algorithm_command(self, algorithm_app: App, algorithm_type: Type[Any]):
        AlgorithmType = NewType("AlgorithmType", algorithm_type)  # type: ignore
        _main_func = self.main_func
        _logger = self._logger

        def meta_algorithm_command(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            algorithm_args: Annotated[AlgorithmType, Parameter(name="*")] = algorithm_type(),  # type: ignore
            hidden_script_args: Annotated[ScriptArgsType, Parameter(parse=False)],
            hidden_sim_config_dict: Annotated[
                Optional[Dict[str, Any]], Parameter(parse=False)
            ] = None,
            config_file: Annotated[
                Optional[cyclopts.types.ExistingYamlPath],
                Parameter(
                    parse=False, show=True, help="Path to a YAML configuration file."
                ),
            ] = None,
        ):  # type: ignore

            additional_kwargs = {
                "hidden_script_args": hidden_script_args,
            }
            hidden_script_args.algorithm_settings = algorithm_args  # type: ignore

            if hidden_sim_config_dict is not None:
                algorithm_app.config = [
                    _ScholaConfig(
                        hidden_sim_config_dict,
                        use_commands_as_keys=self.multiple_simulators,
                        allow_unknown=False,
                        source=f"config:environment:simulator",
                    ),
                ]

            # No simulator sub-apps: ``parse_args(())`` does not resolve to ``main``; call through.
            if self.no_simulator:
                _logger.debug("Arguments: %s", hidden_script_args)
                return _main_func(hidden_script_args)

            command, bound, ignored = algorithm_app.parse_args(tokens)
            return command(*bound.args, **bound.kwargs, **additional_kwargs)

        return meta_algorithm_command

    def make_train_meta_command(self):
        ResolvedScriptArgsType = NewType("ResolvedScriptArgsType", self.script_args_type)  # type: ignore
        _main_func = self.main_func
        _logger = self._logger

        def train_meta_command(
            *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
            script_args: Annotated[
                ResolvedScriptArgsType,  # pyright: ignore[reportInvalidTypeForm]
                Parameter(name="*"),
            ] = self.script_args_type(), 
            hidden_sim_config_dict: Annotated[
                Optional[Dict[str, Any]], Parameter(parse=False)
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
                    # there is no extra processing to do here, we can just call the main function
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
                Optional[cyclopts.types.ExistingYamlPath],
                Parameter(
                    parse=True, show=True, help="Path to a YAML configuration file."
                ),
            ] = None,
        ):
            additional_kwargs: Dict[str, Any] = {}
            config_dict = {}
            if config_file is not None:
                config_dict = load_yaml_file(config_file, self._logger)

            # if we have a simulator, extract ``environment.simulator`` and forward it as a hidden argument
            # this lets us put it at the root level of the app instead of nested in the algorithm command
            sim_config_dict: Optional[Dict[str, Any]] = None
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
        for algorithm in self.algorithm_table:
            algorithm_app = App(
                name=algorithm,
                group_commands=Group("Simulator (Choose One)", sort_key=0),
                help=self.algorithm_help[algorithm],
            )
            algorithm_type = self.algorithm_table[algorithm]
            algorithm_app.meta.default(
                self.make_algorithm_command(algorithm_app, algorithm_type)
            )
            self.maybe_make_simulator_commands(algorithm_app)

            root_app.command(algorithm_app.meta, name=algorithm)
            root_app[algorithm].group = alg_group

    def maybe_make_simulator_commands(self, root_app: App):
        if self.no_simulator:
            return
        elif self.single_simulator:
            self.collapse_simulator_command(root_app)
        else:
            self.make_simulator_commands(root_app)

    def collapse_simulator_command(self, algorithm_app: App):
        simulator_name, simulator_type = list(self.simulator_table.items())[0]
        sim_command = self.make_simulator_command(simulator_type)
        algorithm_app.default(sim_command)
        algorithm_app.command(
            sim_command,
            name=simulator_name,
        )

    def make_simulator_commands(self, algorithm_app: App):
        for simulator_name, simulator_type in self.simulator_table.items():
            sim_command = self.make_simulator_command(simulator_type)
            if simulator_name == self.default_simulator_name:
                algorithm_app.default(sim_command)
            algorithm_app.command(
                sim_command,
                name=simulator_name,
                alias=self.simulator_aliases[simulator_name],
            )
            algorithm_app[simulator_name].help = self.simulator_help[simulator_name]

    @property
    def algorithm_table(self) -> Dict[str, Type[Any]]:
        raise NotImplementedError("algorithm_table must be implemented in the subclass")

    @property
    def algorithm_help(self) -> Dict[str, str]:
        return defaultdict(str)

    @property
    def no_algorithm(self) -> bool:
        return len(self.algorithm_table) == 0

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
        self, config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        return len(self.algorithm_table) > 0
