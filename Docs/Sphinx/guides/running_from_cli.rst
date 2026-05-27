Running with a Standalone Environment from CLI
==============================================

Running Schola from the command line interface (CLI) is a powerful way to interact with the system. This guide will walk you through the steps to run Schola from the CLI, including how to set up your environment and execute commands.


Building Your Environment
-------------------------

Before you can run a standalone executable from the Command-line with Schola, you need to build your environment into a standalone executable. This involves packaging your Unreal Engine project which is detailed in the  `official unreal engine documentation <https://dev.epicgames.com/documentation/en-us/unreal-engine/packaging-unreal-engine-projects>`_.

Running From CLI
----------------

Training CLIs nest an **algorithm** subcommand (for example ``ppo``) and a **simulator** subcommand (``editor``, ``executable``, or ``project``). Simulator-specific options are merged into the same flag list (see :doc:`cli_dataclass_conventions`).

Standalone executable
~~~~~~~~~~~~~~~~~~~~~


To launch a standalone environment (i.e. a game built in Development or Shipping mode), use the ``executable`` simulator and pass the path to your packaged binary:

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE>

    .. group-tab:: Ray RLlib
        .. code-block:: bash
            
            schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE>

Replace ``<PATH_TO_EXECUTABLE>`` with the path to your packaged Unreal Engine executable. To attach to an editor session that is already running, use ``editor`` instead of ``executable`` (no executable path required).

Project simulator
~~~~~~~~~~~~~~~~~

Use the ``project`` simulator when you want Schola to **build** (cook/package) your Unreal project and then **launch** the staged standalone game for training, instead of supplying an already packaged ``.exe``. The runtime type is :py:class:`~schola.core.simulators.unreal.UnrealProject` in ``project_simulator.py``: it resolves a ``.uproject`` file, invokes Unreal Build Tool / RunUAT via the project's engine association, then starts the built binary like :py:class:`~schola.core.simulators.unreal.UnrealExecutable`.

**Requirements**

* A full Unreal Engine install on the same machine so RunUAT / UBT can be discovered (see ``get_ubt_path`` / engine version detection in the project layout).
* A path to the **``.uproject`` file** on the CLI (the training dataclass validator expects a file; the Python class can also accept a directory containing a single ``.uproject``).

**Basic usage**

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash

            schola sb3 train [ppo|sac] project --uproject-path <PATH_TO_UPROJECT>

    .. group-tab:: Ray RLlib
        .. code-block:: bash

            schola rllib train [ppo|sac|impala|appo] project --uproject-path <PATH_TO_UPROJECT>

**Common options** (same names as for ``executable`` where applicable)

* ``--build-dir`` — Staging directory for the packaged build. If omitted, a temporary folder under the system temp directory is used (see ``UnrealProject`` construction in ``project_simulator.py``).
* ``--ubt-path`` — Explicit path to Unreal Build Tool / RunUAT if auto-detection from the project folder fails.
* ``--map`` — Single map to cook and run. The implementation normalizes paths that contain ``/Content/`` into Unreal ``/Game/...`` form and validates the result (see ``try_and_resolve_map``). If you omit ``--map``, the build cooks **all** maps and the game starts on the project default map.
* ``--headless``, ``--fps``, ``--display-logs``, ``--disable-script``, ``--num-simulators`` — Same semantics as for the ``executable`` simulator (see the sections below).

.. note::

    The current training CLI wires ``UnrealProject`` with ``use_cached_build=False`` (see ``UnrealProjectSimulatorConfig.make``), so **each training process triggers a full build step** when you use ``project``. Expect long startup times compared to ``executable`` or ``editor``. If you already have a packaged binary, prefer ``executable``; use ``project`` when you want a repeatable “build then train” path from source.

Headless Mode
~~~~~~~~~~~~~

Schola can be run in headless mode, which is useful for running scripts or automating tasks. To run Schola in headless mode, use the following command:

.. tabs::
    
    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE> --headless

    .. group-tab:: Ray RLlib
        .. code-block:: bash
            
           schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE> --headless


This command will start Schola without the graphical user interface (GUI), allowing for accelerated simulation speeds.

.. note::
    Any features requiring rendering will not work when running in headless mode (e.g. :cpp:class:`UCameraSensor`).

Fixed Simulation Timestep
~~~~~~~~~~~~~~~~~~~~~~~~~

Schola allows you to set a fixed frames per second (FPS) for the simulation. This can be useful for ensuring consistent performance across different runs. To set a fixed FPS, use the following command: 

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE> --fps <FPS>

    .. group-tab:: Ray RLlib
        .. code-block:: bash
            
            schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE> --fps <FPS>

Replace `<FPS>` with the desired frames per second value. For example, to set the FPS to 30, use:

.. tabs::
    
    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
             
             schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE> --fps 30

    .. group-tab:: Ray RLlib
        .. code-block:: bash
           
           schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE> --fps 30

.. note::
    The FPS determines the delta used when calculating updates in Unreal Engine, however the number of timesteps simulated per second is independent of this setting. For example if ``--fps 100`` and Unreal simulates your environment at 1000fps then for every second in the real world, 10 seconds in the environment will be simulated. 

Controlling The Map
~~~~~~~~~~~~~~~~~~~

Schola allows you to specify the map to load when launching the environment. To do this, use the `--map` argument followed by the path to the map. For example:

.. tabs::
    
    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE> --map <MAP_NAME>

    .. group-tab:: Ray RLlib
        .. code-block:: bash
            
            schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE> --map <MAP_NAME>

The map should be specified as a relative path from the ``Content`` folder, with ``Content`` replaced by ``Game``. For example ``/Content/LevelOne/Map`` would be specified as ``Game/LevelOne/Map``.

.. note::
    The map must be a valid Unreal Engine map file. If the map is not found or isn't specified, Schola will default to the main map specified in the project settings.

.. note::
    The map parameter will not work with Shipping builds by default, you need to take additional steps to allow the map to be loaded based on a command line flag. 

Passing Environment Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schola lets you forward arbitrary key/value reset options to your Unreal environment at training or evaluation time via ``--env-options``. The values are delivered to the simulator on the next ``reset()`` (mirroring SB3's ``set_options`` semantics) and are useful for parameters that should not be baked into the binary -- for example, a curriculum level, a difficulty setting, or a per-run seed source.

Use Cyclopts' dotted syntax and repeat the flag once per key:

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash

            schola sb3 train [ppo|sac] executable --executable-path <PATH_TO_EXECUTABLE> \
                --env-options.level=hard --env-options.curriculum=stage2

    .. group-tab:: Ray RLlib
        .. code-block:: bash

            schola rllib train [ppo|sac|impala|appo] executable --executable-path <PATH_TO_EXECUTABLE> \
                --env-options.level=hard --env-options.curriculum=stage2

The same flag works for evaluation:

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash

            schola sb3 eval [ppo|sac] --checkpoint <PATH_TO_CHECKPOINT> executable \
                --executable-path <PATH_TO_EXECUTABLE> --env-options.level=hard

    .. group-tab:: Ray RLlib
        .. code-block:: bash

            schola rllib eval --checkpoint <PATH_TO_CHECKPOINT> executable \
                --executable-path <PATH_TO_EXECUTABLE> --env-options.level=hard

.. note::

    Values arrive at your Unreal environment as **strings** -- ``--env-options`` is typed as ``Dict[str, str]`` and Cyclopts does not perform type inference on the CLI side. Parse or cast them as needed inside your Unreal environment code.

.. note::

    Options are consumed on the **first** ``reset()`` after they are set and then cleared (SB3-style one-shot). To re-apply between resets, call ``env.set_options(...)`` from your training script, or pass the flag again on each invocation.

