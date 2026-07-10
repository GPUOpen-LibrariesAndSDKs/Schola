Schola Fabrica (LLM C++ rewards + SB3)
=====================================

Schola **Fabrica** is an optional workflow inspired by `Eureka: Human-Level Reward Design via Coding Large Language Models <https://github.com/eureka-research/Eureka>`_: an LLM **Deep Agent** explores your Unreal C++ project (and optionally an **Editor Python** world snapshot), then writes **initialization** and **reward** code into a generated ``*.fabrica.gen.cpp`` file. Short **Stable Baselines 3** runs score each candidate.

Install
-------

.. code-block:: bash

   pip install "schola[fabrica]"

Python **3.11+** is required. The reward agent uses ``deepagents`` (installed via ``schola[fabrica]`` on 3.11+); there is no fallback if ``deepagents`` cannot be imported.

C++ base: ``AFabricaEnvironment``
---------------------------------

Subclass ``AFabricaEnvironment`` (module ``ScholaTraining``). The base **implements** ``ICppOnlyMultiAgentEnvironment`` (``InitializeEnvironment`` / ``Reset`` / ``Step`` still exchange ``TMap<FString, …>`` with the trainer). Subclasses implement **single-agent** ``BlueprintNativeEvent`` hooks: ``OnUserInitializeEnvironment`` (one ``FInteractionDefinition&``), ``OnUserReset`` (one ``FInitialAgentState&``), ``OnUserStep`` (one action ``FInstancedStruct`` plus one ``FFabricaAgentState&``). The base adapts those to the multi-agent maps using ``GetFabricaSingleAgentId()`` (default ``"agent"``; override in C++ if you need a different id). Implement hooks in a **Blueprint subclass** (event graph) or in C++ via ``OnUserInitializeEnvironment_Implementation`` and the other ``*_Implementation`` methods. The base then invokes **Fabrica-generated** overrides:

- ``FabricaGeneratedInit`` — populate ``FabricaTrackedActors`` (``TMap<FString, TObjectPtr<AActor>>``).
- ``FabricaGeneratedRewardForAgent`` — write shaped terms into ``FAgentState::Info`` using the prefix from ``UFabricaRewardInfo::GetComponentPrefix()`` (default ``fabrica_r:``), and set ``FAgentState::Reward`` to the **total** (typically the sum of components).

``OnUserStep`` fills one ``FFabricaAgentState``. ``AFabricaEnvironment::Step`` maps it under ``GetFabricaSingleAgentId()``, converts to ``FAgentState``, then runs Fabrica shaping; ``TaskSuccessMetric`` is copied into ``Info`` under ``UFabricaRewardInfo::GetTaskSuccessInfoPrefix()`` (default ``fabrica_ts``) before ``FabricaGeneratedRewardForAgent`` runs. Python training logs read that same key on episode end.

Fabrica hook implementations live in ``*.fabrica.gen.cpp`` (out-of-line overrides on your concrete environment class).

CLI
---

``schola fabrica run`` uses the same ``MetaAlgCommand`` layout as ``schola sb3 train``: choose an **algorithm** (``ppo`` or ``sac``), then the **project** simulator, then Fabrica paths and SB3 flags.

Unlike ``schola sb3 train``, Fabrica only registers the **project** simulator. Each iteration merges new reward C++ into ``*.fabrica.gen.cpp`` (derived from ``--env-header``: ``Public/…/Foo.h`` → ``Private/…/Foo.fabrica.gen.cpp``; override the output directory with ``--code-gen-folder``). That code must be compiled into the binary used for scoring. The ``project`` simulator rebuilds on every run (``use_cached_build=False``), so the staged game picks up the latest generated regions. ``executable``, ``external`` / ``editor``, and other simulators that attach to a pre-built binary would never see those changes.

.. code-block:: bash

   schola fabrica run ppo project ^
     --env-header path/to/YourEnv.h ^
     --task-description path/to/task.txt ^
     --uproject-path path/to/Game.uproject ^
     --code-roots path/to/extra ^
     --code-roots path/to/other

The reward Deep Agent can list, read, and grep files under **code roots**. By default, roots include the parent of ``--env-header`` and the parent of ``--uproject-path``. Pass ``--code-roots`` one or more times to add extra directories (for example another module's ``Source`` tree). Nested roots collapse to the shortest ancestor (``A/B`` and ``A/B/C`` resolve to ``A/B`` only).

Optional ``--config-file`` matches the SB3 train YAML pattern (see ``schola sb3 train --help``). See :doc:`running_from_cli` for ``project`` simulator options (``--map``, ``--headless``, ``--build-dir``, and so on).

Flags include ``--editor-snapshot.enabled`` to run ``UnrealEditor-Cmd`` with the bundled ``fabrica_world_snapshot.py`` once at the beginning of the fabrica run. The snapshot uses the same ``--uproject-path`` and ``--map`` as the ``project`` simulator; the resulting JSON is reused for every iteration (requires Editor Python and a resolvable Engine from the ``.sln``).

References
----------

- `Eureka (ICLR 2024) <https://github.com/eureka-research/Eureka>`_
- `LangChain Deep Agents <https://github.com/langchain-ai/deepagents>`_
- `Unreal Editor Scripting with Python <https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-editor-using-python>`_
