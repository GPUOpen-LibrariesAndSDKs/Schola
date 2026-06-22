Setting Up Inference
====================

This guide will explain how to use your trained RL agents in inference mode (i.e. without connecting to Python).


.. note:: 
    
    This guide assumes you have already done a training run using Schola and have either a saved checkpoint or a model exported to Onnx.

.. note::

    Text spaces are not yet supported for NNE inference. The supported observation and action
    spaces are Box, Discrete, MultiDiscrete, MultiBinary, and Dict (including nested combinations
    of these).


Convert a Checkpoint to Onnx
----------------------------

If you did not export to Onnx during training you will need to convert a checkpoint to Onnx. You can use the following scripts to create an Onnx model from your checkpoint:

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            schola sb3 export --policy-checkpoint-path <CHECKPOINT_PATH> --output-path <ONNX_PATH> --algorithm <ALGORITHM>
    
    .. group-tab:: Ray RLlib
        .. code-block:: bash

            schola rllib export --policy-checkpoint-path <CHECKPOINT_DIR> [--output-path <OUTPUT_DIR>]

For SB3, ``<ALGORITHM>`` must match how the checkpoint was trained (for example ``PPO`` or ``SAC``); allowed values are the same as the export command's ``--algorithm`` choices in ``schola sb3 export --help``.

For RLlib, ``<CHECKPOINT_DIR>`` is the algorithm checkpoint **directory** produced by Ray. ``--output-path`` is optional; if omitted, ONNX output is written alongside the checkpoint (see ``schola rllib export --help``).

These commands produce an ONNX model in Schola's export layout for use in the next section.

Load an Onnx Model into Unreal Engine
-------------------------------------

Schola can export ONNX models directly into your project's ``Content`` folder during training.
When at least one Neural Network Engine (NNE) runtime plugin (for example
``NNERuntimeORT`` or ``NNERuntimeBasicCpu``) is enabled, the Schola editor automatically imports
any ``.onnx`` file written under ``Content`` and creates a corresponding ``UNNEModelData`` asset
in the Content Browser.

To use auto-import:

1. Set your training run's ``CheckpointDir`` to a folder inside ``Content`` (for example
   ``Content/Schola/Models``).
2. Enable ``Export to ONNX`` in the checkpoint settings.
3. Run training from the editor. When export completes, the ONNX asset appears in the matching
   Content Browser folder without manual drag-and-drop.

To update an existing ONNX asset, overwrite the source ``.onnx`` file on disk in the same
``Content`` folder, for example by running another training export or replacing the file in your
file explorer. The Schola watcher will re-import the updated file. Dragging and dropping a
replacement file over an existing ``UNNEModelData`` asset uses Unreal's manual import path instead
of Schola's watcher and may prompt for source-file metadata.

If no NNE runtime plugin is enabled, Schola logs a one-time warning and does not start its
``Content`` auto-import watcher. You can still import manually by dragging and dropping the
``.onnx`` file into the Content Browser if Unreal's native NNE editor importer is available.


Setting up Your Unreal Engine Level
-----------------------------------

Schola's inference system consists of three main components:

1. **Agent** - Any object implementing the :cpp:class:`IAgent` interface that defines observation and action spaces
2. **Policy** - A :cpp:class:`UNNEPolicy` that loads your trained ONNX model and performs inference
3. **Stepper** - A :cpp:class:`USimpleStepper` (or :cpp:class:`UPipelinedStepper`) that coordinates the observation-inference-action loop

Follow these steps to set up inference in your project:

Step 1: Implement the IAgent Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a class (Actor, Component, or any UObject) that implements the :cpp:class:`IAgent` interface. You must implement these methods:

- **Define()** - Specify the observation and action spaces for your agent
- **Observe()** - Collect current observations from the environment
- **Act()** - Execute actions provided by the policy
- **GetStatus() / SetStatus()** - Manage agent state

.. tabs::

    .. group-tab:: Blueprint

        Create a Blueprint class and add the ``Agent`` interface. Implement the ``Define``, ``Observe``, and ``Act`` events.

    .. group-tab:: C++

        .. code-block:: cpp

            UCLASS()
            class AMyAgent : public AActor, public IAgent
            {
                GENERATED_BODY()
                
                virtual void Define_Implementation(FInteractionDefinition& OutDefinition) override;
                virtual void Observe_Implementation(FInstancedStruct& OutObservations) override;
                virtual void Act_Implementation(const FInstancedStruct& InAction) override;
            };

Step 2: Create and Configure the Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a :cpp:class:`UNNEPolicy` object and configure it with your ONNX model:

1. In your Blueprint or C++, create a ``UNNEPolicy`` object
2. Set the ``Model Data`` property to the ONNX model data asset you imported
3. Set the ``Runtime Name`` to your desired inference runtime (e.g., "NNERuntimeORTCpu" or "NNERuntimeORTDml")
4. Call ``Init()`` with the agent's interaction definition

.. tabs::

    .. group-tab:: Blueprint

        - Add a ``UNNEPolicy`` variable to your Blueprint
        - In ``BeginPlay``, call ``Define`` on your agent to get the interaction definition
        - Call ``Init`` on the policy, passing the interaction definition
        - Set the ``Model Data`` and ``Runtime Name`` properties in the details panel

    .. group-tab:: C++

        .. code-block:: cpp

            UNNEPolicy* Policy = NewObject<UNNEPolicy>(this);
            Policy->ModelData = YourOnnxModelDataAsset;
            Policy->RuntimeName = TEXT("NNERuntimeORTCpu");
            
            FInteractionDefinition Definition;
            IAgent::Execute_Define(YourAgent, Definition);
            Policy->Init(Definition);

Step 3: Create and Initialize the Stepper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a :cpp:class:`USimpleStepper` to manage the observation-inference-action loop:

1. Create a ``USimpleStepper`` object
2. Call ``Init()`` with your agent(s) and policy
3. Call ``Step()`` each frame (e.g., in ``Tick()``) to run inference

.. tabs::

    .. group-tab:: Blueprint

        - Add a ``USimpleStepper`` variable to your Blueprint
        - In ``BeginPlay``, call ``Init`` with an array of agents and your policy
        - In ``Tick``, call ``Step`` on the stepper

    .. group-tab:: C++

        .. code-block:: cpp

            USimpleStepper* Stepper = NewObject<USimpleStepper>(this);
            
            TArray<TScriptInterface<IAgent>> Agents;
            Agents.Add(YourAgent);
            
            Stepper->Init(Agents, Policy);
            
            // In your Tick function:
            Stepper->Step();

.. note::
    
    For better performance with slower inference, consider using :cpp:class:`UPipelinedStepper` instead of :cpp:class:`USimpleStepper`. The pipelined stepper overlaps observation collection and action execution with inference.