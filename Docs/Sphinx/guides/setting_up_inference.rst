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
Unreal Engine's built-in Auto Reimport system can monitor that folder and import or re-import
``.onnx`` files as ``UNNEModelData`` assets when they are written.

To use Unreal's Auto Reimport for ONNX exports:

1. Enable at least one Neural Network Engine (NNE) runtime plugin, such as ``NNERuntimeORT`` or
   ``NNERuntimeBasicCpu``, so Unreal's NNE ONNX importer is available.
2. Open **Editor Preferences > Loading & Saving > Auto Reimport**.
3. Enable **Monitor Content Directories**.
4. In **Directories to Monitor**, add your project content directory. You can use the virtual path
   ``/Game/`` or the absolute path to your project's ``Content`` folder. If you use an absolute
   path, set the mount point to ``/Game/``.
5. Add an include wildcard for ``*.onnx`` so the monitor is scoped to ONNX model exports.
6. Ensure **Auto Create Assets** is enabled if you want newly-created ``.onnx`` files to create
   ``UNNEModelData`` assets automatically.
7. Set your training run's ``CheckpointDir`` to a folder inside ``Content`` (for example
   ``Content/Schola/Models``), enable ``Export to ONNX``, and run training from the editor.

When export completes, Unreal imports the ONNX file into the matching Content Browser folder. To
update an existing ONNX asset, overwrite the source ``.onnx`` file on disk in the same ``Content``
folder, for example by running another training export or replacing the file in your file explorer.
Unreal's Auto Reimport system will detect the change and re-import the asset.

If no NNE runtime plugin is enabled, Unreal may report ``Unknown extension 'onnx'`` because no ONNX
import factory is registered. Enable an NNE runtime plugin and restart the editor before importing
or auto-reimporting ONNX models.


Setting up Your Unreal Engine Level
-----------------------------------

Schola's inference system consists of three main components:

1. **Agent** - Any object implementing the :cpp:class:`IAgent` interface that defines observation and action spaces
2. **Policy** - A :cpp:class:`UNNEPolicy` that loads your trained ONNX model, performs inference, and optionally advances recurrent state
3. **Stepper** - A :cpp:class:`USimpleStepper` (or :cpp:class:`UPipelinedStepper`) that coordinates the observation-inference-action loop and owns one policy state per agent

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
4. For recurrent models, set ``Max State Sequence Length`` to the history length expected by the model
5. Call ``Init()`` with the agent's interaction definition

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

Recurrent ONNX Models
"""""""""""""""""""""

``UNNEPolicy`` automatically treats model inputs whose names begin with ``state_in`` and model
outputs whose names begin with ``state_out`` as recurrent state tensors. The policy pairs these
inputs and outputs in model order, so the model must expose the same number of each. Models without
these tensors remain stateless.

After ``Init()``, the stepper calls ``CreateInitialState()`` once for each agent. For a recurrent
model this creates a zero-initialized :cpp:class:`UNNEPolicyState`; for a stateless model it returns
the shared :cpp:class:`UStatelessPolicyState`. Each inference reads that agent's previous state
and, after successful inference, advances it in place with the corresponding ``state_out`` values.

If you call ``Think`` on a ``UNNEPolicy`` directly instead of through a stepper, create one state
per agent yourself. In Blueprint, call **Create Initial State** on the policy after ``Init`` and
store the result in a variable that you pass to every ``Think`` call for that agent.

If a state input has a dynamic sequence dimension, ``Max State Sequence Length`` controls the
allocated history length. On each successful inference Schola drops the oldest entry and appends the
newest state. The default history length is one.

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

The stepper creates and retains a separate policy state for every agent. It manages agents like a
``TArray``: ``AddAgent()`` creates a new initial state and returns the agent's index.
``RemoveAgentAt()`` takes an index and releases that agent's state when it is safe to do so, and
later agents shift down by one. ``RemoveAgent()`` removes every entry for an agent and returns how
many it removed. ``FindAgent()`` returns the index of the first entry for an agent, and
``GetAgent()`` and ``GetState()`` look up an entry by index. The same agent can be added more than
once, with each entry getting its own state. Call ``ResetState()`` or ``ResetStates()`` at episode
boundaries to return recurrent state to its initial value. Resetting the state of a stateless
model does nothing.

.. note::
    
    For better performance with slower inference, consider using :cpp:class:`UPipelinedStepper` instead of :cpp:class:`USimpleStepper`. The pipelined stepper overlaps observation collection and action execution with inference.

    A state reset requested while ``UPipelinedStepper`` has inference in flight is deferred until
    that inference finishes.

Policy State Lifecycle
----------------------

:cpp:class:`IPolicyState` represents data that a policy carries from one inference step to the
next, such as recurrent neural-network tensors, an action history, or an LLM conversation. It does
not store the policy itself. A single policy instance can therefore serve multiple agents while
each agent retains independent state.

The policy defines the state type and creates its initial value through ``CreateInitialState()``.
The caller owns the returned object, keeps it alive with a ``UPROPERTY`` reference, and passes the
same object to every ``Think()`` call for that agent. ``Think()`` advances the object in place; it
must not replace it.

Stateless policies return the shared :cpp:class:`UStatelessPolicyState` from
``UStatelessPolicyState::Get()`` (**Get Stateless Policy State** in Blueprint), which is what the
default implementation does. A state is therefore never null. Policies implement the protected
``CreateInitialStateImpl()``, and callers use the non-virtual ``CreateInitialState()``, which
rejects a state without a native ``IPolicyState``, such as ``None`` from a Blueprint override. An
agent whose policy returns ``None`` therefore fails to be added to a stepper.

State implementations provide the following operations:

* ``Reset()`` restores the state to its value at the beginning of an episode.
* ``CopyFrom()`` copies a compatible state into an existing object without allocating a new
  ``UObject``.
* ``NotifyStateUpdate()`` prepares the previous state to receive the current step's data. Its
  default implementation does nothing and succeeds. An override can shift a recurrent sequence or
  trim a history window.
* ``ToString()`` provides a debugging representation. By default, it returns the implementing
  class name.

A stateful ``Think()`` follows a specific order:

1. Read the previous state and compute the action and any new state values.
2. Call ``NotifyStateUpdate()``.
3. If notification succeeds, write the new values into the existing state object.

If ``Think()`` or ``NotifyStateUpdate()`` returns false, that agent's state must remain unchanged.
A default ``BatchedThink()`` call processes agents in order, so states processed before a later
failure may already have advanced. Batched callers must supply exactly one state per observation.

``IPolicyState`` implementations must be native C++ classes; the interface cannot be implemented
entirely in Blueprint. ``CreateInitialState()`` runs on the game thread, but ``Think()`` and
``NotifyStateUpdate()`` may run on a background thread. Do not create ``UObject`` instances or use
game-thread-only APIs in those methods. With :cpp:class:`UPipelinedStepper`, do not inspect a state
while inference is in flight.

Implementing a Custom Stateful Policy
-------------------------------------

A custom C++ policy implements :cpp:class:`IPolicy`. For example, a policy that runs its own LSTM
keeps the hidden and cell state for each agent, and creates and advances that state as follows:

.. code-block:: cpp

    bool ULSTMPolicy::CreateInitialStateImpl(
        TScriptInterface<IPolicyState>& OutState) const
    {
        ULSTMPolicyState* State = NewObject<ULSTMPolicyState>(GetTransientPackage());
        State->Hidden.Init(0.0f, HiddenSize);
        State->Cell.Init(0.0f, HiddenSize);
        OutState = State;
        return true;
    }

    bool ULSTMPolicy::Think(
        const TInstancedStruct<FPoint>& InObservations,
        const TScriptInterface<IPolicyState>& InOutState,
        TInstancedStruct<FPoint>& OutAction)
    {
        ULSTMPolicyState* State =
            Cast<ULSTMPolicyState>(InOutState.GetObject());
        const FBoxPoint* Observation = InObservations.GetPtr<FBoxPoint>();
        if (!State || !Observation)
        {
            return false;
        }

        // Read the previous hidden and cell state while computing this step's action.
        TArray<float> ActionValues;
        TArray<float> NewHidden;
        TArray<float> NewCell;
        if (!RunLSTMCell(Observation->Values, State->Hidden, State->Cell,
                ActionValues, NewHidden, NewCell))
        {
            return false;
        }
        OutAction.InitializeAs<FBoxPoint>(ActionValues);

        // Let the state prepare for this step's values.
        if (!State->NotifyStateUpdate())
        {
            return false;
        }

        // Only mutate the state after NotifyStateUpdate succeeds.
        State->Hidden = MoveTemp(NewHidden);
        State->Cell = MoveTemp(NewCell);
        return true;
    }

``RunLSTMCell`` stands in for the policy's own inference code. ``ULSTMPolicyState`` implements
:cpp:class:`IPolicyState` with ``Hidden`` and ``Cell`` arrays: ``Reset()`` fills both with zeros, and
``CopyFrom()`` copies both arrays from another ``ULSTMPolicyState``. The default
``NotifyStateUpdate()`` is enough here; a state that keeps a sequence of past values would
override it to drop the oldest entry.

You only need a custom policy for models that ``UNNEPolicy`` cannot run. For ONNX models,
``UNNEPolicy`` handles recurrent state automatically through ``state_in`` and ``state_out`` tensors,
as described in `Recurrent ONNX Models`_.

Blueprint Policies
^^^^^^^^^^^^^^^^^^

Derive from :cpp:class:`UBlueprintPolicy` to implement policy behavior in Blueprint:

1. ``CreateInitialState`` returns a native C++ object implementing ``IPolicyState``. The default
   returns the stateless state; an override for a stateless policy should return
   **Get Stateless Policy State**, never ``None``.
2. ``ComputeAction`` reads the observation and previous state to produce an action.
3. Schola calls ``NotifyStateUpdate()`` on the state.
4. ``WriteState`` writes the current step's data into that same state object.

``IPolicyState`` cannot be implemented entirely in Blueprint because state methods may run on a
background thread. The state class must be native C++, although ``UBlueprintPolicy`` itself and its
events can be implemented in Blueprint.
