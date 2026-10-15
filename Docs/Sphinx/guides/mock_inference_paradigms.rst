Mock NPC Inference Paradigms
===========================

``ScholaInferenceUtils`` exposes ``UNPCInferencePolicy`` in
``Policies/NPCInferencePolicy.h``. Select one of five paradigms with
``FNPCInferenceConfig::Paradigm``. It implements the existing ``IPolicy``
interface, so callers can use ``Think``, ``BatchedThink``, and existing steppers.
The same methods are Blueprint-callable; they do not add a UI or training gRPC endpoints.

All backends are deterministic mocks. No neural network, online optimizer,
background worker, Mass subsystem, or game-tree simulator is loaded. Copied
points stand in for actions, so prototypes should use compatible point shapes.
``Init`` accepts an interaction definition but does not enforce action-space
compatibility. ``UNNEPolicy`` is unchanged. ``UBlueprintPolicy`` gains batch and busy-state events.

Construction and selection
-------------------------

Add ``ScholaInferenceUtils`` to your module dependencies. Hold the policy in a
``UPROPERTY`` or another Unreal GC-safe owner, as with other UObject policies.

.. code-block:: cpp

   #include "Policies/NPCInferencePolicy.h"
   #include "Points/BoxPoint.h"

   TInstancedStruct<FPoint> Behavior, PreferredAction, Action;
   Behavior.InitializeAs<FBoxPoint>(TArray<float>{1.0f});
   PreferredAction.InitializeAs<FBoxPoint>(TArray<float>{2.0f});

   auto* Policy = NewObject<UNPCInferencePolicy>(Owner);
   FNPCInferenceConfig Config;
   Config.Paradigm = ENPCInferenceParadigm::Adaptive;
   if (!Policy->Configure(Config) || !Policy->Init(InteractionDefinition))
   {
       return;
   }
   Policy->Think(Behavior, Action);

Successful ``Configure`` resets initialization, feedback, subscriptions' pending
notifications, registered heads, search candidates, and diagnostics. Call ``Init``
again after changing configuration. Invalid configuration leaves previous state
intact. Check boolean return values before applying output actions. Inference
failures leave caller outputs intact. Empty batches and invalid points fail.
All API calls must be serialized; this mock does not provide concurrent access.

Adaptive NPCs
-------------

``SubmitFeedback`` accepts player behavior, a preferred action and a finite reward.
Subsequent inference replays the most recent nonnegative feedback action for an
identical observation. Negative feedback is recorded without replacing that
action. Unseen behaviors return the observation unchanged. This memory lookup
simulates online adaptation without implementing a learning algorithm.

.. code-block:: cpp

   FPlayerBehaviorFeedback Feedback;
   Feedback.PlayerBehavior = Behavior;
   Feedback.PreferredAction = PreferredAction;
   Feedback.Reward = 1.0f;
   Policy->SubmitFeedback(Feedback);
   Policy->Think(Behavior, Action); // Returns PreferredAction.

Event Based NPCs
----------------

Configure one or more event names; forward notifications from the game's event
system through ``NotifyEvent``. Any subscribed event permits one successful
``Think`` or ``BatchedThink`` call. Notifications coalesce until that call;
failures preserve the pending notification. This models turn-based scheduling
without implementing an event bus. No action is produced between events.

.. code-block:: cpp

   Config.Paradigm = ENPCInferenceParadigm::EventBased;
   Config.Events = {TEXT("TurnStarted"), TEXT("PlayerMoved")};
   Policy->Configure(Config);
   Policy->Init(InteractionDefinition);
   if (Policy->NotifyEvent(TEXT("TurnStarted")) && Policy->Think(Behavior, Action))
   {
       // Apply Action using the existing agent API.
   }

Background NPCs
---------------

``BatchedThink`` simulates dispatch in chunks of ``MaxBatchSize``. ``WorkerCount``
and ``FanOut`` select mock worker lanes: ``Contiguous`` assigns each chunk to
one lane, ``RoundRobin`` assigns successive rows to successive lanes, and
``MassEntity`` maps an entity ID modulo worker count. Execution is synchronous;
chunk dispatch counts and lanes are observable in ``GetStats``. Outputs always
match input order, including a partial final chunk.

.. code-block:: cpp

   Config.Paradigm = ENPCInferenceParadigm::Background;
   Config.MaxBatchSize = 32;
   Config.WorkerCount = 4;
   Config.FanOut = ENPCInferenceFanOut::RoundRobin;
   Policy->Configure(Config);
   Policy->Init(InteractionDefinition);
   TArray<TInstancedStruct<FPoint>> Observations = {Behavior, Behavior};
   TArray<TInstancedStruct<FPoint>> Actions;
   Policy->BatchedThink(Observations, Actions);

For the mock Mass adapter, select ``MassEntity`` and call
``ThinkEntities(EntityIds, Observations, Actions)``. IDs must be unique,
nonnegative, and have the same count as observations. Diagnostics retain IDs in
row order; no actual Mass entities or processors are created. Plain
``BatchedThink`` in this mode has no IDs and uses contiguous lanes.

Shared Policy NPCs
-----------------

Register specialized heads with fixed mock actions. ``ThinkShared`` simulates one
shared trunk evaluation for an observation, followed by one evaluation per
requested head. The API returns actions in requested head order, and validates
all heads before doing any work. ``Think`` and ``BatchedThink`` use the configured
default head. The mock head returns its registered action irrespective of the
copied trunk observation; counters expose the intended reuse contract.

.. code-block:: cpp

   Config.Paradigm = ENPCInferenceParadigm::SharedPolicy;
   Config.Head = TEXT("Combat");
   Policy->Configure(Config);
   Policy->Init(InteractionDefinition);
   Policy->RegisterHead(TEXT("Combat"), PreferredAction);
   Policy->RegisterHead(TEXT("Navigation"), Behavior);
   Policy->ThinkShared(Behavior, {TEXT("Combat"), TEXT("Navigation")}, Actions);
   // GetStats(): one trunk evaluation, two head evaluations.

Advanced Strategy NPCs
---------------------

``SetSearchCandidates`` supplies legal mock actions and scores for simulated
opponent replies from the NPC's perspective. ``MonteCarloTreeSearch`` selects
the greatest mean rollout score; ``Minimax`` selects the greatest worst-reply
score. Ties choose the first candidate. These stand-ins expose strategy selection
without implementing real MCTS, tree expansion or game simulation.

.. code-block:: cpp

   Config.Paradigm = ENPCInferenceParadigm::AdvancedStrategy;
   Config.SearchStrategy = ENPCSearchStrategy::Minimax;
   Config.SearchBudget = 4;
   Policy->Configure(Config);
   Policy->Init(InteractionDefinition);
   FStrategySearchCandidate Risky, Safe;
   Risky.Action = Behavior;
   Risky.OpponentReplyScores = {-2.0f, 10.0f};
   Safe.Action = PreferredAction;
   Safe.OpponentReplyScores = {3.0f, 3.0f};
   Policy->SetSearchCandidates({Risky, Safe});
   Policy->Think(Behavior, Action); // Minimax chooses Safe.

All candidates need valid actions and at least one finite reply score. The total
number of supplied reply scores must fit ``SearchBudget``; an invalid replacement
preserves the previous candidates. The budget is per observation, not per batch.
Candidates remain available until replaced or reconfigured; supply new candidates
when the game state changes. ``SearchEvaluations`` counts all reply evaluations.

Verification
------------

Unreal automation tests are grouped under ``Schola.Inference.Paradigms`` in
``Source/ScholaInferenceUtils/Private/Test/Policies/NPCInferencePolicyTest.cpp``.
They cover adaptation, event gating and failure atomicity, fan-out and Mass ID
mapping, shared computation counts, and opponent-aware strategy selection.

Blueprint API
-------------

``UNPCInferencePolicy`` is ``BlueprintType`` and ``Blueprintable``. Construct an
NPC Inference Policy object using the existing Blueprint object-construction
mechanism, make an ``FNPCInferenceConfig``, then call ``Configure`` and ``Init``.
All selection enums, configuration fields, feedback and search-candidate structs
are available to Blueprint. ``GetConfig`` and ``GetStats`` are pure getters;
statistics fields are read-only. Inference uses instanced-struct pins constrained
to ``FPoint``. Runtime validation rejects unrelated structs and preserves outputs
on failure. Each native method listed above has a corresponding Blueprint call.
The implementations remain mocks and no custom editor UI or assets are added.

``UBlueprintPolicy`` now exposes ``BatchedThink`` and ``IsInferenceBusy`` as
Blueprint Native Events. A Blueprint policy can override ``BatchedThink`` to
handle the entire batch once. Its default implementation calls the existing
``Think`` event per observation. ``IsInferenceBusy`` defaults to false and can be
overridden for policies that perform asynchronous work. Existing ``Think`` and
``Init`` events keep their signatures. The native batch bridge rejects malformed
output counts and unrelated action types before updating caller outputs.
