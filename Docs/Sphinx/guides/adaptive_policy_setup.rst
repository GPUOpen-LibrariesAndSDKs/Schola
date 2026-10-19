Adaptive Policy Records and Local Setup
============================================================

This guide describes the implemented foundation for adaptive NPCs: shared
Unreal data records and validation of the NPC's input/output definitions. The
complete gameplay-to-learning loop remains the proposed design in
:doc:`adaptive_online_learning_contract`.

The policy does not yet send requests, receive actions, or train a model. The
existing ``UNPCInferencePolicy`` mock and its five paradigms are unchanged.

What each type represents
--------------------------------------------------

.. list-table:: Foundation types
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Meaning
   * - ``FAdaptiveDecision``
     - An action and the session, NPC, encounter, request, and policy-version
       identifiers needed to match it to its origin.
   * - ``FAdaptiveExperience``
     - A completed outcome: those identifiers, the originating observation,
       command actually executed, reward, next observation, and separate
       termination/truncation flags.
   * - ``EAdaptiveSessionState``
     - Labels for local initialization and eventual remote connection state.
   * - ``EAdaptiveRequestStatus``
     - Labels for eventual request results: Unknown, Pending, Ready, Failed,
       and Cancelled. No request operations are implemented yet.
   * - ``UAdaptiveNPCPolicy``
     - An Unreal policy object that validates and copies the declared
       observation/action spaces. It implements ``IPolicy`` while immediate
       inference remains unavailable.

Include ``Policies/AdaptiveLearningTypes.h`` for the records and status types,
or ``Policies/AdaptiveNPCPolicy.h`` for the policy. Add ``ScholaInferenceUtils``
to the consuming module's dependencies. All types are available to Blueprint.

Records are passive containers. Assigning a reward or action does not validate,
submit, acknowledge, or learn from an experience. A finite negative reward is
legitimate learning data. Do not put a preferred action or an unrelated fallback
into ``ExecutedAction``.

The initial Unreal representation uses ``FGuid`` for ``SessionId`` and signed
64-bit integers for ``AgentId``, ``EpisodeId``, ``DecisionId``, and
``PolicyVersion``. A default record has an invalid GUID and numeric IDs set to
``INDEX_NONE`` (-1), meaning unassigned; zero is a valid numeric identity/version.
Initialization does not generate session or decision IDs. D's wire format and
session-handshake assignment remain subject to team review; these records do
not define a protocol or silently settle the contract's other open decisions.

Supported local definitions
---------------------------

This first implementation accepts the following numerical subset. Dimensions
are supplied by the game rather than fixed to the pursuit example.

.. list-table:: Accepted spaces
   :header-rows: 1
   :widths: 23 77

   * - Definition
     - Requirements
   * - Observation space
     - A nonempty ``FBoxSpace``. Bounds may be finite, one-sided, or unbounded.
       Every enabled bound must be finite. An enabled lower bound cannot exceed
       an enabled upper bound. Constant observation dimensions are allowed.
   * - Action space
     - A nonempty ``FBoxSpace``. Each dimension needs finite lower and upper
       bounds, with a strictly positive finite difference so action scaling
       cannot divide by zero or overflow.
   * - Shape
     - Empty means a vector. Otherwise every dimension must be positive and
       their product must equal the number of stored bounds. Invalid or
       overflowing shapes are rejected without constructing sample points.

Dictionary, discrete, text, base ``FSpace``, and custom space types are rejected
by this initial policy. Unsupported types need an explicit extension and
learner agreement before use. This is local structural validation; a future
handshake must still check the loaded model's dimensions, preprocessing, units,
and action scaling with B and D.

Schola's existing ``FSpace::Validate`` / ``Contains`` methods check actual point
values against a space. They do not validate the space definition itself. This
policy checks definitions; point validation belongs to the later request and
experience operations.

Initialize and inspect the policy
--------------------------------------------------

In Blueprint, construct an Adaptive NPC Policy, retain it in an owning variable,
and call ``Init`` with an interaction definition. Check the return value and
read ``GetLastError`` if it fails. ``GetInteractionDefinition`` returns an
independent copy of the accepted definitions; before initialization it returns
false without changing the output.

In C++, keep the policy alive with a normal Unreal-owned reference, for example
this member of the owning actor or UObject:

.. code-block:: cpp

   UPROPERTY()
   TObjectPtr<UAdaptiveNPCPolicy> AdaptivePolicy;

Then initialize it from that object's code:

.. code-block:: cpp

   #include "Policies/AdaptiveNPCPolicy.h"
   #include "Spaces/BoxSpace.h"

   TInstancedStruct<FSpace> Observations, Actions;
   Observations.InitializeAs<FBoxSpace>(TArray<FBoxSpaceDimension>{
       FBoxSpaceDimension::Unbounded(), FBoxSpaceDimension::Unbounded()});
   Actions.InitializeAs<FBoxSpace>(TArray<FBoxSpaceDimension>{
       FBoxSpaceDimension(-1.0f, 1.0f), FBoxSpaceDimension(-1.0f, 1.0f)});

   AdaptivePolicy = NewObject<UAdaptiveNPCPolicy>(this);
   if (!AdaptivePolicy->Init(FInteractionDefinition(Observations, Actions)))
   {
       UE_LOG(LogTemp, Warning, TEXT("%s"), *AdaptivePolicy->GetLastError());
       return;
   }
   // Local validation succeeded. A Python connection has not been started.

A new policy starts ``Uninitialized``. A first invalid initialization reports
``Failed`` and remains uninitialized. A valid initialization reports
``Initialized`` (displayed as **Locally Initialized** in Blueprint), clears the
previous error, and owns a copy of the definition. Mutating the caller's input
or a retrieved copy cannot change the policy's stored definition.

An invalid replacement returns false and records an error while preserving the
previous valid definition and initialized state. A later valid initialization
can replace it. Always check ``Init``'s return value when changing definitions;
``IsInitialized`` reports whether an accepted definition still exists.

``Connecting``, ``Ready``, and ``Stopped`` are reserved session states for the
later connection lifecycle. Local initialization never reports ``Ready``.
``IsInferenceBusy`` returns false because no work is dispatched. ``Think`` and
``BatchedThink`` return false promptly, explain that inference is unavailable,
and preserve the action or whole action array supplied by the caller.

Verification
------------

The Unreal automation tests under ``Schola.Inference.Adaptive`` cover supported
definitions, malformed Blueprint payloads, bounds and shapes, recovery after
invalid initialization, owned copies, output preservation, and Blueprint
exposure. Run these alongside ``Schola.Inference.Paradigms`` to check the
existing mock API. These tests establish the local foundation, not a live
communication round trip or neural learning.
