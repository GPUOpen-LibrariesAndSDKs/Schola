Native Online Learning Foundation
=================================

Schola now provides the common lifecycle contracts and native build boundary
needed to integrate RLtools as an in-process learning backend. The common
contracts live in ``ScholaInferenceUtils``. ``ScholaRLTools`` depends on that
module and the pinned ``RLTools`` external module; common policy headers do not
depend on RLtools. This integration currently targets Unreal Engine 5.8 on
Windows. Earlier engine versions and other platforms require separate checks.

What is available
-----------------

``UAdaptiveNPCPolicy`` exposes ``GetSessionInfo`` as a copied session, agent,
episode and readiness record. A successful ``Init`` replaces an old binding;
invalid replacement input preserves the existing binding and decisions.
``Shutdown`` and destruction also release an active binding. The base first
closes admission and invalidates old identities, then calls the protected
``ReleaseLearnerBinding(PreviousSession)`` hook once for that Ready binding.
Repeated shutdown does not release twice. A concrete subclass must clean up
resources acquired during a failed, partially completed initialization itself.

The existing AdaptiveSAC adapter implements this hook, so calling ``Shutdown``
through ``UAdaptiveNPCPolicy`` stops its learner as well. Its
``ShutdownLearner`` function remains a compatibility alias. A future shared
backend must detach only the calling NPC; it stops the logical learner only
when its last binding is released or its world ends.

``ResetEpisode`` retains the learner session and increments only that agent's
episode. It cancels outstanding decisions before calling ``OnEpisodeReset``
with the previous identity. This notification must preserve weights and replay.
Experiences must match the current session, agent and episode, with nonnegative
decision/version IDs. Submit the final outcome before resetting. No decision
history or duplicate-submission detection is implied by those identity checks.

Optional operations and snapshots
---------------------------------

``GetCapabilities`` advertises support for learning control, inference-mode
selection, learning reset, and snapshots. The base advertises none of them.
``SetLearningEnabled``, ``SetInferenceMode``, ``ResetLearning``, and
``CaptureSnapshot`` return ``FAdaptiveOperationResult`` with a status and
diagnostic. Their default status is ``Unsupported`` and they change no state.
A backend that supports an operation assigns its operation ID and implements
readiness checks and completion reporting. Pending does not mean completed.

Training pause and deterministic action selection are separate contracts.
Learning reset applies to learned state and may affect a learner group;
episode reset is agent-local. These declarations do not implement a native
training worker or claim that the old example's combined ``SetFrozen`` control
already implements the new contract.

``UAdaptivePolicySnapshot`` is an abstract handle for immutable inference state.
A concrete backend captures actor weights and preprocessing, then initializes
the handle once with a session, policy version and copied interaction definition.
``Evaluate`` writes its output only after successful backend evaluation. It
cannot submit experience or change the source learner. The backend validates
space compatibility and retains frozen storage independently of live weights.
Handles are runtime objects; serialized or duplicated configuration does not
inherit an initialized snapshot.

Native dependency and compilation
---------------------------------

The single vendored include tree is ``Source/ThirdParty/RLTools``, pinned to
``b32d9985c65a5e098a6bbf190fd994962d288b99``. Its headers and license were moved
without changes from AdaptiveTag. The external module validates the revision
and stages ``RLTools-LICENSE.txt``. Existing standalone examples use the same
tree; they retain their algorithms and game behavior.

``ScholaRLTools`` compiles with C++20, precise floating-point semantics, no RTTI
or exceptions, and no unity merging or precompiled header. RLtools operation
definitions are confined to
``Private/Learning/RLToolsLearner.cpp``. The spaces descriptor carries
template dimensions only; it does not implement game movement or rewards.
The ``Schola.RLTools.NativeBuild`` automation test initializes every registered
CPU SAC, TD3 and PPO configuration and evaluates its actor. Those test shapes
are not a fixed gameplay-profile catalogue or evidence of complete online
learning through the plugin.

Next integration steps
----------------------

The reusable numerical learner is described in :doc:`native_rltools_learning`.
The supplied native policy and world-local learner service are described in
:doc:`native_adaptive_policy`, and project configuration generation is described
in :doc:`rltools_configuration_generation`, and checkpoint save/load is described
in :doc:`native_rltools_checkpoints`. Game/environment coordinators remain
subsequent integration work. Game code
can select the supplied policy and provide task callbacks rather
than implement its own RLtools wrapper. No new Python learning backend is used.

Verification
------------

From ``Examples/AdaptiveSAC``, run:

.. code-block:: powershell

   ./Test-AdaptiveSAC.ps1 -Filter 'Schola.Inference.Adaptive+Schola.RLTools+ScholaOnlineLearning.SAC.Policy'

The adaptive tests cover copied identity, release ordering and idempotence,
episode isolation, old-episode rejection, unsupported operations, Blueprint
reflection, and immutable snapshot behavior. Existing SAC tests include
shutdown through a base-policy reference while the native learner is training.
Build verification does not establish packaged gameplay or cross-version support.
