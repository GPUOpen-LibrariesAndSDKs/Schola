Driving adaptive NPCs with UAdaptiveStepper
===========================================

``UAdaptiveStepper`` runs the observe, decide, act and learn loop for online-learning
NPCs. The other steppers call ``IPolicy::Think`` or ``BatchedThink``, and
``UAdaptiveNPCPolicy::Think`` deliberately returns false: an adaptive action must be
tied to the outcome it causes, which a plain ``Think`` cannot record. The adaptive
stepper uses the ticket API from :doc:`adaptive_decision_lifecycle` instead.

What the game implements
------------------------

Each NPC implements two interfaces:

* ``IAgent`` (existing): ``Define``, ``Observe``, ``Act``, ``GetStatus``.
* ``IAdaptiveAgent`` (``Steppers/AdaptiveAgentInterface.h``):

  * ``ConsumeReward``: return the reward earned since the last call and clear it.
  * ``GetEpisodeStatus``: report ``bTerminated`` (the task ended) and ``bTruncated``
    (a time limit or interruption ended it).
  * ``ResetEpisode``: start a new encounter. Learned state is kept.

All three can be implemented in Blueprint.

Setup
-----

.. code-block:: cpp

   // Template policy: configure it once; the stepper never binds it directly.
   auto* Template = NewObject<URLToolsNPCPolicy>(this);   // outer supplies the world
   Template->Configuration = TEXT("sac/3x2-mlp32");
   Template->SharedGroup = TEXT("Guards");                 // optional: pool learning

   Stepper = NewObject<UAdaptiveStepper>(this);
   Stepper->DecisionTimeoutSeconds = 1.0f;
   Stepper->Init(Agents, Template);

   // Every decision tick (for example at 10 Hz):
   Stepper->Step();

``Init`` duplicates the template once per agent, using the template's outer, and
initializes each copy with that agent's ``Define`` result. Each policy object binds
exactly one agent. Learning is independent per agent unless the backend pools it.
For ``URLToolsNPCPolicy``, give the template a ``SharedGroup`` name so every copy
trains one learner. ``GetPolicy(Index)`` returns an agent's binding for learning
controls, snapshots and checkpoints.

Native learners become Ready asynchronously. ``Step`` skips an agent until its
binding is Ready, so call it every tick from the start. ``AreAllReady`` reports when
every binding is Ready. Agents whose ``GetStatus`` is not ``Running`` are skipped.

What one Step does
------------------

For each Ready, running agent:

1. **Poll.** If a request is outstanding, poll it. A Ready action is applied with
   ``Act`` and starts an interval. A request that is still pending keeps waiting.
   An expired, failed or cancelled request is counted as abandoned. Nothing was
   applied, so nothing is submitted for it.
2. **Close the interval.** If an applied action has no outcome yet, observe the next
   state, read the reward and episode flags, and call ``SubmitExperience``. If the
   episode ended, the agent and then the policy reset the episode, and the agent is
   observed again.
3. **Request.** Request a decision for the current observation and poll it
   immediately. Synchronous backends such as ``URLToolsNPCPolicy`` therefore act in
   the same ``Step``.

Rules the stepper relies on
---------------------------

* **Submit before reset.** The policy checks the experience's episode ID against
  the current episode, so an ending outcome must be submitted before
  ``ResetEpisode``. The stepper always does this.
* **Decisions expire.** A request that is not applied within
  ``DecisionTimeoutSeconds`` expires when it is polled, even if it was Ready. The
  stepper then requests a fresh decision.
* **Reward while pending is discarded.** Reward earned while a request is still
  pending belongs to no decision. The stepper discards it when the new action is
  applied. With synchronous backends this window is empty.
* **Admission is not training.** ``Accepted`` means the learner queued the record.
  ``Dropped``, ``Frozen``, ``Rejected`` and ``NotReady`` are counted in
  ``GetStats().NotAccepted`` and are not retried.

Shutdown
--------

``Shutdown`` shuts down every binding. Call ``Init`` again to start over with fresh
bindings. Destroying the stepper does not shut bindings down early; each policy
releases its own binding when it is destroyed.

Tests
-----

``Schola.Inference.Adaptive.Stepper.*`` (7 tests) covers Init validation, one binding
per agent, waiting for Ready, outcome attribution, episode ends (submit before
reset, terminated versus truncated), pending and expired requests, and shutdown.
They use the controlled test policy and need no learner:

.. code-block:: powershell

   ./Test-AdaptiveSAC.ps1 -Filter 'Schola.Inference.Adaptive.Stepper'
