Adaptive Decision Tickets
=========================

An adaptive NPC asks for an action and keeps the game running while the answer
is prepared. This implementation manages that wait: it issues a ticket,
tracks the answer, and prevents obsolete or invalid actions reaching gameplay.
It builds on :doc:`adaptive_policy_setup` and the team loop in
:doc:`adaptive_online_learning_contract`.

What is implemented
-------------------

``UAdaptiveNPCPolicy`` manages one NPC and one pending decision. It does not
move actors, choose fallback actions, calculate rewards, or train a model.
Those are C's gameplay and B's learning responsibilities. D's actual Python
connection and routing through ``UNPCInferencePolicy`` remain later work.
Existing mocks and the simple/pipelined steppers are unchanged.

The base policy still cannot connect to Python. ``Init`` only validates local
definitions. A protected native ``StartSession`` hook confirms a valid session,
agent and initial episode after the future handshake. Tests use a small native
subclass that supplies this confirmation and controlled answers. Blueprint
cannot manufacture readiness or inject completions.

The gameplay loop
-----------------

Once the session is Ready, C's coordinator can:

1. Call ``RequestDecision(Observation, OutDecisionId, TimeoutSeconds)``. A true
   return supplies a ticket immediately, without waiting for an action. A false
   return leaves the output ID unchanged; read ``GetLastError``.
2. Keep playing and call ``TryGetDecision(DecisionId, OutDecision)`` on later
   updates. This poll does not wait. Only Ready writes the output record.
3. Apply a Ready action once, retaining its identity and policy version for
   later experience. Repeated polls return independent copies of the same
   result; C must track whether it has already executed it.
4. Request a new decision when needed. A pending ticket must first finish,
   expire, or be cancelled. A validated dispatch attempt replaces the previous
   terminal/ready slot; older tickets then report Unknown. There is no growing
   history or hidden queue of decisions.

.. list-table:: Poll results
   :header-rows: 1
   :widths: 22 78

   * - Status
     - Meaning
   * - Unknown
     - This ticket is invalid, has never existed, or has been retired.
   * - Pending
     - Waiting for a matching response. No action is available.
   * - Ready
     - A validated action is available and has not passed its deadline.
   * - Failed
     - Dispatch/service failure or an invalid matching response. Read
       ``GetDecisionError`` for this ticket, or ``GetLastError``.
   * - Cancelled
     - The caller cancelled, the encounter reset, or the policy stopped or
       reinitialized. No action may be used.
   * - Expired
     - The action age limit was reached. Request a fresh decision.

``IsInferenceBusy`` means pending and unexpired, not connected or ready.
Immediate ``Think`` / ``BatchedThink`` remain unavailable and preserve outputs,
even with a Ready session. Existing steppers are not the adaptive coordinator.

Validation and age limits
-------------------------

Requests require a Ready session, a finite positive timeout, and a valid finite
Box observation. Results must match the session, agent, episode and decision
IDs exactly, have a nonnegative policy version, and contain a valid finite Box
action. Schola's existing space validation checks dimensions and bounds;
the bridge additionally checks finite values and explicit shapes. Empty point
shapes are accepted; an explicit shape must match the definition (or the flat
vector length if the definition has no shape). Nothing is silently clamped.

An invalid matching result fails its ticket. Unrelated, duplicate or late
responses cannot replace a result or fail a newer ticket. Lower policy versions
are accepted if otherwise valid; the version identifies which snapshot chose
the action, rather than whether it is the newest available model.

The timeout is an action age limit measured from dispatch on Unreal's local
monotonic clock, including time spent waiting before gameplay reads the answer.
The default is one second for this API, not a team-approved gameplay tuning
value; C should pass a value suitable for its decision cadence. At the exact
deadline both Pending and Ready become Expired. A prompt response cannot be
used indefinitely. No clocks are compared across processes.

Expiry is evaluated when requesting, polling, cancelling, or processing a
completion. There is no timer or ticking actor. ``IsInferenceBusy`` also checks
the deadline without mutating state; poll before reading a timeout diagnostic.

Reset and stop
--------------

``CancelDecision`` invalidates a retained Pending or Ready ticket. This is local
cancellation: it does not promise that Python stops its computation.

``ResetEpisode`` cancels Pending/Ready actions and increments the episode ID.
It retains the session and its readiness, rather than restarting learning.
C owns capturing the final observation and submitting terminal experience
before resetting; experience queues and acknowledgements are not implemented
in this PR.

A successful ``Init`` invalidates old actions and requires a fresh handshake.
An invalid replacement preserves the previously accepted setup and in-flight
work. ``Shutdown`` is idempotent, invalidates the session, and prevents more
requests until ``Init`` and a new handshake. Destruction also invalidates work.
Decision IDs are never reused within the policy object's lifetime. Loaded or
duplicated configuration does not inherit a live session or its tickets.

Native integration boundary
---------------------------

``FAdaptiveDecisionRequest`` contains copied observation data plus the four
identity fields. It is an Unreal record, not an agreed wire schema. The
protected ``DispatchDecision`` hook must return promptly and copy its input
before retaining it. Its base implementation rejects dispatch. Tests override
only this hook and the local clock; there is no general transport framework.

The future native client must marshal copied results/failures to the game
thread before invoking ``CompleteDecision`` / ``FailDecision``. Capture a weak
policy pointer and the original request identity, never an actor or a mutable
reference into the policy. If the weak pointer is no longer valid, discard the
callback. Valid pointers are still subject to session/ticket checks, including
after shutdown. A delayed failure must carry the original identity too.
All policy API calls belong on the game thread. No worker thread may directly
mutate this UObject, and no game-thread call may wait for network activity.

Verification
------------

Run the editor automation filter ``Schola.Inference.Adaptive`` for local setup
and lifecycle tests, or ``Schola.Inference`` to include existing paradigm tests.
Controlled responses cover immediate and delayed success, failure, expiry,
invalid data, wrong identities, duplicate callbacks, reset, reinitialization,
shutdown, destruction, copied records and reflected Blueprint calls. The clock
is advanced deterministically without sleeps. These tests demonstrate request
management; a real network round trip requires D's client integration.
