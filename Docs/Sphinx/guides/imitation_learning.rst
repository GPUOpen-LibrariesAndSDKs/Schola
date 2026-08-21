Imitation Learning from Recorded Demonstrations
===============================================

Imitation learning trains an agent from recorded gameplay. You do not use a reward
function. You play the game. Schola records what you see and what you do. The
agent learns to copy those actions. Use this when the behavior you want is easier
to show than to score. It gives reinforcement learning a competent start policy
instead of a random one.

Schola covers the full path. You record demonstrations from your Unreal
environment in RLlib's offline episode format. You train on that data with RLlib
BC or MARWIL. You export the result to ONNX for
:doc:`inference in Unreal <setting_up_inference>`.

Minari collection (``schola minari collect``) remains available for Farama
ecosystem tools. It is a separate path. RLlib BC and MARWIL read RLlib Parquet,
not Minari HDF5.

Installing
----------

The offline algorithms need RLlib and the msgpack codecs that RLlib uses to
store episodes:

.. code-block:: bash

    pip install ./Plugins/Schola/Resources/python[offline]

The ``[all]`` extra in :doc:`setup_schola` already includes this.

Choosing an Algorithm
---------------------

Schola exposes two offline algorithms from RLlib:

``bc``
    Behavior cloning. Supervised learning on the recorded actions. It ignores
    rewards. It copies the demonstrator, including mistakes. Use it when your
    demonstrations are consistent and you want the fastest path to a working
    policy.

``marwil``
    Monotonic Advantage Re-Weighted Imitation Learning. It weights each
    demonstrated action by its estimated advantage. It can prefer the better
    parts of an inconsistent demonstration set. It can exceed the demonstrator.
    Use it when your recordings mix good and mediocre play. Setting
    ``--beta 0.0`` reduces MARWIL to behavior cloning.

Collect then train
------------------

Set up an imitation environment in Unreal. See the Imitation Learning section
of :doc:`migrating_to_v2`. Then record and train in one command. A simulator
subcommand launches the same simulators as other Schola commands, records until
the Unreal process or gRPC session ends, writes the dataset to ``--output``,
then trains:

.. code-block:: bash

    schola rllib bc executable --executable-path <PATH_TO_EXECUTABLE> \
        --output ./demos --timesteps 100000

    schola rllib marwil executable --executable-path <PATH_TO_EXECUTABLE> \
        --output ./demos --timesteps 100000 --beta 1.0

Use ``--max-steps`` only as a safety cap. The default is to stop recording when
you quit the game, then start training. ``--output`` must not already exist.
Do not pass ``--input`` in this mode.

``external`` / ``editor``, ``project`` and ``gym`` work as they do for
``schola rllib train``. There is no implicit simulator. If you omit the
subcommand, Schola trains from an existing dataset instead of connecting to
Unreal.

Train from an existing dataset
------------------------------

When the Parquet directory already exists, omit the simulator and pass
``--input``. Training does not step an environment:

.. code-block:: bash

    schola rllib bc --input ./demos --timesteps 100000

    schola rllib marwil --input ./demos --timesteps 100000 --beta 1.0

The observation and action spaces come from the dataset sidecar written during
collection. The network, checkpoint, logging and resource flags work as they
do for the online algorithms.

Useful options:

* ``--output`` / ``-o`` — Directory to write when a simulator subcommand is
  given. Required in that mode. Must not already exist.
* ``--input`` / ``-i`` — Existing directory of RLlib Parquet shards. Required
  when no simulator subcommand is given. Cannot be combined with ``--output``.
* ``--timesteps`` — When to stop training. Offline algorithms never sample an
  environment. This counts *trained* steps, not sampled ones.
* ``--offline-data-workers`` and ``--offline-read-cpus`` — Size the Ray Data
  episode transformation pool and the CPU reservation for Parquet reads.
* ``--beta``, ``--vf-coeff``, ``--bc-logstd-coeff`` — MARWIL only. See
  :py:class:`schola.scripts.rllib.settings.MARWILSettings`.

Full option lists are in
:py:class:`schola.scripts.rllib.settings.BCSettings`,
:py:class:`schola.scripts.rllib.settings.MARWILSettings`.

Deploying the Trained Model
---------------------------

Training exports ONNX the same way the online algorithms do. The model has one
input per sensor. It drops into the inference setup in
:doc:`setting_up_inference`:

.. code-block:: bash

    schola rllib bc --input ./demos --export-onnx

Improving Results
-----------------

The learned policy can only be as good as what you recorded. Do two checks
before you blame the algorithm:

**Idle steps.** Human recordings contain steps where you had not yet reacted.
The action is a no-op. These teach the agent to do nothing in states that call
for action. Count them. Prune them if there are many.

**Action balance.** If one action dominates the recording, the agent learns to
always take it. Rare but important actions such as turning need enough
examples. Record extra demonstrations of the situations that need them.

.. note::

    Offline training needs one Tune trial CPU. It needs one CPU per
    ``--offline-data-workers`` worker. It needs ``--offline-read-cpus`` CPUs for
    reads (four CPUs with the defaults). Schola reports the computed local Ray
    allocation before it starts training.

Where to Go Next
----------------

Expect a cloned policy to know the shape of the task. Do not expect it to play
well. Where it fails, the usual next step is more data. Record demonstrations
of the situations it gets wrong.

You can continue with reinforcement learning. That is a manual step today.
``--resume-from`` expects a checkpoint from the same algorithm. An online
algorithm cannot pick up a BC or MARWIL checkpoint from the CLI. You must load
the trained ``RLModule`` yourself (see
:py:func:`schola.rllib.checkpoint.load_rl_module_from_algorithm_checkpoint`) into
the online algorithm config.
