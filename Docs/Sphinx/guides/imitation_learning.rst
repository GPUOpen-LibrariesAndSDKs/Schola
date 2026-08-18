Imitation Learning from Recorded Demonstrations
===============================================

Imitation learning trains an agent from recorded gameplay. You do not use a reward
function. You play the game. Schola records what you see and what you do. The
agent learns to copy those actions. Use this when the behavior you want is easier
to show than to score. It gives reinforcement learning a competent start policy
instead of a random one.

Schola covers the full path. You record a `Minari <https://minari.farama.org/>`_
dataset from your Unreal environment. You train on it with RLlib offline
algorithms. You export the result to ONNX for
:doc:`inference in Unreal <setting_up_inference>`.

Installing
----------

The offline algorithms need RLlib and Minari. They also need the msgpack codecs
that RLlib uses to store episodes:

.. code-block:: bash

    pip install ./Plugins/Schola/Resources/python[offline]

The ``[all]`` extra in :doc:`setup_schola` already includes this.

Collecting a Dataset
--------------------

Set up an imitation environment in Unreal. See the Imitation Learning section
of :doc:`migrating_to_v2`. Then record yourself playing:

.. code-block:: bash

    schola minari collect executable --executable-path <PATH_TO_EXECUTABLE> \
        --dataset-id my-demo-v0 --num-steps 10000

Minari requires the version suffix on the dataset id. ``my-demo-v0`` is valid.
``my-demo`` is not. The dataset is written to the local Minari directory.
``minari.list_local_datasets()`` finds it there.

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

Training
--------

Both algorithms read a dataset. They do not step an environment. They take **no
simulator subcommand**. You do not need Unreal or a Gym environment running:

.. code-block:: bash

    schola rllib train bc --dataset-id my-demo-v0 --timesteps 100000

    schola rllib train marwil --dataset-id my-demo-v0 --timesteps 100000 --beta 1.0

The observation and action spaces come from the dataset. The simulator flags
(``--map``, ``--headless``, ``--executable-path`` and the rest) are not
available. The connection flags (``--port``, ``--url``) have nothing to connect
to. The network, checkpoint, logging and resource flags work as they do for the
online algorithms.

Useful options:

* ``--dataset-id`` / ``-d`` — The Minari dataset to train on. Required.
* ``--timesteps`` — When to stop. Offline algorithms never sample an
  environment. This counts *trained* steps, not sampled ones.
* ``--converted-data-dir`` — Where to write the RLlib copy of the dataset.
  Schola treats this path as a cache root. It writes each successful conversion
  to an owned fingerprinted child directory. It never removes existing files in
  the root. It reuses a matching completed conversion. It discards partial
  conversions before they become visible. By default the cache is temporary.
  It is deleted after training.
* ``--offline-data-workers`` and ``--offline-read-cpus`` — Size the Ray Data
  episode transformation pool and the CPU reservation for Parquet reads.
* ``--beta``, ``--vf-coeff``, ``--bc-logstd-coeff`` — MARWIL only. See
  :py:class:`schola.scripts.rllib.settings.MARWILSettings`.

Start training from the command line. The Unreal training settings launch a run
against the game that is running. The offline algorithms do not need that.

Full option lists are in
:py:class:`schola.scripts.rllib.settings.BCSettings`,
:py:class:`schola.scripts.rllib.settings.MARWILSettings`.

Deploying the Trained Model
---------------------------

Training exports ONNX the same way the online algorithms do. The model has one
input per sensor. It drops into the inference setup in
:doc:`setting_up_inference`:

.. code-block:: bash

    schola rllib train bc --dataset-id my-demo-v0 --export-onnx

Improving Results
-----------------

The learned policy can only be as good as what you recorded. Do two checks
before you blame the algorithm:

**Idle steps.** Human recordings contain steps where you had not yet reacted.
The action is a no-op. These teach the agent to do nothing in states that call
for action. Count them. Prune them if there are many:

.. code-block:: python

    import minari
    import numpy as np

    dataset = minari.load_dataset("my-demo-v0")
    total = idle = 0
    for episode in dataset.iterate_episodes():
        idle += np.sum(np.all(episode.actions == 0, axis=-1))
        total += len(episode.actions)
    print(f"Idle steps: {idle}/{total}")

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
of the situations it gets wrong. Merge them into the dataset with
``minari.combine_datasets()``.

You can continue with reinforcement learning. That is a manual step today.
``--resume-from`` expects a checkpoint from the same algorithm. An online
algorithm cannot pick up a BC or MARWIL checkpoint from the CLI. You must load
the trained ``RLModule`` yourself (see
:py:func:`schola.rllib.checkpoint.load_rl_module_from_algorithm_checkpoint`) into
the online algorithm config.
