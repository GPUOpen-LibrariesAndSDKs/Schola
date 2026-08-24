Imitation Learning from Recorded Demonstrations
===============================================

Imitation learning trains an agent from recorded gameplay, and as such, you do not use a reward
function during training. Rather, as you play the game or interact with the environment, Schola records what you see and what you
do, and the agent learns to copy those actions.

Use this path when the behavior you want is easier to show than to score, because
even an imperfect cloned policy gives reinforcement learning a far more useful
starting point than a random one.

Schola records demonstrations in the `Minari <https://minari.farama.org/>`_ dataset
format, so you can train with any Minari-compatible imitation-learning library and
then export the result to ONNX for :doc:`inference in Unreal <setting_up_inference>`.

Installing
----------

Install the Minari extra:

.. code-block:: bash

    pip install ./Plugins/Schola/Resources/python[minari]

The ``[all]`` extra in :doc:`setup_schola` already includes this.

Set Up the Imitation Environment
---------------------------------

Configure an imitation environment in Unreal before you collect data. See the
Imitation Learning section of :doc:`migrating_to_v2`.

Collect Demonstrations
----------------------

Run ``schola minari collect`` to start a recording session. Schola connects to your
Unreal environment over gRPC and records observations, actions, rewards,
terminations, and truncations as you play.

.. code-block:: bash

    schola minari collect executable --executable-path <PATH_TO_EXECUTABLE> \
        --dataset-id my-demo-v0 --num-steps 10000

``external`` / ``editor`` (the default), ``project``, and ``gym`` work the same way
they do for online training. See :doc:`running_from_cli`.

Minari requires that the dataset ID include a version suffix (for example ``-v0``).

Useful options:

* ``--dataset-id``: Name for the new Minari dataset.
* ``--num-steps`` / ``-t``: Number of demonstration steps to record. The default is
  1000.
* ``--data-path``: Directory for Minari HDF5 storage. If you omit it, Minari uses
  ``MINARI_DATASETS_PATH`` or ``~/.minari/datasets/``.
* ``--seed``: Random seed for the collector.
* ``--record-infos``: Also store the ``info`` dict from each step.
* ``--author``, ``--author-email``, ``--description``, ``--algorithm-name``,
  ``--code-permalink``: Dataset metadata fields.

See :py:class:`schola.scripts.minari.settings.MinariCollectionSettings` for the full
option list.

Inspect the Dataset
-------------------

Load the dataset in Python to check episode count, step count, and space definitions:

.. code-block:: python

    import minari

    print(minari.list_local_datasets())
    dataset = minari.load_dataset("my-demo-v0")
    print(dataset.total_episodes, dataset.total_steps)
    print(dataset.observation_space, dataset.action_space)

Train a Policy
--------------

Schola does not ship a single training command for Minari datasets. Instead, you
choose the library that fits your task. Common options include:

* `imitation <https://imitation.readthedocs.io/>`_ — BC, DAgger, GAIL, and related
  algorithms.
* `d3rlpy <https://d3rlpy.readthedocs.io/>`_ — offline deep RL algorithms.
* A custom PyTorch or TensorFlow script that reads Minari episodes.

The example below uses the ``imitation`` library for behavior cloning (BC). BC treats
the problem as supervised learning on recorded actions: it ignores rewards and copies
the demonstrator directly, including any mistakes. This makes BC the fastest path to a
working policy when your demonstrations are consistent.

.. code-block:: python

    import minari
    import numpy as np
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    dataset = minari.load_dataset("my-demo-v0")

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    for episode in dataset.iterate_episodes():
        all_obs.append(episode.observations[:-1])
        all_next_obs.append(episode.observations[1:])
        all_acts.append(episode.actions)
        dones = np.zeros(len(episode.actions), dtype=bool)
        dones[-1] = episode.terminations[-1] or episode.truncations[-1]
        all_dones.append(dones)

    transitions = Transitions(
        obs=np.concatenate(all_obs),
        acts=np.concatenate(all_acts),
        next_obs=np.concatenate(all_next_obs),
        dones=np.concatenate(all_dones),
        infos=np.array([{}] * sum(len(a) for a in all_acts)),
    )

    bc_trainer = BC(
        observation_space=dataset.observation_space,
        action_space=dataset.action_space,
        demonstrations=transitions,
    )
    bc_trainer.train(n_epochs=100)

Other imitation-learning algorithms (for example DAgger or GAIL) follow the same
pattern: load Minari episodes, convert them to the library format, then train.

Deploy the Trained Model
------------------------

After training, export your policy to ONNX so Unreal can run it. The exact export
call depends on the training library you used, but the example below shows how to
do it for a PyTorch policy from ``imitation``:

.. code-block:: python

    import torch

    policy = bc_trainer.policy
    policy.set_training_mode(False)
    dummy_obs = torch.zeros(1, *dataset.observation_space.shape, dtype=torch.float32)
    torch.onnx.export(
        policy,
        dummy_obs,
        "bc_model.onnx",
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

Import the resulting ONNX file into Unreal and wire it through Schola inference as
described in :doc:`setting_up_inference`.

Improve Results
---------------

The learned policy can only be as good as the data you recorded. Before you change
the algorithm, check for these two common data-quality issues:

**Idle steps.** Human demonstrations often contain steps where you had not yet
reacted, so the recorded action is a no-op. A large number of these idle transitions
teaches the agent to do nothing in states that call for action. Count them and remove
them if they make up a significant fraction of the dataset.

**Action balance.** If one action dominates the demonstration set, the agent learns
to always take it. Rare but important actions — such as turning — need enough
examples to be learned reliably. Record extra demonstrations that focus on the
under-represented situations.

You can combine multiple Minari datasets with ``minari.combine_datasets()`` when you
add targeted recordings.

Next Steps
----------

A cloned policy will typically know the basics of the task, but do not expect it to
play well on its own. Where it fails, record more demonstrations of those situations
and retrain.

You can warm-start reinforcement learning from the cloned weights. Load them into
your online trainer or continue in Unreal with a hybrid workflow. For Schola RLlib
users who trained with ``schola rllib offline-train``, resume online training from
the BC checkpoint:

.. code-block:: bash

    schola rllib train ppo --resume-from <bc-checkpoint> --timesteps 50000

That command loads the cloned ``RLModule`` weights into a new PPO (or IMPALA) run
without restoring the BC optimizer or lifetime step count — ``--timesteps`` sets a
fresh lifetime cap. Note that SAC is not supported on this path and the critic starts
untrained.

Alternative: RLlib Offline Training
-----------------------------------

Schola also ships an integrated offline path that writes RLlib Parquet during
collection and trains with RLlib BC or MARWIL instead of Minari HDF5. To use it,
install the ``rllib-offline`` extra and refer to
:py:class:`schola.scripts.rllib.settings.BCSettings` and
:py:class:`schola.scripts.rllib.settings.MARWILSettings`.

.. code-block:: bash

    pip install ./Plugins/Schola/Resources/python[rllib-offline]

    schola rllib collect executable --executable-path <PATH_TO_EXECUTABLE> \
        --output ./demos --num-steps 1000

    schola rllib offline-train bc --input ./demos --timesteps 100000

Use this path when you want Schola to manage collection, training, and ONNX export
in one RLlib workflow.
