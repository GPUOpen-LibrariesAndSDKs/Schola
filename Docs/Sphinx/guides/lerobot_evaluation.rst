.. Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

Evaluating LeRobot Policies with Schola
=======================================

The ``lerobot_env_schola`` plugin lets `LeRobot
<https://github.com/huggingface/lerobot>`_ evaluate a policy in an Unreal
environment exposed through Schola. It adapts Schola's vector environment,
observations, continuous actions, episode metadata, rendering, and success
information to LeRobot's evaluation contract.

The current integration supports evaluation against one externally managed
Unreal process. The process may expose multiple homogeneous agent slots.

Installation
------------

The plugin requires Python 3.12, LeRobot 0.6, and Schola 2.1. From the Schola
repository, install both Python packages into the environment used to run
LeRobot:

.. code-block:: bash

   pip install -e "./Resources/python"
   pip install -e "./Resources/python/lerobot_env_schola"

LeRobot discovers installed distributions whose names start with
``lerobot_env_``. No manual import or Gymnasium registration is required. You
can confirm discovery by running:

.. code-block:: bash

   lerobot-eval --help

``schola`` should appear among the choices for ``--env.type``.

What to Pass to LeRobot
-----------------------

At minimum, ``lerobot-eval`` needs:

* ``--config_path`` pointing to a YAML file that defines the Schola environment
  and evaluation settings;
* ``--policy.path`` pointing to a local pretrained policy directory or a
  Hugging Face Hub repository; and
* an Unreal environment that is already running and listening on the configured
  gRPC address.

The local policy directory or Hub repository must contain a LeRobot pretrained
policy, including ``config.json`` and ``model.safetensors``.

The YAML file must provide the following values:

``env.type``
   Must be ``schola``.

``env.protocol.url`` and ``env.protocol.port``
   The address of the running Unreal environment. ``url`` defaults to
   ``localhost``; ``port`` must match the port configured in Unreal.

``env.observations``
   A complete mapping of every Schola observation key. The camera names, vector
   names, and resulting shapes must match the pretrained policy's input
   features.

``eval.n_episodes``
   The number of episodes to evaluate.

``eval.batch_size``
   The number of homogeneous agent slots exposed by Unreal. It must match
   exactly, even when there is only one slot.

``eval.use_async_envs``
   Must be ``false`` because Schola already supplies a vector environment.

The policy's action feature size must equal the flattened size of the Schola
action space. LeRobot supplies each action as a batch with shape
``(eval.batch_size, action_dimension)``; the plugin reconstructs nested Schola
``Dict`` actions before stepping the environment.

The shortest typical invocation is:

.. code-block:: bash

   lerobot-eval --config_path schola_eval.yaml \
       --policy.path <PATH_OR_HUB_REPOSITORY>

Optional LeRobot arguments include ``--output_dir`` for results and videos,
``--seed`` for repeatable rollouts, and policy-specific overrides such as
``--policy.device=cuda``.

Configure Observations
----------------------

LeRobot policies use named state and image features, while an Unreal
environment may expose observations under arbitrary Schola keys. The
``observations`` section accounts for every source observation using one of
four mappings:

``cameras``
   Maps a LeRobot camera name to one Schola image key. Images may be
   channel-first or channel-last ``uint8`` data, or floating-point data bounded
   by ``[0, 1]``. The adapter emits channel-last ``uint8`` images under
   ``observation.images.<camera-name>``.

``vectors``
   Maps an output name to an ordered list of Schola ``Box`` observations. Each
   source is flattened and the values are concatenated in the listed order.
   Name the policy's main proprioceptive vector ``agent_pos`` to map it to
   ``observation.state``.

``passthrough``
   Renames one Schola ``Box`` observation without changing its shape. The
   special output name ``environment_state`` maps to
   ``observation.environment_state``.

``ignore``
   Lists source observations intentionally omitted from policy input.

A source key must appear exactly once. Unaccounted keys, duplicate use of a
source, and unknown keys are rejected when the environment is created.

For example, this mapping combines joint positions, velocities, and gripper
state into one policy state while exposing two cameras:

.. code-block:: yaml

   observations:
     cameras:
       front: front_camera
       wrist: wrist_camera
     vectors:
       agent_pos:
         - joint_positions
         - joint_velocities
         - gripper
     passthrough:
       environment_state: target_state
     ignore:
       - debug

Run an Evaluation
-----------------

Start the Unreal environment and note its gRPC port. Create a LeRobot
evaluation configuration such as ``schola_eval.yaml``:

.. code-block:: yaml

   env:
     type: schola
     task: reach
     task_description: Reach the target with the gripper.
     episode_length: 300
     success_key: goal_reached
     render_camera: front
     render_fps: 30
     simulator:
       num_simulators: 1
     protocol:
       url: localhost
       port: 8000
     observations:
       cameras:
         front: front_camera
       vectors:
         agent_pos:
           - joint_positions
           - joint_velocities
       ignore:
         - debug

   eval:
     n_episodes: 10
     batch_size: 1
     use_async_envs: false

Then evaluate a local checkpoint or a policy from the Hugging Face Hub:

.. code-block:: bash

   lerobot-eval --config_path schola_eval.yaml \
       --policy.path <PATH_OR_HUB_REPOSITORY>

For example:

.. code-block:: bash

   lerobot-eval --config_path schola_eval.yaml \
       --policy.path outputs/train/reach/checkpoints/last/pretrained_model \
       --output_dir outputs/eval/reach \
       --policy.device=cuda

LeRobot writes aggregate metrics to ``eval_info.json`` in its evaluation output
directory. When image observations are configured, it can also render
evaluation videos using ``render_camera``. If no render camera is specified,
the first configured camera is used.

Success Metrics
~~~~~~~~~~~~~~~

Set ``success_key`` to the name of a Schola ``info`` value that indicates task
success. The plugin exposes that value to LeRobot as ``is_success``. Unreal
string info values must be ``"true"`` or ``"false"`` (case-insensitive);
boolean values are also accepted. If ``success_key`` is omitted, evaluation
still runs but no successful episodes are reported by this mapping.

Vectorized Evaluation
~~~~~~~~~~~~~~~~~~~~~

Schola performs vectorization inside Unreal, so LeRobot must not add its own
asynchronous vector layer:

.. code-block:: yaml

   eval:
     batch_size: 4
     use_async_envs: false

``batch_size`` must exactly match the number of homogeneous agent slots exposed
by the connected Unreal process. Keep ``simulator.num_simulators`` set to
``1``; multiple externally managed processes are not supported by this
integration.

Feature and Action Inference
----------------------------

When ``features`` and ``features_map`` are omitted, the plugin infers them from
the adapted observation space and flattened action space. It maps:

* ``agent_pos`` to ``observation.state``;
* ``environment_state`` to ``observation.environment_state``;
* cameras to ``observation.images.<camera-name>``;
* other vector outputs to ``observation.<output-name>``; and
* the flattened continuous action to ``action``.

Explicit ``features`` and ``features_map`` may be supplied when a policy needs
custom declarations, but both mappings must contain the same keys.

Schola action spaces must be a ``Box`` or a nested ``Dict`` containing only
``Box`` spaces. Nested actions are flattened for the policy and reconstructed
before each Schola step. Discrete actions are not supported.

Troubleshooting
---------------

Common configuration failures are caused by an observation key missing from all
four mapping sections, a LeRobot batch size that differs from Unreal's agent
count, ``use_async_envs`` being enabled, or an image whose type, bounds, or
channel count is unsupported.
