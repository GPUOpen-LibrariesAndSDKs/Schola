.. Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

Evaluating LeRobot Policies with Schola
=======================================

The ``lerobot_env_schola`` plugin lets `LeRobot
<https://github.com/huggingface/lerobot>`_ evaluate a policy in an Unreal
environment exposed through Schola. It adapts Schola's vector environment,
observations, continuous actions, episode metadata, rendering, and success
information to LeRobot's evaluation contract.

Evaluation uses one externally managed Unreal process. That process may expose
multiple homogeneous agent slots.

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
   A complete mapping from the pretrained policy's input feature names to
   Schola observation source paths.

``eval.n_episodes``
   The number of episodes to evaluate.

``eval.batch_size``
   LeRobot's requested vector size. Schola uses the actual number of homogeneous
   agent slots exposed by Unreal and logs a warning if it differs.

``eval.use_async_envs``
   Must be ``false`` because Schola already supplies a vector environment.

The policy's action feature size must equal the flattened size of the Schola
action space. LeRobot supplies each action as a batch with shape
``(number_of_unreal_slots, action_dimension)``; the plugin reconstructs nested
Schola ``Dict`` actions before stepping the environment.

The shortest typical invocation is:

.. code-block:: bash

   lerobot-eval --config_path schola_eval.yaml \
       --policy.path <PATH_OR_HUB_REPOSITORY>

Optional LeRobot arguments include ``--output_dir`` for results and videos,
``--seed`` for repeatable rollouts, and policy-specific overrides such as
``--policy.device=cuda``.

Configure Observations
----------------------

LeRobot policies use canonical feature names such as
``observation.images.front`` and ``observation.state``, while an Unreal
environment may expose observations under arbitrary Schola keys. Each entry in
``observations`` maps an exact policy feature name to a Schola source path, or
to an ordered list of sources that are flattened and concatenated.

Dots in a source path traverse nested Schola ``Dict`` spaces. For example,
``robot.joint_positions`` selects ``joint_positions`` inside ``robot``.
``__root__`` selects an unnamed top-level observation when Schola exposes a
non-``Dict`` space. Because dots and ``__root__`` are reserved syntax, Schola
dictionary keys may not contain a dot or equal ``__root__``.

Feature behavior is inferred from the policy key; no separate type declaration
is needed:

* ``observation.images.<camera>`` and singular ``observation.image`` each map to
  exactly one image source. Sources may be channel-first or channel-last
  ``uint8`` data, or floating-point data bounded by ``[0, 1]``. The adapter
  emits channel-last ``uint8`` images. ``observation.image`` cannot be combined
  with ``observation.images.*``.
* A single non-image source is passed through with its original ``Box`` shape
  and dtype.
* A YAML list of non-image sources is flattened and concatenated in the order
  written.

Every leaf source must appear exactly once. Unaccounted sources, duplicate use,
unknown paths, and non-``Box`` leaves are rejected when the environment is
created.

For example, this mapping combines joint positions, velocities, and gripper
state into one policy state while exposing two cameras:

.. code-block:: yaml

   observations:
     observation.images.front: sensors.front_camera
     observation.images.wrist: sensors.wrist_camera
     observation.state:
       - robot.joint_positions
       - robot.joint_velocities
       - robot.gripper
     observation.environment_state: target_state

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
       observation.images.front: sensors.front_camera
       observation.state:
         - robot.joint_positions
         - robot.joint_velocities

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
evaluation videos using ``render_camera``. For ``observation.images.*``, omit
``render_camera`` to use the first configured camera. For singular
``observation.image``, omit it or set it to ``image``.

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

``batch_size`` is only a construction request from LeRobot. If it differs from
the connected Unreal process, the plugin logs a warning and uses Unreal's
actual slot count. Keep ``simulator.num_simulators`` set to ``1``; multiple
externally managed processes are not supported.

Feature and Action Inference
----------------------------

When ``features`` and ``features_map`` are omitted, the plugin infers them from
the adapted Gym observation space and flattened action space:

* ``agent_pos`` maps to ``observation.state``;
* ``environment_state`` maps to ``observation.environment_state``;
* ``pixels/<camera-name>`` maps to ``observation.images.<camera-name>``;
* a singular ``pixels`` image maps to ``observation.image``;
* other vector outputs map to ``observation.<output-name>``; and
* the flattened continuous action maps to ``action``.

Explicit ``features`` and ``features_map`` may be supplied when a policy needs
custom declarations, but both mappings must contain the same keys.

Schola action spaces must be a ``Box`` or a nested ``Dict`` containing only
``Box`` spaces. Nested actions are flattened for the policy and reconstructed
before each Schola step. Discrete actions are not supported.

Troubleshooting
---------------

Common configuration failures are caused by an unclaimed Schola leaf, an
unknown dotted source path, ``use_async_envs`` being enabled, or an image whose
type, bounds, or channel count is unsupported.
