.. Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

Evaluating LeRobot Policies with Schola
=======================================

The ``lerobot_env_schola`` plugin lets `LeRobot
<https://github.com/huggingface/lerobot>`_ evaluate a policy in an Unreal
environment exposed through Schola. It adapts Schola's vector environment,
observations, continuous actions, episode metadata, rendering, and success
information to LeRobot's evaluation contract.
Evaluation uses one Schola simulator process, which is either launched by
Schola or is an existing running Unreal Process. That simulator is exposed
to LeRobot as a vectorized environment with one or more homogeneous
sub-environments.

Installation
------------

The plugin requires Python 3.12, LeRobot 0.6.1 or later (but earlier than
0.7), and Schola 2.1 or later. Install LeRobot's evaluation and dataset
dependencies, plus any extras required by the policy being evaluated. Then,
from the Schola repository, install both Schola Python packages into the same
environment:

.. code-block:: bash

   pip install "lerobot[evaluation,dataset]>=0.6.1,<0.7"
   pip install -e "./Resources/python"
   pip install -e "./Resources/python/lerobot_env_schola"

LeRobot discovers installed distributions whose names start with
``lerobot_env_``. No manual import or Gymnasium registration is required. You
can confirm discovery by running:

.. code-block:: bash

   lerobot-eval --help

``schola``, ``schola-external``, ``schola-project``, and
``schola-executable`` should appear among the choices for ``--env.type``.

What to Pass to LeRobot
-----------------------

At minimum, ``lerobot-eval`` needs:

* ``--config_path`` pointing to a YAML file that defines the Schola environment
  and evaluation settings;
* ``--policy.path`` pointing to a local pretrained policy directory or a
  Hugging Face Hub repository; and
* either an existing Unreal environment or a project/executable that the plugin
  can launch.

The local policy directory or Hub repository must contain a LeRobot pretrained
policy, including ``config.json`` and ``model.safetensors``.

The YAML file must provide the following values:

``env.type``
   Selects the simulator lifecycle. ``schola`` and ``schola-external`` both
   connect to an existing process, ``schola-project`` builds and launches an
   Unreal project, and ``schola-executable`` launches a packaged executable.

``env.protocol.url`` and ``env.protocol.port``
   The gRPC address. For ``schola`` and ``schola-external``, ``port`` must match
   the existing Unreal process. Managed project and executable modes may omit
   ``port`` to select an available local port automatically.

``env.observations``
   A LeRobot-shaped observation tree whose values are Schola source paths.

``eval.n_episodes``
   The number of episodes to evaluate.

The policy's action feature size must equal the flattened size of the Schola
action space. LeRobot supplies each action as a batch with shape
``(number of Unreal environments, action_dimension)``; the plugin reconstructs nested
Schola ``Dict`` actions before stepping the environment.

The shortest typical invocation is:

.. code-block:: bash

   lerobot-eval --config_path schola_eval.yaml \
       --policy.path <PATH_OR_HUB_REPOSITORY>

Optional LeRobot arguments include ``--output_dir`` for results and videos,
``--seed`` for repeatable rollouts, and policy-specific overrides such as
``--policy.device=cuda``.

Choose a Simulator Mode
-----------------------

Use ``schola-external`` to connect to an Unreal Editor session or another
process that was started separately:

.. code-block:: yaml

   env:
     type: schola-external
     protocol:
       url: localhost
       port: 8000

``schola`` is an equivalent shorter name for the same external-simulator
mode. ``schola-external`` is preferred when the explicit lifecycle name
improves clarity.

Use ``schola-project`` to build and launch an Unreal project. The project path
is required. The Unreal Built Tool(UBT) path is optional as Schola can
discover the UBT path from the project if it has a corresponding Visual
Studio solution:

.. code-block:: yaml

   env:
     type: schola-project
     simulator:
       uproject_path: ./RobotLab/RobotLab.uproject
       map: /Game/Maps/RobotLab
       headless: true

Use ``schola-executable`` to launch an existing packaged environment:

.. code-block:: yaml

   env:
     type: schola-executable
     simulator:
       executable_path: ./Build/RobotLab.exe
       map: /Game/Maps/RobotLab
       headless: true

All three modes use the same ``observations``, evaluation, protocol, and policy
configuration. The plugin currently supports one simulator process per
evaluation, though that process may expose multiple sub-environments.

Configure Observations
----------------------

The ``observations`` configuration mirrors LeRobot's observation tree. A
top-level field such as ``state`` becomes the policy feature
``observation.state``. Camera entries nested under ``images`` become features
such as ``observation.images.front``. Each field maps to a Schola source path,
or to an ordered list of sources that are flattened and concatenated.

Every Schola source path starts at ``observation``. If the top-level Schola
space is a ``Dict``, dots traverse its nested keys; for example,
``observation.robot.joint_positions`` selects ``joint_positions`` inside
``robot``. If Schola exposes a non-composite top-level space, use
``observation`` by itself.

.. warning::
   Dictionary observation spaces in Schola for environments connected with
   LeRobot must not contain keys with ``.`` as this is the separator used
   for flattening nested dictionaries.

Feature behavior is determined by each field name in ``observations`` and by
the shape and type of its mapped Schola space; it is not inferred from the
policy checkpoint. No separate type declaration is needed:

* ``images.<camera>`` and singular ``image`` each map to exactly one image
  source. Schola camera sources are channel-first ``(C, H, W)`` data: either
  floating-point values bounded by ``[0, 1]``, or ``uint8``. The adapter emits
  channel-last ``uint8`` images. ``image`` cannot be combined with ``images``.
* Every non-image source is flattened to a one-dimensional ``Box`` using
  Gymnasium's standard ``flatten_space`` and ``flatten`` behavior. A ``Box``
  preserves its dtype and element order but not a multidimensional shape. For
  example, a ``Box`` with shape ``(2, 3)`` becomes shape ``(6,)``, while
  ``Discrete(4)`` becomes a four-element one-hot ``Box``.
* A YAML list of non-image sources is flattened using the same Gymnasium
  convention and concatenated in the order written.

Schola sources that are not mapped are ignored and produce a warning. The
same source may be listed under more than one policy feature; the adapter
warns because that is often accidental (for example both ``state`` and
``environment_state`` pointing at the same joints). The policy then sees
the same vector on two independent inputs, which is usually not what those
features mean. Unknown source paths are rejected when the environment is
created.

.. note::
   Each mapped source must flatten to a fixed-shape Gymnasium ``Box``.
   ``Box``, ``Discrete``, ``MultiBinary``, ``MultiDiscrete``, and ``Text``
   all do, as do ``Dict`` and ``Tuple`` trees made only of those spaces.
   Gymnasium cannot pack ``Sequence`` (variable length), ``Graph``, or a
   ``Dict``/``Tuple`` that contains either of those into one ``Box``. Those
   sources are rejected at environment creation. Schola's usual Unreal
   observation spaces are the flattenable kinds above.


Concrete example
~~~~~~~~~~~~~~~~

Consider an `SO-101 <https://huggingface.co/docs/lerobot/so101>`_ follower arm,
which LeRobot supports directly. Its policy state contains five arm joints and
one gripper value. Suppose an SO-101 simulated in Unreal exposes the following
Gymnasium observation space through Schola. Camera observations are
channel-first:

.. code-block:: python

   Dict({
       "cameras": Dict({
           "wrist": Box(0.0, 1.0, shape=(3, 480, 640), dtype=np.float32),
       }),
       "so101": Dict({
           "arm_joint_positions": Box(
               -180.0, 180.0, shape=(5,), dtype=np.float32
           ),
           "gripper_position": Box(
               0.0, 100.0, shape=(1,), dtype=np.float32
           ),
       }),
   })

A typical wrist-camera SO-101 checkpoint has these LeRobot features (shown as
``PolicyFeature`` values for clarity):

.. code-block:: python

   {
       "observation.images.wrist": PolicyFeature(
           type=FeatureType.VISUAL, shape=(3, 480, 640)
       ),
       "observation.state": PolicyFeature(
           type=FeatureType.STATE, shape=(6,)
       ),
       "action": PolicyFeature(
           type=FeatureType.ACTION, shape=(6,)
       ),
   }

The corresponding Schola configuration is:

.. code-block:: yaml

   observations:
     images:
       wrist: observation.cameras.wrist
     state:
       - observation.so101.arm_joint_positions
       - observation.so101.gripper_position

The mirrored field on the left identifies the policy input feature, and the
value on the right identifies the Schola source from which it is built.

.. list-table::
   :header-rows: 1
   :widths: 38 24 38

   * - Policy input feature
     - Mapping behavior
     - Schola source
   * - ``observation.images.wrist`` with shape ``(3, 480, 640)``
     - Read one image and convert CHW float to HWC ``uint8``
     - ``observation.cameras.wrist`` with shape ``(3, 480, 640)``
   * - ``observation.state`` with shape ``(6,)``
     - Flatten and concatenate the listed sources in YAML order
     - ``observation.so101.arm_joint_positions`` ``(5,)`` followed by
       ``observation.so101.gripper_position`` ``(1,)``

The resulting state order is ``shoulder_pan``, ``shoulder_lift``,
``elbow_flex``, ``wrist_flex``, ``wrist_roll``, then ``gripper``. The policy
returns an ``action`` vector in that same semantic six-element order. However,
the observation mapping does not control action ordering. Schola's action space
must independently flatten to the order expected by the checkpoint; the plugin
cannot infer joint semantics or units. There is no dedicated comparison between
the policy and Schola action dimensions, so a mismatch may surface only during
policy inference or when stepping the environment. The body joint values
commonly use degrees for SO-101, while the gripper commonly uses LeRobot's
``RANGE_0_100`` normalization.

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
     protocol:
       url: localhost
       port: 8000
     observations:
       images:
         front: observation.sensors.front_camera
       state:
         - observation.robot.joint_positions
         - observation.robot.joint_velocities

   eval:
     n_episodes: 10

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
string info values must be ``"true"`` or ``"false"`` (case-insensitive).
If ``success_key`` is omitted, evaluation
still runs but no successful episodes are reported by this mapping.

Vectorized Evaluation
~~~~~~~~~~~~~~~~~~~~~

Schola performs vectorization inside Unreal. LeRobot's ``use_async_envs``
and ``eval.batch_size`` settings do not create environments, so they do not
need to be included in the YAML. The number of environments is configured in
Unreal. If LeRobot still passes a ``batch_size`` (including its default),
the plugin logs a warning when that value differs from Unreal's environment
count and uses Unreal's count.

Action Spaces
-------------

The plugin derives LeRobot policy features from the connected Schola spaces
after applying the mirrored ``observations`` configuration. It derives the
action feature from the flattened action space. No separate feature mapping is
required.

Schola action spaces must be a ``Box`` or a nested ``Dict`` containing only
``Box`` spaces. Nested actions are flattened for the policy and reconstructed
before each Schola step. Discrete actions are not supported.

Troubleshooting
---------------

Common configuration failures are caused by an unknown Schola source path or
an image whose type, bounds, or channel count is unsupported.
