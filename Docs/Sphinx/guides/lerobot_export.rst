.. Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

Exporting LeRobot Policies to ONNX
==================================

The ``lerobot_env_schola`` plugin can export a pretrained LeRobot policy to
ONNX in Schola's inference layout. Unreal then runs the file through NNE
without a Python process. This is the LeRobot counterpart of
``schola sb3 export`` and ``schola rllib export``.

Export currently supports LeRobot **ACT** (``ACTPolicy``). Other LeRobot
algorithms are rejected. The policy is loaded on CPU for tracing.

Installation
------------

Use the same environment as :doc:`lerobot_evaluation`: Python 3.12, LeRobot
0.6.1 or later (but earlier than 0.7), and Schola 2.1 or later.

.. code-block:: bash

   pip install "lerobot[evaluation,dataset]>=0.6.1,<0.7"
   pip install -e "./Resources/python"
   pip install -e "./Resources/python/lerobot_env_schola"

That install provides the ``lerobot-to-onnx`` command.

Spaces YAML
-----------

Schola needs a YAML file that describes every observation tensor and the
single action vector the policy expects. Unlike :doc:`lerobot_evaluation`,
this file does **not** map Unreal sources.

``observation`` and ``action`` are configuration sections. Nested keys below
them are flattened on ``.`` to form ONNX tensor names:

* ``observation`` → ``state`` becomes ``state``
* ``observation`` → ``images`` → ``wrist`` becomes ``images.wrist``
* a root action remains ``action``

The LeRobot adapter maps observation tensor names to LeRobot batch keys
internally, for example ``state`` to ``observation.state``. Every observation
must therefore be nested under its feature name, and every input feature in
the policy's ``config.json`` needs an observation; export fails otherwise.
NNE binds buffers by exact ONNX tensor name, so Unreal ``Define()`` dict keys
must use the unprefixed names from the YAML structure.

Mappings without a ``type`` are dictionary containers. Each leaf must include
``type`` and ``shape``. ``box`` also requires ``dtype``.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - ``type``
     - ``shape``
     - Notes
   * - ``box``
     - Tensor axes, for example ``[14]`` or ``[3, 128, 128]``
     - ``dtype`` is required (``float32``, ``uint8``, …). Bounds are not
       used at inference.
   * - ``discrete``
     - ``[n]`` class count
     - The ONNX tensor is a scalar integer, not a vector of length ``n``.
   * - ``multi_discrete``
     - ``nvec`` (one class count per axis)
     - Same meaning as Gymnasium ``MultiDiscrete``.
   * - ``multi_binary``
     - ``[n]``
     - Same meaning as Gymnasium ``MultiBinary``.

Example
-------

The following YAML matches a typical ACT checkpoint that takes a joint
state, two cameras, and a continuous action vector. Save it as
``spaces.yaml`` next to the command you will run.

.. code-block:: yaml

   observation:
     state:
       type: box
       shape: [6]
       dtype: float32
     images:
       wrist:
         type: box
         shape: [3, 128, 128]
         dtype: uint8
       front:
         type: box
         shape: [3, 128, 128]
         dtype: uint8
   action:
     type: box
     shape: [6]
     dtype: float32

That produces these ONNX names:

* inputs: ``state``, ``images.wrist``, ``images.front``
* outputs: ``action`` with shape ``[1, n_action_steps, 6]`` (the actions played from each chunk),
  and ``action_processor.ACTActionProcessor``, which tells Unreal how to play them

The YAML ``action`` describes a single action. The chunk dimension comes from
the policy, see `What the ONNX Graph Contains`_.

Copy shapes and dtypes from the policy's ``config.json``. Inputs use the
corresponding LeRobot features that ``predict_action_chunk`` expects, except that
images declared as ``uint8`` take raw ``0``–``255`` pixels and are scaled to
``[0, 1]`` inside the graph, as during training. Export loads the
saved normalizer and unnormalizer processor steps and embeds LeRobot's
built-in ``MEAN_STD``, ``MIN_MAX``, ``QUANTILES``, or ``QUANTILE10``
transformations in the graph. Other processor steps are not embedded.

Export the Policy
-----------------

Point the CLI at a local pretrained directory (``config.json`` and
``model.safetensors``) or a Hugging Face Hub id:

.. code-block:: bash

   lerobot-to-onnx <PRETRAINED_PATH> <OUTPUT.onnx> spaces.yaml

For example, after training an ACT policy locally:

.. code-block:: bash

   lerobot-to-onnx \
       outputs/train/act_so101/checkpoints/last/pretrained_model \
       Content/Schola/Models/act_so101.onnx \
       spaces.yaml

Add ``--action-processor act`` or ``--action-processor temporal_ensemble`` to choose how Unreal
plays the action chunks, see `What the ONNX Graph Contains`_.

The same conversion is available in Python:

.. code-block:: python

   from lerobot_env_schola.export import convert_pretrained_to_onnx

   convert_pretrained_to_onnx(
       pretrained_path="outputs/train/act_so101/checkpoints/last/pretrained_model",
       export_path="Content/Schola/Models/act_so101.onnx",
       spaces_path="spaces.yaml",
       action_processor="act",
   )

Writing the ``.onnx`` file into a Content folder that Unreal auto-reimports
is the usual next step. See :doc:`setting_up_inference` for NNE import,
``UNNEPolicy``, and the stepper.

What the ONNX Graph Contains
----------------------------

The wrapper packs named observation tensors into a LeRobot batch, normalizes
them, calls ``predict_action_chunk``, and unnormalizes the result. The graph
is stateless and does not keep an action queue or a temporal ensemble.
Every input and the action output have a fixed batch size of 1, because ACT cannot be
exported with a dynamic batch dimension. ``UNNEPolicy`` detects the chunk output
and plays one action per step; see the action chunking section of :doc:`setting_up_inference`.

``--action-processor`` (``action_processor`` in Python) sets how Unreal plays each chunk:

* ``act``: ``action`` is the first ``n_action_steps`` actions of each chunk,
  ``[1, n_action_steps, action_dim]``, which Unreal plays in order before predicting again, like
  LeRobot's action queue.
* ``temporal_ensemble``: ``action`` is the whole chunk, ``[1, chunk_size, action_dim]``. Unreal
  runs the model every step and plays a weighted average of every prediction for the current step,
  like LeRobot's temporal ensemble. The policy must set ``temporal_ensemble_coeff``, which is
  exported as an output of the same name.

Without the flag, export uses ``temporal_ensemble`` if the policy sets ``temporal_ensemble_coeff``,
and ``act`` otherwise. To change how chunks are played, export again. The export also adds an
output named after the processor, such as ``action_processor.ACTActionProcessor``, that tells
Unreal which one to use.

Troubleshooting
---------------

``TypeError`` mentioning a policy class other than ``ACTPolicy``
   Only ACT export is implemented.

``ValueError: Spaces config is missing top-level [...] sections``
   The YAML must have both an ``observation`` and an ``action`` section at
   the top level.

``ValueError: The spaces YAML is missing the policy's input features [...]``
   Add an observation for each listed feature, nested under its name without
   the ``observation.`` prefix. This also happens when the YAML
   ``observation`` section is a single leaf; nest it under its feature name,
   for example ``observation`` → ``state``.

``ValueError: Unknown action processor [...]``
   ``--action-processor`` must name a registered processor, such as ``act`` or
   ``temporal_ensemble``. The error lists every registered name.

``ValueError`` mentioning ``temporal_ensemble_coeff``
   ``temporal_ensemble`` needs the policy's ``temporal_ensemble_coeff``, and the coefficient must
   give finite weights over the chunk. Fix it in the policy's ``config.json``, or export with
   ``--action-processor act``.

``ValueError: unsupported space type``
   A YAML leaf used a ``type`` other than ``box``, ``discrete``,
   ``multi_discrete``, or ``multi_binary``.

ONNX input names do not match Unreal observations
   Tensor names are the dotted keys below the YAML ``observation`` section,
   without an ``observation.`` prefix. Set Unreal ``Define()`` keys to those
   strings.
