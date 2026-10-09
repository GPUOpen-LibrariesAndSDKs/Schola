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
this file does **not** map Unreal sources. It names the ONNX inputs and
outputs using LeRobot batch keys from ``config.json``
``input_features`` / ``output_features``.

Nested mappings are flattened on ``.``, and the root name is kept:

* ``observation.state`` stays ``observation.state``
* ``observation.images.wrist`` stays ``observation.images.wrist``
* ``action`` stays ``action``

Those dotted strings are the ONNX tensor names **and** the keys passed to
``policy.predict_action_chunk``. NNE binds buffers by exact tensor name,
so Unreal ``Define()`` dict keys must be these same strings. Nested Unreal
spaces named ``state`` / ``images`` / ``wrist`` will not match
``observation.state``. Do not rename YAML keys to Unreal-only names; the
policy still requires LeRobot feature names.

Each leaf must include ``type`` and ``shape``. ``box`` also requires
``dtype``.

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

* inputs: ``observation.state``, ``observation.images.wrist``,
  ``observation.images.front``
* output: ``action`` with shape ``[batch, 6]`` (one command per tick)

Copy shapes and dtypes from the policy's ``config.json``. The graph is the
policy only: feed tensors ``predict_action_chunk`` expects (LeRobot batch
keys). It does not apply LeRobot's preprocessor or postprocessor.

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

The same conversion is available in Python:

.. code-block:: python

   from lerobot_env_schola.export import convert_pretrained_to_onnx

   convert_pretrained_to_onnx(
       pretrained_path="outputs/train/act_so101/checkpoints/last/pretrained_model",
       export_path="Content/Schola/Models/act_so101.onnx",
       spaces_path="spaces.yaml",
   )

Writing the ``.onnx`` file into a Content folder that Unreal auto-reimports
is the usual next step. See :doc:`setting_up_inference` for NNE import,
``UNNEPolicy``, and the stepper.

What the ONNX Graph Contains
----------------------------

The wrapper packs named observation tensors into a LeRobot batch and calls
``predict_action_chunk``. Each tick emits **one** action of shape
``[batch, action_dim]``, matching the YAML ``action`` Box and Unreal's
action space. The graph queries the policy every tick (LeRobot
``n_action_steps=1``). It does not keep an action-chunk queue.

Without temporal ensembling
   ``temporal_ensemble_coeff`` is ``None``. The graph is stateless. The
   ``action`` output is ``chunk[:, 0]``.

With temporal ensembling
   ``temporal_ensemble_coeff`` is set (LeRobot ACT's exponential
   ensemble). The graph is stateful. Each tick outputs one command plus
   leftover ensemble buffers that must be fed back on the next tick:

   * ``state_in_ensembled_actions`` /
     ``state_out_ensembled_actions``
   * ``state_in_ensembled_actions_count`` /
     ``state_out_ensembled_actions_count``

   Zero counts mean episode start (the same idea as LeRobot's empty
   ensemble buffers). Schola's NNE policy already wires ``state_in_*`` /
   ``state_out_*`` when the imported model exposes them.

The ensemble math matches LeRobot's ``ACTTemporalEnsembler.update``. Do
not change ``chunk_size`` or ``temporal_ensemble_coeff`` at export time;
they are read from the pretrained config.

Troubleshooting
---------------

``TypeError`` mentioning a policy class other than ``ACTPolicy``
   Only ACT export is implemented.

``ValueError: unsupported space type``
   A YAML leaf used a ``type`` other than ``box``, ``discrete``,
   ``multi_discrete``, or ``multi_binary``.

ONNX input names do not match Unreal observations
   Tensor names are the dotted LeRobot keys, including the
   ``observation.`` prefix. Set Unreal ``Define()`` keys to those
   strings. Changing YAML names would break ``predict_action_chunk``.
