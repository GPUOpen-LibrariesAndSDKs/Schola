Make sure pip is updated before installing the schola python package from `Plugins/Schola/Resources/python`.

.. code-block:: bash
    
    python -m pip install --upgrade pip
    pip install ./Plugins/Schola/Resources/python[all]

``[all]`` installs every training backend. To install a smaller set, pick only the extras you need:

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Extra
      - Installs
    * - ``sb3``
      - Stable Baselines 3, for ``schola sb3 train``.
    * - ``rllib``
      - Ray RLlib, for ``schola rllib train`` and ``schola rllib eval``.
    * - ``minari``
      - Minari, for recording demonstrations with ``schola minari collect``.
    * - ``offline``
      - RLlib, Minari, and the msgpack codecs. Use this to train ``bc`` and ``marwil`` on a recorded dataset (see :doc:`/guides/imitation_learning`).

.. note:: 
        
    Schola installs the cpu version of pytorch by default, to install other versions of pytorch follow the instructions at `Pytorch Get Started <https://pytorch.org/get-started/locally/>`_.

.. note:: 

    To install Pytorch with ROCm on Linux, we recommend following the guide at `Install Pytorch for Radeon GPUs <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html#>`_.