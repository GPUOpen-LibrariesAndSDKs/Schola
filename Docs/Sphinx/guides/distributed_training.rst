Distributed Training with Ray and Kubernetes
=============================================

Schola supports distributed reinforcement learning across multiple machines using `Ray <https://www.ray.io/>`_ and `Kubernetes <https://kubernetes.io/>`_.
This guide covers the architecture, configuration, and deployment of Schola training workloads on a Kubernetes cluster with KubeRay.

.. note::
   Distributed training currently requires **Ray RLlib**. For single-machine training with any framework, see :doc:`running_schola`. For general CLI usage, see :doc:`running_from_cli`.

Use this guide when:

* You need more environment instances than a single machine can run.
* You want to separate GPU-based learning from CPU-bound environment stepping.
* You need reproducible, scalable infrastructure managed by Kubernetes.


Architecture Overview
---------------------

Distributed training in Schola separates three roles across the cluster:

* **Head node** -- Runs the training driver script. Connects to the Ray cluster, performs one-shot space discovery against an Unreal Engine (UE) instance, and submits the PPO training job. It does not perform learning or environment stepping.
* **Learner node(s)** -- Each runs an RLlib learner process. Receives batched experience from workers, computes gradients, and updates model weights. Multiple learners perform data-parallel training across GPUs. In GPU-enabled clusters these nodes would hold the GPUs.
* **Worker nodes** -- Each worker runs a :py:class:`~schola.rllib.env_runner.ScholaEnvRunner` that connects to a UE instance via gRPC and collects experience. Workers do not train the model; they only step the environment and send data to the learner(s).

Each UE instance is managed externally (by Kubernetes, Docker, or a manual process) and is represented in Python by the :py:class:`~schola.core.simulators.external_simulator.ExternalSimulator` simulator class.


Key Components
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Purpose
   * - :py:class:`~schola.core.simulators.external_simulator.ExternalSimulator`
     - No-op simulator for UE instances whose lifecycle is managed outside Python (e.g. by Kubernetes).
   * - :py:class:`~schola.core.protocols.protobuf.grpc_protocol.GrpcProtocol`
     - gRPC client protocol. Supports ``credential_mode="insecure"`` for in-cluster networking.
   * - :py:class:`~schola.rllib.env_runner.ScholaEnvRunner`
     - Custom RLlib ``EnvRunner`` that constructs a Schola environment from serialized ``env_config``. Supports ``{worker_index}`` URL templates for per-worker routing.


Prerequisites
~~~~~~~~~~~~~

* Schola installed with RLlib support (see :doc:`setup_schola`):

  .. code-block:: bash

     pip install -e <path-to-schola>/Resources/python[rllib]

* A Kubernetes cluster (or `minikube <https://minikube.sigs.k8s.io/>`_ for local testing)
* `KubeRay operator <https://docs.ray.io/en/latest/cluster/kubernetes/index.html>`_ installed on the cluster
* A packaged Unreal Engine executable with Schola enabled
* Docker images for Ray workers and UE instances
* Python 3.10+ with ``schola[rllib]`` installed in the worker image


Deployment Topologies
---------------------

Schola supports two deployment patterns depending on how UE is co-located with the Ray worker.

Sidecar Pattern
~~~~~~~~~~~~~~~

UE runs inside the same Kubernetes pod as the Ray worker. The ``ScholaEnvRunner`` connects to ``localhost:50051`` within the pod's shared network namespace.

.. code-block:: text

   Pod (worker-0)
   ├── Container: Ray worker  ──  ScholaEnvRunner connects to localhost:50051
   └── Container: UE instance ──  gRPC server on port 50051

This pattern is simpler to configure but couples UE and Ray worker scaling.

Networked Pattern
~~~~~~~~~~~~~~~~~

UE runs in a separate pod (or on a different machine). The ``ScholaEnvRunner`` connects over the cluster network using a Kubernetes Service name.

.. code-block:: text

   Pod (worker-1)                     Pod (ue-1)
   └── Container: Ray worker  ──────► └── Container: UE instance
       url="ue-{worker_index}" → ue-1      gRPC server on port 50051

The training driver sets ``url="ue-{worker_index}"`` in ``env_config``; ``ScholaEnvRunner`` expands the template on each worker using its ``EnvContext.worker_index``.
This pattern allows independent scaling of UE and Ray resources and supports mounting UE executables via shared volumes to avoid image rebuilds.


Configuring the Training Driver
-------------------------------

The training driver runs on the head node. It performs space discovery, builds the PPO configuration, and submits the job to Ray.

Space Discovery
~~~~~~~~~~~~~~~

Before training begins, the driver connects to one UE instance to discover observation and action spaces:

.. code-block:: python

   from schola.core.protocols.protobuf.grpc_protocol import GrpcProtocol
   from schola.core.simulators.external_simulator import ExternalSimulator
   from schola.rllib.env import RayVecEnv

   protocol = GrpcProtocol(
       url="ue-disc",        # Kubernetes Service name for the discovery UE instance
       port=50051,
       credential_mode="insecure"
   )
   simulator = ExternalSimulator()
   tmp_env = RayVecEnv(protocol, simulator)
   agent_names = {aid: aid for aid in tmp_env.possible_agents}
   tmp_env.close()

PPO Configuration
~~~~~~~~~~~~~~~~~

The ``env_config`` dictionary is serialized and sent to every remote worker. Workers reconstruct the protocol and simulator from these values:

.. code-block:: python

   from ray.rllib.algorithms.ppo import PPOConfig
   from ray.rllib.policy.policy import PolicySpec
   from schola.rllib.connectors import schola_env_to_module_flatten_connector
   from schola.rllib.env_runner import ScholaEnvRunner

   config = (
       PPOConfig()
       .api_stack(
           enable_rl_module_and_learner=True,
           enable_env_runner_and_connector_v2=True,
       )
       .environment(
           env_config={
               "protocol": GrpcProtocol,
               "protocol_args": {
                   "url": "ue-{worker_index}",   # expanded per-worker: ue-1, ue-2, ...
                   "port": 50051,
                   "credential_mode": "insecure",
                   "environment_start_timeout": 120,
               },
               "port_offset_mode": "fixed",
               "simulator": ExternalSimulator,
               "simulator_args": {},
           },
       )
       .framework("torch")
       .env_runners(
           env_runner_cls=ScholaEnvRunner,
           num_env_runners=NUM_WORKERS,           # Y env-runner workers
           env_to_module_connector=schola_env_to_module_flatten_connector,
           custom_resources_per_env_runner={"ue_worker": 1},
       )
       .multi_agent(
           policies={aid: PolicySpec() for aid in agent_names},
           policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
       )
       .training(lr=3e-4, train_batch_size=512)
       .learners(num_learners=NUM_LEARNERS, num_gpus_per_learner=0)  # X learners
   )

Both ``NUM_LEARNERS`` and ``NUM_WORKERS`` are read from environment variables in the Docker simulation
scripts, defaulting to 1 and 2 respectively.

.. note::
   ``port_offset_mode="fixed"`` tells Schola not to add the worker index to the port number.
   The ``{worker_index}`` placeholder in the URL is expanded by each ``ScholaEnvRunner`` using its ``EnvContext.worker_index`` at environment construction time.
   For the sidecar pattern, use ``url="localhost"`` (no template needed, since every worker connects to its own pod).

.. note::
   ``custom_resources_per_env_runner`` pins env runners to nodes that advertise the ``ue_worker`` resource. This prevents Ray from scheduling an environment runner on the learner node, which has no UE instance.


Resource Placement with Custom Resources
-----------------------------------------

Ray schedules actors wherever CPU resources are available. In a dedicated-learner topology, this can cause env runners to land on the learner node (which has no UE server). Schola solves this with **Ray custom resources**.

Worker nodes register a custom resource when joining the cluster:

.. code-block:: bash

   # On each worker node
   ray start --address="$HEAD_ADDR" --num-cpus=1 --resources='{"ue_worker": 1}'

   # On the learner node (no custom resource)
   ray start --address="$HEAD_ADDR" --num-cpus=2

The training driver then requests that resource for each env runner:

.. code-block:: python

   .env_runners(
       custom_resources_per_env_runner={"ue_worker": 1},
   )

For a cluster with X learners and Y workers, resource accounting looks like this:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Node Type
     - CPUs
     - Scheduled Actors
   * - Head (1 node)
     - 0
     - Training driver only (no Ray-scheduled work)
   * - Learner (X nodes)
     - 2 each
     - PPO trial actor on one node; one Learner process per node
   * - Worker (Y nodes)
     - 1 each
     - One ScholaEnvRunner per node (connects to its paired UE instance)

The PPO trial actor requires 1 CPU and is placed on whichever learner node has room.
With X learners at 2 CPUs each, the first learner hosts the PPO trial + one Learner process,
and subsequent learner nodes each host one Learner process.


Per-Worker Routing
------------------

Workers discover their UE instance through the ``env_config`` dictionary, not through environment variables.
All connection details are set in the training driver on the head node and serialized to every worker via RLlib's ``EnvContext``.

**URL templates.** Use ``{worker_index}`` in the ``url`` field (e.g. ``"ue-{worker_index}"``). ``ScholaEnvRunner`` expands this on each worker using its ``EnvContext.worker_index`` — so worker 1 connects to ``ue-1``, worker 2 to ``ue-2``, etc.

**Port offset.** With ``port_offset_mode="per_worker"`` (the default), the base port is incremented by ``worker_index``. With ``port_offset_mode="fixed"``, every worker uses the same port — appropriate when each worker has its own network namespace (K8s pods, Docker containers).

The Docker Compose scripts read a few environment variables at the *script level* on the head node:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Purpose
   * - ``DISCOVERY_GRPC_URL``
     - Hostname of the UE instance used by the head for space discovery.
   * - ``DISCOVERY_GRPC_PORT``
     - Port for the discovery UE instance.
   * - ``NUM_WORKERS``
     - Number of env-runner workers (passed to the training driver, default ``2``).
   * - ``NUM_LEARNERS``
     - Number of dedicated learner processes (passed to the training driver, default ``1``).


Running from the CLI
--------------------

Schola's CLI supports the ``external`` simulator subcommand for environments managed outside Python. The ``external`` subcommand follows the same nesting pattern as other simulators (see :doc:`running_from_cli`):

.. code-block:: bash

   schola rllib train ppo external \
       --credential-mode insecure \
       --port-offset-mode fixed \
       --url ue-service --port 50051

.. note::
   When using the ``external`` simulator, Schola does not start or stop any UE process. You are responsible for ensuring the UE instance is running and reachable at the configured URL and port.


Local Simulation with Docker Compose
-------------------------------------

The ``docker/`` directory contains two Docker Compose files (``docker/cluster-config.yml`` and ``docker/cluster-config-networked.yml``) that simulate the Kubernetes topology on a single machine using mock UE gRPC servers; the supporting Dockerfile, entrypoint scripts, and training drivers live under ``docker/cluster-sim/``.

Sidecar Setup
~~~~~~~~~~~~~

Each worker container runs both a Ray worker and a mock UE server. From the repository root:

.. code-block:: bash

   docker compose -f docker/cluster-config.yml build
   docker compose -f docker/cluster-config.yml up                    # 1 learner, 2 workers (default)
   docker compose -f docker/cluster-config.yml up --scale learner=2  # 2 learners, 2 workers

Networked Setup
~~~~~~~~~~~~~~~

UE servers run in separate containers, communicating over the Docker network. From the repository root:

.. code-block:: bash

   docker compose -f docker/cluster-config-networked.yml up               # 1 learner
   docker compose -f docker/cluster-config-networked.yml up --scale learner=3  # 3 learners

Both setups include dedicated learner container(s) and use custom resources for correct placement.
The networked setup volume-mounts scripts from the host, so you can iterate on the training driver without rebuilding the image.

Set the ``NUM_LEARNERS`` environment variable to match the ``--scale`` value so the training driver
creates the correct number of RLlib learner processes.


Scaling
-------

Learners and workers can be scaled independently.

Scaling Workers (more environment throughput)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add more UE instances (pods or containers), named to match Ray worker indices (e.g. ``ue-3``, ``ue-4``).
2. Add corresponding Ray worker pods that register the ``ue_worker`` custom resource.
3. Set ``NUM_WORKERS`` to match the new count. The ``{worker_index}`` URL template automatically routes each new worker to its UE instance.

The learner and head nodes do not need to change.

Scaling Learners (faster gradient computation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple learners perform **data-parallel training**: the experience batch is sharded across learners,
each computes gradients independently, and the results are all-reduced. This is most useful when each
learner has a GPU.

1. Add more learner nodes to the cluster (each with ``LEARNER_CPUS`` set appropriately).
2. Set ``NUM_LEARNERS`` to match the new count.

Worker and UE nodes do not need to change.

In the Docker Compose simulation, you can scale learners directly:

.. code-block:: bash

   # 3 learners, 2 workers (sidecar)
   NUM_LEARNERS=3 docker compose -f docker/cluster-config.yml up --scale learner=3

   # 3 learners, 2 workers (networked)
   NUM_LEARNERS=3 docker compose -f docker/cluster-config-networked.yml up --scale learner=3

.. warning::
   ``--scale learner=N`` tells Compose to run N replicas of the learner container.
   ``NUM_LEARNERS=N`` tells the training driver to configure RLlib with N learner processes.
   Both values **must** match — a mismatch will cause training to hang waiting for learners that don't exist, or leave learner containers idle.


See Also
--------

* :doc:`running_schola` -- Single-machine training with any framework
* :doc:`running_from_cli` -- CLI subcommand reference and common options
* `Ray RLlib documentation <https://docs.ray.io/en/latest/rllib/index.html>`_
* `KubeRay operator guide <https://docs.ray.io/en/latest/cluster/kubernetes/index.html>`_
* The ``docker/`` directory (Compose files at ``docker/cluster-config*.yml``, image sources under ``docker/cluster-sim/``) for ready-to-run Docker Compose examples
