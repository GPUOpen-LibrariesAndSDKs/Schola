# Training Demo (US4)

1. Install Schola's Python package from the repository root:
   `python -m pip install -e "./Resources/python[sb3]"`.
2. Import **Training Demo** from the AMD Schola package in Package Manager.
3. Select **Tools > Schola > Create Training Demo Scene** and save the scene.
4. Enable **Run In Background** in Player settings so training continues when the
   Python terminal has focus. Enter Play Mode.
5. In the Python environment where Schola was installed, run:
   `schola unity train --timesteps 10000 --checkpoint ckpt/unity_ppo`.

The cube receives three world-space displacements per action, each bounded to
[-0.1, 0.1]. Observations are `agent_position` and `target_position`. Progress
toward the target earns reward, reaching it earns a bonus, and each step has a
small cost. Success terminates the episode; 128 steps truncate it. Schola resets
the agent and preserves the terminal observation for the trainer. Training saves
an SB3 checkpoint; the sample does not load that checkpoint for Unity inference.

The sample has no deterministic controller. Unity waits for Python actions;
movement and resets may look fast or irregular during training. This is a
Transform-based example; each Python step invokes `ScholaEnvironment.Step`
once and does not simulate a physics interval. A short run demonstrates training
and checkpointing, not guaranteed convergence.

Default Unity port: **8001**, bound to loopback only. Run only one trainer against
that port. Change the Training Server's Port and pass `--unity-port` to use another
instance. See `Documentation~/training.md` in the package for the complete guide.
