Task 3 - Reach with Random Target, Random Location
==================================================
.. image:: /_static/examples/xarm_5_example/task3/task3.gif
   :alt: Task 3

For rewards, we are going to change the distance-based reward. Our new mechanism will only reduce the penalty the agent receives if it achieved better progress this episode than it did on the last. Additionally, we are going to tune a couple of other parameters.

In particular, within *Reward*, set:

- Step Penalty: -0.1
- Use Simple Rewards: False
- Use Best Progress Reward: True
- Best Progress Reward Scale: 0.005
- Normalize Best Progress by Initial Distance: True
- Best Progress Improvement Epsilon: 0.1

We also reduce the number of target blocks to 2 to make the task a little easier for the agent to learn. Adjust the following options with *Config*:

- Num Blocks to Spawn: 2
- Randomize Color Permutataion: False
- Randomize Target Color: True
- Use Radial Spawn: True
- Spawn Min Radius: 70.0
- Spawn Max Radius: 120.0
- Use Radial Spawn Arc: True
- Radial Spawn Arc Center Degrees: 0.0
- Radial Spawn Arc Degrees: 110.0
- Resample Radial on Reset: True
- Use Goal Color Override: False

You should be able to see the target as well as the location change now

.. image:: /_static/examples/xarm_5_example/task3/task3-setup.png
   :alt: Task 3 setup

.. image:: /_static/examples/xarm_5_example/task3/task-3-setup-2.png
   :alt: Task 3 setup alt

.. image:: /_static/examples/xarm_5_example/task3/task3-setup-3.png
   :alt: Task 3 setup alt 2

Like Task 2, while the play in editor is running, launch schola with the following command. We are going to continue training from the model trained in Task 2. It is going to take quite a few more timesteps to have an accurate model for this task:

.. code-block:: bash

   schola sb3 train sac editor --resume-from .\ckpt\task2\sac_final.zip --enable-checkpoints --checkpoint-dir .\ckpt\task3 --save-final-policy --port 8000 --timesteps 1000000 --pbar