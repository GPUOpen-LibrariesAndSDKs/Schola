# Training and inference demo

Import this sample and open **Scenes/ScholaTrainingDemo.unity**. Enable **Run In Background**, enter Play Mode, and run from the repository's Python environment:

```powershell
python -m pip install -e "./Resources/python[sb3]"
schola unity train --unity-port 8001 --seed 0 --timesteps 10000 --checkpoint ckpt/unity_ppo --export-onnx
```

The environment rewards progress, terminates at the target, truncates after 128 steps, and resets starting poses. Actions are three world-space displacements bounded to [-0.1, 0.1]. Training advances Transform steps rather than physics intervals. A short run does not guarantee convergence.

Stop Play Mode and Python, open **Scenes/ScholaInferenceDemo.unity**, and press Play to run the bundled ONNX policy on CPU. The overlay announces arrival and stops the runner. To use your output, import `ckpt/unity_ppo.onnx` into Assets and assign it to **Policy Runner > Model**. Do not assign the `.zip` checkpoint. The inference scene has no Training Server.

Builders: **Tools > Schola > Create Training Demo Scene** and **Create Inference Demo Scene**. Duplicate scenes before adapting; reimporting can overwrite sample assets. See **Models/README.md** for model provenance and `Documentation~/setup.md` in the package for the full installation/adaptation guide and troubleshooting.
