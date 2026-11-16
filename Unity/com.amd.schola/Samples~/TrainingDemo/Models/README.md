# Bundled reach-target policy

`reach_target.onnx` is a Schola SB3 PPO policy trained against the **Training Demo** environment through the real Unity training connection. It is supplied for installation/inference checks, not as a general navigation policy.

Regenerate with Unity 6000.3.24f1, Python 3.12, and the repository Python package installed with `[sb3]`. Open `Scenes/ScholaTrainingDemo.unity`, wait for compilation, enable Run In Background, and enter Play Mode:

```powershell
schola unity train --unity-port 8001 --seed 0 --timesteps 10000 --checkpoint ckpt/unity_ppo --export-onnx
```

Replace this model with `ckpt/unity_ppo.onnx`, retaining the existing `.meta` file to preserve scene references. Keep the checkpoint outside sample assets. Inference uses CPU, two three-float inputs (`agent_position`, `target_position`) and a three-float output (`translate`). Each action axis is bounded to [-0.1, 0.1]. Training uses 128-step PPO rollouts and may round the requested step count upward.

The bundled policy is tested from the sample's fixed starting poses. Moving the target, changing coordinate scale, or altering observations/rewards can require retraining. Training results can vary by software/runtime version; see the package's `Documentation~/validation.md` for the recorded environment and model hash.
