// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Chase profiles for the engine-free adaptation harness (ChaseAdaptation.cpp), injected with
// -DSCHOLA_RLTOOLS_PROJECT_PROFILES. Shape matches chase_sim: 10 observations, 2 actions, 64-wide
// 3-layer actor and critics, batch 32. Only the replay capacity differs between them.
// Identifier, observations, actions, hidden width, batch, replay, PPO rollout.
SCHOLA_RLTOOLS_PROFILE("chase-r10k", 10, 2, 64, 32, 10000, 64, FAlgorithmParameters)
SCHOLA_RLTOOLS_PROFILE("chase-r20k", 10, 2, 64, 32, 20000, 64, FAlgorithmParameters)
SCHOLA_RLTOOLS_PROFILE("chase-r60k", 10, 2, 64, 32, 60000, 64, FAlgorithmParameters)
