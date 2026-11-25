// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Profile for the engine-free scaling benchmark (ScalingBenchmark.cpp), injected with
// -DSCHOLA_RLTOOLS_PROJECT_PROFILES. A mid-sized NPC: 16 observations, 4 actions, 64-wide 3-layer
// actor and critics, batch 32, 50k replay.
// Identifier, observations, actions, hidden width, batch, replay, PPO rollout.
SCHOLA_RLTOOLS_PROFILE("scale-16x4", 16, 4, 64, 32, 50000, 64, FAlgorithmParameters)
