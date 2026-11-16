// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola
{
    /// <summary>
    /// Per-step reward hook. Called after <see cref="IAgent.Act"/> and before
    /// <see cref="IAgent.Observe"/>. The component reads scene state the same way a sensor does.
    /// </summary>
    public interface IReward
    {
        /// <summary>
        /// Clears per-episode reward state. Called by <see cref="ScholaEnvironment.Reset"/>
        /// and <see cref="ScholaMultiAgentEnvironment.Reset"/> after restoring snapshot poses
        /// and before collecting initial observations.
        /// </summary>
        void OnEpisodeReset();

        float ComputeReward();
    }
}
