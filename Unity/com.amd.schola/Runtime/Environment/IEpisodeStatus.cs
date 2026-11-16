// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

namespace AMD.Schola
{
    /// <summary>
    /// Per-step episode-end hook. Called after act/reward and before observe.
    /// <c>terminated</c> and <c>truncated</c> are independent, matching proto/Python.
    /// </summary>
    public interface IEpisodeStatus
    {
        /// <summary>
        /// Clears per-episode status state (step counters, latched flags). Called by
        /// <see cref="ScholaEnvironment.Reset"/> and <see cref="ScholaMultiAgentEnvironment.Reset"/>
        /// after restoring snapshot poses and before collecting initial observations.
        /// </summary>
        void OnEpisodeReset();

        void GetStatus(out bool terminated, out bool truncated);
    }
}
