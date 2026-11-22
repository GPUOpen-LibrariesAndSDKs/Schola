// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Per-step training payload. Matches proto <c>AgentState</c> / Unreal <c>FAgentState</c>
    /// and the Python unpack <c>(obs, reward, terminated, truncated, info)</c>.
    /// </summary>
    public sealed class AgentState
    {
        public Point Observations { get; set; }

        public float Reward { get; set; }

        public bool Terminated { get; set; }

        public bool Truncated { get; set; }

        public Dictionary<string, string> Info { get; set; } = new Dictionary<string, string>();

        /// <summary>
        /// Copy used when a finished agent keeps reporting its terminal state. Callers must
        /// not be able to mutate the stored terminal flags or info through a returned map.
        /// </summary>
        public AgentState Copy()
        {
            return new AgentState
            {
                Observations = Observations,
                Reward = Reward,
                Terminated = Terminated,
                Truncated = Truncated,
                Info =
                    Info == null
                        ? new Dictionary<string, string>()
                        : new Dictionary<string, string>(Info),
            };
        }
    }
}
