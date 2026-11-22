// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Multi-agent training environment contract, mirroring Unreal
    /// <c>IMultiAgentScholaEnvironment</c> with C# return values instead of out-params.
    /// Agent identity is the dictionary key. <see cref="AgentState"/> has no agent-id field.
    /// </summary>
    /// <remarks>
    /// The set of agent ids is fixed by <see cref="Initialize"/>. <see cref="Reset"/> and
    /// <see cref="Step"/> return one entry for every one of those ids, every time.
    /// The environment is complete when every agent has reported terminated or truncated;
    /// one finished agent does not end the episode for the others.
    /// </remarks>
    public interface IMultiAgentEnvironment
    {
        /// <summary>Prepares the environment and returns one definition per agent id.</summary>
        IReadOnlyDictionary<string, InteractionDefinition> Initialize();

        void Seed(int seed);

        void SetOptions(IReadOnlyDictionary<string, string> options);

        /// <summary>
        /// Starts a new episode and returns one initial state per agent id. Every agent,
        /// including one that finished in the previous episode, is live again.
        /// </summary>
        IReadOnlyDictionary<string, InitialAgentState> Reset();

        /// <summary>
        /// Advances every live agent by one step and returns one state per agent id.
        /// </summary>
        /// <param name="actions">
        /// One action for every live agent. An agent that already reported terminated or
        /// truncated may be omitted; if an action is present for that agent, it is live
        /// again and stepped (Unreal <c>TScholaEnvironment::Step</c>). An id that
        /// <see cref="Initialize"/> did not return, or a missing action for a live agent,
        /// is an error and must leave the scene unchanged.
        /// </param>
        /// <returns>
        /// A fresh state for every live agent. A finished agent with no action keeps
        /// reporting the state from the step on which it finished.
        /// </returns>
        IReadOnlyDictionary<string, AgentState> Step(IReadOnlyDictionary<string, Point> actions);
    }
}
