// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// US4 placeholder. Receives protocol-shaped training payloads so episode control can
    /// report rewards and episode status at the contract layer without gRPC or protobuf.
    /// A Gym connector will implement this (or replace the call site) and serialize to
    /// <c>GymService.UpdateState</c>.
    /// </summary>
    public interface ITrainingStateSink
    {
        void SubmitState(AgentState state);

        void SubmitInitialState(InitialAgentState state);

        /// <summary>
        /// Multi-agent step payload. Keys are agent ids; <see cref="AgentState"/> itself has none.
        /// </summary>
        void SubmitStates(IReadOnlyDictionary<string, AgentState> states);

        /// <summary>
        /// Multi-agent reset payload. Keys are agent ids.
        /// </summary>
        void SubmitInitialStates(IReadOnlyDictionary<string, InitialAgentState> states);
    }
}
