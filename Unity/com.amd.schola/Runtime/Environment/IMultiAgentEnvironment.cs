// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Multi-agent training environment contract, mirroring Unreal
    /// <c>IMultiAgentScholaEnvironment</c> with C# return values instead of out-params.
    /// Agent identity is the dictionary key. <see cref="AgentState"/> has no agent-id field.
    /// </summary>
    public interface IMultiAgentEnvironment
    {
        IReadOnlyDictionary<string, InteractionDefinition> Initialize();

        void Seed(int seed);

        void SetOptions(IReadOnlyDictionary<string, string> options);

        IReadOnlyDictionary<string, InitialAgentState> Reset();

        IReadOnlyDictionary<string, AgentState> Step(IReadOnlyDictionary<string, Point> actions);
    }
}
