// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;

namespace AMD.Schola
{
    /// <summary>
    /// Presents one <see cref="IEnvironment"/> as an <see cref="IMultiAgentEnvironment"/>
    /// under <see cref="ScholaEnvironment.DefaultAgentId"/>. Matches Unreal's single-agent
    /// adapter, which injects the key <c>SingleAgent</c> into the multi-agent maps.
    /// </summary>
    /// <remarks>
    /// Follows the <see cref="IMultiAgentEnvironment"/> rule for finished agents: once the
    /// wrapped environment reports terminated or truncated it is not stepped again until
    /// <see cref="Reset"/> is called on this adapter, and its terminal state is repeated.
    /// </remarks>
    public sealed class SingleAgentEnvironmentAdapter : IMultiAgentEnvironment
    {
        readonly IEnvironment m_Environment;
        AgentState m_Finished;

        public SingleAgentEnvironmentAdapter(IEnvironment environment)
        {
            m_Environment = environment ?? throw new ArgumentNullException(nameof(environment));
        }

        public IReadOnlyDictionary<string, InteractionDefinition> Initialize()
        {
            return new Dictionary<string, InteractionDefinition>
            {
                [ScholaEnvironment.DefaultAgentId] = m_Environment.Initialize(),
            };
        }

        public void Seed(int seed)
        {
            m_Environment.Seed(seed);
        }

        public void SetOptions(IReadOnlyDictionary<string, string> options)
        {
            m_Environment.SetOptions(options);
        }

        public IReadOnlyDictionary<string, InitialAgentState> Reset()
        {
            var initial = m_Environment.Reset();
            m_Finished = null;
            return new Dictionary<string, InitialAgentState>
            {
                [ScholaEnvironment.DefaultAgentId] = initial,
            };
        }

        public IReadOnlyDictionary<string, AgentState> Step(
            IReadOnlyDictionary<string, Point> actions
        )
        {
            if (actions == null)
            {
                throw new ArgumentNullException(nameof(actions));
            }

            foreach (var pair in actions)
            {
                if (pair.Key != ScholaEnvironment.DefaultAgentId)
                {
                    throw new ArgumentException($"Unknown agent id '{pair.Key}'.", nameof(actions));
                }
            }

            if (m_Finished != null)
            {
                return new Dictionary<string, AgentState>
                {
                    [ScholaEnvironment.DefaultAgentId] = m_Finished,
                };
            }

            if (
                !actions.TryGetValue(ScholaEnvironment.DefaultAgentId, out var action)
                || action == null
            )
            {
                throw new ArgumentException(
                    $"Missing action for agent '{ScholaEnvironment.DefaultAgentId}'.",
                    nameof(actions)
                );
            }

            var state = m_Environment.Step(action);
            if (state != null && (state.Terminated || state.Truncated))
            {
                m_Finished = state;
            }

            return new Dictionary<string, AgentState>
            {
                [ScholaEnvironment.DefaultAgentId] = state,
            };
        }
    }
}
