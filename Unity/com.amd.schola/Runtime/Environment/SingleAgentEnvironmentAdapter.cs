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
    public sealed class SingleAgentEnvironmentAdapter : IMultiAgentEnvironment
    {
        readonly IEnvironment m_Environment;

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
            return new Dictionary<string, InitialAgentState>
            {
                [ScholaEnvironment.DefaultAgentId] = m_Environment.Reset(),
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
                    throw new ArgumentException(
                        $"Unknown agent id '{pair.Key}'.",
                        nameof(actions)
                    );
                }
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

            return new Dictionary<string, AgentState>
            {
                [ScholaEnvironment.DefaultAgentId] = m_Environment.Step(action),
            };
        }
    }
}
