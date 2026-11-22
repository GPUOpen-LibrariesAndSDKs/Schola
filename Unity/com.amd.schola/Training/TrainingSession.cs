// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;

namespace AMD.Schola.Training
{
    /// <summary>
    /// Main-thread, single-environment session implementing Schola's reset modes over a
    /// map of agents. A step calls the environment's synchronous Step once; it does not
    /// advance Unity physics.
    /// </summary>
    /// <remarks>
    /// The environment is complete only when every agent has reported terminated or
    /// truncated, matching Unreal's gym connector. Until then a finished agent keeps its
    /// terminal state and needs no action.
    /// </remarks>
    public sealed class TrainingSession
    {
        readonly IMultiAgentEnvironment m_Environment;
        readonly List<string> m_AgentIds = new List<string>();
        readonly HashSet<string> m_Finished = new HashSet<string>(StringComparer.Ordinal);
        IReadOnlyDictionary<string, InteractionDefinition> m_Definitions;
        TrainingAgentDefinition[] m_EncodedDefinitions;
        bool m_Started;
        bool m_NeedsReset = true;
        int m_AutoReset;

        public TrainingSession(IMultiAgentEnvironment environment)
        {
            m_Environment = environment ?? throw new ArgumentNullException(nameof(environment));
        }

        /// <summary>
        /// Trains a single-agent environment as a one-entry map keyed by
        /// <see cref="ScholaEnvironment.DefaultAgentId"/>.
        /// </summary>
        public TrainingSession(IEnvironment environment)
            : this(new SingleAgentEnvironmentAdapter(environment)) { }

        bool AllAgentsFinished => m_Finished.Count == m_AgentIds.Count;

        public TrainingResponse Handle(TrainingRequest request)
        {
            try
            {
                if (request == null || request.version != TrainingWire.Version)
                    throw new ArgumentException(
                        TrainingWire.VersionMismatch(request?.version ?? 0)
                    );
                if (request.command == "close")
                {
                    m_Started = false;
                    m_NeedsReset = true;
                    return new TrainingResponse();
                }
                if (request.command == "start")
                {
                    if (request.autoreset < 0 || request.autoreset > 2)
                        throw new ArgumentException("Unknown autoreset mode.");
                    // Initialize once: reconnecting must not replace the original reset snapshot.
                    if (m_Definitions == null)
                        m_Definitions =
                            m_Environment.Initialize()
                            ?? throw new InvalidOperationException(
                                "Environment returned no agent definitions."
                            );
                    EncodeDefinitions();
                    m_AutoReset = request.autoreset;
                    m_Started = true;
                    m_NeedsReset = true;
                    m_Finished.Clear();
                    return new TrainingResponse();
                }
                if (!m_Started)
                    throw new InvalidOperationException("Start a training session first.");
                if (request.command == "definition")
                    return new TrainingResponse { agents = m_EncodedDefinitions };
                if (request.command == "reset")
                {
                    var options = new Dictionary<string, string>();
                    foreach (var option in request.options ?? Array.Empty<TrainingEntry>())
                        options.Add(option.key, option.value);
                    if (request.hasSeed)
                        m_Environment.Seed(request.seed);
                    m_Environment.SetOptions(options);
                    return new TrainingResponse { states = Reset() };
                }
                if (request.command != "step")
                    throw new ArgumentException("Unknown training command.");
                if (m_NeedsReset)
                    throw new InvalidOperationException("Reset the environment before stepping.");
                if (AllAgentsFinished)
                {
                    if (m_AutoReset == 1)
                        return new TrainingResponse { states = Reset() }; // NEXT_STEP ignores these actions.
                    throw new InvalidOperationException(
                        "Episode ended; reset before stepping again."
                    );
                }
                // Every action is decoded and checked before the environment is touched.
                var actions = DecodeActions(request.actions);
                var response = new TrainingResponse
                {
                    states = EncodeStates(m_Environment.Step(actions)),
                };
                if (AllAgentsFinished && m_AutoReset == 0)
                {
                    response.hasInitial = true;
                    response.initialStates = Reset();
                }
                return response;
            }
            catch (Exception error)
            {
                // A failed operation can leave user-defined environment hooks partially advanced.
                // Require reset before the next step, rather than silently continuing that episode.
                m_NeedsReset = true;
                return new TrainingResponse { ok = false, error = error.Message };
            }
        }

        void EncodeDefinitions()
        {
            if (m_Definitions.Count == 0)
                throw new InvalidOperationException("Environment declared no agents.");
            var ids = new List<string>();
            var encoded = new List<TrainingAgentDefinition>();
            foreach (var pair in m_Definitions)
            {
                if (string.IsNullOrWhiteSpace(pair.Key) || pair.Value == null)
                    throw new InvalidOperationException(
                        "Every agent needs a non-empty id and a definition."
                    );
                ids.Add(pair.Key);
                encoded.Add(
                    new TrainingAgentDefinition
                    {
                        id = pair.Key,
                        observationSpace = TrainingWire.EncodeSpace(pair.Value.ObservationSpace),
                        actionSpace = TrainingWire.EncodeSpace(pair.Value.ActionSpace),
                    }
                );
            }
            m_AgentIds.Clear();
            m_AgentIds.AddRange(ids);
            m_EncodedDefinitions = encoded.ToArray();
        }

        Dictionary<string, Point> DecodeActions(TrainingAgentAction[] entries)
        {
            var actions = new Dictionary<string, Point>(StringComparer.Ordinal);
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var entry in entries ?? Array.Empty<TrainingAgentAction>())
            {
                if (
                    entry == null
                    || entry.id == null
                    || !m_Definitions.TryGetValue(entry.id, out var definition)
                )
                    throw new ArgumentException($"Unknown agent id '{entry?.id}'.");
                if (!seen.Add(entry.id))
                    throw new ArgumentException($"Duplicate action for agent '{entry.id}'.");
                // A finished agent's action is optional and ignored until the next reset.
                if (m_Finished.Contains(entry.id))
                    continue;
                var action = TrainingWire.DecodePoint(entry.action);
                TrainingWire.Validate(action, definition.ActionSpace);
                actions.Add(entry.id, action);
            }
            foreach (var id in m_AgentIds)
            {
                if (!m_Finished.Contains(id) && !actions.ContainsKey(id))
                    throw new ArgumentException($"Missing action for agent '{id}'.");
            }
            return actions;
        }

        TrainingAgentState[] EncodeStates(IReadOnlyDictionary<string, AgentState> states)
        {
            RequireEveryAgent(states?.Count);
            var encoded = new TrainingAgentState[m_AgentIds.Count];
            for (var i = 0; i < m_AgentIds.Count; i++)
            {
                var id = m_AgentIds[i];
                if (!states.TryGetValue(id, out var state) || state == null)
                    throw new InvalidOperationException(
                        $"Environment returned no state for agent '{id}'."
                    );
                if (float.IsNaN(state.Reward) || float.IsInfinity(state.Reward))
                    throw new InvalidOperationException($"Reward for agent '{id}' must be finite.");
                encoded[i] = TrainingWire.EncodeState(id, state);
                if (state.Terminated || state.Truncated)
                    m_Finished.Add(id);
            }
            return encoded;
        }

        TrainingAgentState[] Reset()
        {
            var initial = m_Environment.Reset();
            RequireEveryAgent(initial?.Count);
            var encoded = new TrainingAgentState[m_AgentIds.Count];
            for (var i = 0; i < m_AgentIds.Count; i++)
            {
                var id = m_AgentIds[i];
                if (!initial.TryGetValue(id, out var state) || state == null)
                    throw new InvalidOperationException(
                        $"Environment returned no initial state for agent '{id}'."
                    );
                encoded[i] = TrainingWire.EncodeInitialState(id, state);
            }
            m_NeedsReset = false;
            m_Finished.Clear();
            return encoded;
        }

        void RequireEveryAgent(int? returned)
        {
            if (returned != m_AgentIds.Count)
                throw new InvalidOperationException(
                    $"Environment returned {returned ?? 0} agent states; "
                        + $"{m_AgentIds.Count} agents were declared."
                );
        }
    }
}
