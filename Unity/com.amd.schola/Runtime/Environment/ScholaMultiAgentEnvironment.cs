// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Multi-agent training environment. Each <see cref="ScholaAgent"/> is keyed by a
    /// string id. Step and Reset return maps rather than a single <see cref="AgentState"/>.
    /// Reward and episode status are resolved on each agent, not once for the environment.
    /// </summary>
    public sealed class ScholaMultiAgentEnvironment : MonoBehaviour, IMultiAgentEnvironment
    {
        [Serializable]
        sealed class AgentSlot
        {
            public string Id;
            public ScholaAgent Agent;
        }

        struct TransformSnapshot
        {
            public Transform Transform;
            public Vector3 LocalPosition;
            public Quaternion LocalRotation;
            public Vector3 LocalScale;
        }

        [SerializeField]
        List<AgentSlot> m_Agents = new List<AgentSlot>();

        ITrainingStateSink m_Sink;
        readonly List<TransformSnapshot> m_PoseSnapshot = new List<TransformSnapshot>();
        bool m_Initialized;

        public void SetAgents(IEnumerable<KeyValuePair<string, ScholaAgent>> agents)
        {
            if (agents == null)
            {
                throw new ArgumentNullException(nameof(agents));
            }

            m_Agents = new List<AgentSlot>();
            foreach (var pair in agents)
            {
                m_Agents.Add(new AgentSlot { Id = pair.Key, Agent = pair.Value });
            }

            m_Initialized = false;
        }

        public void SetTrainingStateSink(ITrainingStateSink sink)
        {
            m_Sink = sink;
        }

        /// <summary>
        /// Inits each agent's sensors and actuators, snapshots poses, and returns one
        /// <see cref="InteractionDefinition"/> per agent id.
        /// </summary>
        public IReadOnlyDictionary<string, InteractionDefinition> Initialize()
        {
            EnsureConfigured();

            foreach (var slot in m_Agents)
            {
                foreach (var sensor in slot.Agent.Sensors)
                {
                    if (sensor != null)
                    {
                        sensor.InitSensor();
                    }
                }

                foreach (var actuator in slot.Agent.Actuators)
                {
                    if (actuator != null)
                    {
                        actuator.InitActuator();
                    }
                }
            }

            SnapshotPoses();
            m_Initialized = true;

            var definitions = new Dictionary<string, InteractionDefinition>();
            foreach (var slot in m_Agents)
            {
                definitions[slot.Id] = new InteractionDefinition(
                    slot.Agent.Define(),
                    slot.Agent.DefineActionSpace()
                );
            }

            return definitions;
        }

        public void Seed(int seed)
        {
            _ = seed;
        }

        public void SetOptions(IReadOnlyDictionary<string, string> options)
        {
            _ = options;
        }

        /// <summary>
        /// Restores every initialize snapshot, notifies each agent's episode hooks, and
        /// returns one <see cref="InitialAgentState"/> per id. Does not auto-reset after a
        /// terminal step.
        /// </summary>
        /// <remarks>
        /// Unity invokes a method named <c>Reset</c> when the component is first added, before
        /// <see cref="SetAgents"/> runs. If no agent is bound yet, this is that editor callback
        /// and is a no-op.
        /// </remarks>
        public IReadOnlyDictionary<string, InitialAgentState> Reset()
        {
            if (!HasBoundAgents())
            {
                return new Dictionary<string, InitialAgentState>();
            }

            if (!m_Initialized)
            {
                Initialize();
            }

            RestorePoses();
            NotifyEpisodeReset();

            var initial = new Dictionary<string, InitialAgentState>();
            foreach (var slot in m_Agents)
            {
                initial[slot.Id] = new InitialAgentState { Observations = slot.Agent.Observe() };
            }

            m_Sink?.SubmitInitialStates(initial);
            return initial;
        }

        /// <summary>
        /// Applies each agent's action, then reports that agent's reward, episode flags, and
        /// a fresh observation. The action map is checked before any <see cref="IAgent.Act"/>.
        /// </summary>
        public IReadOnlyDictionary<string, AgentState> Step(
            IReadOnlyDictionary<string, Point> actions
        )
        {
            EnsureConfigured();
            ValidateActions(actions);

            var states = new Dictionary<string, AgentState>();
            foreach (var slot in m_Agents)
            {
                var agent = slot.Agent;
                agent.Act(actions[slot.Id]);

                var reward = 0f;
                var rewardHook = ResolveHook<IReward>(agent);
                if (rewardHook != null)
                {
                    reward = rewardHook.ComputeReward();
                }

                var terminated = false;
                var truncated = false;
                var statusHook = ResolveHook<IEpisodeStatus>(agent);
                if (statusHook != null)
                {
                    statusHook.GetStatus(out terminated, out truncated);
                }

                states[slot.Id] = new AgentState
                {
                    Observations = agent.Observe(),
                    Reward = reward,
                    Terminated = terminated,
                    Truncated = truncated,
                };
            }

            m_Sink?.SubmitStates(states);
            return states;
        }

        bool HasBoundAgents()
        {
            if (m_Agents == null)
            {
                return false;
            }

            foreach (var slot in m_Agents)
            {
                if (slot != null && slot.Agent != null)
                {
                    return true;
                }
            }

            return false;
        }

        void EnsureConfigured()
        {
            if (m_Agents == null || m_Agents.Count == 0)
            {
                throw new InvalidOperationException(
                    $"{nameof(ScholaMultiAgentEnvironment)} requires an agent."
                );
            }

            var seen = new HashSet<string>();
            foreach (var slot in m_Agents)
            {
                if (slot == null || slot.Agent == null)
                {
                    throw new InvalidOperationException(
                        $"{nameof(ScholaMultiAgentEnvironment)} has an empty agent slot."
                    );
                }

                if (string.IsNullOrWhiteSpace(slot.Id))
                {
                    throw new InvalidOperationException(
                        $"{nameof(ScholaMultiAgentEnvironment)} requires a non-empty agent id."
                    );
                }

                if (!seen.Add(slot.Id))
                {
                    throw new InvalidOperationException(
                        $"{nameof(ScholaMultiAgentEnvironment)} agent id '{slot.Id}' is duplicated."
                    );
                }
            }
        }

        void ValidateActions(IReadOnlyDictionary<string, Point> actions)
        {
            if (actions == null)
            {
                throw new ArgumentNullException(nameof(actions));
            }

            var ids = new HashSet<string>();
            foreach (var slot in m_Agents)
            {
                ids.Add(slot.Id);
            }

            foreach (var pair in actions)
            {
                if (!ids.Contains(pair.Key))
                {
                    throw new ArgumentException(
                        $"Unknown agent id '{pair.Key}'.",
                        nameof(actions)
                    );
                }
            }

            foreach (var slot in m_Agents)
            {
                if (!actions.TryGetValue(slot.Id, out var action) || action == null)
                {
                    throw new ArgumentException(
                        $"Missing action for agent '{slot.Id}'.",
                        nameof(actions)
                    );
                }
            }
        }

        void NotifyEpisodeReset()
        {
            foreach (var slot in m_Agents)
            {
                ResolveHook<IReward>(slot.Agent)?.OnEpisodeReset();
                ResolveHook<IEpisodeStatus>(slot.Agent)?.OnEpisodeReset();
            }
        }

        static T ResolveHook<T>(ScholaAgent agent)
            where T : class
        {
            var onAgent = agent.GetComponent<T>();
            if (onAgent != null)
            {
                return onAgent;
            }

            return agent.GetComponentInChildren<T>(true);
        }

        void SnapshotPoses()
        {
            m_PoseSnapshot.Clear();
            var seen = new HashSet<Transform>();
            foreach (var slot in m_Agents)
            {
                var transforms = slot.Agent.GetComponentsInChildren<Transform>(true);
                foreach (var transform in transforms)
                {
                    CapturePose(transform, seen);
                }

                foreach (var actuator in slot.Agent.Actuators)
                {
                    if (actuator != null)
                    {
                        CapturePose(actuator.transform, seen);
                    }
                }
            }
        }

        void CapturePose(Transform target, HashSet<Transform> seen)
        {
            if (target == null || !seen.Add(target))
            {
                return;
            }

            m_PoseSnapshot.Add(
                new TransformSnapshot
                {
                    Transform = target,
                    LocalPosition = target.localPosition,
                    LocalRotation = target.localRotation,
                    LocalScale = target.localScale,
                }
            );
        }

        void RestorePoses()
        {
            foreach (var pose in m_PoseSnapshot)
            {
                if (pose.Transform == null)
                {
                    continue;
                }

                pose.Transform.localPosition = pose.LocalPosition;
                pose.Transform.localRotation = pose.LocalRotation;
                pose.Transform.localScale = pose.LocalScale;
            }
        }
    }
}
