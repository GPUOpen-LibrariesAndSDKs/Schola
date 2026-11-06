// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Single-agent training environment that composes a <see cref="ScholaAgent"/>.
    /// Step applies an action then reports protocol-shaped <see cref="AgentState"/>.
    /// Reset restores the initialize snapshot and returns <see cref="InitialAgentState"/>.
    /// </summary>
    public sealed class ScholaEnvironment : MonoBehaviour, IEnvironment
    {
        /// <summary>
        /// Agent-id key <see cref="SingleAgentEnvironmentAdapter"/> injects. Matches Unreal.
        /// </summary>
        public const string DefaultAgentId = "SingleAgent";

        struct TransformSnapshot
        {
            public Transform Transform;
            public Vector3 LocalPosition;
            public Quaternion LocalRotation;
            public Vector3 LocalScale;
        }

        [SerializeField]
        ScholaAgent m_Agent;

        IReward m_Reward;
        IEpisodeStatus m_Status;
        ITrainingStateSink m_Sink;
        readonly List<TransformSnapshot> m_PoseSnapshot = new List<TransformSnapshot>();
        bool m_Initialized;

        public void SetAgent(ScholaAgent agent)
        {
            m_Agent = agent;
        }

        public void SetReward(IReward reward)
        {
            m_Reward = reward;
        }

        public void SetEpisodeStatus(IEpisodeStatus status)
        {
            m_Status = status;
        }

        public void SetTrainingStateSink(ITrainingStateSink sink)
        {
            m_Sink = sink;
        }

        /// <summary>
        /// Binds the agent, inits sensors/actuators, snapshots local poses of the agent hierarchy
        /// and configured actuator transforms, and returns obs/action spaces.
        /// </summary>
        public InteractionDefinition Initialize()
        {
            ResolveBindings();
            if (m_Agent == null)
            {
                throw new InvalidOperationException(
                    $"{nameof(ScholaEnvironment)} requires a {nameof(ScholaAgent)}."
                );
            }

            foreach (var sensor in m_Agent.Sensors)
            {
                if (sensor != null)
                {
                    sensor.InitSensor();
                }
            }

            foreach (var actuator in m_Agent.Actuators)
            {
                if (actuator != null)
                {
                    actuator.InitActuator();
                }
            }

            SnapshotPoses();
            m_Initialized = true;
            return new InteractionDefinition(m_Agent.Define(), m_Agent.DefineActionSpace());
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
        /// Restores the initialize snapshot, notifies episode hooks, observes, and returns
        /// <see cref="InitialAgentState"/> (obs + info only). Does not auto-reset after a
        /// terminal step; the caller must invoke this to start the next episode.
        /// </summary>
        /// <remarks>
        /// Unity invokes a method named <c>Reset</c> when the component is first added, before
        /// <see cref="SetAgent"/> runs. If no agent is bound yet, this is that editor callback and is a no-op.
        /// </remarks>
        public InitialAgentState Reset()
        {
            if (m_Agent == null)
            {
                ResolveBindings();
            }

            if (m_Agent == null)
            {
                return new InitialAgentState();
            }

            if (!m_Initialized)
            {
                Initialize();
            }

            RestorePoses();
            NotifyEpisodeReset();
            var initial = new InitialAgentState { Observations = m_Agent.Observe() };
            m_Sink?.SubmitInitialState(initial);
            return initial;
        }

        /// <summary>
        /// Applies <paramref name="action"/>, then reports reward, episode flags, and a fresh observation.
        /// </summary>
        public AgentState Step(Point action)
        {
            if (m_Agent == null)
            {
                throw new InvalidOperationException(
                    $"{nameof(ScholaEnvironment)} requires a {nameof(ScholaAgent)}."
                );
            }

            m_Agent.Act(action);

            var reward = m_Reward != null ? m_Reward.ComputeReward() : 0f;
            var terminated = false;
            var truncated = false;
            if (m_Status != null)
            {
                m_Status.GetStatus(out terminated, out truncated);
            }

            var state = new AgentState
            {
                Observations = m_Agent.Observe(),
                Reward = reward,
                Terminated = terminated,
                Truncated = truncated,
            };
            m_Sink?.SubmitState(state);
            return state;
        }

        void NotifyEpisodeReset()
        {
            if (m_Reward != null)
            {
                m_Reward.OnEpisodeReset();
            }

            if (m_Status != null)
            {
                m_Status.OnEpisodeReset();
            }
        }

        void ResolveBindings()
        {
            if (m_Agent == null)
            {
                m_Agent = GetComponentInChildren<ScholaAgent>();
            }

            if (m_Reward == null)
            {
                m_Reward = GetComponentInChildren<IReward>();
            }

            if (m_Status == null)
            {
                m_Status = GetComponentInChildren<IEpisodeStatus>();
            }
        }

        void SnapshotPoses()
        {
            m_PoseSnapshot.Clear();
            var seen = new HashSet<Transform>();

            var transforms = m_Agent.GetComponentsInChildren<Transform>(true);
            foreach (var t in transforms)
            {
                CapturePose(t, seen);
            }

            // Actuators may be bound to parent or sibling objects, not only descendants.
            foreach (var actuator in m_Agent.Actuators)
            {
                if (actuator != null)
                {
                    CapturePose(actuator.transform, seen);
                }
            }
        }

        void CapturePose(Transform t, HashSet<Transform> seen)
        {
            if (t == null || !seen.Add(t))
            {
                return;
            }

            m_PoseSnapshot.Add(
                new TransformSnapshot
                {
                    Transform = t,
                    LocalPosition = t.localPosition,
                    LocalRotation = t.localRotation,
                    LocalScale = t.localScale,
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
