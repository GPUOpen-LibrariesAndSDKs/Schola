// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEngine;

namespace AMD.Schola.Samples.TrainingDemo
{
    /// <summary>
    /// Reward and episode hooks for one cube of a team, each reaching its own target.
    /// Add one to every agent GameObject so <see cref="ScholaMultiAgentEnvironment"/>
    /// finds it for that agent.
    /// </summary>
    /// <remarks>
    /// The reward is this cube's own progress. The episode status is shared: every agent
    /// terminates on the step the whole team is at its targets, and every agent truncates
    /// after the same number of steps. Finishing together is what lets
    /// <c>schola unity train</c> use one shared policy for the team.
    /// </remarks>
    public sealed class MultiAgentTrainingDemoTask : MonoBehaviour, IReward, IEpisodeStatus
    {
        [SerializeField]
        Transform m_Agent;

        [SerializeField]
        Transform m_Target;

        [SerializeField]
        MultiAgentTrainingDemoTask[] m_Team = new MultiAgentTrainingDemoTask[0];

        [SerializeField]
        int m_MaxSteps = 128;

        [SerializeField]
        float m_ArrivalDistance = 0.2f;
        int m_Steps;
        float m_PreviousDistance;
        bool m_ArrivalRewarded;

        public bool HasArrived =>
            Vector3.Distance(m_Agent.position, m_Target.position) <= m_ArrivalDistance;

        /// <param name="team">Every task in the environment, including this one.</param>
        public void Configure(Transform agent, Transform target, MultiAgentTrainingDemoTask[] team)
        {
            m_Agent = agent;
            m_Target = target;
            m_Team = team;
        }

        public void OnEpisodeReset()
        {
            m_Steps = 0;
            m_ArrivalRewarded = false;
            m_PreviousDistance = Vector3.Distance(m_Agent.position, m_Target.position);
        }

        public float ComputeReward()
        {
            m_Steps++;
            var distance = Vector3.Distance(m_Agent.position, m_Target.position);
            var reward = m_PreviousDistance - distance - 0.01f;
            // Paid once per episode, so waiting at the target for a teammate earns nothing.
            if (distance <= m_ArrivalDistance && !m_ArrivalRewarded)
            {
                reward += 1f;
                m_ArrivalRewarded = true;
            }
            m_PreviousDistance = distance;
            return reward;
        }

        public void GetStatus(out bool terminated, out bool truncated)
        {
            terminated = HasArrived;
            foreach (var task in m_Team)
            {
                if (task != null)
                    terminated &= task.HasArrived;
            }
            truncated = !terminated && m_Steps >= m_MaxSteps;
        }
    }
}
