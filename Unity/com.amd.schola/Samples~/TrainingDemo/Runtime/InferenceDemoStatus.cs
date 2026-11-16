// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
using UnityEngine;

namespace AMD.Schola.Samples.TrainingDemo
{
    /// <summary>Shows inference progress; PolicyRunner remains the only action driver.</summary>
    public sealed class InferenceDemoStatus : MonoBehaviour
    {
        [SerializeField]
        Transform m_Agent;

        [SerializeField]
        Transform m_Target;

        [SerializeField]
        PolicyRunner m_Runner;
        bool m_Arrived;

        public void Configure(Transform agent, Transform target, PolicyRunner runner)
        {
            m_Agent = agent;
            m_Target = target;
            m_Runner = runner;
        }

        void Update()
        {
            if (
                !m_Arrived
                && m_Agent != null
                && m_Target != null
                && Vector3.Distance(m_Agent.position, m_Target.position) <= 0.2f
            )
            {
                m_Arrived = true;
                Debug.Log("Schola Inference Demo: target reached using ONNX; no Python required.");
                m_Runner.StopRunning();
            }
        }

        void OnGUI()
        {
            GUI.Box(
                new Rect(10, 10, 450, 65),
                "Schola ONNX inference (CPU)\n"
                    + (
                        m_Arrived ? "Target reached. Stop and re-enter Play Mode to restart."
                        : m_Runner != null && m_Runner.IsRunning
                            ? "Policy running; move Target to explore the policy."
                        : "Policy stopped. Check the Console and Model assignment."
                    )
            );
        }
    }
}
