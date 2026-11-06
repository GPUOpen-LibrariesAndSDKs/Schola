// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using AMD.Schola;
using UnityEngine;

namespace AMD.Schola.Tests.Environment
{
    /// <summary>
    /// Shared EditMode harness: environment GameObject with a child agent that has
    /// PositionSensor, scripted reward/status, an in-memory sink, and a TranslateActuator
    /// that may sit on the agent, a sibling, or a parent.
    /// Fakes are nested types so Unity can AddComponent them from this Editor test assembly.
    /// </summary>
    public sealed class ScholaEnvironmentTestFixture : IDisposable
    {
        public const float Tolerance = 1e-5f;
        public const string PositionKey = "position";
        public const string TranslateKey = "translate";
        public static readonly Vector3 StartPose = new Vector3(1f, 2f, 3f);

        /// <summary>
        /// Where the configured <see cref="TranslateActuator"/> lives relative to the agent.
        /// </summary>
        public enum ActuatorPlacement
        {
            OnAgent,
            OnSibling,
            OnParent,
        }

        /// <summary>
        /// Test-only <see cref="IReward"/> that returns a scripted scalar.
        /// </summary>
        public sealed class ScriptedReward : MonoBehaviour, IReward
        {
            public float Reward;

            public void OnEpisodeReset() { }

            public float ComputeReward()
            {
                return Reward;
            }
        }

        /// <summary>
        /// Test-only <see cref="IEpisodeStatus"/> that reports scripted terminated/truncated flags.
        /// </summary>
        public sealed class ScriptedEpisodeStatus : MonoBehaviour, IEpisodeStatus
        {
            public bool Terminated;
            public bool Truncated;

            public void OnEpisodeReset()
            {
                Terminated = false;
                Truncated = false;
            }

            public void GetStatus(out bool terminated, out bool truncated)
            {
                terminated = Terminated;
                truncated = Truncated;
            }
        }

        /// <summary>
        /// Test-only <see cref="IEpisodeStatus"/> that truncates after <see cref="MaxSteps"/>
        /// steps in the current episode. Relies on <see cref="OnEpisodeReset"/> to clear its counter.
        /// </summary>
        public sealed class TruncateAfterNStepsStatus : MonoBehaviour, IEpisodeStatus
        {
            public int MaxSteps = 2;
            int m_StepCount;

            public void OnEpisodeReset()
            {
                m_StepCount = 0;
            }

            public void GetStatus(out bool terminated, out bool truncated)
            {
                m_StepCount++;
                terminated = false;
                truncated = m_StepCount >= MaxSteps;
            }
        }

        /// <summary>
        /// Test-only in-memory <see cref="ITrainingStateSink"/>. US4 placeholder stand-in for Python.
        /// </summary>
        public sealed class InMemoryTrainingStateSink : ITrainingStateSink
        {
            public List<AgentState> States { get; } = new List<AgentState>();

            public List<InitialAgentState> InitialStates { get; } = new List<InitialAgentState>();

            public List<IReadOnlyDictionary<string, AgentState>> StateMaps { get; } =
                new List<IReadOnlyDictionary<string, AgentState>>();

            public List<IReadOnlyDictionary<string, InitialAgentState>> InitialStateMaps { get; } =
                new List<IReadOnlyDictionary<string, InitialAgentState>>();

            public void SubmitState(AgentState state)
            {
                States.Add(state);
            }

            public void SubmitInitialState(InitialAgentState state)
            {
                InitialStates.Add(state);
            }

            public void SubmitStates(IReadOnlyDictionary<string, AgentState> states)
            {
                StateMaps.Add(states);
            }

            public void SubmitInitialStates(IReadOnlyDictionary<string, InitialAgentState> states)
            {
                InitialStateMaps.Add(states);
            }
        }

        public GameObject EnvGo { get; private set; }
        public GameObject AgentGo { get; private set; }
        public GameObject ActuatorGo { get; private set; }
        public ScholaEnvironment Environment { get; private set; }
        public ScholaAgent Agent { get; private set; }
        public PositionSensor Sensor { get; private set; }
        public TranslateActuator Actuator { get; private set; }
        public ScriptedReward Reward { get; private set; }
        public ScriptedEpisodeStatus Status { get; private set; }
        public InMemoryTrainingStateSink Sink { get; private set; }

        public static ScholaEnvironmentTestFixture Create(
            ActuatorPlacement actuatorPlacement = ActuatorPlacement.OnAgent
        )
        {
            var fixture = new ScholaEnvironmentTestFixture();

            fixture.EnvGo = new GameObject("ScholaEnvironment");
            fixture.Environment = fixture.EnvGo.AddComponent<ScholaEnvironment>();

            fixture.AgentGo = new GameObject("ScholaAgent");
            fixture.ActuatorGo =
                actuatorPlacement == ActuatorPlacement.OnAgent
                    ? fixture.AgentGo
                    : new GameObject("TranslateActuator");

            switch (actuatorPlacement)
            {
                case ActuatorPlacement.OnParent:
                    fixture.ActuatorGo.transform.SetParent(fixture.EnvGo.transform);
                    SetLocalPose(fixture.ActuatorGo.transform, StartPose);
                    fixture.AgentGo.transform.SetParent(fixture.ActuatorGo.transform);
                    SetLocalPose(fixture.AgentGo.transform, Vector3.zero);
                    break;
                case ActuatorPlacement.OnSibling:
                    fixture.AgentGo.transform.SetParent(fixture.EnvGo.transform);
                    SetLocalPose(fixture.AgentGo.transform, StartPose);
                    fixture.ActuatorGo.transform.SetParent(fixture.EnvGo.transform);
                    SetLocalPose(fixture.ActuatorGo.transform, StartPose);
                    break;
                default:
                    fixture.AgentGo.transform.SetParent(fixture.EnvGo.transform);
                    SetLocalPose(fixture.AgentGo.transform, StartPose);
                    break;
            }

            fixture.Sensor = fixture.AgentGo.AddComponent<PositionSensor>();
            fixture.Sensor.ObservationKey = PositionKey;
            fixture.Actuator = fixture.ActuatorGo.AddComponent<TranslateActuator>();
            fixture.Actuator.ActionKey = TranslateKey;
            fixture.Reward = fixture.AgentGo.AddComponent<ScriptedReward>();
            fixture.Status = fixture.AgentGo.AddComponent<ScriptedEpisodeStatus>();

            fixture.Agent = fixture.AgentGo.AddComponent<ScholaAgent>();
            fixture.Agent.SetSensors(new[] { fixture.Sensor });
            fixture.Agent.SetActuators(new[] { fixture.Actuator });

            fixture.Sink = new InMemoryTrainingStateSink();
            fixture.Environment.SetAgent(fixture.Agent);
            fixture.Environment.SetReward(fixture.Reward);
            fixture.Environment.SetEpisodeStatus(fixture.Status);
            fixture.Environment.SetTrainingStateSink(fixture.Sink);

            fixture.Environment.Initialize();
            return fixture;
        }

        static void SetLocalPose(Transform t, Vector3 localPosition)
        {
            t.localPosition = localPosition;
            t.localRotation = Quaternion.identity;
            t.localScale = Vector3.one;
        }

        public DictPoint TranslateAction(float dx, float dy = 0f, float dz = 0f)
        {
            return new DictPoint
            {
                Points = { [TranslateKey] = new BoxPoint { Values = { dx, dy, dz } } },
            };
        }

        public static Vector3 PositionFromObservations(Point observations)
        {
            var dict = observations as DictPoint;
            if (dict == null || !dict.Points.TryGetValue(PositionKey, out var nested))
            {
                return Vector3.positiveInfinity;
            }

            var box = nested as BoxPoint;
            if (box == null || box.Values == null || box.Values.Count != 3)
            {
                return Vector3.positiveInfinity;
            }

            return new Vector3(box.Values[0], box.Values[1], box.Values[2]);
        }

        public void Dispose()
        {
            if (EnvGo != null)
            {
                UnityEngine.Object.DestroyImmediate(EnvGo);
            }
        }
    }
}
