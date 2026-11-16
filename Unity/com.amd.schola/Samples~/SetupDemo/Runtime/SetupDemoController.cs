// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEngine;

namespace AMD.Schola.Samples
{
    /// <summary>
    /// Installation demonstration using Schola observations and actions. A deterministic
    /// controller chooses displacements; Schola applies them. No training or episode system.
    /// </summary>
    public sealed class SetupDemoController : MonoBehaviour
    {
        public const string AgentPositionKey = "agent_position";
        public const string TargetPositionKey = "target_position";
        public const string TranslateKey = "translate";

        [SerializeField]
        ScholaAgent m_Agent;

        [SerializeField]
        PositionSensor m_TargetSensor;

        [SerializeField, Min(0.01f)]
        float m_Speed = 2f;

        [SerializeField, Min(0f)]
        float m_ArrivalDistance = 0.1f;

        bool m_ArrivalLogged;

        public DictPoint LastObservation { get; private set; }
        public bool HasArrived { get; private set; }

        /// <summary>Assigns scene references; the agent's Sensors list is configured separately.</summary>
        public void Configure(ScholaAgent agent, PositionSensor targetSensor)
        {
            m_Agent = agent;
            m_TargetSensor = targetSensor;
        }

        void Start()
        {
            if (!ValidateConfiguration())
            {
                enabled = false;
                return;
            }

            foreach (var sensor in m_Agent.Sensors)
            {
                sensor.InitSensor();
            }

            foreach (var actuator in m_Agent.Actuators)
            {
                actuator.InitActuator();
            }

            m_Agent.Define();
            m_Agent.DefineActionSpace();
            Debug.Log(
                "Schola Setup Demo: deterministic Schola actions; no training or inference.",
                this
            );
        }

        /// <summary>Reject ambiguous keys and sensors/actuators on the wrong scene objects.</summary>
        public bool ValidateConfiguration()
        {
            if (
                m_Agent == null
                || m_TargetSensor == null
                || m_Speed <= 0f
                || m_ArrivalDistance < 0f
            )
            {
                Debug.LogError(
                    "Schola Setup Demo: assign Agent and Target Sensor; check Speed and Arrival Distance.",
                    this
                );
                return false;
            }

            var sensors = m_Agent.Sensors;
            if (
                sensors.Count != 2
                || sensors[0] == null
                || sensors[0].transform != m_Agent.transform
                || sensors[0].ObservationKey != AgentPositionKey
                || sensors[1] != m_TargetSensor
                || sensors[1].ObservationKey != TargetPositionKey
            )
            {
                Debug.LogError(
                    "Schola Setup Demo: Sensors must be [cube agent_position, target target_position]. See the setup guide.",
                    this
                );
                return false;
            }

            var actuators = m_Agent.Actuators;
            if (
                actuators.Count != 1
                || actuators[0] == null
                || actuators[0].transform != m_Agent.transform
                || actuators[0].ActionKey != TranslateKey
            )
            {
                Debug.LogError(
                    "Schola Setup Demo: Actuators must contain one TranslateActuator on the cube with Action Key 'translate'. See the setup guide.",
                    this
                );
                return false;
            }

            return true;
        }

        void Update()
        {
            Step(Time.deltaTime);
        }

        /// <summary>
        /// Reads Observe() before deciding movement, then reads it again for the display.
        /// Sends the chosen displacement through ScholaAgent.Act and its TranslateActuator.
        /// </summary>
        public void Step(float deltaTime)
        {
            var observation = (DictPoint)m_Agent.Observe();
            var current = ReadPosition(observation, AgentPositionKey);
            var target = ReadPosition(observation, TargetPositionKey);
            HasArrived = Vector3.Distance(current, target) <= m_ArrivalDistance;
            if (!HasArrived)
            {
                var nextPosition = Vector3.MoveTowards(
                    current,
                    target,
                    m_Speed * Mathf.Max(0f, deltaTime)
                );
                var delta = nextPosition - current;
                m_Agent.Act(
                    new DictPoint
                    {
                        Points =
                        {
                            [TranslateKey] = new BoxPoint
                            {
                                Values = { delta.x, delta.y, delta.z },
                            },
                        },
                    }
                );
            }

            LastObservation = (DictPoint)m_Agent.Observe();
            HasArrived =
                Vector3.Distance(ReadPosition(LastObservation, AgentPositionKey), target)
                <= m_ArrivalDistance;
            if (HasArrived && !m_ArrivalLogged)
            {
                Debug.Log(
                    "Schola Setup Demo: target reached. Move Target in Play Mode to continue.",
                    this
                );
            }

            m_ArrivalLogged = HasArrived;
        }

        static Vector3 ReadPosition(DictPoint observation, string key)
        {
            var values = ((BoxPoint)observation.Points[key]).Values;
            return new Vector3(values[0], values[1], values[2]);
        }

        void OnGUI()
        {
            if (LastObservation == null)
            {
                return;
            }

            GUILayout.BeginArea(new Rect(12f, 12f, 470f, 140f), GUI.skin.box);
            GUILayout.Label("Schola Setup Demo - deterministic Schola actions");
            GUILayout.Label("No training, episode logic, or model inference");
            GUILayout.Label(
                $"{AgentPositionKey}: {ReadPosition(LastObservation, AgentPositionKey):F2}"
            );
            GUILayout.Label(
                $"{TargetPositionKey}: {ReadPosition(LastObservation, TargetPositionKey):F2}"
            );
            GUILayout.Label(
                HasArrived ? "Target reached - move Target to continue" : "Moving toward Target"
            );
            GUILayout.EndArea();
        }
    }
}
