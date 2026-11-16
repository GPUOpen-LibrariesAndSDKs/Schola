// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Agent that defines observation/action spaces from sensors and actuators,
    /// aggregates Observe into a DictPoint, and fans out Act to actuators.
    /// </summary>
    public sealed class ScholaAgent : MonoBehaviour, IAgent
    {
        [SerializeField]
        List<PositionSensor> m_Sensors = new List<PositionSensor>();

        [SerializeField]
        List<TranslateActuator> m_Actuators = new List<TranslateActuator>();

        public IReadOnlyList<PositionSensor> Sensors => m_Sensors;

        public IReadOnlyList<TranslateActuator> Actuators => m_Actuators;

        public void SetSensors(IEnumerable<PositionSensor> sensors)
        {
            m_Sensors = new List<PositionSensor>(sensors);
        }

        public void SetActuators(IEnumerable<TranslateActuator> actuators)
        {
            m_Actuators = new List<TranslateActuator>(actuators);
        }

        /// <summary>
        /// Builds the published observation space from attached sensors (DictSpace of sensor spaces).
        /// </summary>
        public Space Define()
        {
            var dictSpace = new DictSpace();
            foreach (var sensor in m_Sensors)
            {
                if (sensor == null)
                {
                    continue;
                }

                dictSpace.Spaces[sensor.ObservationKey] = sensor.GetObservationSpace();
            }

            return dictSpace;
        }

        /// <summary>
        /// Builds the published action space from attached actuators (DictSpace of actuator spaces).
        /// Duplicate <see cref="TranslateActuator.ActionKey"/> values warn and last-write-wins.
        /// </summary>
        public Space DefineActionSpace()
        {
            var dictSpace = new DictSpace();
            foreach (var actuator in m_Actuators)
            {
                if (actuator == null)
                {
                    continue;
                }

                if (dictSpace.Spaces.ContainsKey(actuator.ActionKey))
                {
                    Debug.LogWarning(
                        $"Duplicate action key '{actuator.ActionKey}' on {nameof(ScholaAgent)} '{name}'; later actuator overwrites the earlier action space entry.",
                        this
                    );
                }

                dictSpace.Spaces[actuator.ActionKey] = actuator.GetActionSpace();
            }

            return dictSpace;
        }

        /// <summary>
        /// Collects current observations from attached sensors into a DictPoint.
        /// </summary>
        public Point Observe()
        {
            var dictPoint = new DictPoint();
            foreach (var sensor in m_Sensors)
            {
                if (sensor == null)
                {
                    continue;
                }

                dictPoint.Points[sensor.ObservationKey] = sensor.CollectObservations();
            }

            return dictPoint;
        }

        /// <summary>
        /// Checks that <paramref name="action"/> is a <see cref="DictPoint"/> whose keys and
        /// payloads match every attached actuator. Does not move anything.
        /// </summary>
        public void ValidateAction(Point action)
        {
            var dictPoint = RequireDictPoint(action);

            foreach (var actuator in m_Actuators)
            {
                if (actuator == null)
                {
                    continue;
                }

                if (
                    !dictPoint.Points.TryGetValue(actuator.ActionKey, out var nested)
                    || nested == null
                )
                {
                    throw new ArgumentException(
                        $"Missing action key '{actuator.ActionKey}'; expected keys: {FormatExpectedActionKeys()}.",
                        nameof(action)
                    );
                }
            }

            foreach (var actuator in m_Actuators)
            {
                if (actuator == null)
                {
                    continue;
                }

                actuator.ValidateAction(dictPoint.Points[actuator.ActionKey]);
            }
        }

        /// <summary>
        /// Applies a DictPoint action to attached actuators keyed by <see cref="TranslateActuator.ActionKey"/>.
        /// Keys and payloads are checked before any actuator is applied so a bad action leaves the scene unchanged.
        /// </summary>
        public void Act(Point action)
        {
            ValidateAction(action);

            var dictPoint = (DictPoint)action;
            foreach (var actuator in m_Actuators)
            {
                if (actuator == null)
                {
                    continue;
                }

                actuator.TakeAction(dictPoint.Points[actuator.ActionKey]);
            }
        }

        static DictPoint RequireDictPoint(Point action)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            if (!(action is DictPoint dictPoint))
            {
                throw new ArgumentException(
                    $"Expected {nameof(DictPoint)}, got {action.GetType().Name}.",
                    nameof(action)
                );
            }

            return dictPoint;
        }

        string FormatExpectedActionKeys()
        {
            var builder = new StringBuilder();
            foreach (var actuator in m_Actuators)
            {
                if (actuator == null)
                {
                    continue;
                }

                if (builder.Length > 0)
                {
                    builder.Append(", ");
                }

                builder.Append(actuator.ActionKey);
            }

            return builder.Length > 0 ? builder.ToString() : "(none)";
        }
    }
}
