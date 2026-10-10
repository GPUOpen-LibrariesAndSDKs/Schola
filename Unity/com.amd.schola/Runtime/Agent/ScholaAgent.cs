// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Agent that defines observation space from sensors and aggregates Observe into a DictPoint.
    /// </summary>
    public sealed class ScholaAgent : MonoBehaviour
    {
        [SerializeField]
        List<PositionSensor> m_Sensors = new List<PositionSensor>();

        public IReadOnlyList<PositionSensor> Sensors => m_Sensors;

        public void SetSensors(IEnumerable<PositionSensor> sensors)
        {
            m_Sensors = new List<PositionSensor>(sensors);
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
    }
}
