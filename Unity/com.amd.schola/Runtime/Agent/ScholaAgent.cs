// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Agent that defines observation space from sensors and aggregates Observe.
    /// Stub for Phase 1 (tests red). Phase 2 implements Define / Observe.
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
            // Phase 2: aggregate GetObservationSpace under each sensor ObservationKey.
            return new DictSpace();
        }

        /// <summary>
        /// Collects current observations from attached sensors into a DictPoint.
        /// </summary>
        public Point Observe()
        {
            // Phase 2: aggregate CollectObservations under each sensor ObservationKey.
            return new DictPoint();
        }
    }
}