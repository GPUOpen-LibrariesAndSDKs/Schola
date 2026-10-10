// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Scene-backed sensor that reads <see cref="Transform.position"/> into a fixed-length BoxPoint.
    /// Stub for Phase 1 (tests red). Phase 2 implements CollectObservations / GetObservationSpace.
    /// </summary>
    public sealed class PositionSensor : MonoBehaviour, ISensor
    {
        [SerializeField]
        string m_ObservationKey = "position";

        /// <summary>
        /// Stable dict key used when an agent aggregates this sensor.
        /// </summary>
        public string ObservationKey
        {
            get => m_ObservationKey;
            set => m_ObservationKey = value;
        }

        public void InitSensor()
        {
            // Phase 2: validate configuration.
        }

        public Space GetObservationSpace()
        {
            // Phase 2: return BoxSpace with 3 dimensions matching position x,y,z.
            return new BoxSpace();
        }

        public Point CollectObservations()
        {
            // Phase 2: return BoxPoint from current transform.position.
            return new BoxPoint();
        }
    }
}
