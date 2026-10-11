// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Scene-backed sensor that reads <see cref="Transform.position"/> into a fixed-length BoxPoint (x, y, z).
    /// </summary>
    public sealed class PositionSensor : MonoBehaviour, ISensor
    {
        const int DimensionCount = 3;

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
            if (string.IsNullOrEmpty(m_ObservationKey))
            {
                Debug.LogWarning($"{nameof(PositionSensor)} on '{name}' has an empty {nameof(ObservationKey)}.", this);
            }
        }

        public Space GetObservationSpace()
        {
            var space = new BoxSpace();
            for (var i = 0; i < DimensionCount; i++)
            {
                space.Dimensions.Add(new BoxSpaceDimension());
            }

            return space;
        }

        public Point CollectObservations()
        {
            var position = transform.position;
            return new BoxPoint
            {
                Values = { position.x, position.y, position.z },
            };
        }
    }
}
