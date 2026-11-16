// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using UnityEngine;

namespace AMD.Schola
{
    /// <summary>
    /// Scene-backed actuator that applies a 3-float BoxPoint delta to <see cref="Transform.position"/>.
    /// </summary>
    public sealed class TranslateActuator : MonoBehaviour, IActuator
    {
        const int DimensionCount = 3;

        [SerializeField]
        string m_ActionKey = "translate";

        [SerializeField, Min(0f)]
        float m_MaxDisplacement;

        /// <summary>Per-axis action bound. Zero preserves the original unbounded space.</summary>
        public float MaxDisplacement
        {
            get => m_MaxDisplacement;
            set
            {
                if (float.IsNaN(value) || float.IsInfinity(value) || value < 0f)
                    throw new ArgumentOutOfRangeException(nameof(value));
                m_MaxDisplacement = value;
            }
        }

        /// <summary>
        /// Stable dict key used when an agent aggregates this actuator.
        /// </summary>
        public string ActionKey
        {
            get => m_ActionKey;
            set => m_ActionKey = value;
        }

        public void InitActuator()
        {
            if (string.IsNullOrEmpty(m_ActionKey))
            {
                Debug.LogWarning(
                    $"{nameof(TranslateActuator)} on '{name}' has an empty {nameof(ActionKey)}.",
                    this
                );
            }
        }

        public Space GetActionSpace()
        {
            var space = new BoxSpace();
            for (var i = 0; i < DimensionCount; i++)
            {
                space.Dimensions.Add(
                    m_MaxDisplacement > 0f
                        ? new BoxSpaceDimension(-m_MaxDisplacement, m_MaxDisplacement)
                        : new BoxSpaceDimension()
                );
            }

            return space;
        }

        /// <summary>
        /// Checks that <paramref name="action"/> is a 3-float in-bounds <see cref="BoxPoint"/>.
        /// Does not move the transform.
        /// </summary>
        public void ValidateAction(Point action)
        {
            ReadTranslation(action);
        }

        public void TakeAction(Point action)
        {
            transform.position += ReadTranslation(action);
        }

        Vector3 ReadTranslation(Point action)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            if (!(action is BoxPoint boxPoint))
            {
                throw new ArgumentException(
                    $"Expected {nameof(BoxPoint)}, got {action.GetType().Name}.",
                    nameof(action)
                );
            }

            if (boxPoint.Values == null || boxPoint.Values.Count != DimensionCount)
            {
                var actualCount = boxPoint.Values == null ? 0 : boxPoint.Values.Count;
                throw new ArgumentException(
                    $"Action '{m_ActionKey}' expected {nameof(BoxPoint)} with {DimensionCount} values, got {actualCount}.",
                    nameof(action)
                );
            }

            foreach (var value in boxPoint.Values)
            {
                if (
                    float.IsNaN(value)
                    || float.IsInfinity(value)
                    || (m_MaxDisplacement > 0f && Mathf.Abs(value) > m_MaxDisplacement)
                )
                    throw new ArgumentException(
                        "Translation is outside the configured action bounds.",
                        nameof(action)
                    );
            }

            return new Vector3(boxPoint.Values[0], boxPoint.Values[1], boxPoint.Values[2]);
        }
    }
}
