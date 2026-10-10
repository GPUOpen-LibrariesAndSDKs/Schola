// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using AMD.Schola;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Sensors
{
    /// <summary>
    /// US1 AC1/AC2 sensor tests.
    /// T1: observation values update with the scene.
    /// T2: observation values keep a consistent format matching GetObservationSpace.
    /// </summary>
    public sealed class PositionSensorObservationTests
    {
        const float Tolerance = 1e-5f;

        [Test]
        public void CollectObservations_UpdatesWhenTransformChanges()
        {
            var go = new GameObject("PositionSensor_Freshness");
            try
            {
                var sensor = go.AddComponent<PositionSensor>();
                sensor.InitSensor();

                var stateA = new Vector3(1f, 2f, 3f);
                go.transform.position = stateA;
                var obsA = sensor.CollectObservations() as BoxPoint;

                Assert.That(obsA, Is.Not.Null, "CollectObservations must return a BoxPoint");
                Assert.That(obsA.Values.Count, Is.EqualTo(3));
                Assert.That(obsA.Values[0], Is.EqualTo(stateA.x).Within(Tolerance));
                Assert.That(obsA.Values[1], Is.EqualTo(stateA.y).Within(Tolerance));
                Assert.That(obsA.Values[2], Is.EqualTo(stateA.z).Within(Tolerance));

                var stateB = new Vector3(-4f, 5.5f, 6f);
                go.transform.position = stateB;
                var obsB = sensor.CollectObservations() as BoxPoint;

                Assert.That(obsB, Is.Not.Null, "CollectObservations must return a BoxPoint after scene change");
                Assert.That(obsB.Values.Count, Is.EqualTo(3));
                Assert.That(obsB.Values[0], Is.EqualTo(stateB.x).Within(Tolerance));
                Assert.That(obsB.Values[1], Is.EqualTo(stateB.y).Within(Tolerance));
                Assert.That(obsB.Values[2], Is.EqualTo(stateB.z).Within(Tolerance));

                Assert.That(Mathf.Abs(obsB.Values[0] - obsA.Values[0]), Is.GreaterThan(Tolerance),
                    "Observation must not be stale from state A");
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void CollectObservations_MatchesObservationSpaceFormatAcrossStates()
        {
            var go = new GameObject("PositionSensor_Format");
            try
            {
                var sensor = go.AddComponent<PositionSensor>();
                sensor.InitSensor();

                var space = sensor.GetObservationSpace() as BoxSpace;
                Assert.That(space, Is.Not.Null, "GetObservationSpace must return a BoxSpace");
                Assert.That(space.Dimensions.Count, Is.EqualTo(3),
                    "Position sensor space must declare 3 dimensions (x, y, z)");

                go.transform.position = new Vector3(1f, 2f, 3f);
                var obsA = sensor.CollectObservations() as BoxPoint;
                Assert.That(obsA, Is.Not.Null);
                AssertObservationMatchesSpace(obsA, space);

                go.transform.position = new Vector3(10f, -20f, 0.25f);
                var obsB = sensor.CollectObservations() as BoxPoint;
                Assert.That(obsB, Is.Not.Null);
                AssertObservationMatchesSpace(obsB, space);

                Assert.That(obsB.Values.Count, Is.EqualTo(obsA.Values.Count));
                Assert.That(obsB.Shape, Is.EqualTo(obsA.Shape),
                    "Shape must be stable across CollectObservations calls");
                Assert.That(obsA.Values.Count, Is.EqualTo(3));
                Assert.That(obsB.Values[0], Is.EqualTo(10f).Within(Tolerance), "Ordering: index 0 is x");
                Assert.That(obsB.Values[1], Is.EqualTo(-20f).Within(Tolerance), "Ordering: index 1 is y");
                Assert.That(obsB.Values[2], Is.EqualTo(0.25f).Within(Tolerance), "Ordering: index 2 is z");
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        static void AssertObservationMatchesSpace(BoxPoint point, BoxSpace space)
        {
            Assert.That(point.Values.Count, Is.EqualTo(space.Dimensions.Count),
                "BoxPoint length must match BoxSpace dimension count");

            if (space.Shape != null && space.Shape.Count > 0)
            {
                Assert.That(point.Shape, Is.EqualTo(space.Shape),
                    "BoxPoint shape must match BoxSpace shape when shape is declared");
            }
        }
    }
}
