// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using AMD.Schola;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Agent
{
    /// <summary>
    /// US1 agent aggregation tests.
    /// T3: stable Dict keys; nested values update with the scene.
    /// </summary>
    public sealed class ScholaAgentObserveTests
    {
        const float Tolerance = 1e-5f;
        const string PositionKey = "position";
        const string VelocityProxyKey = "position_b";

        [Test]
        public void DefineAndObserve_UseStableDictKeys_AndNestedValuesUpdate()
        {
            var agentGo = new GameObject("ScholaAgent");
            var sensorAGo = new GameObject("SensorA");
            var sensorBGo = new GameObject("SensorB");
            try
            {
                sensorAGo.transform.SetParent(agentGo.transform);
                sensorBGo.transform.SetParent(agentGo.transform);

                var sensorA = sensorAGo.AddComponent<PositionSensor>();
                sensorA.ObservationKey = PositionKey;
                var sensorB = sensorBGo.AddComponent<PositionSensor>();
                sensorB.ObservationKey = VelocityProxyKey;

                var agent = agentGo.AddComponent<ScholaAgent>();
                agent.SetSensors(new[] { sensorA, sensorB });

                sensorA.InitSensor();
                sensorB.InitSensor();

                var defined = agent.Define() as DictSpace;
                Assert.That(defined, Is.Not.Null, "Define must return a DictSpace for multi-sensor agents");
                Assert.That(defined.Spaces.ContainsKey(PositionKey), Is.True,
                    "Define must include stable key for sensor A");
                Assert.That(defined.Spaces.ContainsKey(VelocityProxyKey), Is.True,
                    "Define must include stable key for sensor B");
                Assert.That(defined.Spaces[PositionKey], Is.TypeOf<BoxSpace>());
                Assert.That(defined.Spaces[VelocityProxyKey], Is.TypeOf<BoxSpace>());

                sensorAGo.transform.position = new Vector3(1f, 0f, 0f);
                sensorBGo.transform.position = new Vector3(0f, 2f, 0f);

                var obsA = agent.Observe() as DictPoint;
                Assert.That(obsA, Is.Not.Null, "Observe must return a DictPoint");
                Assert.That(obsA.Points.Keys, Is.EquivalentTo(new[] { PositionKey, VelocityProxyKey }),
                    "Observe keys must match Define keys");

                var nestedA = obsA.Points[PositionKey] as BoxPoint;
                var nestedB = obsA.Points[VelocityProxyKey] as BoxPoint;
                Assert.That(nestedA, Is.Not.Null);
                Assert.That(nestedB, Is.Not.Null);
                Assert.That(nestedA.Values[0], Is.EqualTo(1f).Within(Tolerance));
                Assert.That(nestedB.Values[1], Is.EqualTo(2f).Within(Tolerance));

                sensorAGo.transform.position = new Vector3(9f, 8f, 7f);
                sensorBGo.transform.position = new Vector3(3f, 4f, 5f);

                var obsB = agent.Observe() as DictPoint;
                Assert.That(obsB, Is.Not.Null);
                Assert.That(obsB.Points.Keys, Is.EquivalentTo(obsA.Points.Keys),
                    "Dict keys must remain stable across Observe calls");

                var nestedA2 = obsB.Points[PositionKey] as BoxPoint;
                var nestedB2 = obsB.Points[VelocityProxyKey] as BoxPoint;
                Assert.That(nestedA2, Is.Not.Null);
                Assert.That(nestedB2, Is.Not.Null);
                Assert.That(nestedA2.Values[0], Is.EqualTo(9f).Within(Tolerance));
                Assert.That(nestedA2.Values[1], Is.EqualTo(8f).Within(Tolerance));
                Assert.That(nestedA2.Values[2], Is.EqualTo(7f).Within(Tolerance));
                Assert.That(nestedB2.Values[0], Is.EqualTo(3f).Within(Tolerance));
                Assert.That(nestedB2.Values[1], Is.EqualTo(4f).Within(Tolerance));
                Assert.That(nestedB2.Values[2], Is.EqualTo(5f).Within(Tolerance));

                Assert.That(Mathf.Abs(nestedA2.Values[0] - nestedA.Values[0]), Is.GreaterThan(Tolerance),
                    "Nested observation must not be stale from prior Observe");
            }
            finally
            {
                Object.DestroyImmediate(agentGo);
            }
        }
    }
}
