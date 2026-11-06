// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using AMD.Schola;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Environment
{
    /// <summary>
    /// Multi-agent environment: agent-id maps, independent rewards and episode flags, and
    /// rejection of an action map that does not match the registered agents.
    /// </summary>
    public sealed class ScholaMultiAgentEnvironmentTests
    {
        const string AgentAId = "agentA";
        const string AgentBId = "agentB";
        static readonly Vector3 StartA = new Vector3(1f, 2f, 3f);
        static readonly Vector3 StartB = new Vector3(4f, 5f, 6f);

        GameObject m_Root;
        ScholaMultiAgentEnvironment m_Environment;
        ScholaAgent m_AgentA;
        ScholaAgent m_AgentB;
        ScholaEnvironmentTestFixture.ScriptedReward m_RewardA;
        ScholaEnvironmentTestFixture.ScriptedReward m_RewardB;
        ScholaEnvironmentTestFixture.ScriptedEpisodeStatus m_StatusA;
        ScholaEnvironmentTestFixture.ScriptedEpisodeStatus m_StatusB;
        ScholaEnvironmentTestFixture.InMemoryTrainingStateSink m_Sink;

        [SetUp]
        public void SetUp()
        {
            m_Root = new GameObject("ScholaMultiAgentEnvironment");
            m_Environment = m_Root.AddComponent<ScholaMultiAgentEnvironment>();
            m_AgentA = CreateAgent("AgentA", StartA, out m_RewardA, out m_StatusA);
            m_AgentB = CreateAgent("AgentB", StartB, out m_RewardB, out m_StatusB);
            m_Sink = new ScholaEnvironmentTestFixture.InMemoryTrainingStateSink();
            m_Environment.SetAgents(
                new[]
                {
                    new KeyValuePair<string, ScholaAgent>(AgentAId, m_AgentA),
                    new KeyValuePair<string, ScholaAgent>(AgentBId, m_AgentB),
                }
            );
            m_Environment.SetTrainingStateSink(m_Sink);
        }

        [TearDown]
        public void TearDown()
        {
            if (m_Root != null)
            {
                UnityEngine.Object.DestroyImmediate(m_Root);
            }
        }

        [Test]
        public void Initialize_ReturnsOneDefinitionPerAgentId()
        {
            var definitions = m_Environment.Initialize();

            Assert.That(definitions.Keys, Is.EquivalentTo(new[] { AgentAId, AgentBId }));
            AssertSpace(definitions[AgentAId]);
            AssertSpace(definitions[AgentBId]);
        }

        [Test]
        public void Step_AppliesEachAgentActionAndKeepsRewardAndStatusIndependent()
        {
            m_Environment.Initialize();
            m_RewardA.Reward = 1.5f;
            m_RewardB.Reward = -0.25f;
            m_StatusA.Terminated = true;
            m_StatusB.Truncated = true;

            var states = m_Environment.Step(
                new Dictionary<string, Point>
                {
                    [AgentAId] = Translate(0.4f),
                    [AgentBId] = Translate(0f, 0.7f),
                }
            );

            Assert.That(states[AgentAId].Reward, Is.EqualTo(1.5f));
            Assert.That(states[AgentAId].Terminated, Is.True);
            Assert.That(states[AgentAId].Truncated, Is.False);
            Assert.That(states[AgentBId].Reward, Is.EqualTo(-0.25f));
            Assert.That(states[AgentBId].Terminated, Is.False);
            Assert.That(states[AgentBId].Truncated, Is.True);
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    states[AgentAId].Observations
                ),
                StartA + new Vector3(0.4f, 0f, 0f)
            );
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    states[AgentBId].Observations
                ),
                StartB + new Vector3(0f, 0.7f, 0f)
            );
            Assert.That(m_Sink.StateMaps, Has.Count.EqualTo(1));
            Assert.That(m_Sink.StateMaps[0][AgentAId].Reward, Is.EqualTo(1.5f));
            Assert.That(m_Sink.StateMaps[0][AgentBId].Truncated, Is.True);
        }

        [Test]
        public void Reset_RestoresBothPosesAndReturnsInitialStates()
        {
            m_Environment.Initialize();
            m_Environment.Step(
                new Dictionary<string, Point>
                {
                    [AgentAId] = Translate(0.4f),
                    [AgentBId] = Translate(0f, 0.7f),
                }
            );
            m_StatusA.Terminated = true;

            var initial = m_Environment.Reset();

            Assert.That(initial[AgentAId], Is.InstanceOf<InitialAgentState>());
            Assert.That(initial[AgentAId], Is.Not.InstanceOf<AgentState>());
            Assert.That(initial[AgentBId], Is.InstanceOf<InitialAgentState>());
            Assert.That(initial[AgentBId], Is.Not.InstanceOf<AgentState>());
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    initial[AgentAId].Observations
                ),
                StartA
            );
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    initial[AgentBId].Observations
                ),
                StartB
            );
            Assert.That(m_Sink.InitialStateMaps, Has.Count.EqualTo(1));
            Assert.That(m_StatusA.Terminated, Is.False);
        }

        [Test]
        public void Step_MissingAgentId_DoesNotMoveAnyAgent()
        {
            AssertActionMapRejected(
                new Dictionary<string, Point> { [AgentAId] = Translate(1f) }
            );
        }

        [Test]
        public void Step_UnknownAgentId_DoesNotMoveAnyAgent()
        {
            AssertActionMapRejected(
                new Dictionary<string, Point>
                {
                    [AgentAId] = Translate(1f),
                    [AgentBId] = Translate(1f),
                    ["other"] = Translate(1f),
                }
            );
        }

        [Test]
        public void Step_NullAction_DoesNotMoveAnyAgent()
        {
            AssertActionMapRejected(
                new Dictionary<string, Point> { [AgentAId] = Translate(1f), [AgentBId] = null }
            );
        }

        [Test]
        public void Step_NullActionMap_Throws()
        {
            m_Environment.Initialize();

            Assert.Throws<ArgumentNullException>(() => m_Environment.Step(null));
        }

        [Test]
        public void Initialize_DuplicateAgentId_Throws()
        {
            m_Environment.SetAgents(
                new[]
                {
                    new KeyValuePair<string, ScholaAgent>(AgentAId, m_AgentA),
                    new KeyValuePair<string, ScholaAgent>(AgentAId, m_AgentB),
                }
            );

            Assert.Throws<InvalidOperationException>(() => m_Environment.Initialize());
        }

        [Test]
        public void Initialize_EmptyAgentId_Throws()
        {
            m_Environment.SetAgents(
                new[] { new KeyValuePair<string, ScholaAgent>(" ", m_AgentA) }
            );

            Assert.Throws<InvalidOperationException>(() => m_Environment.Initialize());
        }

        [Test]
        public void Initialize_NullAgent_Throws()
        {
            m_Environment.SetAgents(
                new[] { new KeyValuePair<string, ScholaAgent>(AgentAId, null) }
            );

            Assert.Throws<InvalidOperationException>(() => m_Environment.Initialize());
        }

        void AssertActionMapRejected(IReadOnlyDictionary<string, Point> actions)
        {
            m_Environment.Initialize();
            var beforeA = m_AgentA.transform.position;
            var beforeB = m_AgentB.transform.position;

            Assert.Throws<ArgumentException>(() => m_Environment.Step(actions));

            AssertPosition(m_AgentA.transform.position, beforeA);
            AssertPosition(m_AgentB.transform.position, beforeB);
            Assert.That(m_Sink.StateMaps, Is.Empty);
        }

        ScholaAgent CreateAgent(
            string name,
            Vector3 localPosition,
            out ScholaEnvironmentTestFixture.ScriptedReward reward,
            out ScholaEnvironmentTestFixture.ScriptedEpisodeStatus status
        )
        {
            var agentGo = new GameObject(name);
            agentGo.transform.SetParent(m_Root.transform);
            agentGo.transform.localPosition = localPosition;
            agentGo.transform.localRotation = Quaternion.identity;
            agentGo.transform.localScale = Vector3.one;

            var sensor = agentGo.AddComponent<PositionSensor>();
            sensor.ObservationKey = ScholaEnvironmentTestFixture.PositionKey;
            var actuator = agentGo.AddComponent<TranslateActuator>();
            actuator.ActionKey = ScholaEnvironmentTestFixture.TranslateKey;
            reward = agentGo.AddComponent<ScholaEnvironmentTestFixture.ScriptedReward>();
            status = agentGo.AddComponent<ScholaEnvironmentTestFixture.ScriptedEpisodeStatus>();

            var agent = agentGo.AddComponent<ScholaAgent>();
            agent.SetSensors(new[] { sensor });
            agent.SetActuators(new[] { actuator });
            return agent;
        }

        static DictPoint Translate(float dx, float dy = 0f, float dz = 0f)
        {
            return new DictPoint
            {
                Points =
                {
                    [ScholaEnvironmentTestFixture.TranslateKey] = new BoxPoint
                    {
                        Values = { dx, dy, dz },
                    },
                },
            };
        }

        static void AssertSpace(InteractionDefinition definition)
        {
            var observationSpace = definition.ObservationSpace as DictSpace;
            Assert.That(observationSpace, Is.Not.Null);
            Assert.That(
                observationSpace.Spaces.ContainsKey(ScholaEnvironmentTestFixture.PositionKey),
                Is.True
            );

            var actionSpace = definition.ActionSpace as DictSpace;
            Assert.That(actionSpace, Is.Not.Null);
            Assert.That(
                actionSpace.Spaces.ContainsKey(ScholaEnvironmentTestFixture.TranslateKey),
                Is.True
            );
        }

        static void AssertPosition(Vector3 actual, Vector3 expected)
        {
            Assert.That(
                actual.x,
                Is.EqualTo(expected.x).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
            Assert.That(
                actual.y,
                Is.EqualTo(expected.y).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
            Assert.That(
                actual.z,
                Is.EqualTo(expected.z).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
        }
    }

    /// <summary>
    /// Single-agent environments stay on <see cref="IEnvironment"/>. The adapter is what
    /// injects <see cref="ScholaEnvironment.DefaultAgentId"/> into the multi-agent maps.
    /// </summary>
    public sealed class SingleAgentEnvironmentAdapterTests
    {
        ScholaEnvironmentTestFixture m_Fixture;

        [SetUp]
        public void SetUp()
        {
            m_Fixture = ScholaEnvironmentTestFixture.Create();
        }

        [TearDown]
        public void TearDown()
        {
            m_Fixture?.Dispose();
        }

        [Test]
        public void ResetAndStep_RoundTripUnderSingleAgentKey()
        {
            var adapter = new SingleAgentEnvironmentAdapter(m_Fixture.Environment);
            var initial = adapter.Reset();

            Assert.That(initial.Keys, Is.EquivalentTo(new[] { ScholaEnvironment.DefaultAgentId }));
            Assert.That(
                initial[ScholaEnvironment.DefaultAgentId],
                Is.InstanceOf<InitialAgentState>()
            );
            Assert.That(
                initial[ScholaEnvironment.DefaultAgentId],
                Is.Not.InstanceOf<AgentState>()
            );
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    initial[ScholaEnvironment.DefaultAgentId].Observations
                ),
                ScholaEnvironmentTestFixture.StartPose
            );

            m_Fixture.Reward.Reward = 2f;
            var states = adapter.Step(
                new Dictionary<string, Point>
                {
                    [ScholaEnvironment.DefaultAgentId] = m_Fixture.TranslateAction(0.5f),
                }
            );

            Assert.That(states.Keys, Is.EquivalentTo(new[] { ScholaEnvironment.DefaultAgentId }));
            Assert.That(states[ScholaEnvironment.DefaultAgentId].Reward, Is.EqualTo(2f));
            AssertPosition(
                ScholaEnvironmentTestFixture.PositionFromObservations(
                    states[ScholaEnvironment.DefaultAgentId].Observations
                ),
                ScholaEnvironmentTestFixture.StartPose + new Vector3(0.5f, 0f, 0f)
            );
        }

        [Test]
        public void Step_UnknownAgentId_DoesNotMoveTheAgent()
        {
            var adapter = new SingleAgentEnvironmentAdapter(m_Fixture.Environment);
            var before = m_Fixture.AgentGo.transform.position;

            Assert.Throws<ArgumentException>(() =>
                adapter.Step(
                    new Dictionary<string, Point> { ["other"] = m_Fixture.TranslateAction(1f) }
                )
            );

            AssertPosition(m_Fixture.AgentGo.transform.position, before);
        }

        static void AssertPosition(Vector3 actual, Vector3 expected)
        {
            Assert.That(
                actual.x,
                Is.EqualTo(expected.x).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
            Assert.That(
                actual.y,
                Is.EqualTo(expected.y).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
            Assert.That(
                actual.z,
                Is.EqualTo(expected.z).Within(ScholaEnvironmentTestFixture.Tolerance)
            );
        }
    }
}
