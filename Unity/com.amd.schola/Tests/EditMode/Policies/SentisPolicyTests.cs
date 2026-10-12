// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;
using System.Text.RegularExpressions;
using AMD.Schola;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace AMD.Schola.Tests.Policies {
    /// <summary>
    /// US5 policy tests using a PPO CartPole model exported by Schola's SB3 exporter.
    /// Model contract: input "obs" float32 [batch, 4], output "action" int64 [batch] (0 = left, 1 = right).
    /// </summary>
    public sealed class SentisPolicyTests {
        const string ModelPath = "Packages/com.amd.schola/Tests/EditMode/Policies/TestModels/cartpole.onnx";
        const string ObservationKey = "obs";
        const string ActionKey = "action";

        ModelAsset m_ModelAsset;
        SentisPolicy m_Policy;

        [SetUp]
        public void SetUp() {
            m_ModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            Assert.That(m_ModelAsset, Is.Not.Null, $"Test model missing at {ModelPath}");
            m_Policy = new SentisPolicy(m_ModelAsset);
        }

        [TearDown]
        public void TearDown() {
            m_Policy.Dispose();
        }

        [Test]
        public void Init_WithMatchingKeys_Succeeds() {
            Assert.That(m_Policy.Init(CartPoleDefinition()), Is.True);
            Assert.That(m_Policy.IsInitialized, Is.True);
        }

        [TestCase(0.05f, 1, TestName = "Think_PoleLeaningRight_PushesRight")]
        [TestCase(-0.05f, 0, TestName = "Think_PoleLeaningLeft_PushesLeft")]
        public void Think_ReturnsTrainedAction(float poleAngle, int expectedAction) {
            Assert.That(m_Policy.Init(CartPoleDefinition()), Is.True);

            Assert.That(m_Policy.Think(CartPoleObservation(poleAngle), out var actions), Is.True);

            var actionDict = actions as DictPoint;
            Assert.That(actionDict, Is.Not.Null, "Think must return a DictPoint");
            Assert.That(actionDict.Points.Keys, Is.EquivalentTo(new[] { ActionKey }));
            var action = actionDict.Points[ActionKey] as DiscretePoint;
            Assert.That(action, Is.Not.Null, "Discrete action space must produce a DiscretePoint");
            Assert.That(action.Value, Is.EqualTo(expectedAction));
        }

        [Test]
        public void Init_ObservationKeyNotInModel_FailsWithClearError() {
            var definition = CartPoleDefinition();
            var observationSpace = (DictSpace)definition.ObservationSpace;
            observationSpace.Spaces["position"] = observationSpace.Spaces[ObservationKey];
            observationSpace.Spaces.Remove(ObservationKey);

            LogAssert.Expect(LogType.Error, new Regex("Model has no input named 'position'"));
            Assert.That(m_Policy.Init(definition), Is.False);
            Assert.That(m_Policy.IsInitialized, Is.False);
        }

        [Test]
        public void Init_ActionKeyNotInModel_FailsWithClearError() {
            var definition = CartPoleDefinition();
            var actionSpace = (DictSpace)definition.ActionSpace;
            actionSpace.Spaces["jump"] = actionSpace.Spaces[ActionKey];
            actionSpace.Spaces.Remove(ActionKey);

            LogAssert.Expect(LogType.Error, new Regex("Model has no output named 'jump'"));
            Assert.That(m_Policy.Init(definition), Is.False);
        }

        [Test]
        public void Think_BeforeInit_Fails() {
            LogAssert.Expect(LogType.Error, new Regex("Init must succeed before Think"));
            Assert.That(m_Policy.Think(CartPoleObservation(0f), out var actions), Is.False);
            Assert.That(actions, Is.Null);
        }

        static InteractionDefinition CartPoleDefinition() {
            var observation = new BoxSpace();
            for (var i = 0; i < 4; i++) {
                observation.Dimensions.Add(new BoxSpaceDimension());
            }

            return new InteractionDefinition(
                new DictSpace { Spaces = { [ObservationKey] = observation } },
                new DictSpace { Spaces = { [ActionKey] = new DiscreteSpace(2) } });
        }

        /// <summary>
        /// CartPole observation: cart position, cart velocity, pole angle, pole angular velocity.
        /// </summary>
        static DictPoint CartPoleObservation(float poleAngle) {
            return new DictPoint {
                Points ={
                    [ObservationKey] = new BoxPoint { Values = new List<float> { 0f, 0f, poleAngle, 0f } },
                },
            };
        }
    }
}
