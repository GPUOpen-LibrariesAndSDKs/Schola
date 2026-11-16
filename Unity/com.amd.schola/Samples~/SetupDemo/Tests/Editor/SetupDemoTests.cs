// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using AMD.Schola.Samples.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AMD.Schola.Samples.Tests
{
    public sealed class SetupDemoTests
    {
        Scene m_Scene;
        SetupDemoController m_Controller;

        [SetUp]
        public void SetUp()
        {
            m_Scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            m_Controller = SetupDemoSceneBuilder.Populate(m_Scene);
        }

        [TearDown]
        public void TearDown()
        {
            EditorSceneManager.CloseScene(m_Scene, true);
        }

        [Test]
        public void Steps_UseScholaObservationsAndActions_ReachTargetWithoutOvershoot_AndFollowMovedTarget()
        {
            Assert.That(m_Controller.ValidateConfiguration(), Is.True);
            var agent = m_Controller.GetComponent<ScholaAgent>();
            var target = agent.Sensors[1];
            var actionSpace = (DictSpace)agent.DefineActionSpace();
            Assert.That(
                actionSpace.Spaces.Keys,
                Is.EquivalentTo(new[] { SetupDemoController.TranslateKey })
            );
            Assert.That(
                ((BoxSpace)actionSpace.Spaces[SetupDemoController.TranslateKey]).Dimensions.Count,
                Is.EqualTo(3)
            );
            Assert.That(
                ((DictSpace)agent.Define()).Spaces.Keys,
                Is.EquivalentTo(
                    new[]
                    {
                        SetupDemoController.AgentPositionKey,
                        SetupDemoController.TargetPositionKey,
                    }
                )
            );

            m_Controller.Step(0.5f);
            Assert.That(agent.transform.position.x, Is.EqualTo(-2f).Within(1e-5f));
            var observed = (BoxPoint)
                m_Controller.LastObservation.Points[SetupDemoController.AgentPositionKey];
            Assert.That(observed.Values[0], Is.EqualTo(-2f).Within(1e-5f));
            m_Controller.Step(100f);
            Assert.That(m_Controller.HasArrived, Is.True);
            Assert.That(
                Vector3.Distance(agent.transform.position, target.transform.position),
                Is.LessThan(1e-5f)
            );

            target.transform.position = new Vector3(0f, 0.5f, 2f);
            m_Controller.Step(0.5f);
            Assert.That(m_Controller.HasArrived, Is.False);
            m_Controller.Step(100f);
            Assert.That(
                Vector3.Distance(agent.transform.position, target.transform.position),
                Is.LessThan(1e-5f)
            );
        }

        [Test]
        public void DuplicateObservationKeys_ReportConfigurationError()
        {
            var agent = m_Controller.GetComponent<ScholaAgent>();
            agent.Sensors[1].ObservationKey = SetupDemoController.AgentPositionKey;
            LogAssert.Expect(
                LogType.Error,
                "Schola Setup Demo: Sensors must be [cube agent_position, target target_position]. See the setup guide."
            );
            Assert.That(m_Controller.ValidateConfiguration(), Is.False);
        }

        [Test]
        public void MissingActuator_ReportsConfigurationError()
        {
            var agent = m_Controller.GetComponent<ScholaAgent>();
            agent.SetActuators(new TranslateActuator[0]);
            LogAssert.Expect(
                LogType.Error,
                "Schola Setup Demo: Actuators must contain one TranslateActuator on the cube with Action Key 'translate'. See the setup guide."
            );
            Assert.That(m_Controller.ValidateConfiguration(), Is.False);
        }

        [Test]
        public void Step_ActionKeyMismatchIsRejectedBySchola_WithoutMovingCube()
        {
            Assert.That(m_Controller.ValidateConfiguration(), Is.True);
            var agent = m_Controller.GetComponent<ScholaAgent>();
            var before = agent.transform.position;
            agent.Actuators[0].ActionKey = "other";
            var error = Assert.Throws<System.ArgumentException>(() => m_Controller.Step(0.5f));
            Assert.That(error.Message, Does.Contain("other"));
            Assert.That(agent.transform.position, Is.EqualTo(before));
        }

        [Test]
        public void SavedScene_ReloadsSensorAndActuatorReferencesAndKeys()
        {
            var path = AssetDatabase.GenerateUniqueAssetPath("Assets/ScholaSetupDemoTest.unity");
            try
            {
                Assert.That(EditorSceneManager.SaveScene(m_Scene, path), Is.True);
                EditorSceneManager.CloseScene(m_Scene, true);
                m_Scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                SetupDemoController loaded = null;
                foreach (var root in m_Scene.GetRootGameObjects())
                {
                    if (root.TryGetComponent<SetupDemoController>(out var controller))
                    {
                        loaded = controller;
                    }
                }

                Assert.That(loaded, Is.Not.Null);
                Assert.That(loaded.ValidateConfiguration(), Is.True);
                var loadedAgent = loaded.GetComponent<ScholaAgent>();
                Assert.That(
                    loadedAgent.Actuators[0],
                    Is.SameAs(loaded.GetComponent<TranslateActuator>())
                );
                Assert.That(
                    loadedAgent.Actuators[0].ActionKey,
                    Is.EqualTo(SetupDemoController.TranslateKey)
                );
                loaded.Step(0.5f);
                Assert.That(loaded.LastObservation.Points.Count, Is.EqualTo(2));
                Assert.That(loadedAgent.transform.position.x, Is.EqualTo(-2f).Within(1e-5f));
            }
            finally
            {
                AssetDatabase.DeleteAsset(path);
            }
        }
    }
}
