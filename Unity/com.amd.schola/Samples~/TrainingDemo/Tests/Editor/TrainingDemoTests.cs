// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
using AMD.Schola.Samples.TrainingDemo.Editor;
using AMD.Schola.Training;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AMD.Schola.Samples.TrainingDemo.Tests
{
    public sealed class TrainingDemoTests
    {
        [UnityEngine.TestTools.UnityTearDown]
        public System.Collections.IEnumerator Cleanup()
        {
            if (Application.isPlaying)
                yield return new UnityEngine.TestTools.ExitPlayMode();
            foreach (var runner in Object.FindObjectsByType<PolicyRunner>(FindObjectsSortMode.None))
                runner.StopRunning();
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        }

        [UnityEngine.TestTools.UnityTest]
        public System.Collections.IEnumerator ShippedInferenceScene_RunsWithoutPythonAndReachesTarget()
        {
            var script = AssetDatabase.FindAssets("TrainingDemoSceneBuilder t:MonoScript")[0];
            var sample = System.IO.Path.GetDirectoryName(
                System.IO.Path.GetDirectoryName(AssetDatabase.GUIDToAssetPath(script))
            );
            EditorSceneManager.OpenScene(sample + "/Scenes/ScholaInferenceDemo.unity");
            yield return new UnityEngine.TestTools.EnterPlayMode();
            var agent = Object.FindFirstObjectByType<ScholaAgent>();
            var target = GameObject.Find("Target").transform;
            for (
                var frame = 0;
                frame < 256 && Vector3.Distance(agent.transform.position, target.position) > 0.2f;
                frame++
            )
                yield return null;
            yield return null;
            Assert.That(
                Vector3.Distance(agent.transform.position, target.position),
                Is.LessThanOrEqualTo(0.2f)
            );
            Assert.That(Object.FindFirstObjectByType<TrainingServer>(), Is.Null);
            Assert.That(Object.FindFirstObjectByType<PolicyRunner>().IsRunning, Is.False);
            yield return new UnityEngine.TestTools.ExitPlayMode();
        }

        [TestCase(false)]
        [TestCase(true)]
        public void GeneratedScene_SavesReloadsAndHasOneActionDriver(bool inference)
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            TrainingDemoSceneBuilder.Populate(inference);
            const string path = "Assets/US6-test-scene.unity";
            try
            {
                Assert.That(
                    EditorSceneManager.SaveScene(
                        UnityEngine.SceneManagement.SceneManager.GetActiveScene(),
                        path
                    ),
                    Is.True
                );
                EditorSceneManager.OpenScene(path);
                var agent = Object.FindFirstObjectByType<ScholaAgent>();
                Assert.That(agent.Sensors.Count, Is.EqualTo(2));
                Assert.That(agent.Actuators.Count, Is.EqualTo(1));
                Assert.That(agent.Actuators[0].MaxDisplacement, Is.EqualTo(0.1f));
                Assert.That(
                    Object.FindFirstObjectByType<TrainingServer>() != null,
                    Is.EqualTo(!inference)
                );
                var runner = Object.FindFirstObjectByType<PolicyRunner>();
                Assert.That(runner != null, Is.EqualTo(inference));
                if (inference)
                {
                    Assert.That(runner.Model, Is.Not.Null);
                    Assert.That(runner.StartRunning(), Is.True);
                    runner.StopRunning();
                }
                else
                {
                    var environment = Object.FindFirstObjectByType<ScholaEnvironment>();
                    environment.Initialize();
                    environment.Reset();
                    // Reach the goal via real environment steps, then verify snapshot reset.
                    var initial = agent.transform.position;
                    var action = new DictPoint();
                    var delta = new BoxPoint();
                    delta.Values.AddRange(new[] { 0.1f, 0f, 0f });
                    action.Points["translate"] = delta;
                    for (var i = 0; i < 20; i++)
                        environment.Step(action);
                    Assert.That(agent.transform.position.x, Is.GreaterThan(initial.x + 1.7f));
                    environment.Reset();
                    Assert.That(agent.transform.position, Is.EqualTo(initial));
                }
            }
            finally
            {
                foreach (
                    var runner in Object.FindObjectsByType<PolicyRunner>(FindObjectsSortMode.None)
                )
                    runner.StopRunning();
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                AssetDatabase.DeleteAsset(path);
            }
        }
    }
}
