// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AMD.Schola.Samples.Editor
{
    /// <summary>Lets Unity create and serialize the scene and built-in asset references.</summary>
    public static class SetupDemoSceneBuilder
    {
        [MenuItem("Tools/AMD Schola/Create Setup Demo Scene")]
        public static void CreateScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning("Stop Play Mode before creating the Schola Setup Demo scene.");
                return;
            }

            var path = EditorUtility.SaveFilePanelInProject(
                "Save Schola Setup Demo",
                "ScholaSetupDemo",
                "unity",
                "Choose a scene path under Assets. Existing scenes require overwrite confirmation."
            );
            if (
                string.IsNullOrEmpty(path)
                || !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()
            )
            {
                return;
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Populate(scene);
            if (EditorSceneManager.SaveScene(scene, path))
            {
                Selection.activeGameObject = scene.GetRootGameObjects()[0];
                Debug.Log(
                    $"Schola Setup Demo saved to {path}. Press Play; see the Game view and Console."
                );
            }
        }

        /// <summary>Builds only into the supplied empty scene, also used by the sample tests.</summary>
        public static SetupDemoController Populate(Scene scene)
        {
            var previous = SceneManager.GetActiveScene();
            SceneManager.SetActiveScene(scene);
            try
            {
                var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.name = "Schola Cube";
                cube.GetComponent<Renderer>().sharedMaterial = Material(
                    "Agent",
                    new Color(0.1f, 0.45f, 0.9f)
                );
                cube.transform.position = new Vector3(-3f, 0.5f, 0f);
                var cubeSensor = cube.AddComponent<PositionSensor>();
                cubeSensor.ObservationKey = SetupDemoController.AgentPositionKey;
                var agent = cube.AddComponent<ScholaAgent>();
                var actuator = cube.AddComponent<TranslateActuator>();
                actuator.ActionKey = SetupDemoController.TranslateKey;
                agent.SetActuators(new[] { actuator });

                var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                target.name = "Target";
                target.GetComponent<Renderer>().sharedMaterial = Material(
                    "Target",
                    new Color(1f, 0.5f, 0.05f)
                );
                target.transform.position = new Vector3(3f, 0.5f, 0f);
                target.transform.localScale = Vector3.one * 0.4f;
                var targetSensor = target.AddComponent<PositionSensor>();
                targetSensor.ObservationKey = SetupDemoController.TargetPositionKey;
                agent.SetSensors(new[] { cubeSensor, targetSensor });

                var controller = cube.AddComponent<SetupDemoController>();
                controller.Configure(agent, targetSensor);

                var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
                ground.name = "Ground";
                ground.GetComponent<Renderer>().sharedMaterial = Material(
                    "Ground",
                    new Color(0.16f, 0.2f, 0.26f)
                );
                RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
                RenderSettings.ambientLight = Color.gray * 0.6f;
                ground.transform.localScale = Vector3.one * 1.5f;

                var cameraObject = new GameObject("Main Camera");
                cameraObject.tag = "MainCamera";
                var camera = cameraObject.AddComponent<Camera>();
                camera.transform.position = new Vector3(0f, 8f, -10f);
                camera.transform.LookAt(new Vector3(0f, 0.5f, 0f));
                camera.orthographic = true;
                camera.orthographicSize = 5f;
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(0.12f, 0.16f, 0.22f);

                var light = new GameObject("Directional Light").AddComponent<Light>();
                light.type = LightType.Directional;
                light.intensity = 1.2f;
                light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

                EditorSceneManager.MarkSceneDirty(scene);
                return controller;
            }
            finally
            {
                SceneManager.SetActiveScene(previous);
            }
        }

        static Material Material(string name, Color color)
        {
            var script = AssetDatabase.FindAssets("SetupDemoSceneBuilder t:MonoScript")[0];
            var sample = System
                .IO.Path.GetDirectoryName(
                    System.IO.Path.GetDirectoryName(AssetDatabase.GUIDToAssetPath(script))
                )
                .Replace('\\', '/');
            var folder = sample + "/Materials";
            if (!AssetDatabase.IsValidFolder(folder))
                AssetDatabase.CreateFolder(sample, "Materials");
            var path = folder + "/" + name + ".mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Standard")) { color = color };
                AssetDatabase.CreateAsset(material, path);
            }
            return material;
        }
    }
}
