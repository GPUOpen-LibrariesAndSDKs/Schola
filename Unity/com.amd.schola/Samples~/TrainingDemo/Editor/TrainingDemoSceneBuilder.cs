// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using AMD.Schola.Training;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AMD.Schola.Samples.TrainingDemo.Editor
{
    public static class TrainingDemoSceneBuilder
    {
        [MenuItem("Tools/Schola/Create Training Demo Scene")]
        public static void CreateScene()
        {
            Create(false);
        }

        [MenuItem("Tools/Schola/Create Inference Demo Scene")]
        public static void CreateInferenceScene() => Create(true);

        static void Create(bool inference)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning("Stop Play Mode before creating a Schola scene.");
                return;
            }
            var path = EditorUtility.SaveFilePanelInProject(
                "Save Schola Scene",
                inference ? "ScholaInferenceDemo" : "ScholaTrainingDemo",
                "unity",
                "Save under Assets."
            );
            if (
                string.IsNullOrEmpty(path)
                || !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()
            )
                return;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Populate(inference);
            EditorSceneManager.SaveScene(scene, path);
        }

        public static ScholaAgent Populate(
            bool inference,
            Unity.InferenceEngine.ModelAsset model = null
        )
        {
            var root = new GameObject("Schola Training Environment");
            var environment = root.AddComponent<ScholaEnvironment>();
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "Agent";
            cube.GetComponent<Renderer>().sharedMaterial = Material(
                "Agent",
                new Color(0.1f, 0.45f, 0.9f)
            );
            cube.transform.SetParent(root.transform);
            cube.transform.position = new Vector3(-1, 0.5f, 0);
            // This example uses Transform displacement, not physics simulation.
            Object.DestroyImmediate(cube.GetComponent<Collider>());
            var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            target.name = "Target";
            target.GetComponent<Renderer>().sharedMaterial = Material(
                "Target",
                new Color(1f, 0.5f, 0.05f)
            );
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = Color.gray * 0.6f;
            target.transform.SetParent(root.transform);
            target.transform.position = new Vector3(1, 0.5f, 0);
            target.transform.localScale = Vector3.one * 0.3f;
            Object.DestroyImmediate(target.GetComponent<Collider>());
            var position = cube.AddComponent<PositionSensor>();
            position.ObservationKey = "agent_position";
            var goal = target.AddComponent<PositionSensor>();
            goal.ObservationKey = "target_position";
            var actuator = cube.AddComponent<TranslateActuator>();
            actuator.MaxDisplacement = 0.1f;
            var agent = cube.AddComponent<ScholaAgent>();
            agent.SetSensors(new[] { position, goal });
            agent.SetActuators(new[] { actuator });
            environment.SetAgent(agent);
            root.AddComponent<TrainingDemoTask>().Configure(cube.transform, target.transform);
            if (inference)
            {
                var runner = root.AddComponent<PolicyRunner>();
                runner.SetAgents(new[] { agent });
                if (model == null)
                {
                    var script = AssetDatabase.FindAssets("TrainingDemoSceneBuilder t:MonoScript")[
                        0
                    ];
                    var sample = System.IO.Path.GetDirectoryName(
                        System.IO.Path.GetDirectoryName(AssetDatabase.GUIDToAssetPath(script))
                    );
                    model = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(
                        sample + "/Models/reach_target.onnx"
                    );
                }
                runner.Model = model;
                root.AddComponent<InferenceDemoStatus>()
                    .Configure(agent.transform, target.transform, runner);
            }
            else
                root.AddComponent<TrainingServer>();
            var camera = new GameObject("Main Camera").AddComponent<Camera>();
            camera.tag = "MainCamera";
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.12f, 0.16f, 0.22f);
            camera.transform.position = new Vector3(0, 4, -7);
            camera.transform.LookAt(new Vector3(0, 0.5f, 0));
            var light = new GameObject("Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(45, -30, 0);
            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(root.scene);
            return agent;
        }

        static Material Material(string name, Color color)
        {
            var script = AssetDatabase.FindAssets("TrainingDemoSceneBuilder t:MonoScript")[0];
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
