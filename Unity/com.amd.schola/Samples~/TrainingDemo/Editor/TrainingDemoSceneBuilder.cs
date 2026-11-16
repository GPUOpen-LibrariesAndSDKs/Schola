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
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                return;
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var root = new GameObject("Schola Training Environment");
            var environment = root.AddComponent<ScholaEnvironment>();
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "Agent";
            cube.transform.SetParent(root.transform);
            cube.transform.position = new Vector3(-1, 0.5f, 0);
            // This example uses Transform displacement, not physics simulation.
            Object.DestroyImmediate(cube.GetComponent<Collider>());
            var target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            target.name = "Target";
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
            root.AddComponent<TrainingServer>();
            var camera = new GameObject("Main Camera").AddComponent<Camera>();
            camera.tag = "MainCamera";
            camera.transform.position = new Vector3(0, 4, -7);
            camera.transform.LookAt(new Vector3(0, 0.5f, 0));
            var light = new GameObject("Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(45, -30, 0);
            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log(
                "Schola Training Demo ready. Save the scene, enter Play Mode, then run schola unity train."
            );
        }
    }
}
