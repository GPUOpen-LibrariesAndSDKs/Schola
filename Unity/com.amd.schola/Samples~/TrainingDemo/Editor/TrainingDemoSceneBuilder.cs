// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.Collections.Generic;
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
            var agent = CreateReachingAgent(root.transform, "Agent", "Target", 0, out var target);
            environment.SetAgent(agent);
            root.AddComponent<TrainingDemoTask>().Configure(agent.transform, target);
            root.AddComponent<TrainingServer>();
            AddCameraAndLight(new Vector3(0, 4, -7));
            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log(
                "Schola Training Demo ready. Save the scene, enter Play Mode, then run schola unity train."
            );
        }

        /// <summary>
        /// Two cubes, each reaching its own target, under one
        /// <see cref="ScholaMultiAgentEnvironment"/>. Both agents have the same spaces and
        /// finish on the same step, so <c>schola unity train</c> trains them as one policy.
        /// </summary>
        [MenuItem("Tools/Schola/Create Multi-Agent Training Demo Scene")]
        public static void CreateMultiAgentScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                return;
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var root = new GameObject("Schola Multi-Agent Training Environment");
            var environment = root.AddComponent<ScholaMultiAgentEnvironment>();
            var lanes = new[] { -1.5f, 1.5f };
            var agents = new List<KeyValuePair<string, ScholaAgent>>();
            var targets = new Transform[lanes.Length];
            var team = new MultiAgentTrainingDemoTask[lanes.Length];
            for (var i = 0; i < lanes.Length; i++)
            {
                var agent = CreateReachingAgent(
                    root.transform,
                    $"Agent {i}",
                    $"Target {i}",
                    lanes[i],
                    out targets[i]
                );
                // On the agent's GameObject, where the environment looks for its hooks.
                team[i] = agent.gameObject.AddComponent<MultiAgentTrainingDemoTask>();
                agents.Add(new KeyValuePair<string, ScholaAgent>($"agent_{i}", agent));
            }
            for (var i = 0; i < team.Length; i++)
                team[i].Configure(agents[i].Value.transform, targets[i], team);
            environment.SetAgents(agents);
            // With no environment assigned, the server uses the one on its GameObject.
            root.AddComponent<TrainingServer>();
            AddCameraAndLight(new Vector3(0, 6, -8));
            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log(
                "Schola Multi-Agent Training Demo ready. Save the scene, enter Play Mode, then run schola unity train."
            );
        }

        /// <summary>
        /// A cube with a bounded translate actuator that observes its own position and
        /// its target's position. The cube starts at x = -1 and the target sits at x = 1.
        /// </summary>
        static ScholaAgent CreateReachingAgent(
            Transform root,
            string agentName,
            string targetName,
            float z,
            out Transform target
        )
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = agentName;
            cube.transform.SetParent(root);
            cube.transform.position = new Vector3(-1, 0.5f, z);
            // This example uses Transform displacement, not physics simulation.
            Object.DestroyImmediate(cube.GetComponent<Collider>());
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = targetName;
            sphere.transform.SetParent(root);
            sphere.transform.position = new Vector3(1, 0.5f, z);
            sphere.transform.localScale = Vector3.one * 0.3f;
            Object.DestroyImmediate(sphere.GetComponent<Collider>());
            var position = cube.AddComponent<PositionSensor>();
            position.ObservationKey = "agent_position";
            var goal = sphere.AddComponent<PositionSensor>();
            goal.ObservationKey = "target_position";
            var actuator = cube.AddComponent<TranslateActuator>();
            actuator.MaxDisplacement = 0.1f;
            var agent = cube.AddComponent<ScholaAgent>();
            agent.SetSensors(new[] { position, goal });
            agent.SetActuators(new[] { actuator });
            target = sphere.transform;
            return agent;
        }

        static void AddCameraAndLight(Vector3 cameraPosition)
        {
            var camera = new GameObject("Main Camera").AddComponent<Camera>();
            camera.tag = "MainCamera";
            camera.transform.position = cameraPosition;
            camera.transform.LookAt(new Vector3(0, 0.5f, 0));
            var light = new GameObject("Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(45, -30, 0);
        }
    }
}
