// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using AMD.Schola.Training;
using UnityEditor;
using UnityEngine;

namespace AMD.Schola.Tests.Training
{
    /// <summary>
    /// Batch-mode host for Test/unity/test_unity_live.py. Uses real US1/2/3 runtime
    /// components and Editor updates to pump the same transport used in Play Mode.
    /// Set <c>SCHOLA_TEST_UNITY_AGENTS=2</c> for a two-agent multi-agent environment.
    /// </summary>
    public static class TrainingIntegrationHost
    {
        static TrainingServer s_Server;

        [InitializeOnLoadMethod]
        static void RestoreAfterReload()
        {
            // Unity can reload once more after executing a batch entry point when
            // test-assembly defines change. Restore only this explicitly requested host.
            if (
                Application.isBatchMode
                && Array.IndexOf(
                    System.Environment.GetCommandLineArgs(),
                    "AMD.Schola.Tests.Training.TrainingIntegrationHost.Run"
                ) >= 0
            )
                EditorApplication.delayCall += Run;
        }

        sealed class Episode : MonoBehaviour, IReward, IEpisodeStatus
        {
            public Transform Agent;
            public Episode[] Team = Array.Empty<Episode>();
            int m_Steps;

            public bool HasArrived => Agent.position.x >= 0.2f;

            public void OnEpisodeReset() => m_Steps = 0;

            public float ComputeReward()
            {
                m_Steps++;
                return Agent.position.x;
            }

            public void GetStatus(out bool terminated, out bool truncated)
            {
                // Shared team completion so SB3's shared-policy path sees every agent finish
                // on the same step.
                terminated = true;
                foreach (var member in Team)
                {
                    if (member == null || !member.HasArrived)
                        terminated = false;
                }
                if (Team.Length == 0)
                    terminated = HasArrived;
                truncated = !terminated && m_Steps >= 4;
            }
        }

        public static void Run()
        {
            if (s_Server != null && s_Server.IsRunning)
                return;
            var previous = GameObject.Find("US4 integration environment");
            if (previous != null)
                UnityEngine.Object.DestroyImmediate(previous);
            var root = new GameObject("US4 integration environment");
            var agentCount = AgentCount();
            var port = int.Parse(
                System.Environment.GetEnvironmentVariable("SCHOLA_TEST_UNITY_PORT") ?? "8001"
            );
            var server = root.AddComponent<TrainingServer>();
            if (agentCount > 1)
                StartMultiAgent(root, server, agentCount, port);
            else
                StartSingleAgent(root, server, port);
            s_Server = server;
            EditorApplication.update += server.PumpRequests;
            // The test process owns and terminates this dedicated batch Editor.
            Debug.Log($"SCHOLA_US4_READY:{server.BoundPort}");
        }

        static int AgentCount()
        {
            var raw = System.Environment.GetEnvironmentVariable("SCHOLA_TEST_UNITY_AGENTS");
            if (string.IsNullOrEmpty(raw))
                return 1;
            return Math.Max(1, int.Parse(raw));
        }

        static void StartSingleAgent(GameObject root, TrainingServer server, int port)
        {
            var agentObject = new GameObject("Agent");
            agentObject.transform.SetParent(root.transform);
            var agent = BindTranslateAgent(agentObject);
            var episode = root.AddComponent<Episode>();
            episode.Agent = agentObject.transform;
            episode.Team = new[] { episode };
            var environment = root.AddComponent<ScholaEnvironment>();
            environment.SetAgent(agent);
            environment.SetReward(episode);
            environment.SetEpisodeStatus(episode);
            server.StartServer(environment, port);
        }

        static void StartMultiAgent(
            GameObject root,
            TrainingServer server,
            int agentCount,
            int port
        )
        {
            var agents = new List<KeyValuePair<string, ScholaAgent>>();
            var team = new Episode[agentCount];
            for (var i = 0; i < agentCount; i++)
            {
                var agentObject = new GameObject($"Agent {i}");
                agentObject.transform.SetParent(root.transform);
                agentObject.transform.position = new Vector3(0, 0, i);
                var agent = BindTranslateAgent(agentObject);
                var episode = agentObject.AddComponent<Episode>();
                episode.Agent = agentObject.transform;
                team[i] = episode;
                agents.Add(new KeyValuePair<string, ScholaAgent>($"agent_{i}", agent));
            }
            foreach (var episode in team)
                episode.Team = team;
            var environment = root.AddComponent<ScholaMultiAgentEnvironment>();
            environment.SetAgents(agents);
            server.StartServer(environment, port);
        }

        static ScholaAgent BindTranslateAgent(GameObject agentObject)
        {
            var sensor = agentObject.AddComponent<PositionSensor>();
            sensor.ObservationKey = "position";
            var actuator = agentObject.AddComponent<TranslateActuator>();
            actuator.MaxDisplacement = 0.1f;
            var agent = agentObject.AddComponent<ScholaAgent>();
            agent.SetSensors(new[] { sensor });
            agent.SetActuators(new[] { actuator });
            return agent;
        }
    }
}
