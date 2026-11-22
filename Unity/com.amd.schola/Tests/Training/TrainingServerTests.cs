// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AMD.Schola.Training;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Training
{
    public sealed class TrainingServerTests
    {
        static void SendJson(NetworkStream stream, string json)
        {
            var bytes = Encoding.UTF8.GetBytes(json);
            var length = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(bytes.Length));
            stream.Write(length, 0, 4);
            stream.Write(bytes, 0, bytes.Length);
        }

        static string ReceiveJson(NetworkStream stream)
        {
            var header = Read(stream, 4);
            var size = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(header, 0));
            return Encoding.UTF8.GetString(Read(stream, size));
        }

        static void SendStart(TcpClient client) =>
            SendJson(
                client.GetStream(),
                "{\"version\":" + TrainingWire.Version + ",\"command\":\"start\"}"
            );

        static string Exchange(TrainingServer server, int port, params string[] requests)
        {
            var reply = Task.Run(() =>
            {
                using (var client = new TcpClient())
                {
                    client.Connect(IPAddress.Loopback, port);
                    client.ReceiveTimeout = 5000;
                    var stream = client.GetStream();
                    string last = null;
                    foreach (var request in requests)
                    {
                        SendJson(stream, request);
                        last = ReceiveJson(stream);
                    }
                    return last;
                }
            });
            var timer = Stopwatch.StartNew();
            while (!reply.IsCompleted && timer.Elapsed.TotalSeconds < 5)
            {
                server.PumpRequests();
                Thread.Sleep(1);
            }
            Assert.That(reply.IsCompleted, Is.True, "Transport did not reply");
            return reply.GetAwaiter().GetResult();
        }

        static ScholaAgent AddTranslateAgent(GameObject parent, string name, string observationKey)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform);
            var sensor = go.AddComponent<PositionSensor>();
            sensor.ObservationKey = observationKey;
            var actuator = go.AddComponent<TranslateActuator>();
            actuator.MaxDisplacement = 1;
            var agent = go.AddComponent<ScholaAgent>();
            agent.SetSensors(new[] { sensor });
            agent.SetActuators(new[] { actuator });
            return agent;
        }

        static void SetSerializedField(object target, string fieldName, object value)
        {
            var field = target
                .GetType()
                .GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, fieldName);
            field.SetValue(target, value);
        }

        [Test]
        public void Stop_WithRequestWaitingForMainThread_ReleasesConnectionAndPort()
        {
            var go = new GameObject("TrainingShutdownTest");
            var agent = go.AddComponent<ScholaAgent>();
            var environment = go.AddComponent<ScholaEnvironment>();
            environment.SetAgent(agent);
            var server = go.AddComponent<TrainingServer>();
            try
            {
                server.StartServer(environment, 0);
                var port = server.BoundPort;
                using (var client = new TcpClient())
                {
                    client.Connect(IPAddress.Loopback, port);
                    SendStart(client);
                    // Deliberately do not pump Update: shutdown must cancel waiting work.
                    var timer = Stopwatch.StartNew();
                    server.StopServer();
                    Assert.That(timer.Elapsed.TotalSeconds, Is.LessThan(2));
                    Assert.That(server.IsRunning, Is.False);
                }
                var replacement = new TcpListener(IPAddress.Loopback, port);
                replacement.Start();
                replacement.Stop();
                // A stopped listener can be restarted without leaking its worker.
                server.StartServer(environment, port);
                Assert.That(server.IsRunning, Is.True);
                server.StopServer();
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void NetworkRequest_ExecutesOnlyWhenMainThreadPumps()
        {
            var go = new GameObject("TrainingPumpTest");
            var agent = AddTranslateAgent(go, "Agent", "position");
            var environment = go.AddComponent<ScholaEnvironment>();
            environment.SetAgent(agent);
            var server = go.AddComponent<TrainingServer>();
            try
            {
                server.StartServer(environment, 0);
                var response = JsonUtility.FromJson<TrainingResponse>(
                    Exchange(
                        server,
                        server.BoundPort,
                        "{\"version\":" + TrainingWire.Version + ",\"command\":\"start\"}"
                    )
                );
                Assert.That(response.ok, Is.True);
                Assert.That(response.version, Is.EqualTo(TrainingWire.Version));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void StartServer_ServesMultiAgentEnvironmentOverTheWire()
        {
            var go = new GameObject("TrainingMultiAgentDefinitionTest");
            var agentA = AddTranslateAgent(go, "A", "position");
            var agentB = AddTranslateAgent(go, "B", "position");
            var environment = go.AddComponent<ScholaMultiAgentEnvironment>();
            environment.SetAgents(
                new[]
                {
                    new KeyValuePair<string, ScholaAgent>("agent_0", agentA),
                    new KeyValuePair<string, ScholaAgent>("agent_1", agentB),
                }
            );
            var server = go.AddComponent<TrainingServer>();
            try
            {
                server.StartServer(environment, 0);
                var response = JsonUtility.FromJson<TrainingResponse>(
                    Exchange(
                        server,
                        server.BoundPort,
                        "{\"version\":" + TrainingWire.Version + ",\"command\":\"start\"}",
                        "{\"version\":" + TrainingWire.Version + ",\"command\":\"definition\"}"
                    )
                );
                Assert.That(response.ok, Is.True);
                Assert.That(response.agents, Has.Length.EqualTo(2));
                Assert.That(
                    new[] { response.agents[0].id, response.agents[1].id },
                    Is.EquivalentTo(new[] { "agent_0", "agent_1" })
                );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void StartServer_UsesAssignedMultiAgentFieldBeforeSingleAgentComponent()
        {
            var go = new GameObject("TrainingResolveMultiAgent");
            var singleAgent = AddTranslateAgent(go, "Single", "position");
            var multiAgent = AddTranslateAgent(go, "Multi", "position");
            var single = go.AddComponent<ScholaEnvironment>();
            single.SetAgent(singleAgent);
            var multi = go.AddComponent<ScholaMultiAgentEnvironment>();
            multi.SetAgents(
                new[] { new KeyValuePair<string, ScholaAgent>("team_agent", multiAgent) }
            );
            var server = go.AddComponent<TrainingServer>();
            SetSerializedField(server, "m_MultiAgentEnvironment", multi);
            try
            {
                server.StartServer();
                var response = JsonUtility.FromJson<TrainingResponse>(
                    Exchange(
                        server,
                        server.BoundPort,
                        "{\"version\":" + TrainingWire.Version + ",\"command\":\"start\"}",
                        "{\"version\":" + TrainingWire.Version + ",\"command\":\"definition\"}"
                    )
                );
                Assert.That(response.ok, Is.True);
                Assert.That(response.agents, Has.Length.EqualTo(1));
                Assert.That(response.agents[0].id, Is.EqualTo("team_agent"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void StartServer_RejectsBothEnvironmentFieldsAssigned()
        {
            var go = new GameObject("TrainingBothEnvironments");
            var singleAgent = go.AddComponent<ScholaAgent>();
            var multiAgent = AddTranslateAgent(go, "Multi", "position");
            var single = go.AddComponent<ScholaEnvironment>();
            single.SetAgent(singleAgent);
            var multi = go.AddComponent<ScholaMultiAgentEnvironment>();
            multi.SetAgents(
                new[] { new KeyValuePair<string, ScholaAgent>("team_agent", multiAgent) }
            );
            var server = go.AddComponent<TrainingServer>();
            SetSerializedField(server, "m_Environment", single);
            SetSerializedField(server, "m_MultiAgentEnvironment", multi);
            try
            {
                Assert.Throws<InvalidOperationException>(() => server.StartServer());
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(go);
            }
        }

        static byte[] Read(Stream stream, int count)
        {
            var data = new byte[count];
            var offset = 0;
            while (offset < count)
            {
                var n = stream.Read(data, offset, count - offset);
                if (n == 0)
                    throw new EndOfStreamException();
                offset += n;
            }
            return data;
        }
    }
}
