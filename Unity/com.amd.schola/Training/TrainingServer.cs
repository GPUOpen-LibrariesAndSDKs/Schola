// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace AMD.Schola.Training
{
    /// <summary>
    /// Loopback TCP endpoint for the packaged Python gRPC adapter. Socket work runs on
    /// a background thread; all environment operations and JSON conversion run in Update.
    /// One endpoint owns one environment and accepts one connection at a time.
    /// </summary>
    /// <remarks>
    /// Serves either a single-agent <see cref="ScholaEnvironment"/> or a
    /// <see cref="ScholaMultiAgentEnvironment"/>. Assign at most one of the two fields.
    /// With neither assigned, a <see cref="ScholaMultiAgentEnvironment"/> on this
    /// GameObject is used, then a <see cref="ScholaEnvironment"/>.
    /// </remarks>
    [DisallowMultipleComponent]
    [AddComponentMenu("Schola/Training Server")]
    public sealed class TrainingServer : MonoBehaviour
    {
        // Wire form of the close command queued when a trainer disconnects.
        static readonly string s_CloseRequest =
            "{\"version\":" + TrainingWire.Version + ",\"command\":\"close\"}";

        [SerializeField]
        ScholaEnvironment m_Environment;

        [SerializeField]
        ScholaMultiAgentEnvironment m_MultiAgentEnvironment;

        [SerializeField]
        int m_Port = 8001;

        [SerializeField]
        int m_RequestTimeoutSeconds = 30;

        [SerializeField]
        bool m_RunScriptOnPlay;

        [SerializeField]
        TrainingScriptSettings m_ScriptSettings = new TrainingScriptSettings();

        LaunchableScript m_Script;

        sealed class Pending
        {
            public string Json;
            public int State; // 0 queued, 1 executing, 2 cancelled
            public readonly TaskCompletionSource<string> Reply = new TaskCompletionSource<string>(
                TaskCreationOptions.RunContinuationsAsynchronously
            );
        }

        readonly ConcurrentQueue<Pending> m_Requests = new ConcurrentQueue<Pending>();
        readonly object m_ConnectionLock = new object();
        TcpListener m_Listener;
        TcpClient m_Client;
        Thread m_Thread;
        CancellationTokenSource m_Cancellation;
        volatile bool m_Running;
        TrainingSession m_Session;
        int m_TimeoutMilliseconds;

        public bool IsRunning => m_Running;
        public int BoundPort { get; private set; }
        public string LastConnectionError { get; private set; }

        /// <summary>
        /// Launch the training script when the server starts.
        /// </summary>
        public bool RunScriptOnPlay
        {
            get { return m_RunScriptOnPlay; }
            set { m_RunScriptOnPlay = value; }
        }

        public TrainingScriptSettings ScriptSettings => m_ScriptSettings;

        public bool IsScriptRunning => m_Script != null && m_Script.IsRunning;

        void OnEnable()
        {
            if (Application.isPlaying)
                StartServer();
        }

        public void StartServer()
        {
            if (m_Environment != null && m_MultiAgentEnvironment != null)
                throw new InvalidOperationException(
                    "Assign either Environment or Multi Agent Environment to the training "
                        + "server, not both."
                );
            if (m_Environment == null && m_MultiAgentEnvironment == null)
            {
                m_MultiAgentEnvironment = GetComponent<ScholaMultiAgentEnvironment>();
                if (m_MultiAgentEnvironment == null)
                    m_Environment = GetComponent<ScholaEnvironment>();
            }
            if (m_MultiAgentEnvironment != null)
                StartServer(m_MultiAgentEnvironment, m_Port, m_RequestTimeoutSeconds);
            else if (m_Environment != null)
                StartServer(m_Environment, m_Port, m_RequestTimeoutSeconds);
            else
                throw new InvalidOperationException(
                    "Assign a ScholaEnvironment or ScholaMultiAgentEnvironment to the "
                        + "training server."
                );

            // Passing -ScholaDisableScript on the command line overrides the setting
            if (m_RunScriptOnPlay && !IsScriptDisabledOnCommandLine() && !IsScriptRunning)
            {
                m_Script = m_ScriptSettings.GetLaunchableScript(BoundPort, m_RequestTimeoutSeconds);
                m_Script.LaunchScript();
            }
        }

        public static bool IsScriptDisabledOnCommandLine()
        {
            foreach (var arg in Environment.GetCommandLineArgs())
            {
                if (string.Equals(arg, "-ScholaDisableScript", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>Start on Unity's main thread. Port zero selects an available port for tests.</summary>
        public void StartServer(IEnvironment environment, int port, int requestTimeoutSeconds = 30)
        {
            if (m_Running)
                return;
            ValidateEndpoint(port, requestTimeoutSeconds);
            Listen(new TrainingSession(environment), port, requestTimeoutSeconds);
        }

        /// <summary>
        /// Start on Unity's main thread, serving every agent of a multi-agent environment.
        /// Port zero selects an available port for tests.
        /// </summary>
        public void StartServer(
            IMultiAgentEnvironment environment,
            int port,
            int requestTimeoutSeconds = 30
        )
        {
            if (m_Running)
                return;
            ValidateEndpoint(port, requestTimeoutSeconds);
            Listen(new TrainingSession(environment), port, requestTimeoutSeconds);
        }

        static void ValidateEndpoint(int port, int requestTimeoutSeconds)
        {
            if (
                port < 0
                || port > 65535
                || requestTimeoutSeconds < 1
                || requestTimeoutSeconds > 3600
            )
                throw new ArgumentOutOfRangeException(
                    nameof(port),
                    "Invalid port or request timeout."
                );
        }

        void Listen(TrainingSession session, int port, int requestTimeoutSeconds)
        {
            m_Session = session;
            m_TimeoutMilliseconds = requestTimeoutSeconds * 1000;
            m_Listener = new TcpListener(IPAddress.Loopback, port);
            m_Listener.Start(1);
            BoundPort = ((IPEndPoint)m_Listener.LocalEndpoint).Port;
            LastConnectionError = null;
            m_Running = true;
            m_Cancellation = new CancellationTokenSource();
            var listener = m_Listener;
            var cancellation = m_Cancellation.Token;
            m_Thread = new Thread(() => Serve(listener, cancellation))
            {
                IsBackground = true,
                Name = "Schola training transport",
            };
            m_Thread.Start();
        }

        void Update() => PumpRequests();

        /// <summary>Process at most one request per Unity update; may also be driven by Editor tests.</summary>
        public void PumpRequests()
        {
            if (!m_Running || !m_Requests.TryDequeue(out var pending))
                return;
            if (Interlocked.CompareExchange(ref pending.State, 1, 0) != 0)
                return;
            TrainingResponse response;
            try
            {
                response = m_Session.Handle(JsonUtility.FromJson<TrainingRequest>(pending.Json));
            }
            catch (Exception error)
            {
                response = new TrainingResponse { ok = false, error = error.Message };
            }
            pending.Reply.TrySetResult(JsonUtility.ToJson(response));
        }

        void Serve(TcpListener listener, CancellationToken cancellation)
        {
            while (!cancellation.IsCancellationRequested)
            {
                try
                {
                    var client = listener.AcceptTcpClient();
                    lock (m_ConnectionLock)
                    {
                        if (cancellation.IsCancellationRequested)
                        {
                            client.Close();
                            break;
                        }
                        m_Client = client;
                    }
                    client.NoDelay = true;
                    client.SendTimeout = m_TimeoutMilliseconds;
                    using (client)
                    using (var stream = client.GetStream())
                    {
                        while (!cancellation.IsCancellationRequested)
                        {
                            var header = ReadExactly(stream, 4);
                            var length = IPAddress.NetworkToHostOrder(
                                BitConverter.ToInt32(header, 0)
                            );
                            if (length <= 0 || length > TrainingWire.MaxMessageBytes)
                                throw new IOException("Invalid training message length.");
                            var pending = new Pending
                            {
                                Json = Encoding.UTF8.GetString(ReadExactly(stream, length)),
                            };
                            cancellation.ThrowIfCancellationRequested();
                            m_Requests.Enqueue(pending);
                            if (!pending.Reply.Task.Wait(m_TimeoutMilliseconds, cancellation))
                            {
                                Interlocked.CompareExchange(ref pending.State, 2, 0);
                                throw new TimeoutException(
                                    "Unity did not process the request before its deadline."
                                );
                            }
                            var bytes = Encoding.UTF8.GetBytes(
                                pending.Reply.Task.GetAwaiter().GetResult()
                            );
                            if (bytes.Length > TrainingWire.MaxMessageBytes)
                                throw new IOException("Training response is too large.");
                            var size = BitConverter.GetBytes(
                                IPAddress.HostToNetworkOrder(bytes.Length)
                            );
                            stream.Write(size, 0, size.Length);
                            stream.Write(bytes, 0, bytes.Length);
                        }
                    }
                }
                catch (Exception error)
                {
                    if (!cancellation.IsCancellationRequested && !(error is EndOfStreamException))
                        LastConnectionError = error.Message;
                }
                finally
                {
                    lock (m_ConnectionLock)
                    {
                        m_Client?.Close();
                        m_Client = null;
                    }
                    // Ordered ahead of the next connection's start request. Never touch the scene here.
                    if (!cancellation.IsCancellationRequested)
                        m_Requests.Enqueue(new Pending { Json = s_CloseRequest });
                }
            }
        }

        static byte[] ReadExactly(Stream stream, int length)
        {
            var bytes = new byte[length];
            var offset = 0;
            while (offset < length)
            {
                var read = stream.Read(bytes, offset, length - offset);
                if (read == 0)
                    throw new EndOfStreamException();
                offset += read;
            }
            return bytes;
        }

        public void StopServer()
        {
            m_Script?.KillScript();
            m_Script = null;
            m_Running = false;
            m_Cancellation?.Cancel();
            m_Listener?.Stop();
            lock (m_ConnectionLock)
                m_Client?.Close();
            while (m_Requests.TryDequeue(out var pending))
            {
                Interlocked.CompareExchange(ref pending.State, 2, 0);
                pending.Reply.TrySetCanceled();
            }
            m_Thread?.Join(1000);
            m_Thread = null;
            m_Cancellation?.Dispose();
            m_Cancellation = null;
            m_Listener = null;
            BoundPort = 0;
        }

        void OnDisable() => StopServer();

        void OnDestroy() => StopServer();
    }
}
