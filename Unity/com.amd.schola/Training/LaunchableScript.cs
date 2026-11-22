// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Diagnostics;
using System.IO;
using Debug = UnityEngine.Debug;

namespace AMD.Schola.Training
{
    /// <summary>
    /// A script launched as a child process, with its output sent to the Console.
    /// </summary>
    public sealed class LaunchableScript
    {
        Process m_Process;
        volatile bool m_Killed;

        public LaunchableScript(string scriptUrl, string args)
        {
            ScriptUrl = scriptUrl;
            Args = args ?? "";
        }

        public string ScriptUrl { get; }

        public string Args { get; private set; }

        public bool IsRunning
        {
            get { return m_Process != null && !m_Process.HasExited; }
        }

        public void AppendArgs(string additionalArgs)
        {
            if (string.IsNullOrEmpty(additionalArgs))
            {
                return;
            }

            Args = string.IsNullOrEmpty(Args) ? additionalArgs : Args + " " + additionalArgs;
        }

        /// <returns>False, with the reason logged, if the script could not be started.</returns>
        public bool LaunchScript()
        {
            if (IsRunning)
            {
                Debug.LogWarning($"{nameof(LaunchableScript)}: Script already running.");
                return false;
            }

            if (string.IsNullOrEmpty(ScriptUrl))
            {
                Debug.LogError($"{nameof(LaunchableScript)}: No script or Python path set.");
                return false;
            }

            Debug.Log($"{nameof(LaunchableScript)}: Launching {ScriptUrl} {Args}");

            var startInfo = new ProcessStartInfo(ScriptUrl, Args)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = Directory.GetCurrentDirectory(),
            };
            // Without this Python holds its output until it exits.
            startInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

            var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            process.OutputDataReceived += (_, e) => LogLine(e.Data);
            process.ErrorDataReceived += (_, e) => LogLine(e.Data);
            process.Exited += (_, __) => LogExit(process);

            try
            {
                process.Start();
            }
            catch (Exception error)
            {
                Debug.LogError(
                    $"{nameof(LaunchableScript)}: Could not launch {ScriptUrl}: {error.Message}"
                );
                process.Dispose();
                return false;
            }

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            m_Killed = false;
            m_Process = process;
            return true;
        }

        public void KillScript()
        {
            if (m_Process == null)
            {
                return;
            }

            try
            {
                if (!m_Process.HasExited)
                {
                    Debug.Log($"{nameof(LaunchableScript)}: Killing {ScriptUrl}");
                    m_Killed = true;
                    KillTree(m_Process);
                }
            }
            catch (Exception error)
            {
                Debug.LogWarning(
                    $"{nameof(LaunchableScript)}: Could not kill script: {error.Message}"
                );
            }
            finally
            {
                m_Process.Dispose();
                m_Process = null;
            }
        }

        // Conda and the schola command start Python as a child process, so kill the whole tree.
        static void KillTree(Process process)
        {
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
            {
                using (
                    var taskkill = Process.Start(
                        new ProcessStartInfo("taskkill", $"/T /F /PID {process.Id}")
                        {
                            UseShellExecute = false,
                            CreateNoWindow = true,
                        }
                    )
                )
                {
                    taskkill?.WaitForExit(5000);
                }
            }

            if (!process.HasExited)
            {
                process.Kill();
            }
        }

        static void LogLine(string line)
        {
            if (!string.IsNullOrEmpty(line))
            {
                Debug.Log("[Schola script] " + line);
            }
        }

        void LogExit(Process process)
        {
            if (m_Killed)
            {
                return;
            }

            int code;
            try
            {
                code = process.ExitCode;
            }
            catch (InvalidOperationException)
            {
                return;
            }

            if (code != 0)
            {
                Debug.LogError(
                    $"{nameof(LaunchableScript)}: Script {ScriptUrl} exited with code {code}"
                );
            }
        }
    }
}
