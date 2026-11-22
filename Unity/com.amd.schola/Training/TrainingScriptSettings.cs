// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AMD.Schola.Training
{
    /// <summary>
    /// The kind of script to launch. Mirrors Unreal EScriptType.
    /// </summary>
    public enum ScriptType
    {
        Python,
        Other,
    }

    /// <summary>
    /// Which Python to launch the script with. Mirrors Unreal EPythonEnvironmentType, without
    /// the Built-in option because Unity does not ship a Python.
    /// </summary>
    public enum PythonEnvironmentType
    {
        [InspectorName("System PATH Python Env")]
        SystemPath,

        [InspectorName("Conda Env")]
        Conda,

        [InspectorName("Custom Python Path")]
        CustomPythonPath,
    }

    /// <summary>
    /// Which Python script to run. Mirrors Unreal EPythonScript.
    /// </summary>
    public enum PythonScriptType
    {
        [InspectorName("Builtin Schola Unity Training Script")]
        Builtin,

        [InspectorName("Custom Training Script")]
        Custom,
    }

    [Serializable]
    public sealed class ScriptArgument
    {
        public string Name = "";
        public string Value = "";
    }

    /// <summary>
    /// A user provided script and its arguments. Mirrors Unreal FCustomTrainingSettings.
    /// </summary>
    [Serializable]
    public sealed class CustomTrainingSettings
    {
        [Tooltip("Path to the script to launch.")]
        public string LaunchScript = "";

        [Tooltip("Passed as --Name Value.")]
        public List<ScriptArgument> Args = new List<ScriptArgument>();

        [Tooltip("Passed as --Flag.")]
        public List<string> Flags = new List<string>();

        public void GenerateTrainingArgs(ScriptArgBuilder builder)
        {
            foreach (var arg in Args)
            {
                if (!string.IsNullOrEmpty(arg.Name))
                {
                    builder.AddStringArg(arg.Name, arg.Value);
                }
            }

            foreach (var flag in Flags)
            {
                if (!string.IsNullOrEmpty(flag))
                {
                    builder.AddFlag(flag);
                }
            }
        }
    }

    /// <summary>
    /// Settings for the builtin schola unity train command.
    /// </summary>
    [Serializable]
    public sealed class BuiltinTrainingSettings
    {
        [Tooltip("The total number of environment timesteps to train for.")]
        public int Timesteps = 10000;

        public int Seed = 0;

        [Tooltip(
            "Where the Stable-Baselines3 checkpoint is saved, relative to the project folder."
        )]
        public string CheckpointPath = "ckpt/unity_ppo";

        public bool ExportFinalModelToOnnx = true;

        [Tooltip("Where to export the ONNX model. Empty saves it next to the checkpoint.")]
        public string OnnxPath = "";

        public void GenerateTrainingArgs(ScriptArgBuilder builder)
        {
            builder.AddPositionalArgument("unity");
            builder.AddPositionalArgument("train");
            builder.AddIntArg("timesteps", Timesteps);
            builder.AddIntArg("seed", Seed);
            builder.AddStringArg("checkpoint", CheckpointPath);

            if (!ExportFinalModelToOnnx)
            {
                return;
            }

            if (string.IsNullOrEmpty(OnnxPath))
            {
                builder.AddFlag("export-onnx");
            }
            else
            {
                builder.AddStringArg("onnx-path", OnnxPath);
            }
        }
    }

    /// <summary>
    /// Which script to launch when training starts and how. Mirrors Unreal FTrainingScriptSettings.
    /// </summary>
    [Serializable]
    public sealed class TrainingScriptSettings
    {
        public const string BuiltinScriptCommand = "-m schola.scripts.launch";

        public ScriptType ScriptType = ScriptType.Python;

        public PythonEnvironmentType EnvType = PythonEnvironmentType.SystemPath;

        public string CondaEnvName = "";

        [Tooltip("Path to python.exe, for example .venv/Scripts/python.exe.")]
        public string CustomPythonPath = "";

        public PythonScriptType PythonScriptType = PythonScriptType.Builtin;

        public BuiltinTrainingSettings BuiltinSettings = new BuiltinTrainingSettings();

        public CustomTrainingSettings CustomPythonScriptSettings = new CustomTrainingSettings();

        public CustomTrainingSettings CustomScriptSettings = new CustomTrainingSettings();

        /// <summary>
        /// The script and full command line to launch, including the server arguments
        /// so the script connects to the right port.
        /// </summary>
        public LaunchableScript GetLaunchableScript(int unityPort, int timeoutSeconds)
        {
            var builder = new ScriptArgBuilder();
            GenerateTrainingArgs(builder);
            builder.AddIntArg("unity-port", unityPort);
            builder.AddIntArg("timeout", timeoutSeconds);

            var script = GetBaseScript();
            script.AppendArgs(builder.Build());
            return script;
        }

        public void GenerateTrainingArgs(ScriptArgBuilder builder)
        {
            if (ScriptType == ScriptType.Other)
            {
                CustomScriptSettings.GenerateTrainingArgs(builder);
            }
            else if (PythonScriptType == PythonScriptType.Custom)
            {
                CustomPythonScriptSettings.GenerateTrainingArgs(builder);
            }
            else
            {
                BuiltinSettings.GenerateTrainingArgs(builder);
            }
        }

        public string GetScriptPath()
        {
            if (ScriptType == ScriptType.Other)
            {
                return CleanPath(CustomScriptSettings.LaunchScript);
            }

            if (PythonScriptType == PythonScriptType.Custom)
            {
                return ScriptArgBuilder.Quote(CleanPath(CustomPythonScriptSettings.LaunchScript));
            }

            return BuiltinScriptCommand;
        }

        LaunchableScript GetBaseScript()
        {
            var scriptCommand = GetScriptPath();

            if (ScriptType == ScriptType.Other)
            {
                return new LaunchableScript(FullPath(scriptCommand), "");
            }

            switch (EnvType)
            {
                case PythonEnvironmentType.Conda:
                {
                    var conda =
                        Application.platform == RuntimePlatform.WindowsEditor
                        || Application.platform == RuntimePlatform.WindowsPlayer
                            ? "conda.bat"
                            : "conda";
                    var args =
                        "run --live-stream -n "
                        + ScriptArgBuilder.Quote(CondaEnvName)
                        + " python "
                        + scriptCommand;
                    return new LaunchableScript(conda, args);
                }
                case PythonEnvironmentType.CustomPythonPath:
                    return new LaunchableScript(FullPath(CustomPythonPath), scriptCommand);
                default:
                    return new LaunchableScript("python", scriptCommand);
            }
        }

        static string FullPath(string path)
        {
            path = CleanPath(path);
            if (string.IsNullOrEmpty(path) || Path.IsPathRooted(path))
            {
                return path;
            }

            return Path.GetFullPath(path);
        }

        // Pasted paths often carry spaces or the quotes from Windows' Copy as path.
        static string CleanPath(string path)
        {
            return path == null ? null : path.Trim().Trim('"').Trim();
        }
    }
}
