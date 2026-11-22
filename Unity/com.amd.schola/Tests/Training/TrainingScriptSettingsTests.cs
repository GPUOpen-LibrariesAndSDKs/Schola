// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using System.IO;
using AMD.Schola.Training;
using NUnit.Framework;
using UnityEngine;

namespace AMD.Schola.Tests.Training
{
    /// <summary>
    /// Tests the command TrainingServer launches when Run Script On Play is ticked.
    /// No test starts a process.
    /// </summary>
    public sealed class TrainingScriptSettingsTests
    {
        [Test]
        public void Default_RunsBuiltinUnityTrainWithSystemPython()
        {
            var script = new TrainingScriptSettings().GetLaunchableScript(8001, 30);

            Assert.That(script.ScriptUrl, Is.EqualTo("python"));
            Assert.That(
                script.Args,
                Is.EqualTo(
                    "-m schola.scripts.launch unity train --timesteps 10000 --seed 0 "
                        + "--checkpoint ckpt/unity_ppo --export-onnx --unity-port 8001 --timeout 30"
                )
            );
        }

        [Test]
        public void Builtin_OnnxPath_ReplacesExportFlagAndIsQuoted()
        {
            var settings = new TrainingScriptSettings();
            settings.BuiltinSettings.OnnxPath = "My Models/cube.onnx";

            var args = settings.GetLaunchableScript(8001, 30).Args;

            Assert.That(args, Does.Contain("--onnx-path \"My Models/cube.onnx\""));
            Assert.That(args, Does.Not.Contain("--export-onnx"));
        }

        [Test]
        public void Builtin_ExportOff_PassesNoExportArgs()
        {
            var settings = new TrainingScriptSettings();
            settings.BuiltinSettings.ExportFinalModelToOnnx = false;
            settings.BuiltinSettings.OnnxPath = "cube.onnx";

            var args = settings.GetLaunchableScript(8001, 30).Args;

            Assert.That(args, Does.Not.Contain("onnx"));
        }

        [Test]
        public void Conda_RunsPythonInsideTheNamedEnv()
        {
            var settings = new TrainingScriptSettings
            {
                EnvType = PythonEnvironmentType.Conda,
                CondaEnvName = "schola",
            };

            var script = settings.GetLaunchableScript(8001, 30);

            Assert.That(Path.GetFileNameWithoutExtension(script.ScriptUrl), Is.EqualTo("conda"));
            Assert.That(
                script.Args,
                Does.StartWith("run --live-stream -n schola python -m schola.scripts.launch")
            );
        }

        [Test]
        public void CustomPythonPath_RelativePath_IsMadeAbsolute()
        {
            var settings = new TrainingScriptSettings
            {
                EnvType = PythonEnvironmentType.CustomPythonPath,
                CustomPythonPath = ".venv/Scripts/python.exe",
            };

            var script = settings.GetLaunchableScript(8001, 30);

            Assert.That(Path.IsPathRooted(script.ScriptUrl), Is.True);
            Assert.That(script.Args, Does.StartWith("-m schola.scripts.launch unity train"));
        }

        [Test]
        public void CustomPythonPath_PastedWithSpacesOrQuotes_IsCleaned()
        {
            var python = Path.Combine(Application.temporaryCachePath, "python.exe");
            var settings = new TrainingScriptSettings
            {
                EnvType = PythonEnvironmentType.CustomPythonPath,
                CustomPythonPath = " \"" + python + "\" ",
            };

            Assert.That(settings.GetLaunchableScript(8001, 30).ScriptUrl, Is.EqualTo(python));
        }

        [Test]
        public void CustomPythonScript_PassesItsArgsFlagsAndServerArgs()
        {
            var settings = new TrainingScriptSettings
            {
                PythonScriptType = PythonScriptType.Custom,
            };
            settings.CustomPythonScriptSettings.LaunchScript = "my train.py";
            settings.CustomPythonScriptSettings.Args.Add(
                new ScriptArgument { Name = "algo", Value = "ppo" }
            );
            settings.CustomPythonScriptSettings.Flags.Add("verbose");

            var script = settings.GetLaunchableScript(9000, 15);

            Assert.That(script.ScriptUrl, Is.EqualTo("python"));
            Assert.That(
                script.Args,
                Is.EqualTo("\"my train.py\" --algo ppo --verbose --unity-port 9000 --timeout 15")
            );
        }

        [Test]
        public void OtherScript_LaunchesTheScriptItself()
        {
            var settings = new TrainingScriptSettings { ScriptType = ScriptType.Other };
            settings.CustomScriptSettings.LaunchScript = Path.Combine(
                Application.temporaryCachePath,
                "train.bat"
            );

            var script = settings.GetLaunchableScript(8001, 30);

            Assert.That(script.ScriptUrl, Is.EqualTo(settings.CustomScriptSettings.LaunchScript));
            Assert.That(script.Args, Is.EqualTo("--unity-port 8001 --timeout 30"));
        }

        [Test]
        public void Quote_OnlyQuotesValuesThatNeedIt()
        {
            Assert.That(ScriptArgBuilder.Quote("ckpt/unity_ppo"), Is.EqualTo("ckpt/unity_ppo"));
            Assert.That(ScriptArgBuilder.Quote("a b"), Is.EqualTo("\"a b\""));
            Assert.That(ScriptArgBuilder.Quote(""), Is.EqualTo("\"\""));
        }

        [Test]
        public void TrainingServer_DoesNotRunScriptByDefault()
        {
            var gameObject = new GameObject("Training Server");
            try
            {
                Assert.That(gameObject.AddComponent<TrainingServer>().RunScriptOnPlay, Is.False);
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }
    }
}
