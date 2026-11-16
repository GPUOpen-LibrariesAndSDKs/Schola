// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AMD.Schola.Samples.TrainingDemo.Editor
{
    public static class SetupSceneAssets
    {
        public static void Generate()
        {
            var script = AssetDatabase.FindAssets("TrainingDemoSceneBuilder t:MonoScript")[0];
            var sample = Path.GetDirectoryName(
                    Path.GetDirectoryName(AssetDatabase.GUIDToAssetPath(script))
                )
                .Replace('\\', '/');
            Directory.CreateDirectory(sample + "/Scenes");
            foreach (var inference in new[] { false, true })
            {
                var scene = EditorSceneManager.NewScene(
                    NewSceneSetup.EmptyScene,
                    NewSceneMode.Single
                );
                TrainingDemoSceneBuilder.Populate(inference);
                EditorSceneManager.SaveScene(
                    scene,
                    sample
                        + "/Scenes/"
                        + (inference ? "ScholaInferenceDemo" : "ScholaTrainingDemo")
                        + ".unity"
                );
            }
            AssetDatabase.Refresh();
            Debug.Log("SCHOLA_US6_SCENES_SAVED");
        }
    }
}
