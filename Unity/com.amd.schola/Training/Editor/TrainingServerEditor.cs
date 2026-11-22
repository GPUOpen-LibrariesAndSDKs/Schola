// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnityEditor;

namespace AMD.Schola.Training.Editor
{
    /// <summary>
    /// Inspector for TrainingServer. Greys out the script settings until Run Script On Play is
    /// ticked and only shows the fields for the chosen script and Python environment, the way
    /// Unreal's EditCondition and EditConditionHides do on the gRPC Gym Connector.
    /// </summary>
    [CustomEditor(typeof(TrainingServer))]
    public sealed class TrainingServerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Environment"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_Port"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("m_RequestTimeoutSeconds"));

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Script Settings", EditorStyles.boldLabel);

            var runScript = serializedObject.FindProperty("m_RunScriptOnPlay");
            EditorGUILayout.PropertyField(runScript);

            using (new EditorGUI.DisabledScope(!runScript.boolValue))
            using (new EditorGUI.IndentLevelScope())
            {
                DrawScriptSettings(serializedObject.FindProperty("m_ScriptSettings"));
            }

            serializedObject.ApplyModifiedProperties();
        }

        static void DrawScriptSettings(SerializedProperty settings)
        {
            var scriptType = settings.FindPropertyRelative("ScriptType");
            EditorGUILayout.PropertyField(scriptType);

            if (scriptType.enumValueIndex == (int)ScriptType.Other)
            {
                EditorGUILayout.LabelField("Custom Script", EditorStyles.miniBoldLabel);
                DrawCustomSettings(settings.FindPropertyRelative("CustomScriptSettings"));
                return;
            }

            EditorGUILayout.LabelField("Python Env Settings", EditorStyles.miniBoldLabel);
            var envType = settings.FindPropertyRelative("EnvType");
            EditorGUILayout.PropertyField(envType);

            if (envType.enumValueIndex == (int)PythonEnvironmentType.Conda)
            {
                EditorGUILayout.PropertyField(settings.FindPropertyRelative("CondaEnvName"));
            }
            else if (envType.enumValueIndex == (int)PythonEnvironmentType.CustomPythonPath)
            {
                EditorGUILayout.PropertyField(settings.FindPropertyRelative("CustomPythonPath"));
            }

            EditorGUILayout.LabelField("Python Script", EditorStyles.miniBoldLabel);
            var pythonScriptType = settings.FindPropertyRelative("PythonScriptType");
            EditorGUILayout.PropertyField(pythonScriptType);

            if (pythonScriptType.enumValueIndex == (int)PythonScriptType.Custom)
            {
                DrawCustomSettings(settings.FindPropertyRelative("CustomPythonScriptSettings"));
            }
            else
            {
                DrawBuiltinSettings(settings.FindPropertyRelative("BuiltinSettings"));
            }
        }

        static void DrawBuiltinSettings(SerializedProperty builtin)
        {
            EditorGUILayout.PropertyField(builtin.FindPropertyRelative("Timesteps"));
            EditorGUILayout.PropertyField(builtin.FindPropertyRelative("Seed"));
            EditorGUILayout.PropertyField(builtin.FindPropertyRelative("CheckpointPath"));

            var export = builtin.FindPropertyRelative("ExportFinalModelToOnnx");
            EditorGUILayout.PropertyField(export);
            using (new EditorGUI.DisabledScope(!export.boolValue))
            {
                EditorGUILayout.PropertyField(builtin.FindPropertyRelative("OnnxPath"));
            }
        }

        static void DrawCustomSettings(SerializedProperty custom)
        {
            EditorGUILayout.PropertyField(custom.FindPropertyRelative("LaunchScript"));
            EditorGUILayout.PropertyField(custom.FindPropertyRelative("Args"), true);
            EditorGUILayout.PropertyField(custom.FindPropertyRelative("Flags"), true);
        }
    }
}
