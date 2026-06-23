// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

using UnrealBuildTool;

public class ScholaEditor : ModuleRules
{
	public ScholaEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		if (System.Environment.GetEnvironmentVariable("SCHOLA_MEASURE_CPP_COVERAGE") == "1")
		{
			OptimizeCode = CodeOptimization.Never;
		}

		PublicIncludePaths.AddRange(new string[] { });
		PrivateIncludePaths.AddRange(new string[] { "ScholaEditor/Private" });
		//TODO figure out if these can be moved to private
		PublicDependencyModuleNames.AddRange(new string[] { "Kismet", "NNE", "BlueprintEditorLibrary", "Schola", "KismetCompiler", });

		PrivateIncludePathModuleNames.AddRange(new string[] { });
        PrivateDependencyModuleNames.AddRange(new string[] { "AIModule", "Engine", "Core", "BlueprintGraph", "UnrealEd", "CoreUObject", "AutomationController", "KismetCompiler", "Schola", "ScholaInteractors", "gRPC", "ScholaTraining", "Slate", "SlateCore", "ToolMenus" });
		DynamicallyLoadedModuleNames.AddRange(new string[] { });
	}
}
