// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnrealBuildTool;

public class ScholaIREE : ModuleRules
{
	public ScholaIREE(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		if (System.Environment.GetEnvironmentVariable("SCHOLA_MEASURE_CPP_COVERAGE") == "1")
		{
			OptimizeCode = CodeOptimization.Never;
		}

		PrivateIncludePaths.AddRange(new string[] { "ScholaIREE/Private" });

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Schola",
			"ScholaInferenceUtils",
			"IREE",
		});

		PrivateDependencyModuleNames.AddRange(new string[] { "Json", "JsonUtilities" });
	}
}