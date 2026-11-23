// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnrealBuildTool;

public class ScholaMjcf : ModuleRules
{
    public ScholaMjcf(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        if (System.Environment.GetEnvironmentVariable("SCHOLA_MEASURE_CPP_COVERAGE") == "1")
        {
            OptimizeCode = CodeOptimization.Never;
        }

        PublicIncludePaths.AddRange(new string[] { });
        PrivateIncludePaths.AddRange(new string[] { "ScholaMjcf/Private" });

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "XmlParser",
        });
    }
}
