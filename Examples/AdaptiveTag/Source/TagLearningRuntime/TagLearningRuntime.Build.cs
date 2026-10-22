// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
using System.IO;
using UnrealBuildTool;

public class TagLearningRuntime : ModuleRules
{
    public TagLearningRuntime(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.NoPCHs;
        bUseUnity = false;
        bEnableExceptions = false;
        bUseRTTI = false;
        CppStandard = CppStandardVersion.Cpp20;
        FPSemantics = FPSemanticsMode.Precise;
        PrivateDependencyModuleNames.Add("Core");
        string Dependency = Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/RLtools"));
        PublicSystemIncludePaths.Add(Path.Combine(Dependency, "include"));
        PrivateDefinitions.AddRange(new string[] { "NOMINMAX", "RL_TOOLS_DEBUG" });
        RuntimeDependencies.Add("$(TargetOutputDir)/RLTools-LICENSE.txt", Path.Combine(Dependency, "LICENSE"), StagedFileType.NonUFS);
    }
}
