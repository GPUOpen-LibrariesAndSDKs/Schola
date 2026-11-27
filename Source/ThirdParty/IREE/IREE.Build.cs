// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

using UnrealBuildTool;
using System;
using System.IO;
using System.Linq;

/**
 * Optional External module for the IREE C runtime.
 *
 * When headers/libs are missing, defines SCHOLA_IREE_HAS_RUNTIME=0 so ScholaIREE
 * still compiles with a stub runtime (Init fails at runtime with a clear log).
 * Populate Win64 via Resources/Build/iree/build_iree_windows.ps1.
 */
public class IREE : ModuleRules
{
    private const string RuntimeApiHeaderRelative = "iree/runtime/api.h";

    public IREE(ReadOnlyTargetRules Target)
        : base(Target)
    {
        Type = ModuleType.External;

        string platformPath = Path.Combine(ModuleDirectory, Target.Platform.ToString());
        string includePath = Path.Combine(platformPath, "include");
        string libraryPath = Path.Combine(platformPath, "lib");
        string runtimeApiHeader = Path.Combine(includePath, RuntimeApiHeaderRelative);

        bool hasHeaders = File.Exists(runtimeApiHeader);
        bool hasLibs = false;
        string searchPattern = Target.Platform == UnrealTargetPlatform.Win64 ? "*.lib" : null;

        if (searchPattern != null && Directory.Exists(libraryPath))
        {
            hasLibs = Directory.GetFiles(libraryPath, searchPattern).Length > 0;
        }

        if (hasHeaders && hasLibs && searchPattern != null)
        {
            PublicSystemIncludePaths.Add(includePath);
            foreach (string library in Directory
                .GetFiles(libraryPath, searchPattern)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                PublicAdditionalLibraries.Add(library);
            }
            PublicDefinitions.Add("SCHOLA_IREE_HAS_RUNTIME=1");
            // IREE headers only declare iree_allocator_system() when the embedder names the system allocator.
            PublicDefinitions.Add("IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl");
            System.Console.WriteLine("Schola IREE: linking prebuilt runtime from " + platformPath);
        }
        else
        {
            PublicDefinitions.Add("SCHOLA_IREE_HAS_RUNTIME=0");
            System.Console.WriteLine(
                "Schola IREE: runtime not found under " + platformPath
                + ". ScholaIREE will build with a stub (populate via Resources/Build/iree/build_iree_windows.ps1)."
            );
        }
    }
}
