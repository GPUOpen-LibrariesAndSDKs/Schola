using UnrealBuildTool;
public class ScholaMass : ModuleRules
{
    public ScholaMass(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine", "MassCore", "MassEntity", "MassSignals", "ScholaInferenceUtils", "Schola" });
    }
}
