// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "RLToolsPolicyTestHelpers.h"
#include "RLToolsConfigurationAsset.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "HAL/FileManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"

#if WITH_DEV_AUTOMATION_TESTS
using namespace RLToolsPolicyTests;
struct FRLToolsConfigurationTestAccess
{
    static bool Write(const URLToolsConfigurationAsset* Asset, const FString& Path, const FString& Source, bool Remove, FString& Error)
    { return Asset->WriteRequest(Path, Source, Remove, Error); }
};
namespace
{
void Configure(URLToolsConfigurationAsset* Asset, const FRLToolsConfiguration& P)
{
    Asset->InteractionDefinition = Definition(P.Observations, P.Actions);
    auto& S = Asset->BuildSettings;
    S.Name = P.Id.Mid(P.Id.Find(TEXT("/project/")) + 9);
    S.Algorithm = P.Algorithm;
    S.ActorHiddenWidth = P.HiddenWidth; S.CriticHiddenWidth = P.CriticHiddenWidth;
    S.ActorLayers = P.ActorLayers; S.CriticLayers = P.CriticLayers;
    S.ActorActivation = P.ActorActivation; S.CriticActivation = P.CriticActivation;
    S.BatchSize = P.BatchSize; S.ReplayCapacity = P.ReplayCapacity; S.RolloutLength = P.RolloutLength;
    S.Discount = P.Discount; S.TargetTau = P.TargetTau;
    S.ActorInterval = P.ActorInterval; S.TargetInterval = P.TargetInterval;
    S.EntropyCoefficient = P.EntropyCoefficient; S.bAdaptiveEntropy = P.bAdaptiveEntropy;
    S.TargetActionNoise = P.TargetActionNoise; S.TargetActionNoiseClip = P.TargetActionNoiseClip;
    S.PPOClip = P.PPOClip; S.GAE = P.GAE; S.PPOEntropyCoefficient = P.PPOEntropyCoefficient;
    S.InitialActionStd = P.InitialActionStd; S.bNormalizeAdvantage = P.bNormalizeAdvantage; S.bLearnActionStd = P.bLearnActionStd;
}
URLToolsConfigurationAsset* FixtureAsset(ERLToolsAlgorithm Algorithm, int32 Index)
{
    auto* Asset = NewObject<URLToolsConfigurationAsset>();
    Asset->InteractionDefinition = Definition(6 + Index, Index == 2 ? 2 : 3 + Index);
    auto& S = Asset->BuildSettings;
    S.Name = Algorithm == ERLToolsAlgorithm::SAC ? TEXT("guards-sac") : Algorithm == ERLToolsAlgorithm::TD3 ? TEXT("guards-td3") : TEXT("guards-ppo");
    S.Algorithm = Algorithm;
    S.ActorHiddenWidth = Index == 1 ? 16 : 24;
    S.CriticHiddenWidth = Index == 1 ? 24 : 16;
    S.ActorLayers = Index == 1 ? 3 : 4;
    S.CriticLayers = Index == 1 ? 4 : 3;
    S.ActorActivation = Index == 1 ? ERLToolsActivation::ReLU : ERLToolsActivation::Tanh;
    S.CriticActivation = Index == 0 ? ERLToolsActivation::ReLU : ERLToolsActivation::Tanh;
    S.BatchSize = 8; S.ReplayCapacity = 256; S.RolloutLength = 32;
    S.Discount = Index == 0 ? 0.97f : Index == 1 ? 0.95f : 0.98f;
    S.TargetTau = Index == 0 ? 0.02f : 0.01f;
    S.ActorInterval = Index == 0 ? 2 : 3; S.TargetInterval = Index == 0 ? 3 : 4;
    S.EntropyCoefficient = 0.3f; S.bAdaptiveEntropy = false;
    S.TargetActionNoise = 0.1f; S.TargetActionNoiseClip = 0.25f;
    S.PPOClip = 0.15f; S.GAE = 0.9f; S.PPOEntropyCoefficient = 0.02f;
    S.InitialActionStd = 0.4f; S.bNormalizeAdvantage = false;
    return Asset;
}
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRLToolsConfigurationValidationTest, "Schola.RLTools.Generation.AssetValidation", Flags)
bool FRLToolsConfigurationValidationTest::RunTest(const FString&)
{
    auto* Asset = NewObject<URLToolsConfigurationAsset>();
    Asset->InteractionDefinition = Definition(6, 3);
    FString Error;
    TestTrue(TEXT("Arbitrary positive Schola dimensions accepted"), Asset->ValidateConfiguration(Error));
    auto Saved = Asset->BuildSettings;
    auto Reject = [&] { TestFalse(TEXT("Invalid build settings rejected"), Asset->ValidateConfiguration(Error)); TestFalse(TEXT("Actionable diagnostic"), Error.IsEmpty()); Asset->BuildSettings = Saved; };
    Asset->BuildSettings.Name = TEXT("Bad Name"); Reject();
    Asset->BuildSettings.ActorLayers = 1; Reject();
    Asset->BuildSettings.CriticHiddenWidth = MAX_int32; Reject();
    Asset->BuildSettings.BatchSize = 1; Reject();
    Asset->BuildSettings.ReplayCapacity = 1; Reject();
    Asset->BuildSettings.Discount = 0; Reject();
    Asset->BuildSettings.ActorActivation = static_cast<ERLToolsActivation>(255); Reject();
    Asset->BuildSettings.Algorithm = ERLToolsAlgorithm::PPO;
    Asset->BuildSettings.RolloutLength = 63;
    TestFalse(TEXT("Invalid PPO rollout rejected"), Asset->ValidateConfiguration(Error));
    Asset->BuildSettings = Saved;
    Asset->BuildSettings.Name = TEXT("never-built-test-configuration");
    FRLToolsConfiguration Sentinel;
    Sentinel.Id = TEXT("unchanged");
    TestFalse(TEXT("Uncompiled configuration fails"), Asset->GetCompiledConfiguration(Sentinel, Error));
    TestTrue(TEXT("Missing configuration explains rebuilding"), Error.Contains(TEXT("rebuild")));
    TestEqual(TEXT("Failure preserves metadata output"), Sentinel.Id, FString(TEXT("unchanged")));
    Asset->GenerateProjectConfiguration();
    TestTrue(TEXT("Transient objects cannot claim a saved asset request"), Asset->GenerationStatus.Contains(TEXT("Save")));
    for (const TCHAR* Name : {TEXT("GenerateProjectConfiguration"), TEXT("RemoveProjectConfiguration"), TEXT("ValidateConfiguration"), TEXT("GetCompiledConfiguration"), TEXT("GetConfigurationId")})
    {
        const auto* Function = Asset->FindFunction(FName(Name));
        TestTrue(TEXT("Asset workflow is Blueprint callable"), Function && Function->HasAnyFunctionFlags(FUNC_BlueprintCallable));
    }

    FWorld F;
    auto* Policy = F.Policy();
    Policy->Init(Definition()); Ready(Policy);
    const auto Session = Policy->GetSessionInfo();
    Policy->ConfigurationAsset = Asset;
    TestFalse(TEXT("Missing generated replacement rejected"), Policy->Init(Definition()));
    TestTrue(TEXT("Invalid generated replacement retains the existing binding"), Policy->GetSessionInfo().SessionId == Session.SessionId);
    FAdaptiveDecision D;
    TestTrue(TEXT("Existing binding still infers"), Decision(Policy, Observation(), D));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRLToolsConfigurationRequestTest, "Schola.RLTools.Generation.ProjectRequest", Flags)
bool FRLToolsConfigurationRequestTest::RunTest(const FString&)
{
    const FString Directory = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Automation/ScholaGeneration"), FGuid::NewGuid().ToString(EGuidFormats::Digits));
    const FString Manifest = FPaths::Combine(Directory, TEXT("ScholaRLTools.json"));
    // A generated, bounded directory under this project's Saved directory.
    ON_SCOPE_EXIT { IFileManager::Get().Delete(*Manifest); IFileManager::Get().DeleteDirectory(*Directory, false, false); };
    FString Error;
    auto* Asset = NewObject<URLToolsConfigurationAsset>();
    Asset->InteractionDefinition = Definition(6, 3);
    Asset->BuildSettings.Name = TEXT("guards");
    TestTrue(TEXT("Write editor request"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Test/Guards.Guards"), false, Error));
    FString First;
    FFileHelper::LoadFileToString(First, *Manifest);
    TestTrue(TEXT("Repeat is idempotent"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Test/Guards.Guards"), false, Error));
    FString Again;
    FFileHelper::LoadFileToString(Again, *Manifest);
    TestEqual(TEXT("Repeated request bytes unchanged"), Again, First);
    TestFalse(TEXT("Different owner cannot replace same configuration"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Other.Other"), false, Error));
    FFileHelper::LoadFileToString(Again, *Manifest);
    TestEqual(TEXT("Collision preserves original request"), Again, First);
    Asset->BuildSettings.Name = TEXT("renamed-guards");
    Asset->BuildSettings.Algorithm = ERLToolsAlgorithm::TD3;
    Asset->BuildSettings.ActorHiddenWidth = 24; Asset->BuildSettings.CriticHiddenWidth = 16;
    Asset->BuildSettings.ActorLayers = 4; Asset->BuildSettings.CriticActivation = ERLToolsActivation::Tanh;
    TestTrue(TEXT("Editing one asset replaces its previous name/algorithm request"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Test/Guards.Guards"), false, Error));
    FFileHelper::LoadFileToString(Again, *Manifest);
    TSharedPtr<FJsonObject> Root;
    TestTrue(TEXT("Generated request is valid JSON"), FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Again), Root));
    const auto& Profiles = Root->GetArrayField(TEXT("profiles"));
    TestEqual(TEXT("Renaming does not accumulate obsolete requests"), Profiles.Num(), 1);
    const auto Entry = Profiles[0]->AsObject();
    TestEqual(TEXT("Flattened Schola observation dimensions"), Entry->GetIntegerField(TEXT("observations")), 6);
    TestEqual(TEXT("Flattened Schola action dimensions"), Entry->GetIntegerField(TEXT("actions")), 3);
    TestEqual(TEXT("New algorithm requested"), Entry->GetStringField(TEXT("algorithm")), FString(TEXT("TD3")));
    TestFalse(TEXT("Inactive PPO fields are not serialized"), Entry->HasField(TEXT("rolloutLength")));
    TestFalse(TEXT("Inactive SAC parameters are not serialized"), Entry->GetObjectField(TEXT("parameters"))->HasField(TEXT("adaptiveEntropy")));
    Asset->BuildSettings.BatchSize = 1; // Removal remains possible after invalid local editing.
    TestTrue(TEXT("Remove owned request without valid current parameters"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Test/Guards.Guards"), true, Error));
    FFileHelper::LoadFileToString(Again, *Manifest);
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Again), Root);
    TestEqual(TEXT("Removed request absent"), Root->GetArrayField(TEXT("profiles")).Num(), 0);
    Asset->BuildSettings.BatchSize = 32;
    FFileHelper::SaveStringToFile(TEXT("{ broken"), *Manifest);
    TestFalse(TEXT("Malformed existing manifest is preserved"), FRLToolsConfigurationTestAccess::Write(Asset, Manifest, TEXT("/Game/Test/Guards.Guards"), false, Error));
    FFileHelper::LoadFileToString(Again, *Manifest);
    TestEqual(TEXT("Malformed source was not overwritten"), Again, FString(TEXT("{ broken")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRLToolsGeneratedPolicyTest, "Schola.RLTools.Generation.CompiledAssetPolicies", Flags)
bool FRLToolsGeneratedPolicyTest::RunTest(const FString&)
{
    int32 Tested = 0;
    for (auto Algorithm : {ERLToolsAlgorithm::SAC, ERLToolsAlgorithm::TD3, ERLToolsAlgorithm::PPO})
    {
        for (const auto& Profile : URLToolsNPCPolicy::GetAvailableConfigurations(Algorithm))
        {
            if (!Profile.Id.Contains(TEXT("/project/"))) continue;
            ++Tested;
            AddInfo(TEXT("Generated profile used through asset/policy: ") + Profile.Id);
            FWorld F;
            auto* Asset = FixtureAsset(Algorithm, static_cast<int32>(Algorithm));
            if (Asset->GetConfigurationId() != Profile.Id) Configure(Asset, Profile);
            auto* P = F.Policy();
            P->ConfigurationAsset = Asset; // Legacy Algorithm/Configuration intentionally remain default SAC/3x2.
            P->Settings.LearningStarts = Profile.BatchSize;
            P->Settings.PPOEpochs = 2;
            P->SharedGroup = TEXT("GeneratedGroup");
            FString Error;
            FRLToolsConfiguration Resolved;
            TestTrue(TEXT("Asset agrees with compiled metadata"), Asset->GetCompiledConfiguration(Resolved, Error));
            TestEqual(TEXT("Critic architecture is actually described"), Resolved.CriticHiddenWidth, Profile.CriticHiddenWidth);
            TestTrue(TEXT("Init through generated asset"), P->Init(Asset->InteractionDefinition));
            if (!TestTrue(TEXT("Generated learner ready"), Ready(P))) return false;
            auto* Shared = F.Policy();
            Shared->ConfigurationAsset = Asset;
            Shared->Settings = P->Settings;
            Shared->SharedGroup = P->SharedGroup;
            TestTrue(TEXT("Shared generated configuration joins"), Shared->Init(Asset->InteractionDefinition));
            if (!Ready(Shared)) return false;
            TestTrue(TEXT("Canonical compiled algorithm/id used for sharing"), Shared->GetSessionInfo().SessionId == P->GetSessionInfo().SessionId);
            const auto Original = Asset->BuildSettings;
            for (int32 FRLToolsBuildSettings::* Member : {&FRLToolsBuildSettings::ActorHiddenWidth, &FRLToolsBuildSettings::CriticHiddenWidth,
                &FRLToolsBuildSettings::ActorLayers, &FRLToolsBuildSettings::CriticLayers})
            {
                Asset->BuildSettings = Original;
                if (Member == &FRLToolsBuildSettings::ActorLayers || Member == &FRLToolsBuildSettings::CriticLayers)
                    Asset->BuildSettings.*Member = (Asset->BuildSettings.*Member) == 16 ? 15 : (Asset->BuildSettings.*Member) + 1;
                else ++(Asset->BuildSettings.*Member);
                TestFalse(TEXT("Edited network reports stale compiled configuration"), Asset->GetCompiledConfiguration(Resolved, Error));
                TestTrue(TEXT("Stale metadata gives rebuild instructions"), Error.Contains(TEXT("stale")));
            }
            Asset->BuildSettings = Original;
            Asset->BuildSettings.Discount = Original.Discount == 0.9f ? 0.8f : 0.9f;
            const auto Session = P->GetSessionInfo();
            TestFalse(TEXT("Changed algorithm template parameter requires rebuild"), P->Init(Asset->InteractionDefinition));
            TestTrue(TEXT("Rejected stale replacement preserves active learner"), P->GetSessionInfo().SessionId == Session.SessionId);
            Asset->BuildSettings = Original;
            Asset->InteractionDefinition.ActionSpaceDefn.GetMutable<FBoxSpace>().Dimensions[0].Low -= 1;
            TestTrue(TEXT("Runtime bounds do not require new numerical templates"), Asset->GetCompiledConfiguration(Resolved, Error));
            const auto Obs = Observation(Profile.Observations);
            const int32 Records = Algorithm == ERLToolsAlgorithm::PPO ? Profile.RolloutLength : FMath::Max(64, Profile.BatchSize);
            for (int32 I = 0; I < Records; ++I)
            {
                FAdaptiveDecision D;
                if (!TestTrue(TEXT("Generated asset chooses action"), Decision(P, Obs, D))) return false;
                auto E = Outcome(D, Obs);
                E.bTerminated = I + 1 == Records;
                if (!TestEqual(TEXT("Generated asset admits outcome"), P->SubmitExperience(E), EAdaptiveSubmitResult::Accepted)) return false;
            }
            TestTrue(TEXT("Generated learner trains and publishes"), Wait([P, Records] { auto S = Stats(P); return S.Stored == Records && S.Version > 0; }));
        }
    }
    AddInfo(FString::Printf(TEXT("Project-generated configurations exercised: %d"), Tested));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRLToolsConfigurationExportTest, "Schola.RLTools.Generation.ExportBuildFixture", Flags)
bool FRLToolsConfigurationExportTest::RunTest(const FString&)
{
    // Test artifact only: never edit the running project's Config or a plugin source file.
    const FString Manifest = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Automation/RLToolsGeneration/ScholaRLTools.json"));
    int32 Index = 0;
    for (auto Algorithm : {ERLToolsAlgorithm::SAC, ERLToolsAlgorithm::TD3, ERLToolsAlgorithm::PPO})
    {
        auto* Asset = FixtureAsset(Algorithm, Index);
        const auto& S = Asset->BuildSettings;
        FString Error;
        const FString Source = TEXT("/Game/ScholaGeneration/") + S.Name;
        if (!TestTrue(TEXT("Generate request directly from Schola definition and asset settings"),
            FRLToolsConfigurationTestAccess::Write(Asset, Manifest, Source, false, Error))) return false;
        ++Index;
    }
    AddInfo(TEXT("Editor-generated build fixture: ") + Manifest);
    return true;
}
#endif
