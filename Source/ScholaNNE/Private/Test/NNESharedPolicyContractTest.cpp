// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
// Fixtures are built from embedded bytes, which needs editor-only UNNEModelData::Init.
#include "Misc/AutomationTest.h"
#include "Policies/NNESharedPolicy.h"
#include "Points/BoxPoint.h"
#include "BackgroundModels.inl"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNESharedPolicyRejectsUnsupportedModels,
    "Schola.Inference.SharedPolicy.NNEPolicy.RejectsUnsupportedModels",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNESharedPolicyRejectsUnsupportedModels::RunTest(const FString&)
{
    // Both fixtures take two inputs, so neither fits the one-observation-tensor contract.
    struct FFixture { const TCHAR* Name; TConstArrayView<uint8> Bytes; };
    const FFixture Fixtures[] = {
        {TEXT("typed arithmetic model"), MakeArrayView(BackgroundTypedONNX)},
        {TEXT("recurrent model"), MakeArrayView(BackgroundRecurrentONNX)},
    };
    for (const FFixture& Fixture : Fixtures)
    {
        auto* Policy = NewObject<UNNESharedPolicy>();
        Policy->ModelData = NewObject<UNNEModelData>();
        Policy->ModelData->Init(TEXT("onnx"), Fixture.Bytes);

        AddExpectedError(TEXT("exactly one observation input"), EAutomationExpectedErrorFlags::Contains, 1);
        TestFalse(FString::Printf(TEXT("%s: Init rejects the model"), Fixture.Name), Policy->Init());
        TestFalse(FString::Printf(TEXT("%s: policy is not ready"), Fixture.Name), Policy->IsReady());
        TSet<FName> Heads;
        Policy->GetRegisteredHeads(Heads);
        TestTrue(FString::Printf(TEXT("%s: no heads are registered"), Fixture.Name), Heads.IsEmpty());

        // A rejected model must not be usable, and must not produce any action.
        TArray<TInstancedStruct<FPoint>> Observations;
        Observations.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float>{0.f, 0.f}));
        TArray<FName> HeadNames = {TEXT("x")};
        TArray<TInstancedStruct<FPoint>> Actions;
        AddExpectedError(TEXT("Policy is not initialized"), EAutomationExpectedErrorFlags::Contains, 1);
        TestFalse(FString::Printf(TEXT("%s: inference is refused"), Fixture.Name),
            Policy->BatchedThinkWithHeads(Observations, HeadNames, Actions));
        TestTrue(FString::Printf(TEXT("%s: no partial actions"), Fixture.Name), Actions.IsEmpty());
        TestEqual(FString::Printf(TEXT("%s: no model call"), Fixture.Name), int32(Policy->GetModelCallCount()), 0);
    }
    return true;
}
#endif
