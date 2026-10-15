// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "Test/Policies/BlueprintBatchTestPolicy.h"
#include "Points/BoxPoint.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueprintPolicyBatchTest, "Schola.Policies.Blueprint.BatchAndBusy",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FBlueprintPolicyBatchTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UBlueprintBatchTestPolicy>();
    IPolicy* Interface = Policy;
    TInstancedStruct<FPoint> Observation;
    Observation.InitializeAs<FBoxPoint>(TArray<float>{1});
    TArray<TInstancedStruct<FPoint>> Observations = {Observation, Observation};
    TArray<TInstancedStruct<FPoint>> Actions;
    TestTrue(TEXT("Default batch falls back to Think events"), Interface->BatchedThink(Observations, Actions));
    TestEqual(TEXT("One Think event per observation"), Policy->ThinkCalls, 2);
    TestTrue(TEXT("Fallback preserves row order"), Actions == Observations);
    Policy->bBatchOverride = true;
    TestTrue(TEXT("Batch override dispatches"), Interface->BatchedThink(Observations, Actions));
    TestEqual(TEXT("One batch event"), Policy->BatchCalls, 1);
    TestEqual(TEXT("Override skips per-row events"), Policy->ThinkCalls, 2);
    Policy->bWrongCount = true;
    TestFalse(TEXT("Malformed override output rejected"), Interface->BatchedThink(Observations, Actions));
    TestTrue(TEXT("Failed override preserves caller output"), Actions == Observations);
    Policy->bBatchOverride = false;
    TestFalse(TEXT("Invalid input rejected"), Interface->BatchedThink({{}}, Actions));
    TestEqual(TEXT("Invalid input skips events"), Policy->ThinkCalls, 2);
    TestFalse(TEXT("Default busy state"), Interface->IsInferenceBusy());
    Policy->bBusy = true;
    TestTrue(TEXT("Busy override dispatches"), Interface->IsInferenceBusy());
    return true;
}
