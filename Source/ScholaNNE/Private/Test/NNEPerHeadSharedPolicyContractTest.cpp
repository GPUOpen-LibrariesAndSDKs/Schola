// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
// Fixtures are built from embedded bytes, which needs editor-only UNNEModelData::Init.
#include "Misc/AutomationTest.h"
#include "Policies/NNEPerHeadSharedPolicy.h"
#include "BackgroundModels.inl"
#include "PerHeadModels.inl"

// Named, not anonymous: unity builds would collide on an anonymous helper.
namespace NNEPerHeadSharedPolicyContractTest
{
    UNNEModelData* MakeModel(TConstArrayView<uint8> Bytes)
    {
        UNNEModelData* ModelData = NewObject<UNNEModelData>();
        ModelData->Init(TEXT("onnx"), Bytes);
        return ModelData;
    }

    /** Asserts that the policy is unusable: not ready, no heads, no observation width. */
    void TestNotReady(FAutomationTestBase& Test, const TCHAR* Label, const UNNEPerHeadSharedPolicy& Policy)
    {
        TSet<FName> Heads;
        Policy.GetRegisteredHeads(Heads);
        Test.TestFalse(FString::Printf(TEXT("%s: policy is not ready"), Label), Policy.IsReady());
        Test.TestTrue(FString::Printf(TEXT("%s: no heads are registered"), Label), Heads.IsEmpty());
        Test.TestEqual(FString::Printf(TEXT("%s: no observation width"), Label), Policy.GetObservationWidth(), 0);
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPerHeadSharedPolicyInitContract,
    "Schola.Inference.SharedPolicy.PerHeadPolicy.InitContract",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNEPerHeadSharedPolicyInitContract::RunTest(const FString&)
{
    using namespace NNEPerHeadSharedPolicyContractTest;

    UNNEModelData* Narrow = MakeModel(MakeArrayView(PerHeadNarrowONNX));
    UNNEModelData* Wide = MakeModel(MakeArrayView(PerHeadWideONNX));
    UNNEModelData* TwoInputs = MakeModel(MakeArrayView(BackgroundTypedONNX));
    const FName Chase(TEXT("chase"));
    const FName Flee(TEXT("flee"));

    auto* Policy = NewObject<UNNEPerHeadSharedPolicy>();
    TestNotReady(*this, TEXT("Before Init"), *Policy);

    // Two heads reading the same observation width load successfully. They may share one model asset.
    Policy->HeadModels = {{Chase, Narrow}, {Flee, Narrow}};
    if (!TestTrue(TEXT("Init succeeds with matching observation widths"), Policy->Init()))
    {
        return false;
    }
    TestTrue(TEXT("Policy is ready"), Policy->IsReady());
    TestEqual(TEXT("Observation width comes from the models"), Policy->GetObservationWidth(), 2);
    TSet<FName> Heads;
    Policy->GetRegisteredHeads(Heads);
    TestEqual(TEXT("Two heads registered"), Heads.Num(), 2);
    TestTrue(TEXT("chase registered"), Heads.Contains(Chase));
    TestTrue(TEXT("flee registered"), Heads.Contains(Flee));
    TestEqual(TEXT("Init makes no model call"), int32(Policy->GetModelCallCount()), 0);

    // Every rejected configuration must leave the policy unusable, even after the earlier success.
    struct FRejected
    {
        const TCHAR* Label;
        TMap<FName, TObjectPtr<UNNEModelData>> HeadModels;
        const TCHAR* ExpectedError;
    };
    const FRejected Rejected[] = {
        {TEXT("Empty HeadModels"), {}, TEXT("HeadModels is empty")},
        {TEXT("Head without a model"), {{Chase, Narrow}, {Flee, nullptr}}, TEXT("has no model")},
        {TEXT("Head without a name"), {{NAME_None, Narrow}}, TEXT("A head has no name")},
        {TEXT("Mismatched observation widths"), {{Chase, Narrow}, {Flee, Wide}},
            TEXT("Every head must read the same observation width")},
        {TEXT("Model with two inputs"), {{Chase, TwoInputs}}, TEXT("exactly one observation input")},
    };
    for (const FRejected& Case : Rejected)
    {
        Policy->HeadModels = Case.HeadModels;
        AddExpectedError(Case.ExpectedError, EAutomationExpectedErrorFlags::Contains, 1);
        TestFalse(FString::Printf(TEXT("%s: Init fails"), Case.Label), Policy->Init());
        TestNotReady(*this, Case.Label, *Policy);
    }

    Policy->HeadModels = {{Chase, Wide}};
    TestTrue(TEXT("Init recovers once the configuration is valid"), Policy->Init());
    TestEqual(TEXT("Observation width follows the new model"), Policy->GetObservationWidth(), 3);
    return true;
}
#endif
