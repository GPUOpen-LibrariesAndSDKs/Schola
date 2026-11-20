// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
// Fixtures are built from embedded bytes, which needs editor-only UNNEModelData::Init.
#include "Misc/AutomationTest.h"
#include "Policies/NNEPerHeadSharedPolicy.h"
#include "Points/BoxPoint.h"
#include "Points/MultiDiscretePoint.h"
#include "BackgroundModels.inl"
#include "PerHeadModels.inl"

#include <limits>

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

    TInstancedStruct<FPoint> MakeObservation(TArray<float> Values)
    {
        return TInstancedStruct<FPoint>::Make<FBoxPoint>(MoveTemp(Values));
    }

    /** Checks one action exactly; the fixtures compute small integers, which float represents exactly. */
    void TestAction(FAutomationTestBase& Test, const FString& Label, const TInstancedStruct<FPoint>& Action,
        const TArray<float>& Expected)
    {
        const FBoxPoint* Box = Action.GetPtr<FBoxPoint>();
        if (!Test.TestNotNull(*FString::Printf(TEXT("%s is a FBoxPoint"), *Label), Box))
        {
            return;
        }
        if (!Test.TestEqual(*FString::Printf(TEXT("%s width"), *Label), Box->Values.Num(), Expected.Num()))
        {
            return;
        }
        for (int32 Column = 0; Column < Expected.Num(); ++Column)
        {
            Test.TestEqual(*FString::Printf(TEXT("%s column %d"), *Label, Column), Box->Values[Column], Expected[Column]);
        }
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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPerHeadSharedPolicyThinkContract,
    "Schola.Inference.SharedPolicy.PerHeadPolicy.ThinkContract",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNEPerHeadSharedPolicyThinkContract::RunTest(const FString&)
{
    using namespace NNEPerHeadSharedPolicyContractTest;

    // chase = Narrow: [2*x0, 3*x1]. flee = Sum: [x0 + x1]. Different widths and formulas show which model served a row.
    const FName Chase(TEXT("chase"));
    const FName Flee(TEXT("flee"));
    auto* Policy = NewObject<UNNEPerHeadSharedPolicy>();
    Policy->HeadModels = {{Chase, MakeModel(MakeArrayView(PerHeadNarrowONNX))},
        {Flee, MakeModel(MakeArrayView(PerHeadSumONNX))}};
    Policy->MaxBatchSize = 2;
    if (!TestTrue(TEXT("Init succeeds"), Policy->Init()))
    {
        return false;
    }

    // Interleaved heads: each row is served by its own head and returned in its original position.
    {
        const TArray<TInstancedStruct<FPoint>> Observations = {
            MakeObservation({1.f, 2.f}), MakeObservation({3.f, 4.f}), MakeObservation({5.f, 6.f}), MakeObservation({7.f, 8.f})};
        const TArray<FName> Heads = {Chase, Flee, Chase, Flee};
        TArray<TInstancedStruct<FPoint>> Actions;
        if (!TestTrue(TEXT("Mixed batch succeeds"), Policy->BatchedThinkWithHeads(Observations, Heads, Actions)))
        {
            return false;
        }
        TestEqual(TEXT("Mixed batch: one model call per head present"), int32(Policy->GetModelCallCount()), 2);
        if (TestEqual(TEXT("Mixed batch: one action per row"), Actions.Num(), 4))
        {
            TestAction(*this, TEXT("Row 0 (chase)"), Actions[0], {2.f, 6.f});
            TestAction(*this, TEXT("Row 1 (flee)"), Actions[1], {7.f});
            TestAction(*this, TEXT("Row 2 (chase)"), Actions[2], {10.f, 18.f});
            TestAction(*this, TEXT("Row 3 (flee)"), Actions[3], {15.f});
        }
    }

    // One head only: one call. MaxBatchSize limits rows per call, so two rows of one head are allowed.
    {
        const TArray<TInstancedStruct<FPoint>> Observations = {MakeObservation({1.f, 1.f}), MakeObservation({2.f, 0.f})};
        const TArray<FName> Heads = {Flee, Flee};
        TArray<TInstancedStruct<FPoint>> Actions;
        const int32 CallsBefore = int32(Policy->GetModelCallCount());
        TestTrue(TEXT("Homogeneous batch succeeds"), Policy->BatchedThinkWithHeads(Observations, Heads, Actions));
        TestEqual(TEXT("Homogeneous batch: one model call"), int32(Policy->GetModelCallCount()) - CallsBefore, 1);
        if (TestEqual(TEXT("Homogeneous batch: one action per row"), Actions.Num(), 2))
        {
            TestAction(*this, TEXT("Homogeneous row 0"), Actions[0], {2.f});
            TestAction(*this, TEXT("Homogeneous row 1"), Actions[1], {2.f});
        }
    }

    // Each rejection happens before any model runs: no call is counted and the output keeps its old contents.
    struct FRejected
    {
        const TCHAR* Label;
        TArray<TInstancedStruct<FPoint>> Observations;
        TArray<FName> Heads;
        const TCHAR* ExpectedError;
    };
    TArray<TInstancedStruct<FPoint>> WrongType = {MakeObservation({1.f, 2.f}), {}};
    WrongType[1].InitializeAs<FMultiDiscretePoint>(TArray<int>{1});
    const FRejected Rejected[] = {
        {TEXT("Empty batch"), {}, {}, TEXT("Empty batch")},
        {TEXT("Fewer heads than observations"), {MakeObservation({1.f, 2.f}), MakeObservation({3.f, 4.f})}, {Chase},
            TEXT("2 observations but 1 heads")},
        {TEXT("Unknown head after valid rows"), {MakeObservation({1.f, 2.f}), MakeObservation({3.f, 4.f})},
            {Chase, TEXT("Combat")}, TEXT("Row 1 names unknown head 'Combat'")},
        {TEXT("Unset head"), {MakeObservation({1.f, 2.f})}, {NAME_None}, TEXT("unknown head 'None'")},
        {TEXT("One head over MaxBatchSize"),
            {MakeObservation({1.f, 2.f}), MakeObservation({1.f, 2.f}), MakeObservation({1.f, 2.f})},
            {Chase, Chase, Chase}, TEXT("Head 'chase' has 3 rows, which exceeds the limit of 2")},
        {TEXT("Observation too short"), {MakeObservation({1.f, 2.f}), MakeObservation({3.f})}, {Chase, Flee},
            TEXT("Row 1 has 1 values, expected 2")},
        {TEXT("Wrong point type"), WrongType, {Chase, Flee}, TEXT("Row 1 is not a FBoxPoint")},
        // The NaN row belongs to the second head, so the first head's model must not have run either.
        {TEXT("NaN in the second head's row"),
            {MakeObservation({1.f, 2.f}), MakeObservation({std::numeric_limits<float>::quiet_NaN(), 4.f})},
            {Chase, Flee}, TEXT("Row 1 has a non-finite value")},
    };
    for (const FRejected& Case : Rejected)
    {
        TArray<TInstancedStruct<FPoint>> Actions = {MakeObservation({7.f})};
        const int32 CallsBefore = int32(Policy->GetModelCallCount());
        AddExpectedError(Case.ExpectedError, EAutomationExpectedErrorFlags::Contains, 1);
        TestFalse(FString::Printf(TEXT("%s: rejected"), Case.Label),
            Policy->BatchedThinkWithHeads(Case.Observations, Case.Heads, Actions));
        TestEqual(FString::Printf(TEXT("%s: no model call"), Case.Label), int32(Policy->GetModelCallCount()) - CallsBefore, 0);
        if (TestEqual(FString::Printf(TEXT("%s: output keeps its size"), Case.Label), Actions.Num(), 1))
        {
            TestAction(*this, FString::Printf(TEXT("%s: output keeps its value"), Case.Label), Actions[0], {7.f});
        }
    }

    {
        auto* Uninitialized = NewObject<UNNEPerHeadSharedPolicy>();
        TArray<TInstancedStruct<FPoint>> Actions;
        AddExpectedError(TEXT("Policy is not initialized"), EAutomationExpectedErrorFlags::Contains, 1);
        TestFalse(TEXT("Uninitialized policy rejects"),
            Uninitialized->BatchedThinkWithHeads({MakeObservation({1.f, 2.f})}, {Chase}, Actions));
        TestTrue(TEXT("Uninitialized policy writes no actions"), Actions.IsEmpty());
    }

    // The rejections must not have damaged the policy.
    {
        TArray<TInstancedStruct<FPoint>> Actions;
        TestTrue(TEXT("A valid request still succeeds"),
            Policy->BatchedThinkWithHeads({MakeObservation({3.f, 4.f}), MakeObservation({1.f, 1.f})}, {Flee, Chase}, Actions));
        if (TestEqual(TEXT("After rejections: one action per row"), Actions.Num(), 2))
        {
            TestAction(*this, TEXT("After rejections row 0 (flee)"), Actions[0], {7.f});
            TestAction(*this, TEXT("After rejections row 1 (chase)"), Actions[1], {2.f, 3.f});
        }
    }
    return true;
}
#endif
