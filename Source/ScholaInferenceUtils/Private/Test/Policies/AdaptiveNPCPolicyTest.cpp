// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "Policies/AdaptiveNPCPolicy.h"
#include "Common/InstancedStructUtils.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "UObject/UnrealType.h"
#include <limits>

namespace
{
FInteractionDefinition AdaptiveDefinition(int32 ObservationSize = 4, int32 ActionSize = 2)
{
    FInteractionDefinition Definition;
    Definition.AgentType = TEXT("AdaptiveTest");
    Definition.ObsSpaceDefn.InitializeAs<FBoxSpace>();
    Definition.ObsSpaceDefn.GetMutable<FBoxSpace>().Dimensions.Init(FBoxSpaceDimension(-1.0f, 1.0f), ObservationSize);
    Definition.ActionSpaceDefn.InitializeAs<FBoxSpace>();
    Definition.ActionSpaceDefn.GetMutable<FBoxSpace>().Dimensions.Init(FBoxSpaceDimension(-1.0f, 1.0f), ActionSize);
    return Definition;
}

TInstancedStruct<FPoint> AdaptivePoint(float Value)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>(TArray<float>{Value});
    return Point;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptivePolicyInitializationTest, "Schola.Inference.Adaptive.Initialization",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptivePolicyInitializationTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UAdaptiveNPCPolicy>();
    TestFalse(TEXT("Starts without a validated definition"), Policy->IsInitialized());
    TestTrue(TEXT("Starts uninitialized"), Policy->GetSessionState() == EAdaptiveSessionState::Uninitialized);
    auto Output = AdaptiveDefinition(6, 3);
    TestFalse(TEXT("No definition to retrieve"), Policy->GetInteractionDefinition(Output));
    TestEqual(TEXT("Failed retrieval preserves caller definition"), Output.ObsSpaceDefn.Get<FBoxSpace>().Dimensions.Num(), 6);

    auto Input = AdaptiveDefinition();
    Input.ObsSpaceDefn.GetMutable<FBoxSpace>().Shape = {2, 2};
    Input.ActionSpaceDefn.GetMutable<FBoxSpace>().Shape = {1, 2};
    TestTrue(TEXT("Valid multidimensional Box definitions accepted"), Policy->Init(Input));
    TestTrue(TEXT("Local definition initialized"), Policy->IsInitialized());
    TestTrue(TEXT("Initialization does not claim remote readiness"), Policy->GetSessionState() == EAdaptiveSessionState::Initialized);
    TestFalse(TEXT("No inference work dispatched"), Policy->IsInferenceBusy());
    TestTrue(TEXT("Successful initialization clears errors"), Policy->GetLastError().IsEmpty());

    // Both input and output must be independent copies, including instanced space contents.
    Input.AgentType = TEXT("ChangedInput");
    Input.ObsSpaceDefn.GetMutable<FBoxSpace>().Dimensions[0].Low = -99.0f;
    TestTrue(TEXT("Accepted definition retrievable"), Policy->GetInteractionDefinition(Output));
    TestEqual(TEXT("Stored agent type is independent of input"), Output.AgentType, FString(TEXT("AdaptiveTest")));
    TestEqual(TEXT("Stored bounds are independent of input"), Output.ObsSpaceDefn.Get<FBoxSpace>().Dimensions[0].Low, -1.0f);
    Output.ActionSpaceDefn.GetMutable<FBoxSpace>().Dimensions.Reset();

    TestFalse(TEXT("Invalid reinitialization rejected"), Policy->Init(FInteractionDefinition()));
    TestTrue(TEXT("Rejected replacement preserves initialization"), Policy->IsInitialized());
    TestTrue(TEXT("Rejected replacement preserves local state"), Policy->GetSessionState() == EAdaptiveSessionState::Initialized);
    TestTrue(TEXT("Rejection has a diagnostic"), Policy->GetLastError().Contains(TEXT("Observation")));
    TestTrue(TEXT("Previous valid definition retained"), Policy->GetInteractionDefinition(Output));
    TestEqual(TEXT("Caller output mutation did not modify stored actions"), Output.ActionSpaceDefn.Get<FBoxSpace>().Dimensions.Num(), 2);
    TestTrue(TEXT("Previous shape retained"), Output.ObsSpaceDefn.Get<FBoxSpace>().Shape == TArray<int>({2, 2}));

    TestTrue(TEXT("Other observation/action lengths accepted"), Policy->Init(AdaptiveDefinition(6, 3)));
    TestTrue(TEXT("Valid replacement clears previous error"), Policy->GetLastError().IsEmpty());
    Policy->GetInteractionDefinition(Output);
    TestEqual(TEXT("Valid replacement stored"), Output.ActionSpaceDefn.Get<FBoxSpace>().Dimensions.Num(), 3);

    auto Unbounded = AdaptiveDefinition();
    auto& Dimensions = Unbounded.ObsSpaceDefn.GetMutable<FBoxSpace>().Dimensions;
    Dimensions[0] = FBoxSpaceDimension::Unbounded();
    Dimensions[0].Low = std::numeric_limits<float>::quiet_NaN(); // Disabled bounds have no meaning.
    Dimensions[1] = FBoxSpaceDimension::LowerBounded(0.0f);
    Dimensions[2] = FBoxSpaceDimension::UpperBounded(10.0f);
    Dimensions[3] = FBoxSpaceDimension(0.0f, 0.0f); // A constant observation is legitimate.
    TestTrue(TEXT("Unbounded, one-sided and constant observations accepted"), Policy->Init(Unbounded));
    auto* Duplicate = DuplicateObject<UAdaptiveNPCPolicy>(Policy, GetTransientPackage());
    TestTrue(TEXT("Unreal object copies retain their accepted initialization"), Duplicate->IsInitialized());
    TestTrue(TEXT("Copied initialization and session state remain consistent"), Duplicate->GetSessionState() == EAdaptiveSessionState::Initialized);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptivePolicyInvalidDefinitionsTest, "Schola.Inference.Adaptive.InvalidDefinitions",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptivePolicyInvalidDefinitionsTest::RunTest(const FString& Parameters)
{
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float Infinity = std::numeric_limits<float>::infinity();
    for (bool bAction : {false, true})
    {
        const FString Label = bAction ? TEXT("Action") : TEXT("Observation");
        auto Reject = [&](const TCHAR* Case, const FInteractionDefinition& Definition)
        {
            auto* Policy = NewObject<UAdaptiveNPCPolicy>();
            TestFalse(Label + TEXT(": ") + Case, Policy->Init(Definition));
            TestFalse(TEXT("Invalid setup never becomes initialized"), Policy->IsInitialized());
            TestTrue(TEXT("First failed initialization has Failed state"), Policy->GetSessionState() == EAdaptiveSessionState::Failed);
            TestTrue(TEXT("Diagnostic identifies the invalid space"), Policy->GetLastError().Contains(Label));
            TestTrue(TEXT("A valid definition can recover from failed setup"), Policy->Init(AdaptiveDefinition()));
            TestTrue(TEXT("Recovery clears the diagnostic"), Policy->GetLastError().IsEmpty());
        };
        auto Definition = AdaptiveDefinition();
        auto& Space = bAction ? Definition.ActionSpaceDefn : Definition.ObsSpaceDefn;
        const auto ValidSpace = Space;

        Space.Reset();
        Reject(TEXT("missing definition"), Definition);
        Space.InitializeAs<FSpace>();
        Reject(TEXT("abstract base space"), Definition);
        Space.InitializeAs<FDiscreteSpace>(3);
        Reject(TEXT("unsupported discrete space"), Definition);
        Space.InitializeAs<FDictSpace>();
        Reject(TEXT("unsupported dictionary space"), Definition);
        ToUntypedInstancedStruct(Space).InitializeAs<FInteractionDefinition>();
        Reject(TEXT("unrelated Blueprint payload"), Definition);
        Space.InitializeAs<FBoxSpace>();
        Reject(TEXT("empty box"), Definition);

        for (const auto& Shape : TArray<TArray<int>>{{0}, {-1}, {1}, {2, 3}, {MAX_int32, MAX_int32, MAX_int32}})
        {
            Space = ValidSpace;
            Space.GetMutable<FBoxSpace>().Shape = Shape;
            Reject(TEXT("invalid or overflowing shape"), Definition);
        }

        for (const auto& Bounds : TArray<FBoxSpaceDimension>{
                 FBoxSpaceDimension(2.0f, 1.0f), FBoxSpaceDimension(NaN, 1.0f),
                 FBoxSpaceDimension(-1.0f, NaN), FBoxSpaceDimension(-Infinity, 1.0f),
                 FBoxSpaceDimension(-1.0f, Infinity)})
        {
            Space = ValidSpace;
            Space.GetMutable<FBoxSpace>().Dimensions[0] = Bounds;
            Reject(TEXT("invalid enabled bounds"), Definition);
        }

        if (bAction)
        {
            for (const auto& Bounds : TArray<FBoxSpaceDimension>{
                     FBoxSpaceDimension::Unbounded(), FBoxSpaceDimension::LowerBounded(-1.0f),
                     FBoxSpaceDimension::UpperBounded(1.0f), FBoxSpaceDimension(1.0f, 1.0f),
                     FBoxSpaceDimension(-MAX_flt, MAX_flt)})
            {
                Space = ValidSpace;
                Space.GetMutable<FBoxSpace>().Dimensions[0] = Bounds;
                Reject(TEXT("action range cannot be scaled safely"), Definition);
            }
        }
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptivePolicyUnavailableInferenceTest, "Schola.Inference.Adaptive.UnavailableInference",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptivePolicyUnavailableInferenceTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UAdaptiveNPCPolicy>();
    IPolicy* Interface = Policy;
    auto Action = AdaptivePoint(9.0f);
    TArray<TInstancedStruct<FPoint>> Actions = {Action};
    TestFalse(TEXT("Immediate inference before initialization fails"), Interface->Think(AdaptivePoint(1.0f), Action));
    TestTrue(TEXT("Failed inference preserves action"), Action == AdaptivePoint(9.0f));
    TestTrue(TEXT("Local initialization succeeds"), Interface->Init(AdaptiveDefinition()));
    TestFalse(TEXT("Local initialization does not enable inference"), Interface->Think(AdaptivePoint(1.0f), Action));
    TestTrue(TEXT("Unavailable inference preserves action"), Action == AdaptivePoint(9.0f));
    TestFalse(TEXT("Default mutating batch fallback is not used"), Interface->BatchedThink({AdaptivePoint(1.0f), AdaptivePoint(2.0f)}, Actions));
    TestTrue(TEXT("Entire batch output preserved"), Actions.Num() == 1 && Actions[0] == AdaptivePoint(9.0f));
    TestTrue(TEXT("Unavailable inference directs callers to nonblocking requests"), Policy->GetLastError().Contains(TEXT("RequestDecision")));
    TestTrue(TEXT("Unavailable inference preserves local setup"), Policy->GetSessionState() == EAdaptiveSessionState::Initialized);
    TestFalse(TEXT("No work is in flight"), Interface->IsInferenceBusy());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptivePolicyBlueprintTest, "Schola.Inference.Adaptive.BlueprintAPI",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptivePolicyBlueprintTest::RunTest(const FString& Parameters)
{
    for (auto* Record : {FAdaptiveDecision::StaticStruct(), FAdaptiveExperience::StaticStruct()})
    {
        TestTrue(TEXT("Record type is available to Blueprint"), Record->HasMetaData(TEXT("BlueprintType")));
        for (TFieldIterator<FProperty> Property(Record); Property; ++Property)
        {
            TestTrue(Property->GetName(), Property->HasAnyPropertyFlags(CPF_BlueprintVisible));
        }
    }

    const FAdaptiveDecision Unassigned;
    const FAdaptiveExperience Experience;
    TestFalse(TEXT("Records do not manufacture a learner session"), Unassigned.SessionId.IsValid());
    TestTrue(TEXT("Default decision identity is explicitly unassigned"), Unassigned.AgentId == INDEX_NONE
        && Unassigned.EpisodeId == INDEX_NONE && Unassigned.DecisionId == INDEX_NONE && Unassigned.PolicyVersion == INDEX_NONE);
    TestTrue(TEXT("Experience also starts without attribution"), !Experience.SessionId.IsValid()
        && Experience.AgentId == INDEX_NONE && Experience.EpisodeId == INDEX_NONE
        && Experience.DecisionId == INDEX_NONE && Experience.PolicyVersion == INDEX_NONE);

    auto* Policy = NewObject<UAdaptiveNPCPolicy>();
    for (const TCHAR* Name : {TEXT("Init"), TEXT("Think"), TEXT("BatchedThink"), TEXT("IsInferenceBusy"),
             TEXT("IsInitialized"), TEXT("GetSessionState"), TEXT("GetLastError"), TEXT("GetInteractionDefinition")})
    {
        UFunction* Function = Policy->FindFunction(FName(Name));
        TestTrue(Name, Function && Function->HasAnyFunctionFlags(FUNC_BlueprintCallable));
    }

    struct FInitParameters
    {
        FInteractionDefinition InDefinition;
        bool ReturnValue = false;
    } Args;
    Args.InDefinition = AdaptiveDefinition();
    ToUntypedInstancedStruct(Args.InDefinition.ActionSpaceDefn).InitializeAs<FInteractionDefinition>();
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Init")), &Args);
    TestFalse(TEXT("Malformed Blueprint space rejected safely"), Args.ReturnValue);
    Args.InDefinition = AdaptiveDefinition();
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Init")), &Args);
    TestTrue(TEXT("Blueprint initialization dispatches successfully"), Args.ReturnValue);
    TestTrue(TEXT("Blueprint initialization remains local"), Policy->GetSessionState() == EAdaptiveSessionState::Initialized);
    return true;
}
