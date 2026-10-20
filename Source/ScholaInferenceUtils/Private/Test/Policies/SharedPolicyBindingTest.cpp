// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "Policies/NPCInferencePolicy.h"
#include "Policies/SharedPolicyTypes.h"
#include "Points/BoxPoint.h"
#include "Test/Steppers/TestStepper.h"
#include "UObject/UnrealType.h"

// Named, not anonymous: in unity builds an anonymous MockPoint would collide with NPCInferencePolicyTest.cpp.
namespace SharedPolicyBindingTest
{
struct FFixture
{
    const FName Combat = TEXT("Combat");
    const FName Support = TEXT("Support");
    const TSet<FName> RegisteredHeads{Combat, Support};
    UTestAgent* Guard1 = NewObject<UTestAgent>();
    UTestAgent* Medic1 = NewObject<UTestAgent>();
    UTestAgent* Guard2 = NewObject<UTestAgent>();

    TArray<FSharedNPCBinding> MixedBindings() const
    {
        return {FSharedNPCBinding(Guard1, Combat), FSharedNPCBinding(Medic1, Support), FSharedNPCBinding(Guard2, Combat)};
    }
};

static bool SameBindings(const TArray<FSharedNPCBinding>& Actual, const TArray<FSharedNPCBinding>& Expected)
{
    if (Actual.Num() != Expected.Num()) { return false; }
    for (int32 Index = 0; Index < Actual.Num(); ++Index)
    {
        if (Actual[Index].Agent.GetObject() != Expected[Index].Agent.GetObject() || Actual[Index].Head != Expected[Index].Head)
        {
            return false;
        }
    }
    return true;
}

static FString Quoted(const UObject* Object)
{
    return FString::Printf(TEXT("'%s'"), *GetNameSafe(Object));
}

static TInstancedStruct<FPoint> MockPoint(float Value)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>(TArray<float>{Value});
    return Point;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingSingleTest, "Schola.Inference.SharedPolicy.Bindings.SingleBinding",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingSingleTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error = TEXT("Unchanged");
    TestTrue(TEXT("Guard1 -> Combat is valid"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat)}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Success leaves the error untouched"), Error == TEXT("Unchanged"));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingMultipleTest, "Schola.Inference.SharedPolicy.Bindings.MultipleBindings",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingMultipleTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    const TArray<FSharedNPCBinding> Bindings = Fixture.MixedBindings();
    TestTrue(TEXT("Different heads in one collection are valid"),
        SharedPolicyUtils::ValidateNPCBindings(Bindings, Fixture.RegisteredHeads));
    TestEqual(TEXT("All bindings kept"), Bindings.Num(), 3);
    TestTrue(TEXT("Each binding keeps its agent and head"), SharedPolicyBindingTest::SameBindings(Bindings,
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Medic1, Fixture.Support),
            FSharedNPCBinding(Fixture.Guard2, Fixture.Combat)}));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingSharedHeadTest, "Schola.Inference.SharedPolicy.Bindings.SharedHead",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingSharedHeadTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    TestTrue(TEXT("Two NPCs may use the same head"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Guard2, Fixture.Combat)},
        Fixture.RegisteredHeads));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingEmptyHeadTest, "Schola.Inference.SharedPolicy.Bindings.EmptyHead",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingEmptyHeadTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error;
    TestFalse(TEXT("NAME_None head rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, NAME_None)}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names the agent without a head"),
        Error.Contains(TEXT("has no head")) && Error.Contains(SharedPolicyBindingTest::Quoted(Fixture.Guard1)));
    TestFalse(TEXT("Empty-string head rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, FName(TEXT("")))}, Fixture.RegisteredHeads));
    const TSet<FName> HeadsWithNone{NAME_None, Fixture.Combat};
    TestFalse(TEXT("NAME_None rejected even when the caller lists it"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, NAME_None)}, HeadsWithNone));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingUnknownHeadTest, "Schola.Inference.SharedPolicy.Bindings.UnknownHead",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingUnknownHeadTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error;
    TestFalse(TEXT("Unregistered head rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, TEXT("Unknown"))}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names the unknown head"), Error.Contains(TEXT("unregistered head 'Unknown'")));
    TestFalse(TEXT("One unknown head rejects the collection"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Medic1, TEXT("Unknown"))},
        Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names the failing binding"),
        Error.Contains(TEXT("[1]")) && Error.Contains(SharedPolicyBindingTest::Quoted(Fixture.Medic1)));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingNullAgentTest, "Schola.Inference.SharedPolicy.Bindings.NullAgent",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingNullAgentTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error;
    TestFalse(TEXT("Null agent rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(TScriptInterface<IAgent>(), Fixture.Combat)}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names the failing binding"), Error.Contains(TEXT("[0] has no valid agent")));
    UTestAgent* Destroyed = NewObject<UTestAgent>();
    Destroyed->MarkAsGarbage();
    TestFalse(TEXT("Garbage agent rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Destroyed, Fixture.Combat)}, Fixture.RegisteredHeads));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingDuplicateAgentTest, "Schola.Inference.SharedPolicy.Bindings.DuplicateAgent",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingDuplicateAgentTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error;
    TestFalse(TEXT("Agent bound to two heads rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Guard1, Fixture.Support)},
        Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names both bindings"),
        Error.Contains(TEXT("NPC binding [1]")) && Error.Contains(TEXT("already bound at [0]")));
    TestFalse(TEXT("Repeated identical binding rejected"), SharedPolicyUtils::ValidateNPCBindings(
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Guard1, Fixture.Combat)},
        Fixture.RegisteredHeads));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingEmptyCollectionTest, "Schola.Inference.SharedPolicy.Bindings.EmptyCollection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingEmptyCollectionTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FString Error;
    TestFalse(TEXT("Empty collection rejected"),
        SharedPolicyUtils::ValidateNPCBindings({}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error explains the empty collection"), Error.Contains(TEXT("bindings are empty")));
    TestFalse(TEXT("Rejected without an error string"), SharedPolicyUtils::ValidateNPCBindings({}, Fixture.RegisteredHeads));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingLookupTest, "Schola.Inference.SharedPolicy.Bindings.HeadLookup",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingLookupTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    const TArray<FSharedNPCBinding> Bindings = Fixture.MixedBindings();
    FName Head;
    TestTrue(TEXT("Guard1 found"), SharedPolicyUtils::FindHeadForAgent(Bindings, Fixture.Guard1, Head));
    TestTrue(TEXT("Guard1 uses Combat"), Head == Fixture.Combat);
    TestTrue(TEXT("Medic1 found"), SharedPolicyUtils::FindHeadForAgent(Bindings, Fixture.Medic1, Head));
    TestTrue(TEXT("Medic1 uses Support"), Head == Fixture.Support);
    TestTrue(TEXT("Guard2 found"), SharedPolicyUtils::FindHeadForAgent(Bindings, Fixture.Guard2, Head));
    TestTrue(TEXT("Guard2 uses Combat"), Head == Fixture.Combat);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingUnknownLookupTest, "Schola.Inference.SharedPolicy.Bindings.UnknownAgentLookup",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingUnknownLookupTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    const TArray<FSharedNPCBinding> Bindings = Fixture.MixedBindings();
    const FName Unchanged = TEXT("Unchanged");
    FName Head = Unchanged;
    TestFalse(TEXT("Unbound agent not found"),
        SharedPolicyUtils::FindHeadForAgent(Bindings, NewObject<UTestAgent>(), Head));
    TestFalse(TEXT("Null agent not found"), SharedPolicyUtils::FindHeadForAgent(Bindings, TScriptInterface<IAgent>(), Head));
    TestFalse(TEXT("Null agent does not match an unassigned binding"), SharedPolicyUtils::FindHeadForAgent(
        {FSharedNPCBinding(TScriptInterface<IAgent>(), Fixture.Combat)}, TScriptInterface<IAgent>(), Head));
    TestFalse(TEXT("Nothing found in empty bindings"), SharedPolicyUtils::FindHeadForAgent({}, Fixture.Guard1, Head));
    TestTrue(TEXT("Failed lookups leave the head unchanged"), Head == Unchanged);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingOrderTest, "Schola.Inference.SharedPolicy.Bindings.PreservesOrder",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingOrderTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    const TArray<FSharedNPCBinding> Bindings = Fixture.MixedBindings();
    FName Head;
    TestTrue(TEXT("Valid against all heads"), SharedPolicyUtils::ValidateNPCBindings(Bindings, Fixture.RegisteredHeads));
    TestFalse(TEXT("Invalid without Support"), SharedPolicyUtils::ValidateNPCBindings(Bindings, {Fixture.Combat}));
    TestTrue(TEXT("Successful lookup"), SharedPolicyUtils::FindHeadForAgent(Bindings, Fixture.Medic1, Head));
    TestFalse(TEXT("Failed lookup"), SharedPolicyUtils::FindHeadForAgent(Bindings, NewObject<UTestAgent>(), Head));
    TestTrue(TEXT("Bindings keep their order"), SharedPolicyBindingTest::SameBindings(Bindings,
        {FSharedNPCBinding(Fixture.Guard1, Fixture.Combat), FSharedNPCBinding(Fixture.Medic1, Fixture.Support),
            FSharedNPCBinding(Fixture.Guard2, Fixture.Combat)}));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingLegacyHeadTest, "Schola.Inference.SharedPolicy.Bindings.LegacyDefaultHead",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingLegacyHeadTest::RunTest(const FString& Parameters)
{
    // The legacy path ignores per-NPC bindings: every row of a batch uses the configured default head.
    SharedPolicyBindingTest::FFixture Fixture;
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    Config.Paradigm = ENPCInferenceParadigm::SharedPolicy;
    Config.Head = Fixture.Combat;
    TestTrue(TEXT("Configure"), Policy->Configure(Config));
    Policy->Init(FInteractionDefinition());
    const TInstancedStruct<FPoint> CombatAction = SharedPolicyBindingTest::MockPoint(2);
    Policy->RegisterHead(Fixture.Combat, CombatAction);
    Policy->RegisterHead(Fixture.Support, SharedPolicyBindingTest::MockPoint(3));
    const TInstancedStruct<FPoint> Observation = SharedPolicyBindingTest::MockPoint(1);
    TArray<TInstancedStruct<FPoint>> Actions;
    TestTrue(TEXT("Legacy batch"), Policy->BatchedThink({Observation, Observation, Observation}, Actions));
    TestTrue(TEXT("Every row uses the default head"),
        Actions == TArray<TInstancedStruct<FPoint>>{CombatAction, CombatAction, CombatAction});
    TestTrue(TEXT("Default head unchanged"), Policy->GetConfig().Head == Fixture.Combat);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingAgentInterfaceTest, "Schola.Inference.SharedPolicy.Bindings.AgentInterface",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingAgentInterfaceTest::RunTest(const FString& Parameters)
{
    SharedPolicyBindingTest::FFixture Fixture;
    FSharedNPCBinding NonAgent(TScriptInterface<IAgent>(), Fixture.Combat);
    NonAgent.Agent.SetObject(NewObject<UTestPolicy>());
    FString Error;
    TestFalse(TEXT("Object without IAgent rejected"),
        SharedPolicyUtils::ValidateNPCBindings({NonAgent}, Fixture.RegisteredHeads, &Error));
    TestTrue(TEXT("Error names the missing interface"), Error.Contains(TEXT("does not implement IAgent")));

    // Blueprint-implemented agents have an object but no native interface pointer.
    FSharedNPCBinding BlueprintStyle(TScriptInterface<IAgent>(), Fixture.Combat);
    BlueprintStyle.Agent.SetObject(Fixture.Guard1);
    BlueprintStyle.Agent.SetInterface(nullptr);
    TestTrue(TEXT("Agent without a native interface pointer accepted"),
        SharedPolicyUtils::ValidateNPCBindings({BlueprintStyle}, Fixture.RegisteredHeads));
    FName Head;
    TestTrue(TEXT("Lookup matches the object, not the interface pointer"),
        SharedPolicyUtils::FindHeadForAgent({BlueprintStyle}, Fixture.Guard1, Head) && Head == Fixture.Combat);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyBindingReflectionTest, "Schola.Inference.SharedPolicy.Bindings.Reflection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyBindingReflectionTest::RunTest(const FString& Parameters)
{
    const UScriptStruct* BindingStruct = FSharedNPCBinding::StaticStruct();
    const auto* AgentProperty = CastField<FInterfaceProperty>(
        BindingStruct->FindPropertyByName(GET_MEMBER_NAME_CHECKED(FSharedNPCBinding, Agent)));
    const auto* HeadProperty = CastField<FNameProperty>(
        BindingStruct->FindPropertyByName(GET_MEMBER_NAME_CHECKED(FSharedNPCBinding, Head)));
    TestTrue(TEXT("Agent is an IAgent interface property"),
        AgentProperty && AgentProperty->InterfaceClass == UAgent::StaticClass());
    TestNotNull(TEXT("Head is a name property"), HeadProperty);
    const auto IsEditableReadWrite = [](const FProperty* Property)
    {
        return Property && Property->HasAllPropertyFlags(CPF_Edit | CPF_BlueprintVisible)
            && !Property->HasAnyPropertyFlags(CPF_BlueprintReadOnly);
    };
    TestTrue(TEXT("Agent is EditAnywhere and BlueprintReadWrite"), IsEditableReadWrite(AgentProperty));
    TestTrue(TEXT("Head is EditAnywhere and BlueprintReadWrite"), IsEditableReadWrite(HeadProperty));
#if WITH_EDITOR
    TestTrue(TEXT("Struct is a Blueprint type"), BindingStruct->GetBoolMetaData(TEXT("BlueprintType")));
#endif
    TestTrue(TEXT("Default head is NAME_None"), FSharedNPCBinding().Head.IsNone());
    return true;
}
