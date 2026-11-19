// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "NNEModelData.h"
#include "Policies/NNEPerHeadSharedPolicy.h"

// Named, not anonymous: unity builds would collide on an anonymous helper.
namespace SharedPolicyPerHeadPolicyTest
{
	constexpr int32 ObservationWidth = 64;

	UNNEModelData* LoadModel(FAutomationTestBase& Test, const TCHAR* ObjectPath)
	{
		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, ObjectPath);
		Test.TestNotNull(*FString::Printf(TEXT("Imported NNE Model Data %s"), ObjectPath), ModelData);
		return ModelData;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyPerHeadInitTest, "Schola.Inference.SharedPolicy.PerHeadPolicy.Init",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyPerHeadInitTest::RunTest(const FString&)
{
	using namespace SharedPolicyPerHeadPolicyTest;

	UNNEModelData* Expert0 = LoadModel(*this, TEXT("/Game/C_expert_0.C_expert_0"));
	UNNEModelData* Expert1 = LoadModel(*this, TEXT("/Game/C_expert_1.C_expert_1"));
	UNNEModelData* AllHeads = LoadModel(*this, TEXT("/Game/A_all_heads.A_all_heads"));
	if (!Expert0 || !Expert1 || !AllHeads)
	{
		return false;
	}
	const FName Expert0Head(TEXT("expert_0"));
	const FName Expert1Head(TEXT("expert_1"));

	UNNEPerHeadSharedPolicy* Policy = NewObject<UNNEPerHeadSharedPolicy>();
	Policy->HeadModels = {{Expert0Head, Expert0}, {Expert1Head, Expert1}};
	if (!TestTrue(TEXT("Init loads the Architecture C models"), Policy->Init()))
	{
		return false;
	}
	TestTrue(TEXT("Policy is ready"), Policy->IsReady());
	TestEqual(TEXT("Observation width from the models"), Policy->GetObservationWidth(), ObservationWidth);
	TSet<FName> Heads;
	Policy->GetRegisteredHeads(Heads);
	TestEqual(TEXT("Two heads registered"), Heads.Num(), 2);
	TestTrue(TEXT("expert_0 registered"), Heads.Contains(Expert0Head));
	TestTrue(TEXT("expert_1 registered"), Heads.Contains(Expert1Head));
	TestEqual(TEXT("Init makes no model call"), int32(Policy->GetModelCallCount()), 0);

	// The all-heads model reads the same observation but has one output per head, so it cannot serve
	// as a single head's model.
	Policy->HeadModels = {{Expert0Head, Expert0}, {Expert1Head, AllHeads}};
	AddExpectedError(TEXT("a per-head model must have exactly one"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init rejects a model with several outputs"), Policy->Init());
	TestFalse(TEXT("Rejected Init leaves the policy not ready"), Policy->IsReady());
	return true;
}

#endif
