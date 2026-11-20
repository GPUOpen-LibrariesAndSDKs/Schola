// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "NNEModelData.h"
#include "Points/BoxPoint.h"
#include "Policies/NNEPerHeadSharedPolicy.h"
#include "Policies/NNESharedPolicy.h"
#include "Policies/SharedPolicyTypes.h"
#include "SharedPolicyNNETestAgent.h"
#include "Steppers/SharedPolicyStepper.h"

// Named, not anonymous: unity builds would collide on an anonymous helper.
namespace SharedPolicyPerHeadPolicyTest
{
	constexpr int32 ObservationWidth = 64;
	constexpr int32 ActionWidth = 2;
	constexpr double AbsoluteTolerance = 1e-5;
	constexpr double RelativeTolerance = 1e-4;

	const FName Expert0Head(TEXT("expert_0"));
	const FName Expert1Head(TEXT("expert_1"));

	UNNEModelData* LoadModel(FAutomationTestBase& Test, const TCHAR* ObjectPath)
	{
		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, ObjectPath);
		Test.TestNotNull(*FString::Printf(TEXT("Imported NNE Model Data %s"), ObjectPath), ModelData);
		return ModelData;
	}

	/** Loads the benchmark's Architecture C models and initializes a per-head policy around them. */
	UNNEPerHeadSharedPolicy* MakePerHeadPolicy(FAutomationTestBase& Test)
	{
		UNNEModelData* Expert0 = LoadModel(Test, TEXT("/Game/C_expert_0.C_expert_0"));
		UNNEModelData* Expert1 = LoadModel(Test, TEXT("/Game/C_expert_1.C_expert_1"));
		if (!Expert0 || !Expert1)
		{
			return nullptr;
		}
		UNNEPerHeadSharedPolicy* Policy = NewObject<UNNEPerHeadSharedPolicy>();
		Policy->HeadModels = {{Expert0Head, Expert0}, {Expert1Head, Expert1}};
		if (!Test.TestTrue(TEXT("UNNEPerHeadSharedPolicy::Init on Architecture C"), Policy->Init()))
		{
			return nullptr;
		}
		return Policy;
	}

	/** Loads the benchmark's Architecture A model, built from the same seeded weights, as the reference. */
	UNNESharedPolicy* MakeAllHeadsPolicy(FAutomationTestBase& Test)
	{
		UNNEModelData* AllHeads = LoadModel(Test, TEXT("/Game/A_all_heads.A_all_heads"));
		if (!AllHeads)
		{
			return nullptr;
		}
		UNNESharedPolicy* Policy = NewObject<UNNESharedPolicy>();
		Policy->ModelData = AllHeads;
		if (!Test.TestTrue(TEXT("UNNESharedPolicy::Init on Architecture A"), Policy->Init()))
		{
			return nullptr;
		}
		return Policy;
	}

	TInstancedStruct<FPoint> MakeObservation(float FirstValue)
	{
		TArray<float> Values;
		Values.SetNumZeroed(ObservationWidth);
		Values[0] = FirstValue;
		TInstancedStruct<FPoint> Point;
		Point.InitializeAs<FBoxPoint>(Values);
		return Point;
	}

	bool ValuesMatch(FAutomationTestBase& Test, const FString& Label, TConstArrayView<float> Actual,
		TConstArrayView<float> Expected)
	{
		if (Actual.Num() != Expected.Num())
		{
			Test.AddError(FString::Printf(TEXT("%s: %d values, expected %d."), *Label, Actual.Num(), Expected.Num()));
			return false;
		}
		bool bMatches = true;
		for (int32 Column = 0; Column < Expected.Num(); ++Column)
		{
			const double Tolerance = AbsoluteTolerance + RelativeTolerance * FMath::Abs(Expected[Column]);
			if (FMath::Abs(Actual[Column] - Expected[Column]) > Tolerance)
			{
				Test.AddError(FString::Printf(TEXT("%s column %d = %.10f, Architecture A gives %.10f."), *Label, Column,
					Actual[Column], Expected[Column]));
				bMatches = false;
			}
		}
		return bMatches;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyPerHeadInitTest, "Schola.Inference.SharedPolicy.PerHeadPolicy.Init",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyPerHeadInitTest::RunTest(const FString&)
{
	using namespace SharedPolicyPerHeadPolicyTest;

	UNNEPerHeadSharedPolicy* Policy = MakePerHeadPolicy(*this);
	UNNEModelData* AllHeads = LoadModel(*this, TEXT("/Game/A_all_heads.A_all_heads"));
	if (!Policy || !AllHeads)
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
	Policy->HeadModels.Add(Expert1Head, AllHeads);
	AddExpectedError(TEXT("a per-head model must have exactly one"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init rejects a model with several outputs"), Policy->Init());
	TestFalse(TEXT("Rejected Init leaves the policy not ready"), Policy->IsReady());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyPerHeadMatchesAllHeadsTest,
	"Schola.Inference.SharedPolicy.PerHeadPolicy.MatchesAllHeadsPolicy",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyPerHeadMatchesAllHeadsTest::RunTest(const FString&)
{
	using namespace SharedPolicyPerHeadPolicyTest;

	UNNEPerHeadSharedPolicy* PerHead = MakePerHeadPolicy(*this);
	UNNESharedPolicy* AllHeads = MakeAllHeadsPolicy(*this);
	if (!PerHead || !AllHeads)
	{
		return false;
	}

	/** Each row is (head index, first observation value). */
	struct FCase
	{
		const TCHAR* Name;
		TArray<TPair<int32, float>> Rows;
		int32 ExpectedCalls;
	};
	TArray<TPair<int32, float>> Large;
	for (int32 Index = 0; Index < 37; ++Index)
	{
		Large.Add({(Index % 3 == 0) ? 1 : 0, float(1 + Index % 5)});
	}
	const FCase Cases[] = {
		{TEXT("mixed shuffled"), {{1, 1.f}, {0, 2.f}, {1, 2.f}, {0, 1.f}}, 2},
		{TEXT("heads swapped, same observations"), {{1, 1.f}, {0, 2.f}}, 2},
		{TEXT("homogeneous expert_0"), {{0, 1.f}, {0, 2.f}}, 1},
		{TEXT("singleton expert_1"), {{1, 2.f}}, 1},
		{TEXT("37 interleaved rows"), Large, 2},
	};

	bool bAllMatch = true;
	for (const FCase& Case : Cases)
	{
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		for (const TPair<int32, float>& Row : Case.Rows)
		{
			Observations.Add(MakeObservation(Row.Value));
			Heads.Add(Row.Key == 0 ? Expert0Head : Expert1Head);
		}

		TArray<TInstancedStruct<FPoint>> Expected;
		if (!TestTrue(FString::Printf(TEXT("%s: Architecture A succeeds"), Case.Name),
				AllHeads->BatchedThinkWithHeads(Observations, Heads, Expected)))
		{
			return false;
		}
		TArray<TInstancedStruct<FPoint>> Actions;
		const int32 CallsBefore = int32(PerHead->GetModelCallCount());
		if (!TestTrue(FString::Printf(TEXT("%s: per-head policy succeeds"), Case.Name),
				PerHead->BatchedThinkWithHeads(Observations, Heads, Actions)))
		{
			return false;
		}
		TestEqual(FString::Printf(TEXT("%s: one model call per head present"), Case.Name),
			int32(PerHead->GetModelCallCount()) - CallsBefore, Case.ExpectedCalls);
		if (!TestEqual(FString::Printf(TEXT("%s: one action per row"), Case.Name), Actions.Num(), Expected.Num()))
		{
			bAllMatch = false;
			continue;
		}
		for (int32 Row = 0; Row < Actions.Num(); ++Row)
		{
			const FBoxPoint* Box = Actions[Row].GetPtr<FBoxPoint>();
			if (!TestNotNull(*FString::Printf(TEXT("%s row %d is a FBoxPoint"), Case.Name, Row), Box))
			{
				bAllMatch = false;
				continue;
			}
			bAllMatch &= ValuesMatch(*this, FString::Printf(TEXT("%s row %d"), Case.Name, Row), Box->Values,
				Expected[Row].Get<FBoxPoint>().Values);
		}
	}
	TestTrue(TEXT("Per-head actions match Architecture A in every batch"), bAllMatch);
	return bAllMatch;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyPerHeadStepperTest, "Schola.Inference.SharedPolicy.PerHeadPolicy.StepperIntegration",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyPerHeadStepperTest::RunTest(const FString&)
{
	using namespace SharedPolicyPerHeadPolicyTest;

	UNNEPerHeadSharedPolicy* PerHead = MakePerHeadPolicy(*this);
	UNNESharedPolicy* AllHeads = MakeAllHeadsPolicy(*this);
	if (!PerHead || !AllHeads)
	{
		return false;
	}

	// Reference actions for observation value 1 under each head, from Architecture A.
	TArray<TInstancedStruct<FPoint>> Expected;
	if (!TestTrue(TEXT("Architecture A reference succeeds"),
			AllHeads->BatchedThinkWithHeads({MakeObservation(1.f), MakeObservation(1.f)}, {Expert0Head, Expert1Head}, Expected)))
	{
		return false;
	}
	const TArray<float>& Expert0Action = Expected[0].Get<FBoxPoint>().Values;
	const TArray<float>& Expert1Action = Expected[1].Get<FBoxPoint>().Values;

	USharedPolicyNNETestAgent* Agents[3];
	for (USharedPolicyNNETestAgent*& Agent : Agents)
	{
		Agent = NewObject<USharedPolicyNNETestAgent>();
		Agent->ObservationValue = 1.f;
	}

	{
		USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();
		AddExpectedError(TEXT("uses unregistered head"), EAutomationExpectedErrorFlags::Contains, 1);
		TestFalse(TEXT("Stepper rejects a head the per-head policy does not register"),
			Stepper->Init({FSharedNPCBinding(Agents[0], TEXT("Combat"))}, PerHead));
	}

	// Two NPCs on expert_0 and one on expert_1: one Step makes one call per head.
	USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();
	if (!TestTrue(TEXT("Stepper accepts the per-head policy"),
			Stepper->Init({FSharedNPCBinding(Agents[0], Expert0Head), FSharedNPCBinding(Agents[1], Expert1Head),
							  FSharedNPCBinding(Agents[2], Expert0Head)},
				PerHead)))
	{
		return false;
	}
	const int32 CallsBefore = int32(PerHead->GetModelCallCount());
	Stepper->Step();
	TestEqual(TEXT("One Step makes one model call per head present"), int32(PerHead->GetModelCallCount()) - CallsBefore, 2);
	for (int32 Index = 0; Index < 3; ++Index)
	{
		TestEqual(*FString::Printf(TEXT("Agent %d acted once"), Index), Agents[Index]->ActCount, 1);
	}
	ValuesMatch(*this, TEXT("Agent 0 on expert_0"), Agents[0]->LastAction, Expert0Action);
	ValuesMatch(*this, TEXT("Agent 1 on expert_1"), Agents[1]->LastAction, Expert1Action);
	ValuesMatch(*this, TEXT("Agent 2 on expert_0"), Agents[2]->LastAction, Expert0Action);
	TestEqual(TEXT("Agent actions are ActionWidth values"), Agents[1]->LastAction.Num(), ActionWidth);
	return true;
}

#endif
