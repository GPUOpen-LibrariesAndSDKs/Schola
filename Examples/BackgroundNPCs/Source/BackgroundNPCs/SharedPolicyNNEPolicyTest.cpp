// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "NNEModelData.h"
#include "Points/BoxPoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Policies/NNESharedPolicy.h"
#include "Policies/SharedPolicyTypes.h"
#include "SharedPolicyNNETestAgent.h"
#include "Steppers/SharedPolicyStepper.h"

#include <limits>

// Named, not anonymous: unity builds would collide on an anonymous helper.
namespace SharedPolicyNNEPolicyTest
{
	constexpr int32 ObservationWidth = 64;
	constexpr int32 ActionWidth = 2;
	constexpr double AbsoluteTolerance = 1e-5;
	constexpr double RelativeTolerance = 1e-4;
	const TCHAR* ModelObjectPath = TEXT("/Game/A_all_heads.A_all_heads");

	// Python reference outputs for A_all_heads (small profile, seed 301), indexed [head][input - 1][column].
	// Input 1 and 2 are all-zero observations whose first value is 1 or 2.
	const double Reference[2][2][ActionWidth] = {
		{
			{-0.1623554528, -0.1630928367},
			{-0.1648618877, -0.1687339097},
		},
		{
			{0.0694475845, -0.0019764062},
			{0.0638729781, -0.0013816468},
		},
	};

	FName HeadName(int32 Head)
	{
		return FName(*FString::Printf(TEXT("expert_%d"), Head));
	}

	TInstancedStruct<FPoint> MakeObservation(float FirstValue, int32 Width = ObservationWidth)
	{
		TArray<float> Values;
		Values.SetNumZeroed(Width);
		if (Width > 0)
		{
			Values[0] = FirstValue;
		}
		TInstancedStruct<FPoint> Point;
		Point.InitializeAs<FBoxPoint>(Values);
		return Point;
	}

	/** One row of a request: which head serves it and which reference input (1 or 2) it observes. */
	struct FRow
	{
		int32 Head;
		int32 Input;
	};

	void BuildRequest(TConstArrayView<FRow> Rows, TArray<TInstancedStruct<FPoint>>& OutObservations,
		TArray<FName>& OutHeads)
	{
		OutObservations.Reset();
		OutHeads.Reset();
		for (const FRow& Row : Rows)
		{
			OutObservations.Add(MakeObservation(float(Row.Input)));
			OutHeads.Add(HeadName(Row.Head));
		}
	}

	bool MatchesReference(FAutomationTestBase& Test, const FString& Label, TConstArrayView<float> Values,
		const FRow& Row)
	{
		bool bMatches = Values.Num() == ActionWidth;
		if (!bMatches)
		{
			Test.AddError(FString::Printf(TEXT("%s: expected %d values, got %d."), *Label, ActionWidth, Values.Num()));
			return false;
		}
		for (int32 Column = 0; Column < ActionWidth; ++Column)
		{
			const double Expected = Reference[Row.Head][Row.Input - 1][Column];
			const double Tolerance = AbsoluteTolerance + RelativeTolerance * FMath::Abs(Expected);
			if (FMath::Abs(Values[Column] - Expected) > Tolerance)
			{
				Test.AddError(FString::Printf(TEXT("%s column %d = %.10f, Python reference %.10f, tolerance %.10f."),
					*Label, Column, Values[Column], Expected, Tolerance));
				bMatches = false;
			}
		}
		return bMatches;
	}

	bool ActionsMatchReference(FAutomationTestBase& Test, const FString& Label,
		const TArray<TInstancedStruct<FPoint>>& Actions, TConstArrayView<FRow> Rows)
	{
		if (!Test.TestEqual(*FString::Printf(TEXT("%s: action count"), *Label), Actions.Num(), Rows.Num()))
		{
			return false;
		}
		bool bAllMatch = true;
		for (int32 Index = 0; Index < Rows.Num(); ++Index)
		{
			const FBoxPoint* Box = Actions[Index].GetPtr<FBoxPoint>();
			if (!Box)
			{
				Test.AddError(FString::Printf(TEXT("%s: row %d is not a FBoxPoint."), *Label, Index));
				bAllMatch = false;
				continue;
			}
			bAllMatch &= MatchesReference(
				Test, FString::Printf(TEXT("%s row %d (head %d, input %d)"), *Label, Index, Rows[Index].Head,
						Rows[Index].Input),
				Box->Values, Rows[Index]);
		}
		return bAllMatch;
	}

	/** Loads /Game/A_all_heads and initializes a policy around it. */
	UNNESharedPolicy* MakePolicy(FAutomationTestBase& Test, int32 MaxBatchSize = 0)
	{
		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, ModelObjectPath);
		if (!Test.TestNotNull(TEXT("Imported NNE Model Data /Game/A_all_heads"), ModelData))
		{
			Test.AddError(TEXT("The NNE policy tests require the imported asset /Game/A_all_heads."));
			return nullptr;
		}
		UNNESharedPolicy* Policy = NewObject<UNNESharedPolicy>();
		Policy->ModelData = ModelData;
		Policy->MaxBatchSize = MaxBatchSize;
		if (!Test.TestTrue(TEXT("UNNESharedPolicy::Init on Architecture A"), Policy->Init()))
		{
			return nullptr;
		}
		return Policy;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyNNEInitTest, "Schola.Inference.SharedPolicy.NNEPolicy.Init",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyNNEInitTest::RunTest(const FString&)
{
	using namespace SharedPolicyNNEPolicyTest;

	UNNESharedPolicy* Policy = MakePolicy(*this);
	if (!Policy)
	{
		return false;
	}
	TestTrue(TEXT("Policy is ready"), Policy->IsReady());
	TestEqual(TEXT("Observation width from the model"), Policy->GetObservationWidth(), ObservationWidth);

	TSet<FName> Heads;
	Policy->GetRegisteredHeads(Heads);
	TestEqual(TEXT("Two heads registered"), Heads.Num(), 2);
	TestTrue(TEXT("expert_0 registered"), Heads.Contains(HeadName(0)));
	TestTrue(TEXT("expert_1 registered"), Heads.Contains(HeadName(1)));
	TestEqual(TEXT("Init makes no model call"), int32(Policy->GetModelCallCount()), 0);

	// A failed Init must leave the policy unusable, even after an earlier success.
	UNNEModelData* ModelData = Policy->ModelData;
	Policy->HeadOutputSuffix = TEXT("_not_a_suffix");
	AddExpectedError(TEXT("no heads"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init fails when no output names a head"), Policy->Init());
	TestFalse(TEXT("Failed Init leaves the policy not ready"), Policy->IsReady());
	Policy->GetRegisteredHeads(Heads);
	TestTrue(TEXT("Failed Init leaves no heads"), Heads.IsEmpty());

	Policy->HeadOutputSuffix = TEXT("");
	AddExpectedError(TEXT("HeadOutputSuffix must not be empty"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init fails with an empty suffix"), Policy->Init());

	Policy->HeadOutputSuffix = TEXT("_actions");
	Policy->ModelData = nullptr;
	AddExpectedError(TEXT("No model data"), EAutomationExpectedErrorFlags::Contains, 1);
	TestFalse(TEXT("Init fails without model data"), Policy->Init());

	Policy->ModelData = ModelData;
	TestTrue(TEXT("Init recovers once the configuration is valid"), Policy->Init());
	Policy->GetRegisteredHeads(Heads);
	TestEqual(TEXT("Heads are rediscovered"), Heads.Num(), 2);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyNNEMixedHeadTest, "Schola.Inference.SharedPolicy.NNEPolicy.MixedHeadRouting",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyNNEMixedHeadTest::RunTest(const FString&)
{
	using namespace SharedPolicyNNEPolicyTest;

	UNNESharedPolicy* Policy = MakePolicy(*this);
	if (!Policy)
	{
		return false;
	}

	struct FCase
	{
		const TCHAR* Name;
		TArray<FRow> Rows;
	};
	const FCase Cases[] = {
		{TEXT("mixed shuffled"), {{1, 1}, {0, 2}, {1, 2}, {0, 1}}},
		{TEXT("alternating"), {{0, 1}, {1, 2}, {0, 2}, {1, 1}}},
		{TEXT("heads in order"), {{0, 1}, {1, 2}}},
		{TEXT("heads swapped, same observations"), {{1, 1}, {0, 2}}},
		{TEXT("homogeneous expert_0"), {{0, 1}, {0, 2}}},
		{TEXT("homogeneous expert_1"), {{1, 1}, {1, 2}}},
		{TEXT("singleton expert_1"), {{1, 2}}},
	};

	bool bAllMatch = true;
	for (const FCase& Case : Cases)
	{
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		BuildRequest(Case.Rows, Observations, Heads);

		TArray<TInstancedStruct<FPoint>> Actions;
		const int32 CallsBefore = int32(Policy->GetModelCallCount());
		if (!TestTrue(FString::Printf(TEXT("%s: BatchedThinkWithHeads succeeds"), Case.Name),
				Policy->BatchedThinkWithHeads(Observations, Heads, Actions)))
		{
			return false;
		}
		TestEqual(FString::Printf(TEXT("%s: exactly one NNE model call"), Case.Name),
			int32(Policy->GetModelCallCount()) - CallsBefore, 1);
		bAllMatch &= ActionsMatchReference(*this, Case.Name, Actions, Case.Rows);
	}

	// The same observations under swapped heads must swap the actions, not merely differ.
	{
		const TArray<FRow> Forward = {{0, 1}, {1, 2}};
		const TArray<FRow> Swapped = {{1, 1}, {0, 2}};
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		TArray<TInstancedStruct<FPoint>> ForwardActions;
		TArray<TInstancedStruct<FPoint>> SwappedActions;
		BuildRequest(Forward, Observations, Heads);
		TestTrue(TEXT("Forward heads succeed"), Policy->BatchedThinkWithHeads(Observations, Heads, ForwardActions));
		BuildRequest(Swapped, Observations, Heads);
		TestTrue(TEXT("Swapped heads succeed"), Policy->BatchedThinkWithHeads(Observations, Heads, SwappedActions));
		if (ForwardActions.Num() == 2 && SwappedActions.Num() == 2)
		{
			const TArray<float>& Expert0Input1 = ForwardActions[0].Get<FBoxPoint>().Values;
			const TArray<float>& Expert1Input1 = SwappedActions[0].Get<FBoxPoint>().Values;
			TestTrue(TEXT("Different heads give different actions for one observation"),
				FMath::Abs(Expert0Input1[0] - Expert1Input1[0]) > 0.01f);
		}
	}
	return bAllMatch;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyNNEReferenceTest, "Schola.Inference.SharedPolicy.NNEPolicy.ReferenceOutputs",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyNNEReferenceTest::RunTest(const FString&)
{
	using namespace SharedPolicyNNEPolicyTest;

	UNNESharedPolicy* Policy = MakePolicy(*this);
	if (!Policy)
	{
		return false;
	}

	// Every (head, input) pair in one batch.
	TArray<FRow> AllPairs = {{0, 1}, {0, 2}, {1, 1}, {1, 2}};

	// An odd, larger batch exercises reshaping the model to new batch sizes, with each row
	// in a different position than in the small batches.
	TArray<FRow> Large;
	for (int32 Index = 0; Index < 37; ++Index)
	{
		Large.Add({(Index % 3 == 0) ? 1 : 0, 1 + (Index % 2)});
	}

	struct FRequest
	{
		const TCHAR* Name;
		const TArray<FRow>* Rows;
	};
	const FRequest Requests[] = {
		{TEXT("all head and input pairs"), &AllPairs},
		{TEXT("37 rows"), &Large},
		{TEXT("all pairs again after resizing"), &AllPairs},
	};

	bool bAllMatch = true;
	for (const FRequest& Request : Requests)
	{
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		BuildRequest(*Request.Rows, Observations, Heads);
		TArray<TInstancedStruct<FPoint>> Actions;
		const int32 CallsBefore = int32(Policy->GetModelCallCount());
		if (!TestTrue(FString::Printf(TEXT("%s: succeeds"), Request.Name),
				Policy->BatchedThinkWithHeads(Observations, Heads, Actions)))
		{
			return false;
		}
		TestEqual(FString::Printf(TEXT("%s: exactly one NNE model call"), Request.Name),
			int32(Policy->GetModelCallCount()) - CallsBefore, 1);
		for (const TInstancedStruct<FPoint>& Action : Actions)
		{
			const FBoxPoint* Box = Action.GetPtr<FBoxPoint>();
			TestTrue(TEXT("Action is a FBoxPoint of ActionWidth values"), Box && Box->Values.Num() == ActionWidth);
		}
		bAllMatch &= ActionsMatchReference(*this, Request.Name, Actions, *Request.Rows);
	}
	TestTrue(TEXT("NNE outputs match the Python ONNX Runtime reference in every batch"), bAllMatch);
	return bAllMatch;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyNNEInvalidRequestTest,
	"Schola.Inference.SharedPolicy.NNEPolicy.RejectsInvalidRequests",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyNNEInvalidRequestTest::RunTest(const FString&)
{
	using namespace SharedPolicyNNEPolicyTest;

	UNNESharedPolicy* Policy = MakePolicy(*this, /*MaxBatchSize=*/3);
	if (!Policy)
	{
		return false;
	}

	const auto Valid = [](std::initializer_list<FRow> Rows)
	{
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		BuildRequest(MakeArrayView(Rows.begin(), int32(Rows.size())), Observations, Heads);
		return TPair<TArray<TInstancedStruct<FPoint>>, TArray<FName>>(MoveTemp(Observations), MoveTemp(Heads));
	};

	// Each case is rejected before the model runs, so the call counter must not move and the
	// output array must keep its previous contents rather than receive partial actions.
	const auto ExpectRejected = [this, Policy](const TCHAR* Name, const FString& ExpectedLog,
									const TArray<TInstancedStruct<FPoint>>& Observations, const TArray<FName>& Heads)
	{
		TArray<TInstancedStruct<FPoint>> Actions;
		Actions.Add(SharedPolicyNNEPolicyTest::MakeObservation(7.f));
		const int32 CallsBefore = int32(Policy->GetModelCallCount());
		AddExpectedError(ExpectedLog, EAutomationExpectedErrorFlags::Contains, 1);
		TestFalse(FString::Printf(TEXT("%s: rejected"), Name), Policy->BatchedThinkWithHeads(Observations, Heads, Actions));
		TestEqual(FString::Printf(TEXT("%s: output keeps its previous size"), Name), Actions.Num(), 1);
		const FBoxPoint* Kept = Actions.Num() == 1 ? Actions[0].GetPtr<FBoxPoint>() : nullptr;
		TestTrue(FString::Printf(TEXT("%s: output keeps its previous value"), Name), Kept && Kept->Values[0] == 7.f);
		TestEqual(FString::Printf(TEXT("%s: no model call was made"), Name),
			int32(Policy->GetModelCallCount()) - CallsBefore, 0);
	};

	{
		// Valid first rows must not leak out when a later row is invalid.
		auto Request = Valid({{0, 1}, {1, 2}, {0, 1}});
		Request.Value[2] = TEXT("Combat");
		ExpectRejected(TEXT("unknown head on the last row"), TEXT("unknown head 'Combat'"), Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}});
		Request.Value[0] = NAME_None;
		ExpectRejected(TEXT("unset head"), TEXT("unknown head 'None'"), Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Value.RemoveAt(1);
		ExpectRejected(TEXT("fewer heads than observations"), TEXT("2 observations but 1 heads"), Request.Key,
			Request.Value);
	}
	{
		auto Request = Valid({{0, 1}});
		Request.Value.Add(HeadName(1));
		ExpectRejected(TEXT("more heads than observations"), TEXT("1 observations but 2 heads"), Request.Key,
			Request.Value);
	}
	ExpectRejected(TEXT("empty batch"), TEXT("Empty batch"), {}, {});
	{
		auto Request = Valid({{0, 1}, {1, 2}, {0, 1}, {1, 2}});
		ExpectRejected(TEXT("batch over MaxBatchSize"), TEXT("exceeds the limit of 3"), Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Key[1] = MakeObservation(1.f, ObservationWidth - 1);
		ExpectRejected(TEXT("observation one value short"), TEXT("Row 1 has 63 values"),
			Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Key[0] = MakeObservation(1.f, ObservationWidth + 1);
		ExpectRejected(TEXT("observation one value long"), TEXT("Row 0 has 65 values"),
			Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Key[1].InitializeAs<FMultiDiscretePoint>(TArray<int>{1});
		ExpectRejected(TEXT("observation of the wrong point type"), TEXT("Row 1 is not a FBoxPoint"),
			Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Key[1].GetMutable<FBoxPoint>().Values[5] = std::numeric_limits<float>::quiet_NaN();
		ExpectRejected(TEXT("NaN observation"), TEXT("Row 1 has a non-finite value"), Request.Key, Request.Value);
	}
	{
		auto Request = Valid({{0, 1}, {1, 2}});
		Request.Key[0].GetMutable<FBoxPoint>().Values[0] = std::numeric_limits<float>::infinity();
		ExpectRejected(TEXT("infinite observation"), TEXT("Row 0 has a non-finite value"), Request.Key,
			Request.Value);
	}
	{
		UNNESharedPolicy* Uninitialized = NewObject<UNNESharedPolicy>();
		auto Request = Valid({{0, 1}});
		TArray<TInstancedStruct<FPoint>> Actions;
		Actions.Add(MakeObservation(7.f));
		AddExpectedError(TEXT("Policy is not initialized"), EAutomationExpectedErrorFlags::Contains, 1);
		TestFalse(TEXT("Uninitialized policy rejects"), Uninitialized->BatchedThinkWithHeads(Request.Key, Request.Value, Actions));
		TestEqual(TEXT("Uninitialized policy leaves the output untouched"), Actions.Num(), 1);
		TestEqual(TEXT("Uninitialized policy makes no model call"), int32(Uninitialized->GetModelCallCount()), 0);
	}

	// None of the rejections may have damaged the policy.
	{
		const TArray<FRow> Rows = {{1, 2}, {0, 1}};
		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<FName> Heads;
		BuildRequest(Rows, Observations, Heads);
		TArray<TInstancedStruct<FPoint>> Actions;
		TestTrue(TEXT("A valid request still succeeds afterwards"),
			Policy->BatchedThinkWithHeads(Observations, Heads, Actions));
		ActionsMatchReference(*this, TEXT("after rejections"), Actions, Rows);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyNNEStepperTest, "Schola.Inference.SharedPolicy.NNEPolicy.StepperIntegration",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyNNEStepperTest::RunTest(const FString&)
{
	using namespace SharedPolicyNNEPolicyTest;

	UNNESharedPolicy* Policy = MakePolicy(*this);
	if (!Policy)
	{
		return false;
	}
	USharedPolicyNNETestAgent* First = NewObject<USharedPolicyNNETestAgent>();
	USharedPolicyNNETestAgent* Second = NewObject<USharedPolicyNNETestAgent>();
	First->ObservationValue = 1.f;
	Second->ObservationValue = 2.f;
	const FName Expert0 = HeadName(0);
	const FName Expert1 = HeadName(1);

	// Heads the real model does not provide are rejected when the stepper is initialised.
	{
		USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();
		AddExpectedError(TEXT("uses unregistered head"), EAutomationExpectedErrorFlags::Contains, 1);
		TestFalse(TEXT("Stepper rejects a head the NNE policy does not register"),
			Stepper->Init({FSharedNPCBinding(First, TEXT("Combat"))}, Policy));
	}

	// First agent on expert_0, second on expert_1.
	{
		USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();
		if (!TestTrue(TEXT("Stepper accepts the NNE policy"),
				Stepper->Init({FSharedNPCBinding(First, Expert0), FSharedNPCBinding(Second, Expert1)}, Policy)))
		{
			return false;
		}
		const int32 CallsBefore = int32(Policy->GetModelCallCount());
		Stepper->Step();
		TestEqual(TEXT("One Step makes one NNE model call"), int32(Policy->GetModelCallCount()) - CallsBefore, 1);
		TestEqual(TEXT("First agent acted once"), First->ActCount, 1);
		TestEqual(TEXT("Second agent acted once"), Second->ActCount, 1);
		MatchesReference(*this, TEXT("first agent on expert_0"), First->LastAction, {0, 1});
		MatchesReference(*this, TEXT("second agent on expert_1"), Second->LastAction, {1, 2});
	}

	// Swapping the bindings swaps which head each agent hears from, with no other change.
	{
		USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();
		if (!TestTrue(TEXT("Stepper accepts swapped bindings"),
				Stepper->Init({FSharedNPCBinding(First, Expert1), FSharedNPCBinding(Second, Expert0)}, Policy)))
		{
			return false;
		}
		Stepper->Step();
		TestEqual(TEXT("First agent acted again"), First->ActCount, 2);
		MatchesReference(*this, TEXT("first agent on expert_1"), First->LastAction, {1, 1});
		MatchesReference(*this, TEXT("second agent on expert_0"), Second->LastAction, {0, 2});
	}
	return true;
}

#endif
