// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Policies/PassthroughBoxPolicy.h"

#include "Points/BoxPoint.h"
#include "Policies/PolicyInterface.h"

#if WITH_AUTOMATION_TESTS

namespace ScholaPolicyBatchedThinkTestPrivate
{
	static bool BoxPointsNearlyEqual(const FBoxPoint& A, const FBoxPoint& B)
	{
		if (A.Values.Num() != B.Values.Num() || A.Shape.Num() != B.Shape.Num())
		{
			return false;
		}
		for (int32 i = 0; i < A.Values.Num(); ++i)
		{
			if (!FMath::IsNearlyEqual(A.Values[i], B.Values[i]))
			{
				return false;
			}
		}
		for (int32 i = 0; i < A.Shape.Num(); ++i)
		{
			if (A.Shape[i] != B.Shape[i])
			{
				return false;
			}
		}
		return true;
	}

	static bool ObservationMatchesAction(const TInstancedStruct<FPoint>& Obs, const TInstancedStruct<FPoint>& Act)
	{
		const FBoxPoint* ObsBox = Obs.GetPtr<FBoxPoint>();
		const FBoxPoint* ActBox = Act.GetPtr<FBoxPoint>();
		if (!ObsBox || !ActBox)
		{
			return false;
		}
		return BoxPointsNearlyEqual(*ObsBox, *ActBox);
	}
} // namespace ScholaPolicyBatchedThinkTestPrivate

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyBatchedThinkEmptyBatchTest,
	"Schola.Policies.IPolicy.BatchedThink Empty Batch",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyBatchedThinkEmptyBatchTest::RunTest(const FString& Parameters)
{
	UPassthroughBoxPolicy* PolicyObj = NewObject<UPassthroughBoxPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);
	TestNotNull(TEXT("Policy object"), PolicyObj);
	TestNotNull(TEXT("IPolicy cast"), Policy);

	TArray<TInstancedStruct<FPoint>> Observations;
	TArray<TInstancedStruct<FPoint>> Actions;
	Actions.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 99.0f }));

	const bool bOk = Policy->BatchedThink(Observations, Actions);
	TestTrue(TEXT("BatchedThink succeeds on empty observations"), bOk);
	TestEqual(TEXT("OutActions overwritten with empty actions"), Actions.Num(), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyBatchedThinkSingleObservationTest,
	"Schola.Policies.IPolicy.BatchedThink Single Observation",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyBatchedThinkSingleObservationTest::RunTest(const FString& Parameters)
{
	UPassthroughBoxPolicy* PolicyObj = NewObject<UPassthroughBoxPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TArray<float> Values = { 0.25f, -1.5f, 2.0f };
	TArray<TInstancedStruct<FPoint>> Observations;
	Observations.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(Values));

	TArray<TInstancedStruct<FPoint>> Actions;
	const bool bOk = Policy->BatchedThink(Observations, Actions);
	TestTrue(TEXT("BatchedThink succeeds"), bOk);
	TestEqual(TEXT("One action per observation"), Actions.Num(), 1);
	TestTrue(
		TEXT("Action matches observation"),
		ScholaPolicyBatchedThinkTestPrivate::ObservationMatchesAction(Observations[0], Actions[0]));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyBatchedThinkMultipleObservationsTest,
	"Schola.Policies.IPolicy.BatchedThink Multiple Observations",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyBatchedThinkMultipleObservationsTest::RunTest(const FString& Parameters)
{
	UPassthroughBoxPolicy* PolicyObj = NewObject<UPassthroughBoxPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TArray<TInstancedStruct<FPoint>> Observations;
	Observations.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f, 2.0f }));
	Observations.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { -0.5f }));
	Observations.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 3.0f, 4.0f, 5.0f, 6.0f }));

	TArray<TInstancedStruct<FPoint>> Actions;
	Actions.Reserve(10);
	Actions.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 123.0f }));

	const bool bOk = Policy->BatchedThink(Observations, Actions);
	TestTrue(TEXT("BatchedThink succeeds"), bOk);
	TestEqual(TEXT("Actions resized to batch"), Actions.Num(), Observations.Num());

	for (int32 i = 0; i < Observations.Num(); ++i)
	{
		const FString Label = FString::Printf(TEXT("Observation %d matches action"), i);
		TestTrue(*Label, ScholaPolicyBatchedThinkTestPrivate::ObservationMatchesAction(Observations[i], Actions[i]));
	}

	return true;
}

#endif // WITH_AUTOMATION_TESTS
