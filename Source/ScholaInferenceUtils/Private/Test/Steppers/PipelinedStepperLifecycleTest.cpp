// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Async/TaskGraphInterfaces.h"
#include "GatedStepperTestTypes.h"
#include "Steppers/PipelinedStepper.h"
#include "UObject/StrongObjectPtr.h"

// Controlled-delay reproductions of pipelined-stepper delivery defects (plan task 5 / gate C5).

namespace
{
/** Runs game-thread tasks (stepper completion callbacks) for a short period. */
void PumpGameThread(double Seconds = 0.05)
{
	const double End = FPlatformTime::Seconds() + Seconds;
	do
	{
		FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
		FPlatformProcess::SleepNoStats(0.001f);
	}
	while (FPlatformTime::Seconds() < End);
}

bool IsStrictlyIncreasing(const TArray<float>& Values)
{
	for (int32 Index = 1; Index < Values.Num(); ++Index)
	{
		if (Values[Index] <= Values[Index - 1])
		{
			return false;
		}
	}
	return true;
}

struct FGatedPipeline
{
	TStrongObjectPtr<USequenceTestAgent> Agent{NewObject<USequenceTestAgent>()};
	TStrongObjectPtr<UGatedTestPolicy> Policy{NewObject<UGatedTestPolicy>()};
	TStrongObjectPtr<UPipelinedStepper> Stepper{NewObject<UPipelinedStepper>()};

	bool Init() const
	{
		return Stepper->Init({TScriptInterface<IAgent>(Agent.Get())}, TScriptInterface<IPolicy>(Policy.Get()));
	}

	~FGatedPipeline()
	{
		// Never leave worker threads blocked on a gate.
		Policy->ReleaseAll();
		Policy->WaitFinished(Policy->GetStartedCalls());
		PumpGameThread();
	}
};
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPipelinedStepperInFlightSlotTest, "Schola.Steppers.PipelinedStepper.InFlightSlotReuse",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FPipelinedStepperInFlightSlotTest::RunTest(const FString& Parameters)
{
	FGatedPipeline Pipeline;
	TestTrue(TEXT("Init"), Pipeline.Init());

	// Three ticks while no inference completes: a two-slot pipeline must not start a third dispatch.
	for (int32 Tick = 0; Tick < 3; ++Tick)
	{
		Pipeline.Stepper->Step();
	}
	PumpGameThread();
	TestTrue(TEXT("In-flight dispatches never exceed pipeline slots"), Pipeline.Policy->GetPeakInFlight() <= PIPELINE_STAGES);

	// Complete in reverse start order; an older result must not replace a newer completed one.
	for (int32 Index = Pipeline.Policy->GetStartedCalls() - 1; Index >= 0; --Index)
	{
		Pipeline.Policy->Release(Index);
		Pipeline.Policy->WaitFinished(Pipeline.Policy->GetStartedCalls() - Index);
		PumpGameThread();
	}
	const float NewestCompleted = Pipeline.Policy->GetMaxCompletedValue();
	Pipeline.Stepper->Step();
	PumpGameThread();
	TestTrue(TEXT("An action was applied"), Pipeline.Agent->Applied.Num() > 0);
	if (Pipeline.Agent->Applied.Num() > 0)
	{
		TestEqual(TEXT("Newest completed result is applied, not a stale overwrite"), Pipeline.Agent->Applied.Last(), NewestCompleted);
	}

	Pipeline.Policy->ReleaseAll();
	for (int32 Tick = 0; Tick < 6; ++Tick)
	{
		Pipeline.Stepper->Step();
		Pipeline.Policy->WaitFinished(Pipeline.Policy->GetStartedCalls());
		PumpGameThread();
	}
	TestTrue(TEXT("Applied observations strictly increase"), IsStrictlyIncreasing(Pipeline.Agent->Applied));
	TestTrue(TEXT("Pipeline keeps delivering after delays"), Pipeline.Agent->Applied.Num() >= 3);
	AddInfo(FString::Printf(TEXT("Applied=[%s] PeakInFlight=%d"),
		*FString::JoinBy(Pipeline.Agent->Applied, TEXT(","), [](float V) { return FString::SanitizeFloat(V); }),
		Pipeline.Policy->GetPeakInFlight()));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPipelinedStepperStaleCallbackTest, "Schola.Steppers.PipelinedStepper.StaleCallbackAfterInit",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FPipelinedStepperStaleCallbackTest::RunTest(const FString& Parameters)
{
	FGatedPipeline Pipeline;
	TestTrue(TEXT("Init"), Pipeline.Init());
	Pipeline.Stepper->Step(); // Dispatches observation 1, held by the gate.
	const float LastPreInitObservation = static_cast<float>(Pipeline.Agent->ObservationCounter);

	TestTrue(TEXT("Reinit"), Pipeline.Init());
	Pipeline.Stepper->Step(); // New generation dispatch, also held.

	// The pre-reinitialization dispatch completes after the new generation started.
	Pipeline.Policy->Release(0);
	Pipeline.Policy->WaitFinished(1);
	PumpGameThread();

	Pipeline.Policy->ReleaseAll();
	for (int32 Tick = 0; Tick < 6; ++Tick)
	{
		Pipeline.Stepper->Step();
		Pipeline.Policy->WaitFinished(Pipeline.Policy->GetStartedCalls());
		PumpGameThread();
	}
	const bool bStaleApplied = Pipeline.Agent->Applied.ContainsByPredicate(
		[LastPreInitObservation](float Value) { return Value <= LastPreInitObservation; });
	TestFalse(TEXT("No result from before reinitialization is applied"), bStaleApplied);
	TestTrue(TEXT("New generation delivers actions"), Pipeline.Agent->Applied.Num() > 0);
	TestTrue(TEXT("Applied observations strictly increase"), IsStrictlyIncreasing(Pipeline.Agent->Applied));
	AddInfo(FString::Printf(TEXT("Applied=[%s]"),
		*FString::JoinBy(Pipeline.Agent->Applied, TEXT(","), [](float V) { return FString::SanitizeFloat(V); })));
	return true;
}
