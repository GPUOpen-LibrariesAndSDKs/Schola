// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "TestStepper.h"
#include "LogScholaInferenceUtils.h"
#include "Points/BoxPoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Points/Point.h"
#include "Misc/ScopeExit.h"

// Static tracking variables (internal linkage)
static TSet<uint32> GTestPolicyThreadIds; // threads that executed Think
static std::atomic<bool> GTestPolicySawNonGameThread { false };
static uint32 GTestPolicyGameThreadId = 0; // captured at first reset
static std::atomic<uint32> GThinkCounter{ 0 }; // Global think counter


UTestAgent::UTestAgent()
{
	TArray<float> Low = { -1.0f, -1.0f, -1.0f };
	TArray<float> High = { 1.0f, 1.0f, 1.0f };
	TArray<int>	  ActionHigh = { 3 };

	this->Defn = FInteractionDefinition(
		TInstancedStruct<FSpace>::Make<FBoxSpace>(Low, High),
		TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(ActionHigh)
	);
}

void UTestAgent::SetStatus_Implementation(EAgentStatus NewStatus)
{
	Status = NewStatus;
}

EAgentStatus UTestAgent::GetStatus_Implementation()
{
	return Status;
}

void UTestAgent::Define_Implementation(FInteractionDefinition& OutDefinition)
{
	OutDefinition = Defn;
}

void UTestAgent::Act_Implementation(const FInstancedStruct& InAction)
{
	if (Status == EAgentStatus::Running)
	{
		if (const FMultiDiscretePoint* DiscreteAction = InAction.GetPtr<FMultiDiscretePoint>())
		{
			if (DiscreteAction->Values.Num() > 0)
			{
				// Single-Dim action space
				int ActionIndex = DiscreteAction->Values[0];

				if (ActionIndex >= 0 && ActionIndex < Defn.ActionSpaceDefn.Get<FMultiDiscreteSpace>().GetFlattenedSize())
				{
					LastActionReceived = ActionIndex;

					switch (ActionIndex)
					{
						case 0:
							UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Case 0");
							break;
						case 1:
							UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Case 1");
							break;
						case 2:
							UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Case 2");
							break;
					}
				}
				else
				{
					// Invalid Action
					UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStepper: Invalid Action index: {0}", ActionIndex);
				}
			}
		}
		else if (const FBoxPoint* BoxAction = InAction.GetPtr<FBoxPoint>())
		{
			// Expected a Discrete Action
			UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStepper: Got FBoxPoint Action");
		}
		else
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStepper: Got unknown Action type");
		}
	}
	else
	{
		// Agent must be running to act
		UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStepper: Agent must be running to act");
	}
}

void UTestAgent::Observe_Implementation(FInstancedStruct& OutObservations)
{
	// Random Fixed values to test, that fit within the observation definition range
	TArray<float> MockObservation = { 0.5f, -0.1f, 0.2f };
	
	OutObservations.InitializeAs<FBoxPoint>(MockObservation);

	UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: After move to OutObservations: OutObservations state: {0}", OutObservations.IsValid() ? FString(TEXT("Valid")) : FString(TEXT("Invalid")));

	if (const FBoxPoint* BoxPoint = OutObservations.GetPtr<FBoxPoint>())
	{
		UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Successfully created FBoxPoint with {0} values", BoxPoint->Values.Num());
	}
	else
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStepper: Failed to create FBoxPoint");
	}
}

UTestPolicy::UTestPolicy()
{
}

bool UTestPolicy::Think(
	const TInstancedStruct<FPoint>&		  InObservations,
	const TScriptInterface<IPolicyState>& InOutState,
	TInstancedStruct<FPoint>&			  OutAction)
{
	bool Expected = false;
	if(!bInferenceInFlight.compare_exchange_strong(Expected, true, std::memory_order_acq_rel))
	{
		UE_LOGFMT(LogScholaInferenceUtils, Verbose, "TestStepper: Skipped - inference already in flight");
		return false; // signal stepper to try next tick
	}
	ON_SCOPE_EXIT{ bInferenceInFlight.store(false, std::memory_order_release); };
	const uint32 Count = GThinkCounter.fetch_add(1, std::memory_order_relaxed) + 1;
	UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Think #{0} ThreadId={1}", static_cast<uint32>(Count), FPlatformTLS::GetCurrentThreadId());

	// Thread instrumentation
	const uint32 CurrentTid = FPlatformTLS::GetCurrentThreadId();
	GTestPolicyThreadIds.Add(CurrentTid);
	if (IsInGameThread())
	{
		if (GTestPolicyGameThreadId == 0) { GTestPolicyGameThreadId = CurrentTid; }
	}
	else
	{
		GTestPolicySawNonGameThread.store(true, std::memory_order_relaxed);
	}

	TArray<int> ActionChoice = { 1 };
	OutAction.InitializeAs<FMultiDiscretePoint>(ActionChoice);

	UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: After move to OutAction: OutAction state: {0}", OutAction.IsValid() ? FString(TEXT("Valid")) : FString(TEXT("Invalid")));
	UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Policy chose action: 1");
	return true;
}

bool UTestPolicy::Init(const FInteractionDefinition& InPolicyDefinition)
{
	PolicyDefinition = InPolicyDefinition;
	UE_LOGFMT(LogScholaInferenceUtils, Display, "TestStepper: Policy initiated");
	return true;
}

bool UTestPolicy::IsInferenceBusy() const
{
	return bInferenceInFlight.load(std::memory_order_relaxed);
}

void UTestPolicy::ResetThreadTracking()
{
	GTestPolicyThreadIds.Reset();
	GTestPolicySawNonGameThread.store(false, std::memory_order_relaxed);
	GTestPolicyGameThreadId = FPlatformTLS::GetCurrentThreadId(); // snapshot
	GThinkCounter.store(0, std::memory_order_relaxed);
}

TSet<uint32> UTestPolicy::GetThreadIdsCopy()
{
	return GTestPolicyThreadIds; // copy
}

bool UTestPolicy::SawNonGameThread()
{
	return GTestPolicySawNonGameThread.load(std::memory_order_relaxed);
}

bool UTestStatefulPolicy::Think(
	const TInstancedStruct<FPoint>&		  InObservations,
	const TScriptInterface<IPolicyState>& InOutState,
	TInstancedStruct<FPoint>&			  OutAction)
{
	UTestCounterState* Counter = Cast<UTestCounterState>(InOutState.GetObject());
	if (!Counter)
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "TestStatefulPolicy: Expected a UTestCounterState");
		return false;
	}

	OutAction.InitializeAs<FMultiDiscretePoint>(TArray<int> { 1 });

	if (!Counter->NotifyStateUpdate())
	{
		return false;
	}

	Counter->Count++;
	ThinkCount.fetch_add(1, std::memory_order_relaxed);
	return true;
}

bool UTestStatefulPolicy::CreateInitialState(TScriptInterface<IPolicyState>& OutState) const
{
	OutState = NewObject<UTestCounterState>(GetTransientPackage());
	return true;
}

int32 UTestStatefulPolicy::GetCount(const TScriptInterface<IPolicyState>& State)
{
	const UTestCounterState* Counter = Cast<UTestCounterState>(State.GetObject());
	return Counter ? Counter->Count : -1;
}
