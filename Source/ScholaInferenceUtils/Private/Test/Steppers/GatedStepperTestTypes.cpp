// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "GatedStepperTestTypes.h"
#include "Points/BoxPoint.h"
#include "HAL/PlatformProcess.h"

void USequenceTestAgent::Act_Implementation(const FInstancedStruct& InAction)
{
	if (OnAct) OnAct();
	const FBoxPoint* Box = InAction.GetPtr<FBoxPoint>();
	Applied.Add(Box && Box->Values.Num() > 0 ? Box->Values[0] : -1.0f);
}

void USequenceTestAgent::Observe_Implementation(FInstancedStruct& OutObservations)
{
	if (OnObserve) OnObserve();
	OutObservations.InitializeAs<FBoxPoint>(TArray<float>{static_cast<float>(++ObservationCounter)});
}

void UGatedTestPolicy::BeginDestroy()
{
	ReleaseAll();
	// Calls still running hold their gate; only return gates once every call has finished.
	if (WaitFinished(Started.load(), 5.0))
	{
		FScopeLock ScopeLock(&Lock);
		for (FEvent* Gate : Gates)
		{
			FPlatformProcess::ReturnSynchEventToPool(Gate);
		}
		Gates.Reset();
	}
	Super::BeginDestroy();
}

FEvent* UGatedTestPolicy::GetGate(int32 Index)
{
	FScopeLock ScopeLock(&Lock);
	while (Gates.Num() <= Index)
	{
		Gates.Add(FPlatformProcess::GetSynchEventFromPool(true));
	}
	return Gates[Index];
}

bool UGatedTestPolicy::Think(const TInstancedStruct<FPoint>& InObservation, TInstancedStruct<FPoint>& OutAction)
{
	TArray<TInstancedStruct<FPoint>> Actions;
	if (!BatchedThink({InObservation}, Actions))
	{
		return false;
	}
	OutAction = Actions[0];
	return true;
}

bool UGatedTestPolicy::BatchedThink(const TArray<TInstancedStruct<FPoint>>& InObservations,
	TArray<TInstancedStruct<FPoint>>& OutActions)
{
	const int32 Index = Started.fetch_add(1);
	const int32 InFlight = Index + 1 - Finished.load();
	for (int32 Peak = PeakInFlight.load(); InFlight > Peak && !PeakInFlight.compare_exchange_weak(Peak, InFlight);)
	{
	}
	FEvent* Gate = nullptr;
	{
		// Check and create under the lock so ReleaseAll cannot miss a gate created concurrently.
		FScopeLock ScopeLock(&Lock);
		if (!bOpen.load())
		{
			Gate = GetGate(Index);
		}
	}
	if (Gate)
	{
		Gate->Wait();
	}
	OutActions = InObservations;
	for (const TInstancedStruct<FPoint>& Observation : InObservations)
	{
		const FBoxPoint* Box = Observation.GetPtr<FBoxPoint>();
		const float Value = Box && Box->Values.Num() > 0 ? Box->Values[0] : -1.0f;
		for (float Max = MaxCompleted.load(); Value > Max && !MaxCompleted.compare_exchange_weak(Max, Value);)
		{
		}
	}
	Finished.fetch_add(1);
	return true;
}

int32 UGatedTestPolicy::GetStartedCalls() const
{
	return Started.load();
}

bool UGatedTestPolicy::Release(int32 Index, double TimeoutSeconds)
{
	const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
	while (Started.load() <= Index)
	{
		if (FPlatformTime::Seconds() > Deadline)
		{
			return false;
		}
		FPlatformProcess::SleepNoStats(0.0005f);
	}
	GetGate(Index)->Trigger();
	return true;
}

void UGatedTestPolicy::ReleaseAll()
{
	FScopeLock ScopeLock(&Lock);
	bOpen.store(true);
	for (FEvent* Gate : Gates)
	{
		Gate->Trigger();
	}
}

bool UGatedTestPolicy::WaitFinished(int32 Count, double TimeoutSeconds) const
{
	const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
	while (Finished.load() < Count)
	{
		if (FPlatformTime::Seconds() > Deadline)
		{
			return false;
		}
		FPlatformProcess::SleepNoStats(0.0005f);
	}
	return true;
}
