// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Agent/AgentInterface.h"
#include "Policies/PolicyInterface.h"
#include "HAL/Event.h"
#include "GatedStepperTestTypes.generated.h"

/** Agent whose Nth observation has value N; records every applied action value in order. */
UCLASS()
class USequenceTestAgent : public UObject, public IAgent
{
	GENERATED_BODY()
public:
	void Define_Implementation(FInteractionDefinition& OutDefinition) override {}
	void Act_Implementation(const FInstancedStruct& InAction) override;
	void Observe_Implementation(FInstancedStruct& OutObservations) override;
	void SetStatus_Implementation(EAgentStatus NewStatus) override {}
	EAgentStatus GetStatus_Implementation() override { return EAgentStatus::Running; }

	/** Values of applied actions, in application order. */
	TArray<float> Applied;
	int32 ObservationCounter = 0;
	TFunction<void()> OnObserve;
	TFunction<void()> OnAct;
};

/**
 * Policy whose BatchedThink blocks until the test releases that specific call.
 * Actions echo observations, so each applied value identifies its dispatch.
 */
UCLASS()
class UGatedTestPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()
public:
	virtual void BeginDestroy() override;

	bool Think(const TInstancedStruct<FPoint>& InObservation, TInstancedStruct<FPoint>& OutAction) override;
	bool BatchedThink(const TArray<TInstancedStruct<FPoint>>& InObservations,
		TArray<TInstancedStruct<FPoint>>& OutActions) override;
	bool Init(const FInteractionDefinition& InDefinition) override { return true; }
	bool IsInferenceBusy() const override { return false; }

	/** Number of BatchedThink calls that have started. */
	int32 GetStartedCalls() const;
	/** Number of BatchedThink calls that have returned. */
	int32 GetFinishedCalls() const { return Finished.load(); }
	/** Releases call Index (0-based, in start order); waits until it has started first. */
	bool Release(int32 Index, double TimeoutSeconds = 5.0);
	/** Releases every started call and any later call. */
	void ReleaseAll();
	/** Largest first observation value returned by any finished call (-1 if none). */
	float GetMaxCompletedValue() const { return MaxCompleted.load(); }
	/** Largest number of calls that were simultaneously started and not finished. */
	int32 GetPeakInFlight() const { return PeakInFlight.load(); }
	/** Waits until at least Count calls have finished. */
	bool WaitFinished(int32 Count, double TimeoutSeconds = 5.0) const;

private:
	FEvent* GetGate(int32 Index);
	mutable FCriticalSection Lock;
	TArray<FEvent*> Gates;
	std::atomic<int32> Started{0};
	std::atomic<int32> Finished{0};
	std::atomic<bool> bOpen{false};
	std::atomic<float> MaxCompleted{-1.0f};
	std::atomic<int32> PeakInFlight{0};
};
