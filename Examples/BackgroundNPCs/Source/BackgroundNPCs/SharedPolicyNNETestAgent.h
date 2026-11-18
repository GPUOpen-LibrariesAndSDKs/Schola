// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Agent/AgentInterface.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"
#include "SharedPolicyNNETestAgent.generated.h"

/**
 * @brief Test-only NPC for Architecture A (64 observation values, 2 action values).
 *
 * Observes a zero vector whose first value is ObservationValue, matching the Python reference inputs,
 * and records the Box action it receives.
 */
UCLASS()
class USharedPolicyNNETestAgent : public UObject, public IAgent
{
	GENERATED_BODY()

public:
	static constexpr int32 ObservationWidth = 64;
	static constexpr int32 ActionWidth = 2;

	float ObservationValue = 0.f;
	TArray<float> LastAction;
	int32 ActCount = 0;

	void Define_Implementation(FInteractionDefinition& OutDefinition) override
	{
		TArray<float> ObsLow;
		TArray<float> ObsHigh;
		ObsLow.Init(-1000.f, ObservationWidth);
		ObsHigh.Init(1000.f, ObservationWidth);
		TArray<float> ActLow;
		TArray<float> ActHigh;
		ActLow.Init(-1000.f, ActionWidth);
		ActHigh.Init(1000.f, ActionWidth);
		OutDefinition = FInteractionDefinition(
			TInstancedStruct<FSpace>::Make<FBoxSpace>(ObsLow, ObsHigh),
			TInstancedStruct<FSpace>::Make<FBoxSpace>(ActLow, ActHigh));
	}

	void Observe_Implementation(FInstancedStruct& OutObservations) override
	{
		TArray<float> Values;
		Values.SetNumZeroed(ObservationWidth);
		Values[0] = ObservationValue;
		OutObservations.InitializeAs<FBoxPoint>(Values);
	}

	void Act_Implementation(const FInstancedStruct& InAction) override
	{
		if (const FBoxPoint* Box = InAction.GetPtr<FBoxPoint>())
		{
			LastAction = Box->Values;
			++ActCount;
		}
	}

	void SetStatus_Implementation(EAgentStatus NewStatus) override { Status = NewStatus; }

	EAgentStatus GetStatus_Implementation() override { return Status; }

private:
	EAgentStatus Status = EAgentStatus::Running;
};
