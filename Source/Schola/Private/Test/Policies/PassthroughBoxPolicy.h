// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Policies/PolicyInterface.h"
#include "PassthroughBoxPolicy.generated.h"

/**
 * Minimal policy for tests: copies FBoxPoint observations to actions.
 * Uses the default IPolicy::BatchedThink (per-element Think).
 */
UCLASS()
class UPassthroughBoxPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:
	bool Think(const TInstancedStruct<FPoint>& InObservations, TInstancedStruct<FPoint>& OutAction) override;
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;
	bool IsInferenceBusy() const override;
};
