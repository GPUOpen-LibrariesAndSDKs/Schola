// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Policies/PolicyStateInterface.h"
#include "StatelessPolicyState.generated.h"

/**
 * @class UStatelessPolicyState
 * @brief The state of a policy that carries nothing between Think calls.
 *
 * Stateless policies return this from CreateInitialState instead of null, so that a null
 * or non-native state always indicates an error. Like any other state, each agent gets its
 * own instance from Create().
 */
UCLASS(NotBlueprintable)
class SCHOLA_API UStatelessPolicyState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	void Reset() override {}

	bool CopyFrom(const IPolicyState& Other) override;

	/**
	 * @brief Creates a new stateless policy state in the transient package.
	 *
	 * Must be called on the game thread, e.g. from CreateInitialState.
	 *
	 * @return The new state. The caller keeps it alive through a UPROPERTY reference.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Policy", meta = (DisplayName = "Create Stateless Policy State"))
	static TScriptInterface<IPolicyState> Create();
};
