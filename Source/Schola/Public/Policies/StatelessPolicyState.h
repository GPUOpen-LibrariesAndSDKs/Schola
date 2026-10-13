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
 * or non-native state always indicates an error. It holds no data, so every agent of every
 * stateless policy can share the instance returned by Get().
 */
UCLASS(NotBlueprintable)
class SCHOLA_API UStatelessPolicyState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	void Reset() override {}

	bool CopyFrom(const IPolicyState& Other) override;

	/**
	 * @brief Gets the shared stateless policy state.
	 *
	 * Must be called on the game thread, e.g. from CreateInitialState.
	 *
	 * @return The shared instance, kept alive for the lifetime of the program.
	 */
	UFUNCTION(BlueprintPure, Category = "Schola|Policy", meta = (DisplayName = "Get Stateless Policy State"))
	static TScriptInterface<IPolicyState> Get();
};
