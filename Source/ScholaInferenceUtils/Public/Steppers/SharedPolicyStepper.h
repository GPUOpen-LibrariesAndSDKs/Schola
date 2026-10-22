// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/ScriptInterface.h"
#include "Policies/SharedPolicyInterface.h"
#include "Policies/SharedPolicyTypes.h"
#include "SharedPolicyStepper.generated.h"

/**
 * @brief Synchronous stepper that routes each NPC through a named shared-policy head.
 *
 * IStepper::Init only accepts IPolicy and cannot carry per-NPC heads, so this
 * class is not an IStepper. Init takes FSharedNPCBinding rows plus ISharedPolicy.
 * Step() collects observations, looks up heads with FindHeadForAgent, calls
 * BatchedThinkWithHeads, then applies actions in binding order.
 */
UCLASS(BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLAINFERENCEUTILS_API USharedPolicyStepper : public UObject
{
	GENERATED_BODY()

public:
	/** NPC-to-head assignments, in batch order. */
	UPROPERTY(BlueprintReadOnly, Category = "Schola|SharedPolicy")
	TArray<FSharedNPCBinding> NPCBindings;

	/** Policy that implements BatchedThinkWithHeads. */
	UPROPERTY(BlueprintReadOnly, Category = "Schola|SharedPolicy")
	TScriptInterface<ISharedPolicy> Policy;

	/**
	 * @brief Stores bindings and policy after validation against the policy's registered heads.
	 *
	 * @param[in] InBindings One head per NPC. Empty, duplicate, or unknown heads fail.
	 * @param[in] InPolicy Native ISharedPolicy. Blueprint-only implementations are rejected.
	 * @return True if the stepper is ready for Step().
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	bool Init(const TArray<FSharedNPCBinding>& InBindings, const TScriptInterface<ISharedPolicy>& InPolicy);

	/**
	 * @brief One observe → BatchedThinkWithHeads → act cycle.
	 *
	 * Applies no actions if head lookup, inference, or the action count fails.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void Step();
};
