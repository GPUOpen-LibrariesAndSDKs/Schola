// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/Point.h"
#include "SharedPolicyInterface.generated.h"

/**
 * @brief Multi-head batch inference for a shared-policy model.
 *
 * IPolicy::BatchedThink cannot name a head per NPC. This interface is the
 * contract a future ScholaNNE implementation should satisfy; the stepper lives
 * in ScholaInferenceUtils so that module does not depend on ScholaNNE.
 *
 * Parallel arrays, one row per NPC:
 *   Observations = [ObsA, ObsB]
 *   Heads        = [Combat, Support]
 *   OutActions   = [ActionA, ActionB]
 */
UINTERFACE(BlueprintType)
class SCHOLAINFERENCEUTILS_API USharedPolicy : public UInterface
{
	GENERATED_BODY()
};

class SCHOLAINFERENCEUTILS_API ISharedPolicy
{
	GENERATED_BODY()

public:
	/**
	 * @brief Infers one action per NPC using that NPC's head.
	 *
	 * @param[in] Observations Observation row for each NPC, in batch order.
	 * @param[in] Heads Head name for each NPC. Must have the same length as Observations.
	 * @param[out] OutActions On success, one action per NPC in the same order. Unchanged on failure.
	 * @return True if every row produced an action.
	 */
	virtual bool BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations,
		const TArray<FName>& Heads, TArray<TInstancedStruct<FPoint>>& OutActions)
		PURE_VIRTUAL(ISharedPolicy::BatchedThinkWithHeads, return false;);

	/**
	 * @brief Heads this policy can serve. The stepper passes this set to ValidateNPCBindings.
	 *
	 * @param[out] OutHeads Replaced with the registered head names.
	 */
	virtual void GetRegisteredHeads(TSet<FName>& OutHeads) const
		PURE_VIRTUAL(ISharedPolicy::GetRegisteredHeads, return;);
};
