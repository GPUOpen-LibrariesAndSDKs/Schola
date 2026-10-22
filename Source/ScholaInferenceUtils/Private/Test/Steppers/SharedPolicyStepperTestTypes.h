// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Policies/SharedPolicyInterface.h"
#include "Points/MultiDiscretePoint.h"
#include "SharedPolicyStepperTestTypes.generated.h"

/**
 * @brief Test-only shared policy. Each registered head returns a fixed discrete action.
 *
 * Does not load ONNX or call NNE. ForcedActionCount and bThinkSucceeds let tests
 * exercise stepper failure paths.
 */
UCLASS()
class UMockSharedPolicy : public UObject, public ISharedPolicy
{
	GENERATED_BODY()

public:
	UPROPERTY()
	TMap<FName, TInstancedStruct<FPoint>> HeadActions;

	TArray<TInstancedStruct<FPoint>> LastObservations;
	TArray<FName> LastHeads;
	int32 ThinkCalls = 0;
	int32 ForcedActionCount = INDEX_NONE;
	bool bThinkSucceeds = true;

	void RegisterHead(FName Head, int32 ActionIndex)
	{
		TInstancedStruct<FPoint> Action;
		Action.InitializeAs<FMultiDiscretePoint>(TArray<int>{ActionIndex});
		HeadActions.Add(Head, Action);
	}

	void GetRegisteredHeads(TSet<FName>& OutHeads) const override
	{
		OutHeads.Reset();
		for (const TPair<FName, TInstancedStruct<FPoint>>& Pair : HeadActions)
		{
			OutHeads.Add(Pair.Key);
		}
	}

	bool BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations, const TArray<FName>& Heads,
		TArray<TInstancedStruct<FPoint>>& OutActions) override
	{
		++ThinkCalls;
		LastObservations = Observations;
		LastHeads = Heads;

		if (!bThinkSucceeds)
		{
			return false;
		}
		if (Observations.Num() != Heads.Num())
		{
			return false;
		}
		if (ForcedActionCount != INDEX_NONE)
		{
			OutActions.Reset();
			OutActions.SetNum(ForcedActionCount);
			return true;
		}

		TArray<TInstancedStruct<FPoint>> Results;
		Results.Reserve(Heads.Num());
		for (const FName Head : Heads)
		{
			const TInstancedStruct<FPoint>* Action = HeadActions.Find(Head);
			if (!Action)
			{
				return false;
			}
			Results.Add(*Action);
		}
		OutActions = MoveTemp(Results);
		return true;
	}
};
