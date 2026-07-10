// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Points/Point.h"
#include "StructUtils/InstancedStruct.h"
#include "FabricaAgentState.generated.h"

/**
 * @brief State produced by AFabricaEnvironment::OnUserStep (before Fabrica reward shaping).
 * @details The base maps this into FAgentState before FabricaGeneratedRewardForAgent. Optional
 *          task-success values are copied into FAgentState::Info under the UFabricaRewardInfo task-success prefix.
 */
USTRUCT(BlueprintType)
struct SCHOLATRAINING_API FFabricaAgentState
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadWrite, Category = "Schola|Fabrica")
	TInstancedStruct<FPoint> Observations;

	UPROPERTY(BlueprintReadWrite, Category = "Schola|Fabrica")
	TMap<FString, FString> Info;

	UPROPERTY(BlueprintReadWrite, Category = "Schola|Fabrica")
	bool bTerminated = false;

	UPROPERTY(BlueprintReadWrite, Category = "Schola|Fabrica")
	bool bTruncated = false;

	UPROPERTY(BlueprintReadWrite, Category = "Schola|Fabrica")
	float TaskSuccessMetric = 0.0f;

};
