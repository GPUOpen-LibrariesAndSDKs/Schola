// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/ActionProcessor.h"
#include "TemporalEnsembleActionProcessor.generated.h"

/**
 * @brief Runs the model every step and averages every prediction made for the current step.
 *
 * The model outputs a coefficient k with each chunk in CoeffOutputName, and the i-th
 * prediction for a step, counting from the oldest, is weighted exp(-k * i), matching
 * LeRobot's ACTTemporalEnsembler with temporal_ensemble_coeff k.
 */
USTRUCT(BlueprintType, meta = (DisplayName = "Temporal Ensemble"))
struct SCHOLA_API FTemporalEnsembleActionProcessor : public FActionProcessor
{
	GENERATED_BODY()

	/** Model output holding the coefficient k as one float. */
	static constexpr const TCHAR* CoeffOutputName = TEXT("temporal_ensemble_coeff");

	bool Init(int32 InChunkSize, int32 InActionDim) override;

	TArray<FString> GetExtraOutputNames() const override { return { CoeffOutputName }; }

	/** Adds Chunk to the ensemble and plays its weighted average for the current step. The coefficient output must hold one float. */
	bool Process(const FActionChunk& Chunk, TArrayView<float> OutAction) override;

	void Reset() override;

	/** Flattened [chunk_size, action_dim] weighted sum of the predictions for each upcoming step. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	TArray<float> WeightedSums;

	/** Sum of the weights added to each upcoming step. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	TArray<float> WeightTotals;

	/** Number of predictions added to each upcoming step. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	TArray<int32> PredictionCounts;

private:
	/** Drops the step just played and opens an empty slot for the last step of the new chunk. */
	void ShiftEnsemble();
};
