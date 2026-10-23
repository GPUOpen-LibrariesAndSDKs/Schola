// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "ActionChunker.generated.h"

/**
 * @brief How a chunked policy turns each predicted action chunk into one env-step action.
 */
UENUM(BlueprintType)
enum class EActionChunkMode : uint8
{
	/** Run inference when the previous chunk is exhausted, then play actions in order. */
	Queue,
	/** Run inference every step and average overlapping predictions for the current time. */
	TemporalEnsemble
};

/**
 * @brief How an FActionChunker plays each predicted chunk.
 *
 * These are inference settings, not part of a trained model, so the defaults work for any chunked model.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FActionChunkerSettings
{
	GENERATED_BODY()

	/** Queue plays each chunk in order; TemporalEnsemble predicts every step and averages overlapping predictions. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Action Chunking")
	EActionChunkMode ChunkMode = EActionChunkMode::Queue;

	/** Actions played from each chunk before predicting again. 0 or more than the chunk size plays the whole chunk. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = 0, EditCondition = "ChunkMode == EActionChunkMode::Queue", EditConditionHides), Category = "Action Chunking")
	int32 NumActionSteps = 0;

	/** Coefficient k of the ensemble weights exp(-k * i), where i counts predictions for a step from the oldest. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = 0, EditCondition = "ChunkMode == EActionChunkMode::TemporalEnsemble", EditConditionHides), Category = "Action Chunking")
	float TemporalEnsembleCoeff = 0.01f;
};

/**
 * @brief Queue or temporal-ensemble playback of a predicted action chunk.
 *
 * Independent of any neural-network backend. A policy state can own one of these
 * alongside other recurrent data (for example NNE tensors).
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FActionChunker
{
	GENERATED_BODY()

	/**
	 * @brief Allocates the queue or ensemble for a chunk layout.
	 *
	 * Safe to call again with the same sizes. Changing sizes reallocates.
	 *
	 * @param[in] InChunkSize Number of actions the model predicts per inference.
	 * @param[in] InActionDim Size of one action vector.
	 * @param[in] InSettings How each chunk is played.
	 */
	void Init(int32 InChunkSize, int32 InActionDim, const FActionChunkerSettings& InSettings = FActionChunkerSettings());

	/** @return True if the next Consume must be given a new model chunk. Always true in ensemble mode. */
	bool NeedsNewChunk() const;

	/**
	 * @brief Incorporates an optional new chunk and writes the action for this env step.
	 *
	 * In Queue mode, Chunk is required when NeedsNewChunk() is true and ignored otherwise.
	 * In TemporalEnsemble mode, Chunk is always required. Leaves the state unchanged on failure.
	 *
	 * @param[in] Chunk Flattened [chunk_size, action_dim] prediction, row-major.
	 * @param[out] OutAction Receives one action of length ActionDim.
	 * @return True if OutAction was written.
	 */
	bool Consume(TConstArrayView<float> Chunk, TArrayView<float> OutAction);

	/** Zeroes the queue or ensemble, matching the state at the start of an episode. */
	void Reset();

	/** @return True if Other has the same sizes and mode, so CopyFrom would succeed. */
	bool HasSameLayout(const FActionChunker& Other) const;

	/**
	 * @brief Copies another chunker into this one in place.
	 *
	 * Arrays are never reallocated. Layout (sizes and mode) must already match.
	 *
	 * @param[in] Other The chunker to copy from.
	 * @return True if Other has the same layout and the copy succeeded.
	 */
	bool CopyFrom(const FActionChunker& Other);

	/** Flattened [chunk_size, action_dim] queue, or weighted-sum ensemble. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<float> ChunkBuffer;

	/** Sum of the weights of the predictions added to each step of the ensemble. Unused in Queue mode. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<float> EnsembleWeights;

	/**
	 * Number of predictions added to each step of the ensemble. Unused in Queue mode.
	 *
	 * The i-th prediction for a step, counting from the oldest, is weighted exp(-k * i),
	 * matching LeRobot's ACTTemporalEnsembler.
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<int32> EnsembleCounts;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Properties")
	int32 ChunkSize = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Properties")
	int32 ActionDim = 0;

	/** Settings from Init, with NumActionSteps resolved to between 1 and ChunkSize. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Properties")
	FActionChunkerSettings Settings;

	/** Next action index in Queue mode. At least Settings.NumActionSteps when a new chunk is needed. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	int32 NextIndex = 0;

private:
	int32 NumChunkElements() const { return ChunkSize * ActionDim; }

	bool ConsumeQueue(TConstArrayView<float> Chunk, TArrayView<float> OutAction);
	bool ConsumeEnsemble(TConstArrayView<float> Chunk, TArrayView<float> OutAction);
	void ShiftEnsemble();
	void AddChunkToEnsemble(TConstArrayView<float> Chunk);
	void WriteCurrentAction(TArrayView<float> OutAction) const;
};
