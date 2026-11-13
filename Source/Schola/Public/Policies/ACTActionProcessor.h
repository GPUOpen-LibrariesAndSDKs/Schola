// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/ActionProcessor.h"
#include "ACTActionProcessor.generated.h"

/**
 * @brief Plays each predicted chunk in order, like LeRobot ACT's action queue.
 *
 * Runs the model, plays every action of its chunk one per step, then runs the model again.
 * Exporters that play only part of each chunk, like LeRobot's n_action_steps, output only that part.
 */
USTRUCT(BlueprintType, meta = (DisplayName = "ACT"))
struct SCHOLA_API FACTActionProcessor : public FActionProcessor
{
	GENERATED_BODY()

	bool Init(int32 InChunkSize, int32 InActionDim) override;

	/** @return True once every action of the current chunk has been played. */
	bool NeedsNewChunk() const override { return NextIndex >= ChunkSize; }

	/**
	 * @brief Plays the next action of the current chunk, first replacing it with Chunk if NeedsNewChunk() is true.
	 *
	 * Chunk is ignored while the current chunk still has actions to play.
	 */
	bool Process(const FActionChunk& Chunk, TArrayView<float> OutAction) override;

	void Reset() override;

	/** Flattened [chunk_size, action_dim] chunk being played. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	TArray<float> CurrentChunk;

	/** Index of the next action to play from CurrentChunk. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	int32 NextIndex = 0;
};
