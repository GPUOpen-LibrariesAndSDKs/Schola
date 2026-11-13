// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "StructUtils/InstancedStruct.h"
#include "ActionProcessor.generated.h"

/** What a model that predicts action chunks output in one inference. */
struct FActionChunk
{
	/** Flattened [chunk_size, action_dim] actions, row-major. */
	TConstArrayView<float> Actions;

	/** The other outputs the processor reads, one per name of its GetExtraOutputNames() and in that order. */
	TConstArrayView<TConstArrayView<float>> ExtraOutputs;
};

/**
 * @brief Turns the action chunks a policy predicts into one action per step.
 *
 * Models such as LeRobot ACT predict a chunk of several future actions per inference.
 * A policy holds one processor as a template, the one the exported model names, and gives
 * each state its own copy, so every agent plays its own chunk.
 *
 * The base processor predicts a new chunk every step and plays its first action. To play
 * chunks differently, derive from this struct, override NeedsNewChunk, Process and Reset,
 * and GetExtraOutputNames if it reads other model outputs. Exported models name it by its
 * struct name, see MarkerOutputPrefix, so lerobot-to-onnx also needs a Python processor whose
 * unreal_struct is that name. Overrides of Init must call
 * Super::Init. Processors are copied with their UScriptStruct, so any data that must
 * survive a copy has to be a UPROPERTY.
 *
 * Process may run off the game thread, so processors must not create UObjects in it.
 */
USTRUCT(BlueprintType)
struct SCHOLA_API FActionProcessor
{
	GENERATED_BODY()

	virtual ~FActionProcessor() = default;

	/** Start of the output name an exported model names its processor in, followed by the processor's struct name without its F, e.g. ACTActionProcessor. The output's value is unused. */
	static constexpr const TCHAR* MarkerOutputPrefix = TEXT("action_processor.");

	/**
	 * @brief Creates the processor an exported model names in its marker output.
	 * @param[in] OutputNames Names of all the model's outputs, exactly one of which must start with MarkerOutputPrefix.
	 * @return The new processor, not yet initialized, or an empty struct if the model does not name exactly one struct derived from FActionProcessor, logging an error.
	 */
	static TInstancedStruct<FActionProcessor> CreateForModel(TConstArrayView<FString> OutputNames);

	/**
	 * @brief Sets up the processor for a model's chunk layout, matching the start of an episode.
	 * @param[in] InChunkSize Number of actions the model predicts per inference.
	 * @param[in] InActionDim Size of one action.
	 * @return True if the layout is valid for this processor.
	 */
	virtual bool Init(int32 InChunkSize, int32 InActionDim);

	/** @return Names of the model outputs besides the actions that Process reads in Chunk.ExtraOutputs, in that order. Each must be a float tensor with a fixed shape. */
	virtual TArray<FString> GetExtraOutputNames() const { return {}; }

	/** @return True if the model must run this step so Process gets a new chunk. */
	virtual bool NeedsNewChunk() const { return true; }

	/**
	 * @brief Turns an optional new chunk into this step's action.
	 * @param[in] Chunk The model's output. Empty when NeedsNewChunk() was false, so the model did not run.
	 * @param[out] OutAction Receives one action of length ActionDim.
	 * @return True if OutAction was written. Processors that return false must leave themselves unchanged.
	 */
	virtual bool Process(const FActionChunk& Chunk, TArrayView<float> OutAction);

	/** Clears any chunk progress, matching the start of an episode. */
	virtual void Reset() {}

	/**
	 * @brief Checks that two processors can be copied into each other.
	 * @return True if both are empty, or both hold the same processor type with the same chunk layout.
	 */
	static bool IsSameLayout(const TInstancedStruct<FActionProcessor>& A, const TInstancedStruct<FActionProcessor>& B);

	/**
	 * @brief Copies Source into Target without changing Target's processor type.
	 * @return True if IsSameLayout(Target, Source) and the copy succeeded, false otherwise, leaving Target unchanged.
	 */
	static bool CopyInPlace(TInstancedStruct<FActionProcessor>& Target, const TInstancedStruct<FActionProcessor>& Source);

	/** Number of actions the model predicts per inference, set by Init. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	int32 ChunkSize = 0;

	/** Size of one action, set by Init. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Action Processor")
	int32 ActionDim = 0;

protected:
	int32 NumChunkElements() const { return ChunkSize * ActionDim; }

	/** @return True if Init has run and OutAction holds one action, logging an error otherwise. */
	bool IsActionValid(TArrayView<float> OutAction) const;

	/** @return True if Chunk holds a whole chunk of actions, logging an error otherwise. */
	bool IsChunkValid(const FActionChunk& Chunk) const;

	/**
	 * @brief Copies one action out of a flattened [num_actions, action_dim] array.
	 * @param[in] Actions The actions to copy from, e.g. a chunk.
	 * @param[in] Index Index of the action to copy. Must be less than Actions.Num() / ActionDim.
	 * @param[out] OutAction Receives the action. Must hold ActionDim floats.
	 */
	void CopyAction(TConstArrayView<float> Actions, int32 Index, TArrayView<float> OutAction) const;
};
