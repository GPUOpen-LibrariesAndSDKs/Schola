// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/ACTActionProcessor.h"

bool FACTActionProcessor::Init(int32 InChunkSize, int32 InActionDim)
{
	if (!Super::Init(InChunkSize, InActionDim))
	{
		return false;
	}
	CurrentChunk.SetNumZeroed(NumChunkElements());
	Reset();
	return true;
}

bool FACTActionProcessor::Process(const FActionChunk& Chunk, TArrayView<float> OutAction)
{
	if (!IsActionValid(OutAction))
	{
		return false;
	}
	if (NeedsNewChunk())
	{
		if (!IsChunkValid(Chunk))
		{
			return false;
		}
		FMemory::Memcpy(CurrentChunk.GetData(), Chunk.Actions.GetData(), NumChunkElements() * sizeof(float));
		NextIndex = 0;
	}
	CopyAction(CurrentChunk, NextIndex, OutAction);
	++NextIndex;
	return true;
}

void FACTActionProcessor::Reset()
{
	if (CurrentChunk.Num() > 0)
	{
		FMemory::Memzero(CurrentChunk.GetData(), CurrentChunk.Num() * sizeof(float));
	}
	NextIndex = ChunkSize;
}
