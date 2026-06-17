// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "NNEUtils/NNEBufferAllocator.h"
#include "Spaces/DictSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"

//TODO this needs some error handling for when things don't line up properly, e.g. if the space is not found in the buffer map
// ~acann, 7/4/2025

void FNNEBufferAllocator::operator()(const FDictSpace& InSpace)
{   
    this->Buffer.InitializeAs<FNNEDictBuffer>();
      
    for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : InSpace.Spaces)
    {
		TInstancedStruct<FNNEPointBuffer>& SubBufferRef = this->Buffer.GetMutable<FNNEDictBuffer>().Buffers.Add(Pair.Key);
        FNNEBufferAllocator::AllocateBuffer(Pair.Value, SubBufferRef);
    }
}

void FNNEBufferAllocator::operator()(const FBoxSpace& InSpace)
{
    this->Buffer.InitializeAs<FNNEBoxBuffer>(InSpace.GetFlattenedSize());
}

void FNNEBufferAllocator::operator()(const FMultiBinarySpace& InSpace)
{
    this->Buffer.InitializeAs<FNNEMultiBinaryBuffer>(InSpace.GetFlattenedSize());
}

void FNNEBufferAllocator::operator()(const FDiscreteSpace& InSpace)
{
	this->Buffer.InitializeAs<FNNEDiscreteBuffer>();
}

void FNNEBufferAllocator::operator()(const FMultiDiscreteSpace& InSpace)
{
	this->Buffer.InitializeAs<FNNEMultiDiscreteBuffer>(InSpace.GetNumDimensions());
}
