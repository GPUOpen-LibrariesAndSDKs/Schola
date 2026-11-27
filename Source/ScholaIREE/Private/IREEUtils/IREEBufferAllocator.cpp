// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "IREEUtils/IREEBufferAllocator.h"
#include "Spaces/DictSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"

void FIREEBufferAllocator::operator()(const FDictSpace& InSpace)
{
	this->Buffer.InitializeAs<FIREEDictBuffer>();

	for (const TPair<FString, TInstancedStruct<FSpace>>& Pair : InSpace.Spaces)
	{
		TInstancedStruct<FIREEPointBuffer>& SubBufferRef = this->Buffer.GetMutable<FIREEDictBuffer>().Buffers.Add(Pair.Key);
		FIREEBufferAllocator::AllocateBuffer(Pair.Value, SubBufferRef);
	}
}

void FIREEBufferAllocator::operator()(const FBoxSpace& InSpace)
{
	this->Buffer.InitializeAs<FIREEBoxBuffer>(InSpace.GetFlattenedSize());
}

void FIREEBufferAllocator::operator()(const FMultiBinarySpace& InSpace)
{
	this->Buffer.InitializeAs<FIREEMultiBinaryBuffer>(InSpace.GetFlattenedSize());
}

void FIREEBufferAllocator::operator()(const FDiscreteSpace& InSpace)
{
	this->Buffer.InitializeAs<FIREEDiscreteBuffer>();
}

void FIREEBufferAllocator::operator()(const FMultiDiscreteSpace& InSpace)
{
	this->Buffer.InitializeAs<FIREEMultiDiscreteBuffer>(InSpace.GetNumDimensions());
}
