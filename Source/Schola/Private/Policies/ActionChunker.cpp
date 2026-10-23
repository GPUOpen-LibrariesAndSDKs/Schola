// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/ActionChunker.h"
#include "Common/LogSchola.h"

void FActionChunker::Init(int32 InChunkSize, int32 InActionDim, const FActionChunkerSettings& InSettings)
{
	ChunkSize = FMath::Max(InChunkSize, 0);
	ActionDim = FMath::Max(InActionDim, 0);
	Settings = InSettings;
	if (Settings.NumActionSteps <= 0 || Settings.NumActionSteps > ChunkSize)
	{
		Settings.NumActionSteps = ChunkSize;
	}
	ChunkBuffer.SetNumZeroed(NumChunkElements());
	if (Settings.ChunkMode == EActionChunkMode::TemporalEnsemble)
	{
		EnsembleWeights.SetNumZeroed(ChunkSize);
		EnsembleCounts.SetNumZeroed(ChunkSize);
	}
	else
	{
		EnsembleWeights.Reset();
		EnsembleCounts.Reset();
	}
	NextIndex = ChunkSize;
}

bool FActionChunker::NeedsNewChunk() const
{
	if (ChunkSize <= 0 || ActionDim <= 0)
	{
		return true;
	}
	if (Settings.ChunkMode == EActionChunkMode::TemporalEnsemble)
	{
		return true;
	}
	return NextIndex >= Settings.NumActionSteps;
}

void FActionChunker::Reset()
{
	if (ChunkBuffer.Num() > 0)
	{
		FMemory::Memzero(ChunkBuffer.GetData(), ChunkBuffer.Num() * sizeof(float));
	}
	if (EnsembleWeights.Num() > 0)
	{
		FMemory::Memzero(EnsembleWeights.GetData(), EnsembleWeights.Num() * sizeof(float));
	}
	if (EnsembleCounts.Num() > 0)
	{
		FMemory::Memzero(EnsembleCounts.GetData(), EnsembleCounts.Num() * sizeof(int32));
	}
	NextIndex = ChunkSize;
}

bool FActionChunker::HasSameLayout(const FActionChunker& Other) const
{
	return Other.ChunkSize == ChunkSize
		&& Other.ActionDim == ActionDim
		&& Other.Settings.ChunkMode == Settings.ChunkMode
		&& Other.ChunkBuffer.Num() == ChunkBuffer.Num()
		&& Other.EnsembleWeights.Num() == EnsembleWeights.Num()
		&& Other.EnsembleCounts.Num() == EnsembleCounts.Num();
}

bool FActionChunker::CopyFrom(const FActionChunker& Other)
{
	if (!HasSameLayout(Other))
	{
		return false;
	}
	Settings = Other.Settings;
	NextIndex = Other.NextIndex;
	if (ChunkBuffer.Num() > 0)
	{
		FMemory::Memcpy(ChunkBuffer.GetData(), Other.ChunkBuffer.GetData(), ChunkBuffer.Num() * sizeof(float));
	}
	if (EnsembleWeights.Num() > 0)
	{
		FMemory::Memcpy(EnsembleWeights.GetData(), Other.EnsembleWeights.GetData(), EnsembleWeights.Num() * sizeof(float));
	}
	if (EnsembleCounts.Num() > 0)
	{
		FMemory::Memcpy(EnsembleCounts.GetData(), Other.EnsembleCounts.GetData(), EnsembleCounts.Num() * sizeof(int32));
	}
	return true;
}

bool FActionChunker::Consume(TConstArrayView<float> Chunk, TArrayView<float> OutAction)
{
	if (ChunkSize <= 0 || ActionDim <= 0)
	{
		UE_LOGFMT(LogSchola, Error, "FActionChunker::Consume(): Call Init before Consume");
		return false;
	}
	if (OutAction.Num() != ActionDim)
	{
		UE_LOGFMT(LogSchola, Error, "FActionChunker::Consume(): OutAction has {0} elements, expected ActionDim {1}",
			OutAction.Num(), ActionDim);
		return false;
	}
	if (Settings.ChunkMode == EActionChunkMode::TemporalEnsemble)
	{
		return ConsumeEnsemble(Chunk, OutAction);
	}
	return ConsumeQueue(Chunk, OutAction);
}

bool FActionChunker::ConsumeQueue(TConstArrayView<float> Chunk, TArrayView<float> OutAction)
{
	if (NeedsNewChunk())
	{
		if (Chunk.Num() != NumChunkElements())
		{
			UE_LOGFMT(LogSchola, Error, "FActionChunker::Consume(): Queue is empty and chunk has {0} floats, expected {1}",
				Chunk.Num(), NumChunkElements());
			return false;
		}
		FMemory::Memcpy(ChunkBuffer.GetData(), Chunk.GetData(), NumChunkElements() * sizeof(float));
		NextIndex = 0;
	}
	WriteCurrentAction(OutAction);
	++NextIndex;
	return true;
}

bool FActionChunker::ConsumeEnsemble(TConstArrayView<float> Chunk, TArrayView<float> OutAction)
{
	if (Chunk.Num() != NumChunkElements())
	{
		UE_LOGFMT(LogSchola, Error, "FActionChunker::Consume(): Ensemble chunk has {0} floats, expected {1}",
			Chunk.Num(), NumChunkElements());
		return false;
	}
	ShiftEnsemble();
	AddChunkToEnsemble(Chunk);
	WriteCurrentAction(OutAction);
	return true;
}

void FActionChunker::ShiftEnsemble()
{
	if (ChunkSize > 1)
	{
		FMemory::Memmove(ChunkBuffer.GetData(), ChunkBuffer.GetData() + ActionDim, (ChunkSize - 1) * ActionDim * sizeof(float));
		FMemory::Memmove(EnsembleWeights.GetData(), EnsembleWeights.GetData() + 1, (ChunkSize - 1) * sizeof(float));
		FMemory::Memmove(EnsembleCounts.GetData(), EnsembleCounts.GetData() + 1, (ChunkSize - 1) * sizeof(int32));
	}
	FMemory::Memzero(ChunkBuffer.GetData() + (ChunkSize - 1) * ActionDim, ActionDim * sizeof(float));
	EnsembleWeights[ChunkSize - 1] = 0.0f;
	EnsembleCounts[ChunkSize - 1] = 0;
}

void FActionChunker::AddChunkToEnsemble(TConstArrayView<float> Chunk)
{
	for (int32 Step = 0; Step < ChunkSize; ++Step)
	{
		const float Weight = FMath::Exp(-Settings.TemporalEnsembleCoeff * static_cast<float>(EnsembleCounts[Step]));
		EnsembleWeights[Step] += Weight;
		++EnsembleCounts[Step];
		const int32 Offset = Step * ActionDim;
		for (int32 Dim = 0; Dim < ActionDim; ++Dim)
		{
			ChunkBuffer[Offset + Dim] += Weight * Chunk[Offset + Dim];
		}
	}
}

void FActionChunker::WriteCurrentAction(TArrayView<float> OutAction) const
{
	if (Settings.ChunkMode == EActionChunkMode::TemporalEnsemble)
	{
		const float Weight = EnsembleWeights[0];
		const float InvWeight = Weight > 0.0f ? 1.0f / Weight : 0.0f;
		for (int32 Dim = 0; Dim < ActionDim; ++Dim)
		{
			OutAction[Dim] = ChunkBuffer[Dim] * InvWeight;
		}
		return;
	}
	const int32 Offset = NextIndex * ActionDim;
	for (int32 Dim = 0; Dim < ActionDim; ++Dim)
	{
		OutAction[Dim] = ChunkBuffer[Offset + Dim];
	}
}
