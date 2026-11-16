// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Policies/TemporalEnsembleActionProcessor.h"
#include "Common/LogSchola.h"

bool FTemporalEnsembleActionProcessor::Init(int32 InChunkSize, int32 InActionDim)
{
	if (!Super::Init(InChunkSize, InActionDim))
	{
		return false;
	}
	WeightedSums.SetNumZeroed(NumChunkElements());
	WeightTotals.SetNumZeroed(ChunkSize);
	PredictionCounts.SetNumZeroed(ChunkSize);
	Reset();
	return true;
}

bool FTemporalEnsembleActionProcessor::Process(const FActionChunk& Chunk, TArrayView<float> OutAction)
{
	const TConstArrayView<float> Coeff = Chunk.ExtraOutputs.IsValidIndex(0) ? Chunk.ExtraOutputs[0] : TConstArrayView<float>();
	if (!IsActionValid(OutAction) || !IsChunkValid(Chunk))
	{
		return false;
	}
	if (Coeff.Num() != 1)
	{
		UE_LOGFMT(LogSchola, Error, "FTemporalEnsembleActionProcessor::Process(): Got {0} floats in {1}, expected one", Coeff.Num(), CoeffOutputName);
		return false;
	}

	ShiftEnsemble();
	for (int32 Step = 0; Step < ChunkSize; ++Step)
	{
		const float Weight = FMath::Exp(-Coeff[0] * PredictionCounts[Step]);
		WeightTotals[Step] += Weight;
		++PredictionCounts[Step];
		const int32 Offset = Step * ActionDim;
		for (int32 Dim = 0; Dim < ActionDim; ++Dim)
		{
			WeightedSums[Offset + Dim] += Weight * Chunk.Actions[Offset + Dim];
		}
	}

	const float InvWeight = 1.0f / WeightTotals[0];
	for (int32 Dim = 0; Dim < ActionDim; ++Dim)
	{
		OutAction[Dim] = WeightedSums[Dim] * InvWeight;
	}
	return true;
}

void FTemporalEnsembleActionProcessor::Reset()
{
	if (WeightedSums.Num() > 0)
	{
		FMemory::Memzero(WeightedSums.GetData(), WeightedSums.Num() * sizeof(float));
	}
	if (WeightTotals.Num() > 0)
	{
		FMemory::Memzero(WeightTotals.GetData(), WeightTotals.Num() * sizeof(float));
	}
	if (PredictionCounts.Num() > 0)
	{
		FMemory::Memzero(PredictionCounts.GetData(), PredictionCounts.Num() * sizeof(int32));
	}
}

void FTemporalEnsembleActionProcessor::ShiftEnsemble()
{
	const int32 Last = ChunkSize - 1;
	if (Last > 0)
	{
		FMemory::Memmove(WeightedSums.GetData(), WeightedSums.GetData() + ActionDim, Last * ActionDim * sizeof(float));
		FMemory::Memmove(WeightTotals.GetData(), WeightTotals.GetData() + 1, Last * sizeof(float));
		FMemory::Memmove(PredictionCounts.GetData(), PredictionCounts.GetData() + 1, Last * sizeof(int32));
	}
	FMemory::Memzero(WeightedSums.GetData() + Last * ActionDim, ActionDim * sizeof(float));
	WeightTotals[Last] = 0.0f;
	PredictionCounts[Last] = 0;
}
