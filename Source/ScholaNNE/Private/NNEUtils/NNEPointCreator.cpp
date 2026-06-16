// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "NNEUtils/NNEPointCreator.h"
#include "Points/DictPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Spaces/DictSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "LogScholaNNE.h"


int GetMaxIndex(TConstArrayView<float>& InArray)
{
	float CurrMax = InArray[0];
	int	  Index = 0;
	int	  CurrIndex = 0;
	for (const float& Value : InArray)
	{
		if (Value > CurrMax)
		{
			CurrMax = Value;
			Index = CurrIndex;
		}
		CurrIndex += 1;
	}
	return Index;
}


void FNNEPointCreator::operator()(const FNNEDictBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDictPoint>();
	}

    for(const TPair<FString, TInstancedStruct<FNNEPointBuffer>>& Pair : InBuffer.Buffers)
    {
        const FString& Key = Pair.Key;
		const TInstancedStruct<FNNEPointBuffer>& Buffer = Pair.Value;
        if (this->Space.Get<FDictSpace>().Spaces.Contains(Key))
        {
            // Use const references to avoid unnecessary copies during cleanup
            const TInstancedStruct<FSpace>& SubSpace = this->Space.Get<FDictSpace>().Spaces[Key];
			//This might not exist so we use FindOrAdd
			TInstancedStruct<FPoint>&		SubPoint = this->OutputPoint.GetMutable<FDictPoint>().Points.FindOrAdd(Key);
			FNNEPointCreator::CreatePoint(Buffer, SubPoint, SubSpace);
        }
        else
        {
			UE_LOGFMT(LogScholaNNE, Error, "FNNEPointCreator::operator(): Key {0} not found in Dict Space", Key);
        }
    }
}

void FNNEPointCreator::operator()(const FNNEBoxBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FBoxPoint>();
	}

    this->OutputPoint.GetMutable<FBoxPoint>().Values = InBuffer.Buffer;
}

void FNNEPointCreator::operator()(const FNNEMultiBinaryBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiBinaryPoint>();
	}

	this->OutputPoint.GetMutable<FMultiBinaryPoint>().Values = InBuffer.Buffer;
}

void FNNEPointCreator::operator()(const FNNEDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDiscretePoint>();
	}
	this->OutputPoint.GetMutable<FDiscretePoint>().Value = InBuffer.Buffer[0];
}

void FNNEPointCreator::operator()(const FNNEMultiDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiDiscretePoint>();
	}

	const FMultiDiscreteSpace* DiscreteSpace = this->Space.GetPtr<FMultiDiscreteSpace>();
	if (!DiscreteSpace)
	{
		UE_LOGFMT(LogScholaNNE, Error, "FNNEPointCreator::operator()(const FNNEMultiDiscreteBuffer&): Space type mismatch for MultiDiscrete conversion");
		return;
	}

	if (InBuffer.Buffer.Num() != DiscreteSpace->GetNumDimensions())
	{
		UE_LOGFMT(LogScholaNNE, Error, "FNNEPointCreator::operator()(const FNNEMultiDiscreteBuffer&): Buffer dimensions ({0}) don't match DiscreteSpace dimensions ({1})",
			InBuffer.Buffer.Num(), DiscreteSpace->GetNumDimensions());
		return;
	}

	for (int DimIndex = 0; DimIndex < InBuffer.Buffer.Num(); DimIndex++)
	{
		const int64 Value = InBuffer.Buffer[DimIndex];
		const int DimSize = DiscreteSpace->High[DimIndex];
		if (Value < 0 || Value > static_cast<int64>(MAX_int32) || Value >= DimSize)
		{
			UE_LOGFMT(LogScholaNNE, Error, "FNNEPointCreator::operator()(const FNNEMultiDiscreteBuffer&): MultiDiscrete buffer value out of bounds at dimension {0} - value={1}, valid range=[0, {2})",
				DimIndex, Value, DimSize);
			return;
		}
	}

	FMultiDiscretePoint&	   OutputPointRef = this->OutputPoint.GetMutable<FMultiDiscretePoint>();
	OutputPointRef.Values.Reset(InBuffer.Buffer.Num());
	for (const int64 Value : InBuffer.Buffer)
	{
		OutputPointRef.Values.Add(static_cast<int>(Value));
	}
}

