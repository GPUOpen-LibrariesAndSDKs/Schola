// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "IREEUtils/IREEPointCreator.h"
#include "Points/DictPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Spaces/DictSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "LogScholaIREE.h"

void FIREEPointCreator::operator()(const FIREEDictBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDictPoint>();
	}

	for (const TPair<FString, TInstancedStruct<FIREEPointBuffer>>& Pair : InBuffer.Buffers)
	{
		const FString&							  Key = Pair.Key;
		const TInstancedStruct<FIREEPointBuffer>& Buffer = Pair.Value;
		if (this->Space.Get<FDictSpace>().Spaces.Contains(Key))
		{
			const TInstancedStruct<FSpace>& SubSpace = this->Space.Get<FDictSpace>().Spaces[Key];
			// This might not exist so we use FindOrAdd
			TInstancedStruct<FPoint>& SubPoint = this->OutputPoint.GetMutable<FDictPoint>().Points.FindOrAdd(Key);
			FIREEPointCreator::CreatePoint(Buffer, SubPoint, SubSpace);
		}
		else
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointCreator::operator(): Key {0} not found in Dict Space", Key);
		}
	}
}

void FIREEPointCreator::operator()(const FIREEBoxBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FBoxPoint>();
	}

	this->OutputPoint.GetMutable<FBoxPoint>().Values = InBuffer.Buffer;
}

void FIREEPointCreator::operator()(const FIREEMultiBinaryBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiBinaryPoint>();
	}

	this->OutputPoint.GetMutable<FMultiBinaryPoint>().Values = InBuffer.Buffer;
}

void FIREEPointCreator::operator()(const FIREEDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FDiscretePoint>();
	}
	this->OutputPoint.GetMutable<FDiscretePoint>().Value = InBuffer.Buffer[0];
}

void FIREEPointCreator::operator()(const FIREEMultiDiscreteBuffer& InBuffer)
{
	if (!this->OutputPoint.IsValid())
	{
		this->OutputPoint.InitializeAs<FMultiDiscretePoint>();
	}

	const FMultiDiscreteSpace* DiscreteSpace = this->Space.GetPtr<FMultiDiscreteSpace>();
	if (!DiscreteSpace)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointCreator::operator()(const FIREEMultiDiscreteBuffer&): Space type mismatch for MultiDiscrete conversion");
		return;
	}

	if (InBuffer.Buffer.Num() != DiscreteSpace->GetNumDimensions())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointCreator::operator()(const FIREEMultiDiscreteBuffer&): Buffer dimensions ({0}) don't match DiscreteSpace dimensions ({1})",
			InBuffer.Buffer.Num(), DiscreteSpace->GetNumDimensions());
		return;
	}

	for (int DimIndex = 0; DimIndex < InBuffer.Buffer.Num(); DimIndex++)
	{
		const int64 Value = InBuffer.Buffer[DimIndex];
		const int	DimSize = DiscreteSpace->High[DimIndex];
		if (Value < 0 || Value > static_cast<int64>(MAX_int32) || Value >= DimSize)
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointCreator::operator()(const FIREEMultiDiscreteBuffer&): MultiDiscrete buffer value out of bounds at dimension {0} - value={1}, valid range=[0, {2})",
				DimIndex, Value, DimSize);
			return;
		}
	}

	FMultiDiscretePoint& OutputPointRef = this->OutputPoint.GetMutable<FMultiDiscretePoint>();
	OutputPointRef.Values.Reset(InBuffer.Buffer.Num());
	for (const int64 Value : InBuffer.Buffer)
	{
		OutputPointRef.Values.Add(static_cast<int>(Value));
	}
}
