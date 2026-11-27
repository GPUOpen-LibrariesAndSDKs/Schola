// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "IREEUtils/IREEPointToBufferConverter.h"
#include "Points/DictPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Spaces/DictSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"

void FIREEPointToBufferConverter::operator()(const FIREEDictBuffer& InBuffer)
{
	if (!InputPoint.GetPtr<FDictPoint>() || !Space.GetPtr<FDictSpace>())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDictBuffer&): Point and Space type mismatch for Dict conversion");
		return;
	}

	const FDictPoint& DictPoint = InputPoint.Get<FDictPoint>();
	const FDictSpace& DictSpace = Space.Get<FDictSpace>();
	FIREEDictBuffer&  DictBuffer = OutputBuffer.GetMutable<FIREEDictBuffer>();

	for (const auto& SpacePair : DictSpace.Spaces)
	{
		const FString& Key = SpacePair.Key;
		if (!DictBuffer.Buffers.Contains(Key))
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDictBuffer&): Dict buffer missing key '{0}'", Key);
			continue;
		}

		const TInstancedStruct<FPoint>*		SubPointPtr = DictPoint.Points.Find(Key);
		TInstancedStruct<FIREEPointBuffer>* SubBufferPtr = DictBuffer.Buffers.Find(Key);
		if (!SubPointPtr || !SubBufferPtr || !SubBufferPtr->IsValid())
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDictBuffer&): Invalid sub-point or sub-buffer for key '{0}'", Key);
			continue;
		}

		if (!SubPointPtr->IsValid())
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDictBuffer&): Sub-point for key '{0}' is invalid", Key);
			continue;
		}

		ConvertPointToBuffer(*SubPointPtr, *SubBufferPtr, SpacePair.Value);
	}
}

void FIREEPointToBufferConverter::operator()(const FIREEBoxBuffer& InBuffer)
{
	if (!InputPoint.GetPtr<FBoxPoint>())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEBoxBuffer&): Point type mismatch for Box conversion");
		return;
	}

	const FBoxPoint& BoxPoint = InputPoint.Get<FBoxPoint>();
	FIREEBoxBuffer&	 BoxBuffer = OutputBuffer.GetMutable<FIREEBoxBuffer>();

	if (BoxBuffer.Buffer.Num() != BoxPoint.Values.Num())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEBoxBuffer&): Box buffer size mismatch - buffer={0}, point={1}. Avoiding resize to preserve bindings.",
			BoxBuffer.Buffer.Num(), BoxPoint.Values.Num());
		return;
	}
	if (BoxBuffer.Buffer.Num() > 0)
	{
		FMemory::Memcpy(BoxBuffer.Buffer.GetData(), BoxPoint.Values.GetData(), BoxBuffer.Buffer.Num() * sizeof(float));
	}
}

void FIREEPointToBufferConverter::operator()(const FIREEMultiDiscreteBuffer& InBuffer)
{
	if (!InputPoint.GetPtr<FMultiDiscretePoint>() || !Space.GetPtr<FMultiDiscreteSpace>())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiDiscreteBuffer&): Point and Space type mismatch for MultiDiscrete conversion");
		return;
	}

	const FMultiDiscretePoint& DiscretePoint = InputPoint.Get<FMultiDiscretePoint>();
	FIREEMultiDiscreteBuffer&  DiscreteBuffer = OutputBuffer.GetMutable<FIREEMultiDiscreteBuffer>();
	const FMultiDiscreteSpace& DiscreteSpace = Space.Get<FMultiDiscreteSpace>();

	if (DiscretePoint.Values.Num() != DiscreteSpace.GetNumDimensions())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiDiscreteBuffer&): DiscretePoint dimensions ({0}) don't match DiscreteSpace dimensions ({1})",
			DiscretePoint.Values.Num(), DiscreteSpace.GetNumDimensions());
		return;
	}

	if (DiscreteBuffer.Buffer.Num() != DiscreteSpace.GetNumDimensions())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiDiscreteBuffer&): Buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
			DiscreteBuffer.Buffer.Num(), DiscreteSpace.GetNumDimensions());
		return;
	}

	for (int DimIndex = 0; DimIndex < DiscretePoint.Values.Num(); DimIndex++)
	{
		const int SelectedValue = DiscretePoint.Values[DimIndex];
		const int DimSize = DiscreteSpace.High[DimIndex];
		if (SelectedValue < 0 || SelectedValue >= DimSize)
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiDiscreteBuffer&): MultiDiscrete value out of bounds at dimension {0} - value={1}, valid range=[0, {2})",
				DimIndex, SelectedValue, DimSize);
			return;
		}
	}

	for (int DimIndex = 0; DimIndex < DiscretePoint.Values.Num(); DimIndex++)
	{
		DiscreteBuffer.Buffer[DimIndex] = static_cast<int64>(DiscretePoint.Values[DimIndex]);
	}
}

void FIREEPointToBufferConverter::operator()(const FIREEDiscreteBuffer& InBuffer)
{
	if (!InputPoint.GetPtr<FDiscretePoint>() || !Space.GetPtr<FDiscreteSpace>())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDiscreteBuffer&): Point and Space type mismatch for Discrete conversion");
		return;
	}

	const FDiscretePoint& DiscretePoint = InputPoint.Get<FDiscretePoint>();
	FIREEDiscreteBuffer&  DiscreteBuffer = OutputBuffer.GetMutable<FIREEDiscreteBuffer>();
	const FDiscreteSpace& DiscreteSpace = Space.Get<FDiscreteSpace>();

	if (DiscreteBuffer.Buffer.Num() != DiscreteSpace.GetNumDimensions())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDiscreteBuffer&): Buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
			DiscreteBuffer.Buffer.Num(), DiscreteSpace.GetNumDimensions());
		return;
	}

	const int SelectedValue = DiscretePoint.Value;
	const int DimSize = DiscreteSpace.High;
	if (SelectedValue < 0 || SelectedValue >= DimSize)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEDiscreteBuffer&): Discrete value out of bounds - value={0}, valid range=[0, {1})",
			SelectedValue, DimSize);
		return;
	}

	DiscreteBuffer.Buffer[0] = static_cast<int64>(SelectedValue);
}

void FIREEPointToBufferConverter::operator()(const FIREEMultiBinaryBuffer& InBuffer)
{
	if (!InputPoint.GetPtr<FMultiBinaryPoint>() || !Space.GetPtr<FMultiBinarySpace>())
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiBinaryBuffer&): Point and Space type mismatch for Binary conversion");
		return;
	}

	const FMultiBinaryPoint& BinaryPoint = InputPoint.Get<FMultiBinaryPoint>();
	FIREEMultiBinaryBuffer&	 BinaryBuffer = OutputBuffer.GetMutable<FIREEMultiBinaryBuffer>();
	const FMultiBinarySpace& BinarySpace = Space.Get<FMultiBinarySpace>();
	const int				 ExpectedDims = BinarySpace.GetNumDimensions();

	if (BinaryPoint.Values.Num() != ExpectedDims)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiBinaryBuffer&): BinaryPoint dimensions ({0}) don't match BinarySpace dimensions ({1})",
			BinaryPoint.Values.Num(), ExpectedDims);
		return;
	}

	if (BinaryBuffer.Buffer.Num() != ExpectedDims)
	{
		UE_LOGFMT(LogScholaIREE, Error, "FIREEPointToBufferConverter::operator()(const FIREEMultiBinaryBuffer&): Binary buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
			BinaryBuffer.Buffer.Num(), ExpectedDims);
		return;
	}

	for (int i = 0; i < ExpectedDims; i++)
	{
		BinaryBuffer.Buffer[i] = BinaryPoint.Values[i];
	}
}
