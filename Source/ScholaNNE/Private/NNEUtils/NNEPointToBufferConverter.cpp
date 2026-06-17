// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "NNEUtils/NNEPointToBufferConverter.h"
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
#include "LogScholaNNE.h"

void FNNEPointToBufferConverter::operator()(const FNNEDictBuffer& InBuffer)
{
    if (!InputPoint.GetPtr<FDictPoint>() || !Space.GetPtr<FDictSpace>())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDictBuffer&): Point and Space type mismatch for Dict conversion");
        return;
    }

    const FDictPoint& DictPoint = InputPoint.Get<FDictPoint>();
    const FDictSpace& DictSpace = Space.Get<FDictSpace>();
    FNNEDictBuffer& DictBuffer = OutputBuffer.GetMutable<FNNEDictBuffer>();

    for (const auto& SpacePair : DictSpace.Spaces)
    {
        const FString& Key = SpacePair.Key;
        if (!DictBuffer.Buffers.Contains(Key))
        {
            // Missing buffer entry; skip but log
            UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDictBuffer&): Dict buffer missing key '{0}'", Key);
            continue;
        }

        const TInstancedStruct<FPoint>* SubPointPtr = DictPoint.Points.Find(Key);
        TInstancedStruct<FNNEPointBuffer>* SubBufferPtr = DictBuffer.Buffers.Find(Key);
        if (!SubPointPtr || !SubBufferPtr || !SubBufferPtr->IsValid())
        {
            UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDictBuffer&): Invalid sub-point or sub-buffer for key '{0}'", Key);
            continue;
        }

        // If sub-point missing, we can't synthesize here; just continue.
        if (!SubPointPtr->IsValid())
        {
            UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDictBuffer&): Sub-point for key '{0}' is invalid", Key);
            continue;
        }

        ConvertPointToBuffer(*SubPointPtr, *SubBufferPtr, SpacePair.Value);
    }
}

void FNNEPointToBufferConverter::operator()(const FNNEBoxBuffer& InBuffer)
{
    if (!InputPoint.GetPtr<FBoxPoint>())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEBoxBuffer&): Point type mismatch for Box conversion");
        return;
    }

    const FBoxPoint& BoxPoint = InputPoint.Get<FBoxPoint>();
    FNNEBoxBuffer& BoxBuffer = OutputBuffer.GetMutable<FNNEBoxBuffer>();

    if (BoxBuffer.Buffer.Num() != BoxPoint.Values.Num())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEBoxBuffer&): Box buffer size mismatch - buffer={0}, point={1}. Avoiding resize to preserve bindings.",
            BoxBuffer.Buffer.Num(), BoxPoint.Values.Num());
        return;
    }
    if (BoxBuffer.Buffer.Num() > 0)
    {
        FMemory::Memcpy(BoxBuffer.Buffer.GetData(), BoxPoint.Values.GetData(), BoxBuffer.Buffer.Num() * sizeof(float));
    }
}

void FNNEPointToBufferConverter::operator()(const FNNEMultiDiscreteBuffer& InBuffer)
{
    if (!InputPoint.GetPtr<FMultiDiscretePoint>() || !Space.GetPtr<FMultiDiscreteSpace>())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiDiscreteBuffer&): Point and Space type mismatch for MultiDiscrete conversion");
        return;
    }

    const FMultiDiscretePoint& DiscretePoint = InputPoint.Get<FMultiDiscretePoint>();
    FNNEMultiDiscreteBuffer& DiscreteBuffer = OutputBuffer.GetMutable<FNNEMultiDiscreteBuffer>();
    const FMultiDiscreteSpace& DiscreteSpace = Space.Get<FMultiDiscreteSpace>();

    if (DiscretePoint.Values.Num() != DiscreteSpace.GetNumDimensions())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiDiscreteBuffer&): DiscretePoint dimensions ({0}) don't match DiscreteSpace dimensions ({1})",
            DiscretePoint.Values.Num(), DiscreteSpace.GetNumDimensions());
        return;
    }

    if (DiscreteBuffer.Buffer.Num() != DiscreteSpace.GetNumDimensions())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiDiscreteBuffer&): Buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
            DiscreteBuffer.Buffer.Num(), DiscreteSpace.GetNumDimensions());
        return;
    }

    for (int DimIndex = 0; DimIndex < DiscretePoint.Values.Num(); DimIndex++)
    {
        const int SelectedValue = DiscretePoint.Values[DimIndex];
        const int DimSize = DiscreteSpace.High[DimIndex];
        if (SelectedValue < 0 || SelectedValue >= DimSize)
        {
            UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiDiscreteBuffer&): MultiDiscrete value out of bounds at dimension {0} - value={1}, valid range=[0, {2})",
                DimIndex, SelectedValue, DimSize);
            return;
        }
    }

    for (int DimIndex = 0; DimIndex < DiscretePoint.Values.Num(); DimIndex++)
    {
        DiscreteBuffer.Buffer[DimIndex] = static_cast<int64>(DiscretePoint.Values[DimIndex]);
    }
}

void FNNEPointToBufferConverter::operator()(const FNNEDiscreteBuffer& InBuffer)
{
    if (!InputPoint.GetPtr<FDiscretePoint>() || !Space.GetPtr<FDiscreteSpace>())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDiscreteBuffer&): Point and Space type mismatch for Discrete conversion");
        return;
    }

    const FDiscretePoint& DiscretePoint = InputPoint.Get<FDiscretePoint>();
    FNNEDiscreteBuffer& DiscreteBuffer = OutputBuffer.GetMutable<FNNEDiscreteBuffer>();
    const FDiscreteSpace& DiscreteSpace = Space.Get<FDiscreteSpace>();

    if (DiscreteBuffer.Buffer.Num() != DiscreteSpace.GetNumDimensions())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDiscreteBuffer&): Buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
            DiscreteBuffer.Buffer.Num(), DiscreteSpace.GetNumDimensions());
        return;
    }

    const int SelectedValue = DiscretePoint.Value;
    const int DimSize = DiscreteSpace.High;
    if (SelectedValue < 0 || SelectedValue >= DimSize)
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEDiscreteBuffer&): Discrete value out of bounds - value={0}, valid range=[0, {1})",
            SelectedValue, DimSize);
        return;
    }

    DiscreteBuffer.Buffer[0] = static_cast<int64>(SelectedValue);
}



void FNNEPointToBufferConverter::operator()(const FNNEMultiBinaryBuffer& InBuffer)
{
    if (!InputPoint.GetPtr<FMultiBinaryPoint>() || !Space.GetPtr<FMultiBinarySpace>())
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiBinaryBuffer&): Point and Space type mismatch for Binary conversion");
        return;
    }

    const FMultiBinaryPoint& BinaryPoint = InputPoint.Get<FMultiBinaryPoint>();
    FNNEMultiBinaryBuffer& BinaryBuffer = OutputBuffer.GetMutable<FNNEMultiBinaryBuffer>();
    const FMultiBinarySpace& BinarySpace = Space.Get<FMultiBinarySpace>();
    const int ExpectedDims = BinarySpace.GetNumDimensions();

    if (BinaryPoint.Values.Num() != ExpectedDims)
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiBinaryBuffer&): BinaryPoint dimensions ({0}) don't match BinarySpace dimensions ({1})",
            BinaryPoint.Values.Num(), ExpectedDims);
        return;
    }

    if (BinaryBuffer.Buffer.Num() != ExpectedDims)
    {
        UE_LOGFMT(LogScholaNNE, Error, "FNNEPointToBufferConverter::operator()(const FNNEMultiBinaryBuffer&): Binary buffer size mismatch - buffer={0}, expected={1}. Avoiding resize to preserve bindings.",
            BinaryBuffer.Buffer.Num(), ExpectedDims);
        return;
    }

    for (int i = 0; i < ExpectedDims; i++)
    {
        BinaryBuffer.Buffer[i] = BinaryPoint.Values[i] ? 1.0f : 0.0f;
    }
}

