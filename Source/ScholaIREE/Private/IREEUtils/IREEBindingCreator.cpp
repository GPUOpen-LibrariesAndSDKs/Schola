// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "IREEUtils/IREEBindingCreator.h"
#include "LogScholaIREE.h"

void FIREEBindingCreator::operator()(const FIREEDictBuffer& InBuffer)
{
	for (; Index < TensorDescs.Num(); ++Index)
	{
		const FString& TensorName = TensorDescs[Index].Name;
		// In typical use, IREE doesn't need to do this to skip state tensors, as we have full metadata.
		// However, this is left here for symmetry with how this works in ONNX, to hopefully avoid confusion.
		if (TensorName.StartsWith(TEXT("state_in")) || TensorName.StartsWith(TEXT("state_out")))
		{
			continue;
		}

		if (!Bindings.IsValidIndex(Index))
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEBindingCreator: binding index {0} out of range", Index);
			bError = true;
			return;
		}

		const TInstancedStruct<FIREEPointBuffer>* PointBuffer = InBuffer.Buffers.Find(TensorName);
		if (PointBuffer && PointBuffer->IsValid())
		{
			PointBuffer->Get<FIREEPointBuffer>().Accept(*this);
		}
		else
		{
			UE_LOGFMT(LogScholaIREE, Error, "FIREEBindingCreator: key {0} not found in dict buffer", TensorName);
			bError = true;
			return;
		}
	}
}

void FIREEBindingCreator::operator()(const FIREEBoxBuffer& InBuffer)
{
	Bindings[Index] = InBuffer.MakeBinding();
}

void FIREEBindingCreator::operator()(const FIREEMultiBinaryBuffer& InBuffer)
{
	Bindings[Index] = InBuffer.MakeBinding();
}

void FIREEBindingCreator::operator()(const FIREEDiscreteBuffer& InBuffer)
{
	Bindings[Index] = InBuffer.MakeBinding();
}

void FIREEBindingCreator::operator()(const FIREEMultiDiscreteBuffer& InBuffer)
{
	Bindings[Index] = InBuffer.MakeBinding();
}

bool FIREEBindingCreator::CreateBindings(
	const TInstancedStruct<FSpace>& InSpace,
	TInstancedStruct<FIREEPointBuffer>& InBuffer,
	TConstArrayView<FScholaIREETensorInfo> InTensorDescs,
	TArray<FIREETensorBinding>& OutBindings)
{
	FIREEBindingCreator Creator(InBuffer, InTensorDescs, OutBindings);
	InBuffer.Get<FIREEPointBuffer>().Accept(Creator);
	return Creator.bError;
}
