// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NNESharedPolicyUtils.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"

namespace
{
using namespace UE::NNE;

/** Accepts only a float32 tensor of shape [batch, fixed width] and reports its width. */
bool ReadFloatRowWidth(const FTensorDesc& Desc, int32& OutWidth, FString& OutError)
{
	const TConstArrayView<int32> Dims = Desc.GetShape().GetData();
	if (Desc.GetDataType() != ENNETensorDataType::Float)
	{
		OutError = FString::Printf(TEXT("Tensor '%s' is not float32."), *Desc.GetName());
		return false;
	}
	if (Dims.Num() != 2 || Dims[1] <= 0)
	{
		OutError = FString::Printf(TEXT("Tensor '%s' must have shape [batch, width] with a fixed width."),
			*Desc.GetName());
		return false;
	}
	OutWidth = Dims[1];
	return true;
}

/** The provider matches tensors to a space; the bounds are irrelevant because rows are unbounded floats. */
TInstancedStruct<FSpace> MakeUnboundedBox(int32 Width)
{
	TArray<float> Low;
	TArray<float> High;
	Low.Init(TNumericLimits<float>::Lowest(), Width);
	High.Init(TNumericLimits<float>::Max(), Width);
	return TInstancedStruct<FSpace>::Make<FBoxSpace>(Low, High);
}
}

namespace NNESharedPolicyUtils
{
bool BuildDefinition(UNNEModelData* ModelData, const FString& RuntimeName, FInteractionDefinition& OutDefinition,
	FString& OutError)
{
	using namespace UE::NNE;

	TWeakInterfacePtr<INNERuntimeCPU> Runtime = GetRuntime<INNERuntimeCPU>(RuntimeName);
	if (!Runtime.IsValid())
	{
		OutError = FString::Printf(TEXT("CPU runtime '%s' is not registered. Enable its plugin."), *RuntimeName);
		return false;
	}
	TSharedPtr<IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
	if (!Model.IsValid())
	{
		OutError = FString::Printf(TEXT("Runtime '%s' cannot create the model."), *RuntimeName);
		return false;
	}
	TSharedPtr<IModelInstanceCPU> Probe = Model->CreateModelInstanceCPU();
	if (!Probe.IsValid())
	{
		OutError = TEXT("The runtime created the model but not a model instance.");
		return false;
	}

	const TConstArrayView<FTensorDesc> Inputs = Probe->GetInputTensorDescs();
	if (Inputs.Num() != 1)
	{
		OutError = FString::Printf(TEXT("Expected exactly one observation input, found %d."), Inputs.Num());
		return false;
	}
	int32 ObservationWidth = 0;
	if (!ReadFloatRowWidth(Inputs[0], ObservationWidth, OutError))
	{
		return false;
	}

	const TConstArrayView<FTensorDesc> Outputs = Probe->GetOutputTensorDescs();
	if (Outputs.IsEmpty())
	{
		OutError = TEXT("The model has no outputs.");
		return false;
	}
	FDictSpace ActionSpace;
	for (const FTensorDesc& Output : Outputs)
	{
		int32 Width = 0;
		if (!ReadFloatRowWidth(Output, Width, OutError))
		{
			return false;
		}
		if (ActionSpace.Spaces.Contains(Output.GetName()))
		{
			OutError = FString::Printf(TEXT("Duplicate output name '%s'."), *Output.GetName());
			return false;
		}
		ActionSpace.Spaces.Add(Output.GetName(), MakeUnboundedBox(Width));
	}

	OutDefinition = FInteractionDefinition(
		MakeUnboundedBox(ObservationWidth), TInstancedStruct<FSpace>::Make<FDictSpace>(ActionSpace));
	return true;
}
} // namespace NNESharedPolicyUtils
