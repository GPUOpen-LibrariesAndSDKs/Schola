// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NNESharedPolicy.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "LogScholaNNE.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "Points/BoxPoint.h"
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

/**
 * Reads the model's tensor names and widths from a throwaway instance, then derives the space
 * definition the batch provider needs. The probe is released before this returns.
 */
bool BuildDefinition(UNNEModelData* ModelData, const FString& RuntimeName, FInteractionDefinition& OutDefinition,
	FString& OutError)
{
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
}

bool UNNESharedPolicy::Init()
{
	Context.Reset();
	Backend.Reset();
	Codec.Reset();
	HeadOutputs.Reset();
	ObservationWidth = 0;
	ActionRowWidth = 0;
	ModelCallCount = 0;

	// Failures leave the policy not ready; success is committed only at the end.
	const auto Fail = [](const FString& Message)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNESharedPolicy::Init(): {0}", Message);
		return false;
	};

	if (!ModelData)
	{
		return Fail(TEXT("No model data."));
	}
	if (HeadOutputSuffix.IsEmpty())
	{
		return Fail(TEXT("HeadOutputSuffix must not be empty."));
	}

	FString Error;
	FInteractionDefinition Definition;
	if (!BuildDefinition(ModelData, RuntimeName, Definition, Error))
	{
		return Fail(Error);
	}

	UNNEBatchBackendProvider* Provider = NewObject<UNNEBatchBackendProvider>(this);
	Provider->ModelData = ModelData;
	Provider->RuntimeName = RuntimeName;
	Provider->MaxBatchSize = MaxBatchSize;
	FNPCRowLayout Layout;
	TSharedPtr<IBatchInferenceBackend> NewBackend = Provider->CreateBackend(Definition, Layout, Error);
	if (!NewBackend.IsValid())
	{
		return Fail(Error);
	}
	TOptional<FNPCRowCodec> NewCodec = FNPCRowCodec::Create(Definition, Layout, Error);
	if (!NewCodec.IsSet())
	{
		return Fail(Error);
	}

	TMap<FName, FHeadOutput> NewHeads;
	for (const FNPCRowElement& Output : Layout.Outputs)
	{
		if (!Output.Name.EndsWith(HeadOutputSuffix, ESearchCase::CaseSensitive)
			|| Output.Name.Len() == HeadOutputSuffix.Len())
		{
			continue;
		}
		const FName Head(*Output.Name.LeftChop(HeadOutputSuffix.Len()));
		if (NewHeads.Contains(Head))
		{
			return Fail(FString::Printf(TEXT("Output '%s' maps to head '%s', which another output already uses."),
				*Output.Name, *Head.ToString()));
		}
		FHeadOutput Slice;
		Slice.Offset = Output.Offset;
		Slice.Width = Output.Width;
		NewHeads.Add(Head, Slice);
	}
	if (NewHeads.IsEmpty())
	{
		return Fail(FString::Printf(TEXT("No output name ends with '%s', so the model has no heads."),
			*HeadOutputSuffix));
	}

	TUniquePtr<IBatchInferenceContext> NewContext = NewBackend->CreateContext();
	if (!NewContext.IsValid())
	{
		return Fail(TEXT("The backend could not create an execution context."));
	}

	ObservationWidth = Layout.ObsSize;
	ActionRowWidth = Layout.ActSize;
	HeadOutputs = MoveTemp(NewHeads);
	Codec = MoveTemp(NewCodec);
	Backend = MoveTemp(NewBackend);
	Context = MoveTemp(NewContext);
	return true;
}

void UNNESharedPolicy::GetRegisteredHeads(TSet<FName>& OutHeads) const
{
	OutHeads.Reset();
	for (const TPair<FName, FHeadOutput>& Pair : HeadOutputs)
	{
		OutHeads.Add(Pair.Key);
	}
}

bool UNNESharedPolicy::BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations,
	const TArray<FName>& Heads, TArray<TInstancedStruct<FPoint>>& OutActions)
{
	// Nothing below writes OutActions until the model has run and every row is selected.
	const auto Fail = [](const FString& Message)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNESharedPolicy::BatchedThinkWithHeads(): {0}", Message);
		return false;
	};

	if (!Context.IsValid() || !Codec.IsSet())
	{
		return Fail(TEXT("Policy is not initialized. Call Init() first."));
	}
	const int32 Rows = Observations.Num();
	if (Rows == 0)
	{
		return Fail(TEXT("Empty batch."));
	}
	if (Rows != Heads.Num())
	{
		return Fail(FString::Printf(TEXT("%d observations but %d heads."), Rows, Heads.Num()));
	}
	const int32 Limit = Backend->GetMaxBatchSize();
	if (Limit > 0 && Rows > Limit)
	{
		return Fail(FString::Printf(TEXT("Batch of %d exceeds the limit of %d."), Rows, Limit));
	}
	if (int64(Rows) * FMath::Max(ObservationWidth, ActionRowWidth) > MAX_int32)
	{
		return Fail(FString::Printf(TEXT("Batch of %d rows is too large to stage."), Rows));
	}

	TArray<const FHeadOutput*> RowHeads;
	RowHeads.Reserve(Rows);
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		const FHeadOutput* Head = HeadOutputs.Find(Heads[Row]);
		if (!Head)
		{
			return Fail(FString::Printf(TEXT("Row %d names unknown head '%s'."), Row, *Heads[Row].ToString()));
		}
		RowHeads.Add(Head);
	}

	TArray<float> Input;
	Input.SetNumUninitialized(Rows * ObservationWidth);
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		if (!Codec->Encode(Observations[Row], MakeArrayView(Input).Slice(Row * ObservationWidth, ObservationWidth)))
		{
			return Fail(FString::Printf(
				TEXT("Row %d is not a finite FBoxPoint of %d values."), Row, ObservationWidth));
		}
	}

	TArray<float> Output;
	Output.SetNumUninitialized(Rows * ActionRowWidth);
	++ModelCallCount;
	if (!Context->RunBatch(Input.GetData(), Rows, Output.GetData()))
	{
		return Fail(TEXT("The NNE model call failed or returned invalid tensors."));
	}

	TArray<TInstancedStruct<FPoint>> Actions;
	Actions.Reserve(Rows);
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		const FHeadOutput& Head = *RowHeads[Row];
		Actions.AddDefaulted_GetRef().InitializeAs<FBoxPoint>(
			Output.GetData() + Row * ActionRowWidth + Head.Offset, Head.Width);
	}
	OutActions = MoveTemp(Actions);
	return true;
}
