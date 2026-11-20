// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NNESharedPolicy.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "LogScholaNNE.h"
#include "Points/BoxPoint.h"
#include "Policies/NNESharedPolicyUtils.h"

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
	if (!NNESharedPolicyUtils::BuildDefinition(ModelData, RuntimeName, Definition, Error))
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
		const FBoxPoint* Box = Observations[Row].GetPtr<FBoxPoint>();
		if (!Box)
		{
			return Fail(FString::Printf(TEXT("Row %d is not a FBoxPoint."), Row));
		}
		if (Box->Values.Num() != ObservationWidth)
		{
			return Fail(FString::Printf(
				TEXT("Row %d has %d values, expected %d."), Row, Box->Values.Num(), ObservationWidth));
		}
		if (!Codec->Encode(Observations[Row], MakeArrayView(Input).Slice(Row * ObservationWidth, ObservationWidth)))
		{
			return Fail(FString::Printf(TEXT("Row %d has a non-finite value."), Row));
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
