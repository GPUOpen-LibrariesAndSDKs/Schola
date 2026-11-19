// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NNEPerHeadSharedPolicy.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "LogScholaNNE.h"
#include "Policies/NNESharedPolicyUtils.h"

bool UNNEPerHeadSharedPolicy::Init()
{
	Heads.Reset();
	ObservationWidth = 0;
	ModelCallCount = 0;

	// Failures leave the policy not ready; success is committed only at the end.
	const auto Fail = [](const FString& Message)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPerHeadSharedPolicy::Init(): {0}", Message);
		return false;
	};

	if (HeadModels.IsEmpty())
	{
		return Fail(TEXT("HeadModels is empty. Add one model per head."));
	}

	TMap<FName, FHeadModel> NewHeads;
	int32 NewObservationWidth = 0;
	for (const TPair<FName, TObjectPtr<UNNEModelData>>& Entry : HeadModels)
	{
		const FName Head = Entry.Key;
		const FString HeadLabel = Head.ToString();
		if (Head.IsNone())
		{
			return Fail(TEXT("A head has no name."));
		}
		UNNEModelData* ModelData = Entry.Value;
		if (!ModelData)
		{
			return Fail(FString::Printf(TEXT("Head '%s' has no model."), *HeadLabel));
		}

		FString Error;
		FInteractionDefinition Definition;
		if (!NNESharedPolicyUtils::BuildDefinition(ModelData, RuntimeName, Definition, Error))
		{
			return Fail(FString::Printf(TEXT("Head '%s': %s"), *HeadLabel, *Error));
		}

		UNNEBatchBackendProvider* Provider = NewObject<UNNEBatchBackendProvider>(this);
		Provider->ModelData = ModelData;
		Provider->RuntimeName = RuntimeName;
		Provider->MaxBatchSize = MaxBatchSize;
		FNPCRowLayout Layout;
		TSharedPtr<IBatchInferenceBackend> Backend = Provider->CreateBackend(Definition, Layout, Error);
		if (!Backend.IsValid())
		{
			return Fail(FString::Printf(TEXT("Head '%s': %s"), *HeadLabel, *Error));
		}
		if (Layout.Outputs.Num() != 1)
		{
			return Fail(FString::Printf(
				TEXT("Head '%s': the model has %d outputs, but a per-head model must have exactly one."),
				*HeadLabel, Layout.Outputs.Num()));
		}
		if (NewObservationWidth == 0)
		{
			NewObservationWidth = Layout.ObsSize;
		}
		else if (Layout.ObsSize != NewObservationWidth)
		{
			return Fail(FString::Printf(
				TEXT("Head '%s' reads %d observation values, but another head reads %d. Every head must read the same observation width."),
				*HeadLabel, Layout.ObsSize, NewObservationWidth));
		}

		TOptional<FNPCRowCodec> Codec = FNPCRowCodec::Create(Definition, Layout, Error);
		if (!Codec.IsSet())
		{
			return Fail(FString::Printf(TEXT("Head '%s': %s"), *HeadLabel, *Error));
		}
		TUniquePtr<IBatchInferenceContext> Context = Backend->CreateContext();
		if (!Context.IsValid())
		{
			return Fail(FString::Printf(TEXT("Head '%s': the backend could not create an execution context."), *HeadLabel));
		}

		FHeadModel Model;
		Model.Backend = MoveTemp(Backend);
		Model.Context = MoveTemp(Context);
		Model.Codec = MoveTemp(Codec);
		Model.ActionWidth = Layout.ActSize;
		NewHeads.Add(Head, MoveTemp(Model));
	}

	ObservationWidth = NewObservationWidth;
	Heads = MoveTemp(NewHeads);
	return true;
}

void UNNEPerHeadSharedPolicy::GetRegisteredHeads(TSet<FName>& OutHeads) const
{
	OutHeads.Reset();
	for (const TPair<FName, FHeadModel>& Entry : Heads)
	{
		OutHeads.Add(Entry.Key);
	}
}

bool UNNEPerHeadSharedPolicy::BatchedThinkWithHeads(const TArray<TInstancedStruct<FPoint>>& Observations,
	const TArray<FName>& InHeads, TArray<TInstancedStruct<FPoint>>& OutActions)
{
	UE_LOGFMT(LogScholaNNE, Error, "UNNEPerHeadSharedPolicy::BatchedThinkWithHeads(): Not implemented yet.");
	return false;
}
