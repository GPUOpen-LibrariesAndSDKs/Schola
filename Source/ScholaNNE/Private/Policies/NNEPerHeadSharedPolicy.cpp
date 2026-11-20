// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NNEPerHeadSharedPolicy.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "LogScholaNNE.h"
#include "Points/BoxPoint.h"
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
	// Nothing below writes OutActions until every head's model has run.
	const auto Fail = [](const FString& Message)
	{
		UE_LOGFMT(LogScholaNNE, Error, "UNNEPerHeadSharedPolicy::BatchedThinkWithHeads(): {0}", Message);
		return false;
	};

	if (Heads.IsEmpty())
	{
		return Fail(TEXT("Policy is not initialized. Call Init() first."));
	}
	const int32 Rows = Observations.Num();
	if (Rows == 0)
	{
		return Fail(TEXT("Empty batch."));
	}
	if (Rows != InHeads.Num())
	{
		return Fail(FString::Printf(TEXT("%d observations but %d heads."), Rows, InHeads.Num()));
	}

	/** The rows one head serves in this batch, in their original order. */
	struct FGroup
	{
		FName Head;
		FHeadModel* Model = nullptr;
		TArray<int32> RowIndices;
		TArray<float> Input;
	};
	TArray<FGroup> Groups;
	TMap<FName, int32> GroupIndexByHead;
	for (int32 Row = 0; Row < Rows; ++Row)
	{
		const FName Head = InHeads[Row];
		const int32* GroupIndex = GroupIndexByHead.Find(Head);
		if (!GroupIndex)
		{
			FHeadModel* Model = Heads.Find(Head);
			if (!Model)
			{
				return Fail(FString::Printf(TEXT("Row %d names unknown head '%s'."), Row, *Head.ToString()));
			}
			FGroup& NewGroup = Groups.AddDefaulted_GetRef();
			NewGroup.Head = Head;
			NewGroup.Model = Model;
			GroupIndex = &GroupIndexByHead.Add(Head, Groups.Num() - 1);
		}
		Groups[*GroupIndex].RowIndices.Add(Row);
	}

	for (FGroup& Group : Groups)
	{
		const int32 GroupRows = Group.RowIndices.Num();
		const int32 Limit = Group.Model->Backend->GetMaxBatchSize();
		if (Limit > 0 && GroupRows > Limit)
		{
			return Fail(FString::Printf(TEXT("Head '%s' has %d rows, which exceeds the limit of %d."),
				*Group.Head.ToString(), GroupRows, Limit));
		}
		if (int64(GroupRows) * FMath::Max(ObservationWidth, Group.Model->ActionWidth) > MAX_int32)
		{
			return Fail(FString::Printf(TEXT("Head '%s' has %d rows, too many to stage."), *Group.Head.ToString(),
				GroupRows));
		}
	}

	// Encode every row before running any model, so an invalid row cannot leave some heads already run.
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
	}
	for (FGroup& Group : Groups)
	{
		Group.Input.SetNumUninitialized(Group.RowIndices.Num() * ObservationWidth);
		for (int32 Index = 0; Index < Group.RowIndices.Num(); ++Index)
		{
			const int32 Row = Group.RowIndices[Index];
			if (!Group.Model->Codec->Encode(
					Observations[Row], MakeArrayView(Group.Input).Slice(Index * ObservationWidth, ObservationWidth)))
			{
				return Fail(FString::Printf(TEXT("Row %d has a non-finite value."), Row));
			}
		}
	}

	TArray<TInstancedStruct<FPoint>> Actions;
	Actions.SetNum(Rows);
	TArray<float> Output;
	for (FGroup& Group : Groups)
	{
		const int32 GroupRows = Group.RowIndices.Num();
		const int32 ActionWidth = Group.Model->ActionWidth;
		Output.SetNumUninitialized(GroupRows * ActionWidth);
		++ModelCallCount;
		if (!Group.Model->Context->RunBatch(Group.Input.GetData(), GroupRows, Output.GetData()))
		{
			return Fail(FString::Printf(TEXT("The NNE model call for head '%s' failed or returned invalid tensors."),
				*Group.Head.ToString()));
		}
		for (int32 Index = 0; Index < GroupRows; ++Index)
		{
			Actions[Group.RowIndices[Index]].InitializeAs<FBoxPoint>(
				Output.GetData() + Index * ActionWidth, ActionWidth);
		}
	}
	OutActions = MoveTemp(Actions);
	return true;
}
