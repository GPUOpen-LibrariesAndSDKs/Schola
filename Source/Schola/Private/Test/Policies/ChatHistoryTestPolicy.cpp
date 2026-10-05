// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Test/Policies/ChatHistoryTestPolicy.h"

#include "Points/BoxPoint.h"

void UTestChatHistoryState::Reset()
{
	Messages.Reset();
}

bool UTestChatHistoryState::CopyFrom(const IPolicyState& Other)
{
	const UTestChatHistoryState* OtherState = Cast<UTestChatHistoryState>(Other._getUObject());
	if (!OtherState)
	{
		return false;
	}
	Messages = OtherState->Messages;
	return true;
}

FString UTestChatHistoryState::ToString() const
{
	return FString::Join(Messages, TEXT("\n"));
}

bool UChatHistoryTestPolicy::Think(
	const TInstancedStruct<FPoint>&		  InObservations,
	const TScriptInterface<IPolicyState>& InState,
	TInstancedStruct<FPoint>&			  OutAction,
	TScriptInterface<IPolicyState>&		  OutState)
{
	const UTestChatHistoryState* InHistory = Cast<UTestChatHistoryState>(InState.GetObject());
	UTestChatHistoryState*		 OutHistory = Cast<UTestChatHistoryState>(OutState.GetObject());
	if (!InHistory || !OutHistory || InHistory == OutHistory || !InObservations.IsValid())
	{
		return false;
	}

	OutHistory->CopyFrom(*InHistory);
	OutHistory->Messages.Add(InObservations.Get().ToString());
	OutAction.InitializeAs<FBoxPoint>(TArray<float> { static_cast<float>(OutHistory->Messages.Num()) });
	return true;
}

bool UChatHistoryTestPolicy::CreateInitialState(UObject* InOuter, TScriptInterface<IPolicyState>& OutState) const
{
	OutState = NewObject<UTestChatHistoryState>(InOuter ? InOuter : GetTransientPackage());
	return true;
}

bool UChatHistoryTestPolicy::Init(const FInteractionDefinition& InPolicyDefinition)
{
	return true;
}

bool UChatHistoryTestPolicy::IsInferenceBusy() const
{
	return false;
}
