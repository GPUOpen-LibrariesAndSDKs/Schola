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
	const TScriptInterface<IPolicyState>& InOutState,
	TInstancedStruct<FPoint>&			  OutAction)
{
	UTestChatHistoryState* History = Cast<UTestChatHistoryState>(InOutState.GetObject());
	if (!History || !InObservations.IsValid())
	{
		return false;
	}

	const FString NewMessage = InObservations.Get().ToString();
	OutAction.InitializeAs<FBoxPoint>(TArray<float> { static_cast<float>(History->Messages.Num() + 1) });

	if (!History->NotifyStateUpdate())
	{
		return false;
	}

	History->Messages.Add(NewMessage);
	return true;
}

bool UChatHistoryTestPolicy::CreateInitialState(TScriptInterface<IPolicyState>& OutState) const
{
	OutState = NewObject<UTestChatHistoryState>(GetTransientPackage());
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
