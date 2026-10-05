// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "Policies/PolicyInterface.h"
#include "Policies/PolicyStateInterface.h"
#include "ChatHistoryTestPolicy.generated.h"

/**
 * Test state holding an LLM-style chat history, to exercise non-tensor policy state.
 */
UCLASS()
class UTestChatHistoryState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	UPROPERTY()
	TArray<FString> Messages;

	void Reset() override;
	bool CopyFrom(const IPolicyState& Other) override;
	FString ToString() const override;
};

/**
 * Minimal stateful policy for tests: appends each observation to the chat history
 * and returns the history length as a single-value FBoxPoint action.
 */
UCLASS()
class UChatHistoryTestPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:
	bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InState,
		TInstancedStruct<FPoint>&			  OutAction,
		TScriptInterface<IPolicyState>&		  OutState) override;
	bool CreateInitialState(UObject* InOuter, TScriptInterface<IPolicyState>& OutState) const override;
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;
	bool IsInferenceBusy() const override;
};
