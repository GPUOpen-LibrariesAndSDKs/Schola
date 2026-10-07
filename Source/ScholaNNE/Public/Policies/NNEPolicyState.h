// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "NNEUtils/NNEBuffer.h"
#include "Policies/PolicyStateInterface.h"
#include "NNEPolicyState.generated.h"

/**
 * @brief Recurrent state for an NNE policy, e.g. the hidden and cell tensors of an LSTM.
 *
 * Holds one buffer per state tensor pair in the model (inputs named state_in*, outputs
 * named state_out*), in the order they appear in the model.
 * Created by UNNEPolicy::CreateInitialState.
 */
UCLASS(BlueprintType)
class SCHOLANNE_API UNNEPolicyState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	/** One buffer per recurrent state tensor in the model */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<FNNEStateBuffer> Buffers;

	/** Zeroes every state buffer, matching the state at the start of an episode. */
	void Reset() override
	{
		for (FNNEStateBuffer& Buffer : Buffers)
		{
			Buffer.Reset();
		}
	}

	/**
	 * @brief Copies the buffers of another UNNEPolicyState into this one.
	 * @param[in] Other The state to copy from.
	 * @return True if Other is a UNNEPolicyState, false otherwise.
	 */
	bool CopyFrom(const IPolicyState& Other) override
	{
		const UNNEPolicyState* OtherState = Cast<UNNEPolicyState>(Other._getUObject());
		if (!OtherState)
		{
			return false;
		}
		Buffers = OtherState->Buffers;
		return true;
	}

	/**
	 * @brief Notifies the state that a new model output is ready and drops the oldest entry of each state sequence.
	 *
	 * The policy then writes the model's newest state into the last slot of each buffer.
	 * @return Always true.
	 */
	bool NotifyStateUpdate() override
	{
		for (FNNEStateBuffer& Buffer : Buffers)
		{
			Buffer.Update();
		}
		return true;
	}
};
