// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "IREEUtils/IREEBuffer.h"
#include "Policies/PolicyStateInterface.h"
#include "IREEPolicyState.generated.h"

/**
 * @brief Recurrent state for an IREE policy, e.g. the hidden and cell tensors of an LSTM.
 *
 * Holds one buffer per state tensor pair in the model (inputs named state_in*, outputs
 * named state_out*), in the order they appear in the model.
 * Created by UIREEPolicy::CreateInitialState.
 */
UCLASS(BlueprintType)
class SCHOLAIREE_API UIREEPolicyState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	/** One buffer per recurrent state tensor in the model */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	TArray<FIREEStateBuffer> Buffers;

	/** Zeroes every state buffer, matching the state at the start of an episode. */
	void Reset() override
	{
		for (FIREEStateBuffer& Buffer : Buffers)
		{
			Buffer.Reset();
		}
	}

	/**
	 * @brief Copies the buffers of another UIREEPolicyState into this one in place.
	 *
	 * The buffers are never reallocated, since UIREEPolicy binds the model tensors to them.
	 * @param[in] Other The state to copy from.
	 * @return True if Other is a UIREEPolicyState with the same buffer sizes, false otherwise.
	 */
	bool CopyFrom(const IPolicyState& Other) override
	{
		const UIREEPolicyState* OtherState = Cast<UIREEPolicyState>(Other._getUObject());
		if (!OtherState || OtherState->Buffers.Num() != Buffers.Num())
		{
			return false;
		}
		for (int i = 0; i < Buffers.Num(); i++)
		{
			if (OtherState->Buffers[i].StateBuffer.Num() != Buffers[i].StateBuffer.Num())
			{
				return false;
			}
		}
		for (int i = 0; i < Buffers.Num(); i++)
		{
			FMemory::Memcpy(Buffers[i].StateBuffer.GetData(), OtherState->Buffers[i].StateBuffer.GetData(), Buffers[i].StateBuffer.Num() * sizeof(float));
		}
		return true;
	}

	/**
	 * @brief Commits the model's newest state, which it wrote into the pending slot of each buffer.
	 *
	 * Drops the oldest entry of each state sequence and moves the pending state into the
	 * last position. Without a sequence dimension, the pending state replaces the state.
	 * @return Always true.
	 */
	bool NotifyStateUpdate() override
	{
		for (FIREEStateBuffer& Buffer : Buffers)
		{
			Buffer.Update();
		}
		return true;
	}
};
