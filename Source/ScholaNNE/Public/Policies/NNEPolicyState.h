// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "NNEUtils/NNEBuffer.h"
#include "Policies/ActionChunker.h"
#include "Policies/PolicyStateInterface.h"
#include "NNEPolicyState.generated.h"

/**
 * @brief Recurrent state for an NNE policy, e.g. the hidden and cell tensors of an LSTM.
 *
 * Holds one buffer per state tensor pair in the model (inputs named state_in*, outputs
 * named state_out*), in the order they appear in the model. For models that predict
 * action chunks, such as ACT, also holds the chunk being played in ActionChunker.
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

	/** Action-chunk queue or temporal ensemble. Not bound to NNE tensors. Empty unless the model predicts action chunks. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Policy Data")
	FActionChunker ActionChunker;

	/** Zeroes every state buffer and the action chunk, matching the state at the start of an episode. */
	void Reset() override
	{
		for (FNNEStateBuffer& Buffer : Buffers)
		{
			Buffer.Reset();
		}
		ActionChunker.Reset();
	}

	/**
	 * @brief Copies the buffers and action chunk of another UNNEPolicyState into this one in place.
	 *
	 * The buffers are never reallocated, since UNNEPolicy binds the model tensors to them.
	 * @param[in] Other The state to copy from.
	 * @return True if Other is a UNNEPolicyState with the same buffer sizes and chunk layout, false otherwise.
	 */
	bool CopyFrom(const IPolicyState& Other) override
	{
		const UNNEPolicyState* OtherState = Cast<UNNEPolicyState>(Other._getUObject());
		if (!OtherState || OtherState->Buffers.Num() != Buffers.Num() || !ActionChunker.HasSameLayout(OtherState->ActionChunker))
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
		return ActionChunker.CopyFrom(OtherState->ActionChunker);
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
		for (FNNEStateBuffer& Buffer : Buffers)
		{
			Buffer.Update();
		}
		return true;
	}
};
