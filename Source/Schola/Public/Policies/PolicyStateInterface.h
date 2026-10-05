// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "PolicyStateInterface.generated.h"

/**
 * @brief UInterface wrapper for policy state implementations.
 */
UINTERFACE(BlueprintType, meta = (CannotImplementInterfaceInBlueprint))
class SCHOLA_API UPolicyState : public UInterface
{
	GENERATED_BODY()
};

/**
 * @class IPolicyState
 * @brief Interface for the internal state a policy carries between Think calls.
 *
 * Policy state is any data a policy needs from one step to the next, such as
 * recurrent tensors for an LSTM or the chat history of an LLM-driven agent.
 * The state is owned by the caller (e.g. a stepper or a StateTree node), not by the
 * policy, so a single policy instance can serve many agents, each with its own state.
 *
 * State objects are created by the policy (see IPolicy::CreateInitialState) and must be
 * kept alive by the caller through a UPROPERTY reference. A null state represents a
 * stateless policy.
 *
 * Methods are native-only because Think may run off the game thread, where Blueprint
 * events cannot be executed.
 */
class SCHOLA_API IPolicyState
{
	GENERATED_BODY()

public:
	/**
	 * @brief Resets the state to the value it would have at the start of an episode.
	 */
	virtual void Reset() PURE_VIRTUAL(IPolicyState::Reset, );

	/**
	 * @brief Overwrites this state with the contents of another state of the same type.
	 *
	 * Lets callers keep pre-allocated state objects and copy between them, instead of
	 * creating new UObjects every step (which is unsafe off the game thread).
	 *
	 * @param[in] Other The state to copy from.
	 * @return True if Other is a compatible state type and the copy succeeded, false otherwise.
	 */
	virtual bool CopyFrom(const IPolicyState& Other) PURE_VIRTUAL(IPolicyState::CopyFrom, return false;);

	/**
	 * @brief Converts this state to a string representation.
	 *
	 * The default returns the name of the implementing class. Override to include the state's contents.
	 *
	 * @return A string representation of this state for debugging and logging.
	 */
	virtual FString ToString() const { return _getUObject()->GetClass()->GetName(); }
};
