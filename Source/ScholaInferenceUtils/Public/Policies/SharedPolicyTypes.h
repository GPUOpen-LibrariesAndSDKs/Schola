// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/ScriptInterface.h"
#include "Agent/AgentInterface.h"
#include "SharedPolicyTypes.generated.h"

/**
 * @brief Assigns one NPC agent to one specialized head of a shared policy.
 *
 * Any number of agents may share a head, but each agent may be bound only once; see
 * SharedPolicyUtils::ValidateNPCBindings. Store bindings in a UPROPERTY so that garbage
 * collection tracks the agent reference.
 */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FSharedNPCBinding
{
    GENERATED_BODY()

    FSharedNPCBinding() = default;
    FSharedNPCBinding(const TScriptInterface<IAgent>& InAgent, FName InHead) : Agent(InAgent), Head(InHead) {}

    /** NPC that uses Head. May be a native or a Blueprint implementation of IAgent. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
    TScriptInterface<IAgent> Agent;

    /** Head registered on the shared policy, for example Combat or Support. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|SharedPolicy")
    FName Head = NAME_None;
};

/** Configuration helpers for per-NPC head bindings. They never run inference. */
namespace SharedPolicyUtils
{
/**
 * @brief Checks that bindings are non-empty, that each names a valid agent and a registered head,
 * and that no agent is bound twice.
 *
 * Bindings are checked in order and the first failure is reported. Nothing is logged and the
 * bindings are never modified, so callers decide how to report OutError.
 *
 * @param[in] Bindings NPC-to-head assignments, in batch order.
 * @param[in] RegisteredHeads Heads known to the policy that will serve the bindings.
 * @param[out] OutError If non-null, receives the reason for a failure. Unchanged on success.
 * @return True if every binding is valid.
 */
SCHOLAINFERENCEUTILS_API bool ValidateNPCBindings(const TArray<FSharedNPCBinding>& Bindings,
    const TSet<FName>& RegisteredHeads, FString* OutError = nullptr);

/**
 * @brief Finds the head bound to an agent by comparing object identity.
 *
 * Intended for bindings that passed ValidateNPCBindings. If an agent is bound more than once,
 * the first binding wins.
 *
 * @param[in] Bindings NPC-to-head assignments to search.
 * @param[in] Agent Agent to look up. A null agent never matches.
 * @param[out] OutHead Receives the bound head on success. Unchanged on failure.
 * @return True if Agent has a binding.
 */
SCHOLAINFERENCEUTILS_API bool FindHeadForAgent(const TArray<FSharedNPCBinding>& Bindings,
    const TScriptInterface<IAgent>& Agent, FName& OutHead);
}
