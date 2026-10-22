// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/SharedPolicyTypes.h"
#include "UObject/Class.h"

namespace SharedPolicyUtils
{
bool ValidateNPCBindings(const TArray<FSharedNPCBinding>& Bindings, const TSet<FName>& RegisteredHeads,
    FString* OutError)
{
    auto Fail = [OutError](FString&& Message)
    {
        if (OutError) { *OutError = MoveTemp(Message); }
        return false;
    };

    if (Bindings.IsEmpty())
    {
        return Fail(TEXT("NPC bindings are empty; bind at least one agent to a head."));
    }
    TMap<const UObject*, int32> FirstIndexByAgent;
    FirstIndexByAgent.Reserve(Bindings.Num());
    for (int32 Index = 0; Index < Bindings.Num(); ++Index)
    {
        const FSharedNPCBinding& Binding = Bindings[Index];
        const UObject* Agent = Binding.Agent.GetObject();
        if (!IsValid(Agent))
        {
            return Fail(FString::Printf(TEXT("NPC binding [%d] has no valid agent."), Index));
        }
        // Check the class: the native interface pointer is null for Blueprint-implemented agents.
        if (!Agent->Implements<UAgent>())
        {
            return Fail(FString::Printf(TEXT("NPC binding [%d]: '%s' does not implement IAgent."),
                Index, *Agent->GetName()));
        }
        if (Binding.Head.IsNone())
        {
            return Fail(FString::Printf(TEXT("NPC binding [%d]: agent '%s' has no head."), Index, *Agent->GetName()));
        }
        if (!RegisteredHeads.Contains(Binding.Head))
        {
            return Fail(FString::Printf(TEXT("NPC binding [%d]: agent '%s' uses unregistered head '%s'."),
                Index, *Agent->GetName(), *Binding.Head.ToString()));
        }
        if (const int32* FirstIndex = FirstIndexByAgent.Find(Agent))
        {
            return Fail(FString::Printf(
                TEXT("NPC binding [%d]: agent '%s' is already bound at [%d]; an agent may have only one head."),
                Index, *Agent->GetName(), *FirstIndex));
        }
        FirstIndexByAgent.Add(Agent, Index);
    }
    return true;
}

bool FindHeadForAgent(const TArray<FSharedNPCBinding>& Bindings, const TScriptInterface<IAgent>& Agent, FName& OutHead)
{
    const UObject* AgentObject = Agent.GetObject();
    if (!AgentObject)
    {
        return false;
    }
    const FSharedNPCBinding* Binding = Bindings.FindByPredicate(
        [AgentObject](const FSharedNPCBinding& Candidate) { return Candidate.Agent.GetObject() == AgentObject; });
    if (!Binding)
    {
        return false;
    }
    OutHead = Binding->Head;
    return true;
}
}
