// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Steppers/SharedPolicyStepper.h"
#include "Agent/AgentInterface.h"
#include "LogScholaInferenceUtils.h"

bool USharedPolicyStepper::Init(const TArray<FSharedNPCBinding>& InBindings,
	const TScriptInterface<ISharedPolicy>& InPolicy)
{
	NPCBindings.Reset();
	Policy = nullptr;

	ISharedPolicy* Shared = InPolicy.GetInterface();
	if (!InPolicy.GetObject() || !Shared)
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error,
			"USharedPolicyStepper::Init(): Initialized with no native ISharedPolicy!");
		return false;
	}

	TSet<FName> RegisteredHeads;
	Shared->GetRegisteredHeads(RegisteredHeads);
	FString Error;
	if (!SharedPolicyUtils::ValidateNPCBindings(InBindings, RegisteredHeads, &Error))
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "USharedPolicyStepper::Init(): {0}", Error);
		return false;
	}

	NPCBindings = InBindings;
	Policy = InPolicy;
	return true;
}

void USharedPolicyStepper::Step()
{
	if (NPCBindings.Num() == 0)
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "USharedPolicyStepper::Step(): No NPC bindings to step!");
		return;
	}

	ISharedPolicy* Shared = Policy.GetInterface();
	if (!Policy.GetObject() || !Shared)
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "USharedPolicyStepper::Step(): No policy to use for stepping!");
		return;
	}

	TArray<TInstancedStruct<FPoint>> Observations;
	TArray<FName> Heads;
	Observations.Reserve(NPCBindings.Num());
	Heads.Reserve(NPCBindings.Num());

	for (const FSharedNPCBinding& Binding : NPCBindings)
	{
		UObject* AgentObject = Binding.Agent.GetObject();
		if (!IsValid(AgentObject))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USharedPolicyStepper::Step(): Binding has no valid agent!");
			return;
		}

		FName Head;
		if (!SharedPolicyUtils::FindHeadForAgent(NPCBindings, Binding.Agent, Head))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error,
				"USharedPolicyStepper::Step(): No head for agent '{0}'!", AgentObject->GetName());
			return;
		}

		TInstancedStruct<FPoint> Observation;
		IAgent::Execute_Observe(AgentObject, Observation);
		Observations.Add(MoveTemp(Observation));
		Heads.Add(Head);
	}

	TArray<TInstancedStruct<FPoint>> Actions;
	if (!Shared->BatchedThinkWithHeads(Observations, Heads, Actions))
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error, "USharedPolicyStepper::Step(): Policy failed to think!");
		return;
	}
	if (Actions.Num() != NPCBindings.Num())
	{
		UE_LOGFMT(LogScholaInferenceUtils, Error,
			"USharedPolicyStepper::Step(): Number of actions ({0}) does not match number of agents ({1})!",
			Actions.Num(), NPCBindings.Num());
		return;
	}

	for (int32 Index = 0; Index < NPCBindings.Num(); ++Index)
	{
		UObject* AgentObject = NPCBindings[Index].Agent.GetObject();
		if (IsValid(AgentObject))
		{
			IAgent::Execute_Act(AgentObject, Actions[Index]);
		}
	}
}
