// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
#include "ScholaEditor.h"
#include "EdGraphUtilities.h"

#define LOCTEXT_NAMESPACE "ScholaEditor"

void FScholaEditorModule::StartupModule()
{
	// Agent
	FKismetEditorUtilities::FOnBlueprintCreated AgentCallback;
	/*
	RegisterDefaultFunction(ABlueprintTrainer, AgentCallback, ComputeReward);
	RegisterDefaultFunction(ABlueprintTrainer, AgentCallback, ComputeStatus);
	RegisterDefaultFunction(ABlueprintTrainer, AgentCallback, GetInfo);

	// Static Environment
	FKismetEditorUtilities::FOnBlueprintCreated StaticEnvironmentCallback;
	RegisterDefaultFunction(ABlueprintStaticScholaEnvironment, StaticEnvironmentCallback, RegisterAgents);
	RegisterDefaultEvent(ABlueprintStaticScholaEnvironment, ResetEnvironment);
	RegisterDefaultEvent(ABlueprintStaticScholaEnvironment, InitializeEnvironment);
	RegisterDefaultEvent(ABlueprintStaticScholaEnvironment, SetEnvironmentOptions);
	RegisterDefaultEvent(ABlueprintStaticScholaEnvironment, SeedEnvironment);

	//Dynamic Environment
	FKismetEditorUtilities::FOnBlueprintCreated DynamicEnvironmentCallback;
	RegisterDefaultFunction(ABlueprintDynamicScholaEnvironment, DynamicEnvironmentCallback, RegisterAgents);
	RegisterDefaultEvent(ABlueprintDynamicScholaEnvironment, ResetEnvironment);
	RegisterDefaultEvent(ABlueprintDynamicScholaEnvironment, InitializeEnvironment);
	RegisterDefaultEvent(ABlueprintDynamicScholaEnvironment, SetEnvironmentOptions);
	RegisterDefaultEvent(ABlueprintDynamicScholaEnvironment, SeedEnvironment);

	// Observers
	
	FKismetEditorUtilities::FOnBlueprintCreated DiscreteObserverCallback;
	RegisterDefaultFunction(UBlueprintDiscreteObserver, DiscreteObserverCallback, GetObservationSpace);
	RegisterDefaultFunction(UBlueprintDiscreteObserver, DiscreteObserverCallback, CollectObservations);

	FKismetEditorUtilities::FOnBlueprintCreated BinaryObserverCallback;
	RegisterDefaultFunction(UBlueprintBinaryObserver, BinaryObserverCallback, GetObservationSpace);
	RegisterDefaultFunction(UBlueprintBinaryObserver, BinaryObserverCallback, CollectObservations);

	FKismetEditorUtilities::FOnBlueprintCreated BoxObserverCallback;
	RegisterDefaultFunction(UBlueprintBoxObserver, BoxObserverCallback, GetObservationSpace);
	RegisterDefaultFunction(UBlueprintBoxObserver, BoxObserverCallback, CollectObservations);
	

	//Actuators
	
	FKismetEditorUtilities::FOnBlueprintCreated DiscreteActuatorCallback;
	RegisterDefaultFunction(UBlueprintDiscreteActuator, DiscreteActuatorCallback, GetActionSpace);
	RegisterDefaultFunction(UBlueprintDiscreteActuator, DiscreteActuatorCallback, TakeAction);

	FKismetEditorUtilities::FOnBlueprintCreated BinaryActuatorCallback;
	RegisterDefaultFunction(UBlueprintBinaryActuator, BinaryActuatorCallback, GetActionSpace);
	RegisterDefaultFunction(UBlueprintBinaryActuator, BinaryActuatorCallback, TakeAction);

	FKismetEditorUtilities::FOnBlueprintCreated BoxActuatorCallback;
	RegisterDefaultFunction(UBlueprintBoxActuator, BoxActuatorCallback, GetActionSpace);
	RegisterDefaultFunction(UBlueprintBoxActuator, BoxActuatorCallback, TakeAction);
	*/
	
}

void FScholaEditorModule::ShutdownModule()
{
	FKismetEditorUtilities::UnregisterAutoBlueprintNodeCreation(this);
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FScholaEditorModule, ScholaEditor);