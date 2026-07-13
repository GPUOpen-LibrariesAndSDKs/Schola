// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/FabricaEnvironment.h"

namespace UFabricaRewardInfo
{
	const FString& GetComponentPrefix()
	{
		static FString Prefix(DefaultComponentPrefix);
		return Prefix;
	}

	const FString& GetTaskSuccessInfoPrefix()
	{
		static FString Prefix(DefaultTaskSuccessInfoPrefix);
		return Prefix;
	}
}

namespace
{
FAgentState ConvertFabricaAgentStateToAgentState(const FFabricaAgentState& FabricaState)
{
	FAgentState AgentState;
	AgentState.Observations = FabricaState.Observations;
	AgentState.bTerminated = FabricaState.bTerminated;
	AgentState.bTruncated = FabricaState.bTruncated;
	AgentState.Reward = 0; // Set from the sum of fabrica_r: Info keys after generated reward runs.

	const FString& RewardPrefix = UFabricaRewardInfo::GetComponentPrefix();
	const FString& TaskSuccessPrefix = UFabricaRewardInfo::GetTaskSuccessInfoPrefix();

	// Drop reserved Fabrica keys from user Info; reward components are added later by generated code.
	for (const TPair<FString, FString>& Pair : FabricaState.Info)
	{
		if (!Pair.Key.StartsWith(RewardPrefix) && !Pair.Key.StartsWith(TaskSuccessPrefix))
		{
			AgentState.Info.Add(Pair.Key, Pair.Value);
		}
	}

	AgentState.Info.Add(TaskSuccessPrefix, FString::SanitizeFloat(FabricaState.TaskSuccessMetric));

	return AgentState;
}

float SumFabricaRewardComponents(const TMap<FString, FString>& Info)
{
	const FString& Prefix = UFabricaRewardInfo::GetComponentPrefix();
	float Total = 0.f;
	for (const TPair<FString, FString>& Pair : Info)
	{
		if (Pair.Key.StartsWith(Prefix))
		{
			Total += FCString::Atof(*Pair.Value);
		}
	}
	return Total;
}
} // namespace

AFabricaEnvironment::AFabricaEnvironment()
{
	PrimaryActorTick.bCanEverTick = false;
}

const FString& AFabricaEnvironment::GetFabricaSingleAgentId() const
{
	static const FString DefaultId(TEXT("agent"));
	return DefaultId;
}

void AFabricaEnvironment::BeginPlay()
{
	Super::BeginPlay();
	OnUserBeginPlay();
}

void AFabricaEnvironment::InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
{
	FInteractionDefinition AgentDefinition;
	OnUserInitializeEnvironment(AgentDefinition);
	OutAgentDefinitions.Empty();
	OutAgentDefinitions.Add(GetFabricaSingleAgentId(), AgentDefinition);

	// Tracked actors are resolved exactly once here so generated init is not run
	ClearFabricaTrackedActors();
	FabricaGeneratedInit();
}

void AFabricaEnvironment::SeedEnvironment(int Seed)
{
	OnUserSeedEnvironment(Seed);
}

void AFabricaEnvironment::SetEnvironmentOptions(const TMap<FString, FString>& Options)
{
	OnUserSetEnvironmentOptions(Options);
}

void AFabricaEnvironment::Reset(TMap<FString, FInitialAgentState>& OutAgentState)
{
	FInitialAgentState AgentState;
	OnUserReset(AgentState);
	OutAgentState.Empty();
	OutAgentState.Add(GetFabricaSingleAgentId(), AgentState);
}

void AFabricaEnvironment::Step(const TMap<FString, FInstancedStruct>& InActions, TMap<FString, FAgentState>& OutAgentStates)
{
	const FString& AgentId = GetFabricaSingleAgentId();
	FInstancedStruct Action;
	if (const FInstancedStruct* Found = InActions.Find(AgentId))
	{
		Action = *Found;
	}

	FFabricaAgentState FabricaState;
	OnUserStep(Action, FabricaState);

	TMap<FString, FFabricaAgentState> FabricaStepStates;
	FabricaStepStates.Add(AgentId, FabricaState);

	OutAgentStates.Empty();
	OutAgentStates.Reserve(FabricaStepStates.Num());
	for (const TPair<FString, FFabricaAgentState>& Pair : FabricaStepStates)
	{
		OutAgentStates.Add(Pair.Key, ConvertFabricaAgentStateToAgentState(Pair.Value));
	}

	for (TPair<FString, FAgentState>& Pair : OutAgentStates)
	{
		FabricaGeneratedRewardForAgent(Pair.Key, Pair.Value.Info);
		Pair.Value.Reward = SumFabricaRewardComponents(Pair.Value.Info);
	}
}

void AFabricaEnvironment::ClearFabricaTrackedActors()
{
	FabricaTrackedActors.Empty();
}

void AFabricaEnvironment::FabricaGeneratedInit()
{
	// Default: no-op. Fabrica codegen overrides on concrete subclasses.
}

void AFabricaEnvironment::FabricaGeneratedRewardForAgent(const FString& AgentId, TMap<FString, FString>& RewardComponents)
{
	// Default: no-op. Fabrica codegen overrides on concrete subclasses.
	(void)AgentId;
	(void)RewardComponents;
}

void AFabricaEnvironment::OnUserInitializeEnvironment_Implementation(FInteractionDefinition& OutAgentDefinition)
{
	(void)OutAgentDefinition;
}

void AFabricaEnvironment::OnUserSeedEnvironment_Implementation(int Seed)
{
	(void)Seed;
}

void AFabricaEnvironment::OnUserSetEnvironmentOptions_Implementation(const TMap<FString, FString>& Options)
{
	(void)Options;
}

void AFabricaEnvironment::OnUserReset_Implementation(FInitialAgentState& OutAgentState)
{
	(void)OutAgentState;
}

void AFabricaEnvironment::OnUserStep_Implementation(const FInstancedStruct& InAction, FFabricaAgentState& OutFabricaAgentState)
{
	(void)InAction;
	(void)OutFabricaAgentState;
}

void AFabricaEnvironment::OnUserBeginPlay_Implementation()
{
}
