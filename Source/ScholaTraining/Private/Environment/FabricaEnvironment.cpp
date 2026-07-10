// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/FabricaEnvironment.h"

#include "LogScholaTraining.h"

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
	AgentState.Reward = 0; // 0 because this will be filled by a generated reward function

	const FString& RewardPrefix = UFabricaRewardInfo::GetComponentPrefix();
	const FString& TaskSuccessPrefix = UFabricaRewardInfo::GetTaskSuccessInfoPrefix();

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
	TryRunFabricaGeneratedInit(TEXT("BeginPlay"));
}

void AFabricaEnvironment::InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
{
	FInteractionDefinition AgentDefinition;
	OnUserInitializeEnvironment(AgentDefinition);
	OutAgentDefinitions.Empty();
	OutAgentDefinitions.Add(GetFabricaSingleAgentId(), AgentDefinition);
	TryRunFabricaGeneratedInit(TEXT("InitializeEnvironment"));
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
	ClearFabricaTrackedActors();
	FInitialAgentState AgentState;
	OnUserReset(AgentState);
	OutAgentState.Empty();
	OutAgentState.Add(GetFabricaSingleAgentId(), AgentState);
	TryRunFabricaGeneratedInit(TEXT("Reset"));
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
		StripFabricaComponentInfoKeys(Pair.Value);
		FabricaGeneratedRewardForAgent(Pair.Key, Pair.Value);
	}
}

void AFabricaEnvironment::ClearFabricaTrackedActors()
{
	FabricaTrackedActors.Empty();
}

void AFabricaEnvironment::StripFabricaComponentInfoKeys(FAgentState& OutState) const
{
	const FString& Prefix = UFabricaRewardInfo::GetComponentPrefix();
	TArray<FString> KeysToRemove;
	for (const TPair<FString, FString>& InfoPair : OutState.Info)
	{
		if (InfoPair.Key.StartsWith(Prefix))
		{
			KeysToRemove.Add(InfoPair.Key);
		}
	}
	for (const FString& Key : KeysToRemove)
	{
		OutState.Info.Remove(Key);
	}
}

void AFabricaEnvironment::TryRunFabricaGeneratedInit(const TCHAR* DebugReason)
{
	UE_LOG(LogScholaTraining, Verbose, TEXT("AFabricaEnvironment::FabricaGeneratedInit (%s)"), DebugReason);
	FabricaGeneratedInit();
}

void AFabricaEnvironment::FabricaGeneratedInit()
{
	// Default: no-op. Fabrica codegen overrides on concrete subclasses.
}

void AFabricaEnvironment::FabricaGeneratedRewardForAgent(const FString& AgentId, FAgentState& OutState)
{
	// Default: no-op. Fabrica codegen overrides on concrete subclasses.
	(void)AgentId;
	(void)OutState;
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
