// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ImitationUtils/ImitationConnectorManager.h"
#include "ImitationConnectors/AbstractImitationConnector.h"
#include "Environment/ImitationEnvironmentInterface.h"
#include "Engine/World.h"

// Sets default values
AImitationConnectorManager::AImitationConnectorManager()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
}

// Called when the game starts or when spawned
void AImitationConnectorManager::BeginPlay()
{
	Super::BeginPlay();
	if (Connector)
	{
		TArray<TScriptInterface<IBaseImitationScholaEnvironment>> Environments;
		Connector->CollectEnvironments(Environments);
		Connector->Init(Environments);
	}
}

// Called every frame
void AImitationConnectorManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (Connector)
	{
		Connector->Step();
	}
}
