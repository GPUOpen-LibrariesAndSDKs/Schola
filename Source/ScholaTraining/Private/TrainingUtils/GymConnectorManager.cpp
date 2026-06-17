// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "TrainingUtils/GymConnectorManager.h"
#include "GymConnectors/AbstractGymConnector.h"
#include "Environment/EnvironmentInterface.h"
#include "Engine/World.h"

// Sets default values
AGymConnectorManager::AGymConnectorManager()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
}

// Called when the game starts or when spawned
void AGymConnectorManager::BeginPlay()
{
	Super::BeginPlay();
	if (Connector)
	{
		TArray<TScriptInterface<IBaseScholaEnvironment>> Environments;
		Connector->CollectEnvironments(Environments);
		Connector->Init(Environments);
	}
}

// Called every frame
void AGymConnectorManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (Connector)
	{
		Connector->Step();
	}
}