// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GymConnectorManager.generated.h"

class UAbstractGymConnector;

/**
 * @brief Actor-based container for gym connectors in the Unreal Engine world. Provides a simple way to add a gym connector to your level.
 * @details This actor manages the lifecycle of a gym connector, handling initialization and per-frame updates.
 * Place this actor in your level to manage training connections. It will automatically collect environments from the connector and step the connector every frame.
 * To handle more complex training scenarios, you can implement your own AActor to manage the gym connector lifecycle.
 */
UCLASS()
class SCHOLATRAINING_API AGymConnectorManager : public AActor
{
	GENERATED_BODY()
	
public:
	/**
	 * @brief Default constructor for the gym connector manager.
	 */
	AGymConnectorManager();

	/**
	 * @brief The gym connector instance managed by this actor.
	 * @details Set this to the type of connector you want to use (e.g., gRPC, Manual).
	 */
	UPROPERTY(EditAnywhere, Instanced, BlueprintReadWrite, Category = "Schola|Training")
	UAbstractGymConnector* Connector = nullptr;

	/**
	 * @brief Called every frame to update the connector state.
	 * @param[in] DeltaTime The time elapsed since the last frame.
	 */
	virtual void Tick(float DeltaTime) override;

protected:
	/**
	 * @brief Called when the game starts or when the actor is spawned.
	 * @details Initializes the connector and starts the training process.
	 */
	virtual void BeginPlay() override;

};
