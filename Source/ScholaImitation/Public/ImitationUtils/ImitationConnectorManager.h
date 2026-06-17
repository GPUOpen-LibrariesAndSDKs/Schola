// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ImitationConnectorManager.generated.h"

class UAbstractImitationConnector;

/**
 * @brief Actor-based container for imitation connectors in the Unreal Engine world. Provides a simple way to add an imitation connector to your level.
 * @details This actor manages the lifecycle of an imitation connector, handling initialization and per-frame updates.
 * Place this actor in your level to manage imitation learning connections. It will automatically collect environments from the connector and step the connector every frame.
 * To handle more complex imitation learning scenarios, you can implement your own AActor to manage the imitation connector lifecycle.
 */
UCLASS()
class SCHOLAIMITATION_API AImitationConnectorManager : public AActor
{
	GENERATED_BODY()
	
public:
	/**
	 * @brief Default constructor for the imitation connector manager.
	 */
	AImitationConnectorManager();

	/**
	 * @brief The imitation connector instance managed by this actor.
	 * @details Set this to the type of connector you want to use (e.g., gRPC, Manual).
	 */
	UPROPERTY(EditAnywhere, Instanced, BlueprintReadWrite, Category = "Schola|Imitation")
	UAbstractImitationConnector* Connector = nullptr;

	/**
	 * @brief Called every frame to update the connector state.
	 * @param[in] DeltaTime The time elapsed since the last frame.
	 */
	virtual void Tick(float DeltaTime) override;

protected:
	/**
	 * @brief Called when the game starts or when the actor is spawned.
	 * @details Initializes the connector and starts the imitation learning process.
	 */
	virtual void BeginPlay() override;

};
