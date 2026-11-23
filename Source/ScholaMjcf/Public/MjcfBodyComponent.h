// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Components/PrimitiveComponent.h"
#include "CoreMinimal.h"

#include "MjcfBodyComponent.generated.h"

/**
 * One MuJoCo body in a Blueprint generated from an MJCF file. The body's geoms are attached to it as child components.
 *
 * It derives from UPrimitiveComponent so that it can later hold the body's collision shapes and simulate as one rigid body;
 * for now it draws nothing and has no collision.
 */
UCLASS(ClassGroup = (Schola), meta = (BlueprintSpawnableComponent))
class SCHOLAMJCF_API UMjcfBodyComponent : public UPrimitiveComponent
{
	GENERATED_BODY()

public:
	UMjcfBodyComponent();

	/** The body's name in the MJCF file. Empty for unnamed bodies and for the world body. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "MuJoCo")
	FString BodyName;
};
