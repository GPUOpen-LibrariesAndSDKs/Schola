// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

class FMjcfMaterialBuilder;
class UBlueprint;
class UMaterialInterface;
class USCS_Node;
class UStaticMesh;
struct FMjcfBody;
struct FMjcfGeom;
struct FMjcfModel;

/**
 * Builds the Actor Blueprint that assembles a model: one body component per MuJoCo body, nested like the bodies, with the geoms
 * MuJoCo shows by default attached as static mesh components. Primitive geoms use the engine's basic shapes, except planes,
 * which get their own meshes so that textures can repeat across them.
 */
class FMjcfBlueprintBuilder
{
public:
	/**
	 * @param Meshes The imported mesh assets, by MJCF mesh name.
	 * @param Folder Where plane meshes go.
	 * @param UsedNames Asset names already taken in Folder.
	 * @param OutAssets Receives the plane meshes.
	 */
	FMjcfBlueprintBuilder(const FMjcfModel& Model, const TMap<FString, UStaticMesh*>& Meshes, FMjcfMaterialBuilder& Materials, const FString& Folder,
		TSet<FString>& UsedNames, TArray<UObject*>& OutAssets);

	/** Creates the Blueprint Name in Package, or replaces the components of the one already there, and compiles it. */
	UBlueprint* Build(UObject* Package, FName Name);

private:
	void AddBody(const FMjcfBody& Body, USCS_Node* ParentNode);
	void AddGeom(const FMjcfGeom& Geom, const FString& BodyLabel, USCS_Node& BodyNode);
	USCS_Node* AddNode(UClass* Class, const FString& Name, USCS_Node* ParentNode, const FTransform& Transform);
	void AddShape(USCS_Node& ParentNode, const FString& Name, UStaticMesh* Mesh, const FTransform& Transform, UMaterialInterface* Material);

	/** Adds Shape scaled and centered to fill a box HalfExtent centimeters across each way, before Transform. */
	void AddFittedShape(USCS_Node& ParentNode, const FString& Name, UStaticMesh* Shape, const FVector& HalfExtent, const FTransform& Transform,
		UMaterialInterface* Material);

	UStaticMesh* GetPlaneMesh(const FMjcfGeom& Geom, const FString& Label);

	const FMjcfModel& Model;
	const TMap<FString, UStaticMesh*>& Meshes;
	FMjcfMaterialBuilder& Materials;
	FString Folder;
	TSet<FString>& UsedNames;
	TArray<UObject*>& Assets;

	UBlueprint* Blueprint = nullptr;
	UStaticMesh* Cube = nullptr;
	UStaticMesh* Sphere = nullptr;
	UStaticMesh* Cylinder = nullptr;
	TMap<FString, UStaticMesh*> Planes;
	int32 UnnamedBodies = 0;
	TSet<int32> ReportedTypes;
};
